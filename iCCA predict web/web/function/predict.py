# netlify/functions/predict.py
import json
import os
import traceback
import numpy as np
import pandas as pd
import joblib


FUNCTION_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(FUNCTION_DIR, 'models')
PREPROCESSOR_DIR = os.path.join(FUNCTION_DIR, 'preprocessors')


os_model = None
os_preprocessor = None
dfs_model = None
dfs_preprocessor = None


try:
    os_model = joblib.load(os.path.join(MODEL_DIR, 'os_ssvm_selected.joblib'))
    os_preprocessor = joblib.load(os.path.join(PREPROCESSOR_DIR, 'os_ssvm_selected.joblib'))
    dfs_model = joblib.load(os.path.join(MODEL_DIR, 'dfs_ssvm_selected.joblib'))
    dfs_preprocessor = joblib.load(os.path.join(PREPROCESSOR_DIR, 'os_ssvm_selected.joblib'))  # 假设DFS预处理器同OS
except Exception as e:
    traceback.print_exc()

OS_COLS = [
    'Ascites', 'differentiation grade', 'Lymph node metastasis', 'Nerve invasion',
    'CA199 grade', 'CEA grade', 'GGT grade', 'SII', 'TyG', 'Globulin', 'Neutrophil'
]
DFS_COLS = [
    'PT', 'TyG', 'WBC', 'GGT grade', 'differentiation grade', 'Lymph node metastasis',
    'Nerve invasion', 'MVI', 'CEA grade'
]

try:
    CONTINUOUS_COLS = os_preprocessor.get('continuous_cols', [
        'TG', 'Blood_glucose', 'Neutrophil', 'Platelet', 'Lymphocyte',
        'Globulin', 'PT', 'WBC', 'CEA', 'GGT', 'CA199'
    ])
    CATEGORICAL_COLS = os_preprocessor.get('categorical_cols', [
        'Ascites', 'differentiation grade', 'Lymph node metastasis',
        'Nerve invasion', 'MVI'
    ])
except:
    CONTINUOUS_COLS = ['TG', 'Blood_glucose', 'Neutrophil', 'Platelet', 'Lymphocyte',
                       'Globulin', 'PT', 'WBC', 'CEA', 'GGT', 'CA199']
    CATEGORICAL_COLS = ['Ascites', 'differentiation grade', 'Lymph node metastasis',
                        'Nerve invasion', 'MVI']

STANDARD_CONFIG = {
    'os': {
        'min': -0.3148108,
        'max': 1.654315,
        'cutoffs': [0.42, 0.58],
        'group_mean': {'Low': 35.8, 'Medium': 21.2, 'High': 12.8}
    },
    'dfs': {
        'min': -0.2574518,
        'max': 1.43266,
        'cutoffs': [0.30, 0.56],
        'group_mean': {'Low': 28.4, 'Medium': 16.7, 'High': 7.1}
    }
}

def standardize_risk_score(raw_score, mode='os'):
    config = STANDARD_CONFIG[mode]
    standardized_score = (raw_score - config['min']) / (config['max'] - config['min'])
    standardized_score = np.clip(standardized_score, 0, 1)
    return round(standardized_score, 4)

def risk_score_to_group(standardized_score, mode='os'):
    config = STANDARD_CONFIG[mode]
    cut1, cut2 = config['cutoffs']
    if standardized_score < cut1:
        risk_group = 'Low'
    elif standardized_score < cut2:
        risk_group = 'Medium'
    else:
        risk_group = 'High'
    group_mean = config['group_mean'][risk_group]
    return risk_group, group_mean


def predict_ssvm_survival(model, features, mode='os'):
    try:
        raw_risk_score = model.predict(features)[0]
        standardized_score = standardize_risk_score(raw_risk_score, mode)
        risk_group, group_mean = risk_score_to_group(standardized_score, mode)
        return {
            'raw_risk_score': round(float(raw_risk_score), 4),
            'standardized_risk_score': standardized_score,
            'risk_group': risk_group,
            f'{mode}_mean': group_mean
        }
    except Exception as e:
        print(f"❌ {mode.upper()}评分计算失败: {str(e)}")
        raise e


def classify_column(data, column_name, new_column_name, threshold):
    if column_name in data.columns:
        data[column_name] = pd.to_numeric(data[column_name], errors='coerce').fillna(0)
        data[new_column_name] = data[column_name].apply(lambda x: 1 if x > threshold else 0)
    else:
        data[new_column_name] = 0
    return data


def calculate_tyg(data):
    data['TG'] = pd.to_numeric(data['TG'], errors='coerce').fillna(0)
    data['Blood_glucose'] = pd.to_numeric(data['Blood_glucose'], errors='coerce').fillna(0)
    data['TG'] = np.clip(data['TG'], 0.1, 10)
    data['Blood_glucose'] = np.clip(data['Blood_glucose'], 3.0, 20.0)
    data['TyG'] = data.apply(
        lambda row: np.log(row['TG'] * row['Blood_glucose'] / 2)
        if (row['TG'] > 0 and row['Blood_glucose'] > 0)
        else np.log(0.1 * 3 / 2),
        axis=1
    )
    return data


def calculate_sii(data):
    for col in ['Neutrophil', 'Platelet', 'Lymphocyte']:
        data[col] = pd.to_numeric(data[col], errors='coerce').fillna(1)
        data[col] = np.clip(data[col], 0.1, 1000)
    data['SII'] = data.apply(
        lambda row: (row['Neutrophil'] * row['Platelet']) / row['Lymphocyte'],
        axis=1
    )
    data['SII'] = np.clip(data['SII'], 0, 10000)
    return data


def preprocess_data(df, preprocessor, continuous_cols, categorical_cols):
    df = df.copy()
    for col in continuous_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(float)
            df[col] = np.clip(df[col], 0, 1000)
        else:
            df[col] = 0.0

    for col in categorical_cols:
        if col in df.columns and col in preprocessor.get('label_encoders', {}):
            le = preprocessor['label_encoders'][col]
            df[col] = df[col].astype(str)
            df[col] = df[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
            df[col] = le.transform(df[col]).astype(float)
        elif col in df.columns:
            df[col] = 0.0
        else:
            df[col] = 0.0
    if 'scaler' in preprocessor and continuous_cols:
        scaler_features = preprocessor['scaler'].feature_names_in_ if hasattr(preprocessor['scaler'],
                                                                              'feature_names_in_') else []
        cont_cols = [c for c in continuous_cols if c in df.columns and c in scaler_features]
        if cont_cols:
            scaler_input = df[cont_cols].copy()
            for feat in scaler_features:
                if feat not in scaler_input.columns:
                    scaler_input[feat] = 0.0
            scaler_input = scaler_input[scaler_features]
            scaled_data = preprocessor['scaler'].transform(scaler_input)
            scaled_df = pd.DataFrame(scaled_data, columns=scaler_features, index=df.index)
            for col in cont_cols:
                df[col] = scaled_df[col]
    df = df.astype(float)
    return df


def extract_features(data, target_cols):
    name_mapping = {
        'Blood_glucose': 'Blood glucose',
        'differentiation_grade': 'differentiation grade',
        'Lymph_node_metastasis': 'Lymph node metastasis',
        'Nerve_invasion': 'Nerve invasion',
        'CA199_grade': 'CA199 grade',
        'CEA_grade': 'CEA grade',
        'GGT_grade': 'GGT grade',
        'Ascites': 'Ascites',
        'Globulin': 'Globulin',
        'Neutrophil': 'Neutrophil',
        'SII': 'SII',
        'TyG': 'TyG',
        'PT': 'PT',
        'WBC': 'WBC',
        'MVI': 'MVI'
    }
    mapped_data = {}
    for k, v in data.items():
        mapped_key = name_mapping.get(k, k)
        mapped_data[mapped_key] = v
    features = []
    for col in target_cols:
        val = mapped_data.get(col, 0.0)
        features.append(val)
    feat_df = pd.DataFrame([features], columns=target_cols)
    feat_df = feat_df.astype(float)
    return feat_df


# ===================== Netlify 函数入口 =====================
def handler(event, context):
    # 跨域响应头（必须）
    headers = {
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'POST, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type'
    }


    if event['httpMethod'] == 'OPTIONS':
        return {
            'statusCode': 200,
            'headers': headers,
            'body': json.dumps({'message': 'CORS preflight success'})
        }


    if not os_model or not dfs_model:
        return {
            'statusCode': 500,
            'headers': headers,
            'body': json.dumps({'success': False, 'message': '模型加载失败'})
        }

    try:

        if not event.get('body'):
            return {
                'statusCode': 400,
                'headers': headers,
                'body': json.dumps({'success': False, 'message': '未收到有效数据'})
            }

        req_data = json.loads(event['body'])
）
        df = pd.DataFrame([req_data])
        df = classify_column(df, 'CEA', 'CEA grade', 3.4)
        df = classify_column(df, 'GGT', 'GGT grade', 50)
        df = classify_column(df, 'CA199', 'CA199 grade', 37)
        df = calculate_tyg(df)
        df = calculate_sii(df)
        df = df.fillna(0)

        processed_data = df.iloc[0].to_dict()
        os_df = extract_features(processed_data, OS_COLS)
        dfs_df = extract_features(processed_data, DFS_COLS)

        os_features = preprocess_data(os_df, os_preprocessor, CONTINUOUS_COLS, CATEGORICAL_COLS)
        dfs_features = preprocess_data(dfs_df, dfs_preprocessor, CONTINUOUS_COLS, CATEGORICAL_COLS)

        if hasattr(os_model, 'feature_names_in_'):
            os_features = os_features.reindex(columns=os_model.feature_names_in_, fill_value=0.0)
        if hasattr(dfs_model, 'feature_names_in_'):
            dfs_features = dfs_features.reindex(columns=dfs_model.feature_names_in_, fill_value=0.0)

        os_result = predict_ssvm_survival(os_model, os_features, mode='os')
        dfs_result = predict_ssvm_survival(dfs_model, dfs_features, mode='dfs')

        response = {
            'success': True,
            'os_result': os_result,
            'dfs_result': dfs_result,
            'message': 'success!'
        }

        return {
            'statusCode': 200,
            'headers': headers,
            'body': json.dumps(response)
        }

    except Exception as e:
        print(f"❌ fail: {traceback.format_exc()}")
        return {
            'statusCode': 500,
            'headers': headers,
            'body': json.dumps({
                'success': False,
                'message': f'fail：{str(e)}'
            })
        }