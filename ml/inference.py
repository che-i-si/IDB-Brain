import numpy as np
import pandas as pd
import os, re, json
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple

# ── Settings ──────────────────────────────────────────────
AVAILABLE_TARGETs = {
    "VCC_reduction_1w": "공정 종료 1주일 간 생균 감쇠량",
    "VCC_1w": "공정 종료 1주 후 생균 수",
}

CUTOFFs = {
    "VCC_reduction_1w": 901.4682080924855,
    "VCC_1w": 6000,
}

FeatureTagMap = {
    '2차_접종량': 'Inoculum_1st',
    '중간_접종량': 'Inoculum_2nd',
    '본_접종량': 'Inoculum_main',
    '1차_액량': 'Volume_pre',
    '2차_액량': 'Volume_1st',
    '중간_액량': 'Volume_2nd',
    '배양액량': 'Volume_main',
    '균액량': 'Volume_cc',
    '2차_종료PH': 'Final_pH_1st',
    '중간_종료PH': 'Final_pH_2nd',
    '본배양PH': 'Preinoculation pH',
    '본_시작PH': 'Postinoculation pH',
    '본_종료PH': 'Final_pH_main',
    '2차_DC': 'TCC_1st',
    '중간_DC': 'TCC_2nd',
    'DC': 'TCC_main',
    '배양액균수': 'VCC_main',
    '균액균수': 'VCC_cc',
    '배양압력_avg': 'Process_pressure_avg',
    '배양압력_min': 'Process_pressure_min',
    '배양압력_max': 'Process_pressure_max',
    '펌프_avg': 'Pump_outlet_avg',
    '펌프_min': 'Pump_outlet_min',
    '펌프_max': 'Pump_outlet_max',
    '백프레셔_avg': 'Back_pressure_avg',
    '백프레셔_min': 'Back_pressure_min',
    '백프레셔_max': 'Back_pressure_max',
    '유량_avg': 'Flow_rate_avg',
    '유량_min': 'Flow_rate_min',
    '유량_max': 'Flow_rate_max',
    '분산매량': 'CP_volume',
    'n_discharge': 'Discharge_n',
    'avg_discharge': 'Discharge_avg',
    '소요시간': 'Centrifugation_time',
    '체침균수': 'VCC',
    '생산량': 'Total_production',
    '비스킷량': 'Freeze_dried_cake',
}

REALTIME_VARs = ['배양압력', '펌프', '백프레셔', '유량']

PH_COLUMNs = ['2차_종료PH', '중간_종료PH', '본배양PH', '본_시작PH', '본_종료PH']


# ── Validation ────────────────────────────────────────────
def validate_data(df: pd.DataFrame) -> list:
    """
    업로드 데이터 검증. 오류 목록 반환 (빈 리스트면 통과).
    - 수치형이 아닌 값 검출
    - pH 컬럼: 0~14 범위 초과
    - 나머지 수치 컬럼: 음수 검출
    """
    errors = []

    # feature 컬럼만 검사 대상
    feature_cols = set(FeatureTagMap.keys())
    realtime_cols = set()
    for var in REALTIME_VARs:
        for col in df.columns:
            if re.match(rf"^{var}_\d차$", col):
                realtime_cols.add(col)
    check_cols = [c for c in df.columns if c in feature_cols or c in realtime_cols]

    if not check_cols:
        return errors

    # 1) 수치형 변환 불가 검사
    for col in check_cols:
        for idx, val in df[col].items():
            if pd.isna(val):
                continue
            try:
                float(val)
            except (ValueError, TypeError):
                errors.append({
                    'type': '타입 오류',
                    'row': idx + 1 if isinstance(idx, int) else idx,
                    'column': col,
                    'value': str(val),
                    'reason': '숫자가 아닌 값',
                })

    # 타입 오류가 있으면 범위 검사는 스킵
    if errors:
        return errors

    # 수치 변환
    numeric_df = df[check_cols].apply(pd.to_numeric, errors='coerce')

    # 2) pH 범위 검사 (0~14)
    ph_cols = [c for c in PH_COLUMNs if c in numeric_df.columns]
    for col in ph_cols:
        mask = numeric_df[col].notna() & ((numeric_df[col] < 0) | (numeric_df[col] > 14))
        for idx in mask[mask].index:
            errors.append({
                'type': '범위 초과',
                'row': idx + 1 if isinstance(idx, int) else idx,
                'column': col,
                'value': str(df.at[idx, col]),
                'reason': 'pH는 0~14 범위여야 합니다',
            })

    # 3) 나머지 컬럼 음수 검사
    non_ph_cols = [c for c in check_cols if c not in ph_cols]
    for col in non_ph_cols:
        mask = numeric_df[col].notna() & (numeric_df[col] < 0)
        for idx in mask[mask].index:
            errors.append({
                'type': '범위 초과',
                'row': idx + 1 if isinstance(idx, int) else idx,
                'column': col,
                'value': str(df.at[idx, col]),
                'reason': '음수 값은 허용되지 않습니다',
            })

    return errors


# ── Processing Functions ──────────────────────────────────
def select_columns(df: pd.DataFrame) -> list:
    target_vars = [*FeatureTagMap.keys()]
    data_vars = [*df.columns]
    in_vars = np.intersect1d(target_vars, data_vars).tolist()
    in_vars.extend([
        var for var in data_vars
        if any(bool(re.match(rf"^{rv}_\d차$", var)) for rv in REALTIME_VARs)
    ])
    return in_vars


def process_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    dtype_validity = True
    dtype_msg = ""
    for column in df.columns:
        try:
            df[column] = df[column].astype(float)
        except:
            dtype_validity = False
            for idx, v in df[column].items():
                try:
                    float(v)
                except:
                    dtype_msg += f"\n\t>> {idx}, {column} | invalid value {v}"
    if not dtype_validity:
        raise ValueError("Invalid data type" + dtype_msg)
    return df


def process_valid_range(df: pd.DataFrame) -> pd.DataFrame:
    cols = [col for col in df.columns if col.endswith('PH')]
    df[cols] = df[cols].where((df[cols] >= 0) & (df[cols] <= 14))
    cols = ['n_discharge']
    if all(c in df.columns for c in cols):
        tmp = df[cols].apply(pd.to_numeric, errors='coerce')
        df[cols] = df[cols].where(tmp.notna() & (tmp % 1 == 0))
        df[cols] = df[cols].where((df[cols] >= 0))
    return df


def process_realtime_variable(df: pd.DataFrame) -> pd.DataFrame:
    for var in REALTIME_VARs:
        cols = [col for col in df.columns if col.startswith(var)]
        if not cols:
            continue
        df[var + '_avg'] = df[cols].dropna(axis=0, how='all').mean(axis=1)
        df[var + '_min'] = df[cols].dropna(axis=0, how='all').min(axis=1)
        df[var + '_max'] = df[cols].dropna(axis=0, how='all').max(axis=1)
        df = df.drop(columns=cols)
    return df


def process_drop_sparse_rows(df: pd.DataFrame, cutoff: float, return_drops: bool = False):
    null_ratio = df.isna().sum(axis=1) / (df.shape[1] - 1)
    drop_labels = null_ratio[null_ratio >= cutoff].index.values
    target_cols = ['Discharge_n', 'Discharge_avg', 'Centrifugation_time']
    existing = [c for c in target_cols if c in df.columns]
    if existing:
        null_ratio2 = df[existing].isna().sum(axis=1) / len(existing)
        drop_labels = np.union1d(drop_labels, null_ratio2[null_ratio2 == 1].index.values)
    if return_drops:
        return df.drop(index=drop_labels), drop_labels.tolist()
    return df.drop(index=drop_labels)


def process_null_imputation(df: pd.DataFrame) -> pd.DataFrame:
    defaults = {
        'Flow_rate_avg': 7.7, 'Flow_rate_min': 7.7, 'Flow_rate_max': 7.9,
        'Process_pressure_avg': 0.933333, 'Process_pressure_min': 0.8, 'Process_pressure_max': 1.,
        'Back_pressure_avg': 220., 'Back_pressure_min': 220., 'Back_pressure_max': 220.,
        'Pump_outlet_avg': 220., 'Pump_outlet_min': 220., 'Pump_outlet_max': 220.,
        'Volume_pre': 8000., 'Volume_1st': 100., 'Volume_2nd': 1000.,
        'Volume_main': 2000., 'Volume_cc': 616.005714,
        'Inoculum_1st': 8., 'Inoculum_2nd': 100., 'Inoculum_main': 1000.,
        'TCC_main': 112.047879, 'TCC_1st': 58.40355, 'TCC_2nd': 52.590909,
        'Final_pH_1st': 4.205172, 'Final_pH_2nd': 4.568375,
        'Preinoculation pH': 4.548639, 'Postinoculation pH': 6.590780,
        'Final_pH_main': 4.543191,
        'VCC_main': 96.803468, 'VCC_cc': 3257.982857,
        'CP_volume': 223.405714,
        'Discharge_n': 44.541667, 'Discharge_avg': 14.917143,
        'Centrifugation_time': 3.072343,
        'VCC': 6216.342857, 'Total_production': 236.053143,
        'Freeze_dried_cake': 183.7951515151515,
    }
    for col, val in defaults.items():
        if col in df.columns:
            df[col] = df[col].fillna(val)
    return df


# ── Main Pipeline ─────────────────────────────────────────
def data_processing(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    dups = df.columns[df.columns.duplicated()]
    if len(dups) > 0:
        raise ValueError(f"중복 컬럼 발견: {list(dups)}")

    if 'ID' not in df.columns:
        raise ValueError("`ID` 컬럼이 필요합니다.")
    df = df.set_index(['ID'])
    ALL_SAMPLES = df.index.values.tolist()

    in_columns = select_columns(df)
    df = df[in_columns]
    df = process_dtypes(df)
    df = process_valid_range(df)
    df = process_realtime_variable(df)
    df = df.rename(columns=FeatureTagMap, errors='raise')
    df, dropped = process_drop_sparse_rows(df, cutoff=0.225, return_drops=True)
    df = process_null_imputation(df)

    remaining_nulls = df.isna().sum().sum()
    if remaining_nulls > 0:
        raise ValueError(f"전처리 후에도 결측치 {remaining_nulls}개 남음")

    return df, ALL_SAMPLES


def load_model(target: str, model_dir: str = "./trained_models"):
    """학습된 XGBoost 모델과 스케일러를 로드"""
    booster = xgb.Booster()
    booster.load_model(os.path.join(model_dir, f"{target}_xgb.json"))

    with open(os.path.join(model_dir, f"{target}_sc.json"), "r") as f:
        sc_params = json.load(f)
    sc = StandardScaler()
    sc.mean_ = np.array(sc_params['mean_'])
    sc.scale_ = np.array(sc_params['scale_'])
    sc.feature_names_in_ = np.array(sc_params['feature_names_in_'])

    return booster, sc


def predict(df: pd.DataFrame, target: str, model_dir: str = "./trained_models") -> pd.DataFrame:
    """
    메인 추론 함수.
    Returns: DataFrame with columns ['ID', 'probability']
    """
    if target not in AVAILABLE_TARGETs:
        raise ValueError(f"'{target}'은 유효하지 않습니다. 선택 가능: {list(AVAILABLE_TARGETs.keys())}")

    # 전처리
    data, all_samples = data_processing(df.copy())

    # 모델 로드 & 추론
    booster, sc = load_model(target, model_dir)
    y_prob = booster.predict(xgb.DMatrix(sc.transform(data)))

    # 결과 조립
    result = pd.DataFrame({'ID': all_samples})
    prob_series = pd.Series(y_prob, index=data.index)
    result['probability'] = result['ID'].map(prob_series)
    result['excluded'] = result['probability'].isna()

    return result
