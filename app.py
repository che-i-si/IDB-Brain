import streamlit as st
import pandas as pd
from ml.inference import (
    predict, validate_required_columns, validate_flow_rate,
    validate_dtype, detect_range_warnings, AVAILABLE_TARGETs, CUTOFFs,
)

st.set_page_config(
    page_title="바이오공정 생균 예측",
    page_icon="🔬",
    layout="wide",
)

# ── Header ────────────────────────────────────────────────
st.image("assets/logo_v1.png", width=200)
st.title("바이오공정 생균 예측 시스템")
st.markdown("공정 데이터를 업로드하면 XGBoost 모델이 생균 감쇠 확률을 예측합니다.")

st.divider()

# ── Sidebar: 설정 ─────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ 설정")

    target = st.selectbox(
        "예측 대상",
        options=list(AVAILABLE_TARGETs.keys()),
        format_func=lambda x: f"{x} — {AVAILABLE_TARGETs[x]}",
    )
    st.caption(f"Cutoff: **{CUTOFFs[target]:,.2f}**")

    st.divider()
    st.markdown("**필수 조건**")
    st.markdown("- `ID` 컬럼 포함\n- 한글 컬럼명 사용\n- CSV 또는 Excel 형식")

# ── File Upload ───────────────────────────────────────────
uploaded = st.file_uploader(
    "공정 데이터 파일 업로드",
    type=["csv", "xlsx"],
    help="CSV 또는 Excel 파일을 업로드하세요.",
)

if uploaded is None:
    st.info("📁 파일을 업로드하면 예측이 시작됩니다.")
    st.stop()

# ── Load Data ─────────────────────────────────────────────
try:
    if uploaded.name.endswith(".csv"):
        df = pd.read_csv(uploaded, na_values=[None, " "])
    else:
        df = pd.read_excel(uploaded, na_values=[None, " "])
except Exception as e:
    st.error(f"파일 읽기 실패: {e}")
    st.stop()

st.subheader("📋 업로드된 데이터")
col1, col2 = st.columns(2)
col1.metric("샘플 수", f"{len(df):,}개")
col2.metric("컬럼 수", f"{len(df.columns):,}개")

with st.expander("원본 데이터 미리보기", expanded=False):
    st.dataframe(df.head(10), use_container_width=True)

# ── Validate (차단 검증) ──────────────────────────────────
st.divider()

# 1) 필수 컬럼 검증
try:
    validate_required_columns(df)
except ValueError as e:
    st.error(f"🚫 {e}")
    st.stop()

# 2) 유량 컬럼 검증
try:
    validate_flow_rate(df)
except ValueError as e:
    st.error(f"🚫 {e}")
    st.stop()

# 3) 타입 오류 검증 (숫자가 아닌 값)
dtype_errors = validate_dtype(df)
if dtype_errors:
    st.subheader("🚫 데이터 타입 오류")
    st.error(f"총 {len(dtype_errors)}건의 타입 오류가 발견되어 예측을 실행할 수 없습니다. 숫자가 아닌 값을 수정 후 다시 업로드하세요.")
    error_df = pd.DataFrame(dtype_errors)
    error_df.columns = ['오류 유형', '행', '컬럼', '값', '사유']
    st.dataframe(error_df, use_container_width=True, hide_index=True)
    st.stop()

# 4) 범위 초과 경고 (차단 안 함, NaN 처리 안내)
range_warnings = detect_range_warnings(df)
if range_warnings:
    with st.expander(f"⚠️ 범위 초과 값 {len(range_warnings)}건 (NaN 처리 후 추론 진행)", expanded=False):
        warn_df = pd.DataFrame(range_warnings)
        warn_df.columns = ['유형', '행', '컬럼', '값', '처리']
        st.dataframe(warn_df, use_container_width=True, hide_index=True)

# ── Predict ───────────────────────────────────────────────
if st.button("🚀 예측 실행", type="primary", use_container_width=True):
    try:
        with st.spinner("전처리 및 모델 추론 중..."):
            result = predict(df, target)

        st.subheader("📊 예측 결과")

        predicted = result[~result['excluded']]
        excluded = result[result['excluded']]

        c1, c2, c3 = st.columns(3)
        c1.metric("전체 샘플", f"{len(result)}개")
        c2.metric("예측 완료", f"{len(predicted)}개")
        c3.metric("제외됨 (결측)", f"{len(excluded)}개")

        if not predicted.empty:
            display = predicted[['ID', 'probability']].copy()
            display.columns = ['ID', 'Probability']
            st.dataframe(
                display.style.format({display.columns[1]: '{:.4f}'}),
                use_container_width=True,
                hide_index=True,
            )

            # 다운로드
            csv_str = display.to_csv(index=False)
            csv_data = ("\ufeff" + csv_str).encode("utf-8")
            st.download_button(
                label="📥 결과 CSV 다운로드",
                data=csv_data,
                file_name=f"result_{target}.csv",
                mime="text/csv",
            )

        if not excluded.empty:
            with st.expander(f"⚠️ 제외된 샘플 ({len(excluded)}개)", expanded=False):
                st.caption("결측률이 높거나 필수 변수가 누락되어 예측에서 제외된 샘플입니다.")
                st.dataframe(excluded[['ID']], use_container_width=True, hide_index=True)

    except ValueError as e:
        st.error(f"❌ 데이터 오류: {e}")
    except FileNotFoundError:
        st.error(f"❌ 모델 파일을 찾을 수 없습니다. `trained_models/{target}_xgb.json` 경로를 확인하세요.")
    except Exception as e:
        st.error(f"❌ 예측 실패: {e}")
