import streamlit as st
import pandas as pd
import plotly.express as px
from ml.inference import (
    predict, validate_required_columns, validate_flow_rate,
    validate_dtype, detect_range_warnings, AVAILABLE_TARGETs, CUTOFFs,
)

st.set_page_config(
    page_title="바이오공정 생균 예측",
    page_icon="🔬",
    layout="wide",
)

# ── 공통 상수 ──────────────────────────────────────────────
LABEL_COLORS = {
    '출하 승인 대상':    '#2ecc71',
    '조기 관리 대상':    '#f39c12',
    '유통기한 조정 대상': '#e67e22',
    '폐기 검토 대상':    '#e74c3c',
}
LABEL_ORDER = ['출하 승인 대상', '조기 관리 대상', '유통기한 조정 대상', '폐기 검토 대상']
CUTOFF_LINES = [
    (0.80, "dash",    "#2ecc71", "출하 승인 (0.80)"),
    (0.60, "dot",     "#f39c12", "조기 관리 (0.60)"),
    (0.50, "dashdot", "#e67e22", "유통기한 조정 (0.50)"),
]

def assign_label(p):
    if p >= 0.80:   return '출하 승인 대상'
    elif p >= 0.60: return '조기 관리 대상'
    elif p >= 0.50: return '유통기한 조정 대상'
    else:           return '폐기 검토 대상'

# ── PDF 출력 버튼 스타일 주입 ──────────────────────────────
st.markdown("""
<style>
@media print {
    /* 사이드바·헤더·버튼 숨김 */
    [data-testid="stSidebar"],
    [data-testid="stToolbar"],
    [data-testid="stHeader"],
    .stButton, .stDownloadButton,
    [data-testid="stFileUploader"],
    [data-testid="stExpander"] { display: none !important; }

    /* 여백·배경 초기화 */
    .main .block-container { padding: 0 !important; max-width: 100% !important; }
    body { background: white !important; }

    /* 차트 페이지 잘림 방지 */
    .stPlotlyChart { page-break-inside: avoid; }
}
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────
st.image("assets/logo_v1.png", width=200)
st.title("바이오공정 생균 예측 시스템")
st.markdown("공정 데이터를 업로드하면 XGBoost 모델이 기준 충족 확률을 예측합니다.")
st.divider()

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.header("설정")
    target = st.selectbox(
        "예측 대상",
        options=list(AVAILABLE_TARGETs.keys()),
        format_func=lambda x: f"{x} — {AVAILABLE_TARGETs[x]}",
    )
    st.caption(f"📍 Cutoff: **{CUTOFFs[target]:,.2f}**")
    st.divider()
    st.markdown("**데이터 주의 사항**")
    st.markdown("- `ID` 열 포함\n- 필수 변수를 정확한 명칭으로 포함\n- CSV 또는 Excel 형식\n- 더 자세한 내용은 메뉴얼 참고")

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
    df = pd.read_csv(uploaded, na_values=[None, " "]) if uploaded.name.endswith(".csv") \
         else pd.read_excel(uploaded, na_values=[None, " "])
except Exception as e:
    st.error(f"파일 읽기 실패: {e}")
    st.stop()

st.divider()
st.subheader("☑️ 업로드된 데이터")
col1, col2 = st.columns(2)
col1.metric("샘플 수", f"{len(df):,}개")
col2.metric("컬럼 수", f"{len(df.columns):,}개")

with st.expander("원본 데이터 미리보기", expanded=False):
    st.dataframe(df.head(10), use_container_width=True)

# ── Validate ──────────────────────────────────────────────
st.divider()

try:
    validate_required_columns(df)
except ValueError as e:
    st.error(f"🚫 {e}"); st.stop()

try:
    validate_flow_rate(df)
except ValueError as e:
    st.error(f"🚫 {e}"); st.stop()

dtype_errors = validate_dtype(df)
if dtype_errors:
    st.subheader("🚫 데이터 타입 오류")
    st.error(f"총 {len(dtype_errors)}건의 타입 오류가 발견되어 예측을 실행할 수 없습니다. 숫자가 아닌 값을 수정 후 다시 업로드하세요.")
    error_df = pd.DataFrame(dtype_errors)
    error_df.columns = ['오류 유형', '행', '컬럼', '값', '사유']
    st.dataframe(error_df, use_container_width=True, hide_index=True)
    st.stop()

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

        predicted = result[~result['excluded']].copy()
        excluded  = result[result['excluded']]

        # ── 요약 메트릭 ───────────────────────────────────────
        st.subheader("📊 예측 결과")
        c1, c2, c3 = st.columns(3)
        c1.metric("전체 샘플",     f"{len(result)}개")
        c2.metric("예측 완료",     f"{len(predicted)}개")
        c3.metric("제외됨 (결측)", f"{len(excluded)}개")

        if not predicted.empty:
            predicted['Label'] = predicted['probability'].apply(assign_label)
            display = predicted[['ID', 'probability', 'Label']].copy()
            display.columns = ['ID', 'Probability', 'Label']

            # 결과 테이블
            st.dataframe(
                display.style.format({'Probability': '{:.4f}'}),
                use_container_width=True,
                hide_index=True,
            )

            # 다운로드 + PDF 출력 버튼
            csv_data = ("\ufeff" + display.to_csv(index=False)).encode("utf-8")
            st.download_button(
                label="📥 결과 CSV 다운로드",
                data=csv_data,
                file_name=f"result_{target}.csv",
                mime="text/csv",
                use_container_width=True,
            )

            # ── 시각화 (탭 없이 세로 배치) ────────────────────────
            st.divider()
            st.subheader("📈 결과 시각화")

            # 1) 확률 분포 히스토그램 ──────────────────────────────
            st.markdown("##### 예측 확률 분포 히스토그램")
            fig_hist = px.histogram(
                display,
                x="Probability",
                color="Label",
                nbins=20,
                opacity=0.85,
                color_discrete_map=LABEL_COLORS,
                category_orders={"Label": LABEL_ORDER},
                labels={"Probability": "예측 확률", "count": "샘플 수"},
            )
            for cutoff, dash, color, label in CUTOFF_LINES:
                fig_hist.add_vline(
                    x=cutoff, line_dash=dash, line_color=color, line_width=2,
                    annotation_text=label,
                    annotation_position="top right",
                    annotation_font_size=11,
                )
            fig_hist.update_layout(
                bargap=0.05,
                xaxis=dict(range=[0, 1], dtick=0.1, title="예측 확률"),
                yaxis_title="샘플 수",
                legend_title_text="Label",
                height=420,
            )
            st.plotly_chart(fig_hist, use_container_width=True)

            # 2) Label별 파이 + 바 차트 ────────────────────────────
            st.markdown("##### Label별 샘플 분포")
            label_counts = (
                display['Label']
                .value_counts()
                .reindex(LABEL_ORDER, fill_value=0)
                .reset_index()
            )
            label_counts.columns = ['Label', 'Count']
            label_counts['비율(%)'] = (
                label_counts['Count'] / label_counts['Count'].sum() * 100
            ).round(1)

            pie_col, bar_col = st.columns(2)

            with pie_col:
                fig_pie = px.pie(
                    label_counts,
                    names="Label", values="Count",
                    color="Label",
                    color_discrete_map=LABEL_COLORS,
                    hole=0.4,
                )
                fig_pie.update_traces(
                    textinfo="label+percent",
                    pull=[0.03] * len(LABEL_ORDER),
                )
                fig_pie.update_layout(showlegend=False, height=360)
                st.plotly_chart(fig_pie, use_container_width=True)

            with bar_col:
                fig_bar = px.bar(
                    label_counts,
                    x="Label", y="Count",
                    color="Label", text="비율(%)",
                    color_discrete_map=LABEL_COLORS,
                    category_orders={"Label": LABEL_ORDER},
                    labels={"Count": "샘플 수", "Label": ""},
                )
                fig_bar.update_traces(
                    texttemplate="%{text}%",
                    textposition="outside",
                )
                fig_bar.update_layout(
                    showlegend=False,
                    yaxis_title="샘플 수",
                    height=360,
                )
                st.plotly_chart(fig_bar, use_container_width=True)

        # 제외 샘플
        if not excluded.empty:
            with st.expander(f"⚠️ 제외된 샘플 ({len(excluded)}개)", expanded=False):
                st.caption("결측률이 높아 예측에서 제외된 샘플입니다.")
                st.dataframe(excluded[['ID']], use_container_width=True, hide_index=True)

    except ValueError as e:
        st.error(f"❌ 데이터 오류: {e}")
    except FileNotFoundError:
        st.error(f"❌ 모델 파일을 찾을 수 없습니다. `trained_models/{target}_xgb.json` 경로를 확인하세요.")
    except Exception as e:
        st.error(f"❌ 예측 실패: {e}")
