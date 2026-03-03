import streamlit as st
import zipfile
import pdfplumber
import re
import tempfile
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="한동인성교육 채점 시스템",
    page_icon="📘",
    layout="wide"
)

# ----------------------------
# 🎨 UI 스타일 (모바일 최적화 포함)
# ----------------------------
st.markdown("""
<style>
.block-container {
    padding-top: 1rem;
    padding-bottom: 2rem;
}
.big-title {
    text-align:center;
    font-size:28px;
    font-weight:700;
    color:#1f4e79;
}
.sub-title {
    text-align:center;
    color:gray;
    margin-bottom:20px;
}
@media (max-width: 768px) {
    .big-title {font-size:22px;}
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='big-title'>📘 한동인성교육 채점 시스템</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>for 예림 · 하영</div>", unsafe_allow_html=True)

# ----------------------------
# ⚙️ 사이드바 설정
# ----------------------------
st.sidebar.header("⚙️ 채점 설정")

uploaded_zip = st.sidebar.file_uploader("📦 PDF ZIP 업로드", type="zip")
threshold = st.sidebar.slider("표절 의심 기준 (%)", 10, 100, 70)
min_char_limit = st.sidebar.number_input("최소 글자 수 기준", 0, 10000, 800)

st.sidebar.markdown("---")
st.sidebar.subheader("📌 레이아웃 제외 설정")

default_layout_text = """이름과 학번:
작성일자:
강의주제:
1. 강의 내용 요약
2. 질문과 논의사항
3. 강의를 통한 자기 성찰

[작성안내]
위 1번, 2번, 3번 항목에 대하여 총 한글 300단어 이상 혹은 800자 이상으로
작성하시기 바랍니다. 참고를 위하여, 제출한 강의노트는 해당 강의를 하셨던
교수님께 전달될 수 있습니다.
"""

layout_block = st.sidebar.text_area(
    "글자 수 계산 시 제외할 블록 (편집 가능)",
    value=default_layout_text,
    height=200
)

apply_layout_removal = st.sidebar.checkbox("레이아웃 제거 적용", value=True)

# ----------------------------
# 📛 이름 추출 함수 (최종 확정 버전)
# ----------------------------
def extract_name(filename):
    name_only = os.path.splitext(filename)[0]
    match = re.match(r"([^\d]+)", name_only)
    if match:
        return match.group(1).replace(" ", "").strip()
    return name_only

# ----------------------------
# 📄 PDF 텍스트 추출
# ----------------------------
def extract_text(path):
    text = ""
    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text += t + "\n"
    except:
        pass
    return text

# ----------------------------
# 🚀 분석 시작
# ----------------------------
if uploaded_zip and st.button("🚀 분석 시작", use_container_width=True):

    progress = st.progress(0)
    status = st.empty()

    with tempfile.TemporaryDirectory() as tmpdir:

        zip_path = os.path.join(tmpdir, "upload.zip")
        with open(zip_path, "wb") as f:
            f.write(uploaded_zip.read())

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(tmpdir)

        pdf_files = []
        for root, _, files in os.walk(tmpdir):
            for file in files:
                if file.lower().endswith(".pdf"):
                    pdf_files.append(os.path.join(root, file))

        if len(pdf_files) < 2:
            st.warning("⚠️ 최소 2개 이상의 PDF 파일이 필요합니다.")
            st.stop()

        raw_texts = []
        names = []

        # 1단계
        status.text("① PDF 텍스트 추출 중...")
        for path in pdf_files:
            raw_texts.append(extract_text(path))
            names.append(extract_name(os.path.basename(path)))
        progress.progress(30)

        # 2단계
        status.text("② 레이아웃 제거 적용 중...")
        before_counts = []
        after_counts = []
        processed_docs = []

        for text in raw_texts:
            original_clean = re.sub(r"\s+", "", text)
            before_counts.append(len(original_clean))

            if apply_layout_removal:
                text = text.replace(layout_block, "")

            cleaned = re.sub(r"\s+", "", text)
            after_counts.append(len(cleaned))
            processed_docs.append(text)

        progress.progress(60)

        # 3단계
        status.text("③ 유사도 분석 중...")
        vectorizer = TfidfVectorizer()
        tfidf = vectorizer.fit_transform(processed_docs)
        sim = cosine_similarity(tfidf)
        progress.progress(100)

        status.text("✅ 분석 완료")

    # ----------------------------
    # 📊 글자 수 결과
    # ----------------------------
    st.subheader("📊 글자 수 분석")

    df = pd.DataFrame({
        "이름": names,
        "원본 글자 수": before_counts,
        "레이아웃 제거 후": after_counts,
        "감소량": [b - a for b, a in zip(before_counts, after_counts)]
    })

    def highlight_short(val):
        if val < min_char_limit:
            return "background-color:#ffcccc;"
        return ""

    st.dataframe(
        df.style.applymap(highlight_short, subset=["레이아웃 제거 후"]),
        use_container_width=True
    )

    # ----------------------------
    # 🚨 표절 의심 결과
    # ----------------------------
    st.subheader("🚨 표절 의심 분석")

    suspect_rows = []

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            score = sim[i][j] * 100
            if score >= threshold:
                suspect_rows.append([
                    names[i],
                    names[j],
                    round(score, 2)
                ])

    if not suspect_rows:
        st.success("의심 파일 없음 🎉")
    else:
        suspect_df = pd.DataFrame(
            suspect_rows,
            columns=["이름1", "이름2", "유사도(%)"]
        )
        st.dataframe(suspect_df, use_container_width=True)

    # ----------------------------
    # 📥 엑셀 다운로드
    # ----------------------------
    st.subheader("📥 결과 다운로드")

    output_path = os.path.join(tempfile.gettempdir(), "한동인성교육_채점결과.xlsx")

    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name="글자수", index=False)
        if suspect_rows:
            suspect_df.to_excel(writer, sheet_name="표절의심", index=False)

    with open(output_path, "rb") as f:
        st.download_button(
            label="📥 엑셀 다운로드",
            data=f,
            file_name="한동인성교육_채점결과.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
