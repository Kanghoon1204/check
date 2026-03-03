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

# -----------------------------
# 스타일 개선
# -----------------------------
st.markdown("""
<style>
.big-title {text-align:center; font-size:36px; font-weight:700; color:#1f4e79;}
.sub-title {text-align:center; color:gray; margin-bottom:30px;}
.metric-box {
    padding:15px;
    border-radius:12px;
    background-color:#f5f7fa;
    text-align:center;
}
.section-card {
    padding:20px;
    border-radius:14px;
    background-color:white;
    box-shadow:0 4px 10px rgba(0,0,0,0.05);
    margin-bottom:20px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='big-title'>📘 한동인성교육 채점 시스템</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>for 예림 · 하영</div>", unsafe_allow_html=True)

# -----------------------------
# 사이드바 설정
# -----------------------------
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
    "제외할 기본 레이아웃 블록",
    value=default_layout_text,
    height=200
)

apply_layout_removal = st.sidebar.checkbox(
    "레이아웃 제거 적용",
    value=True
)

# -----------------------------
# PDF 추출 함수
# -----------------------------
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

# -----------------------------
# 분석 시작
# -----------------------------
if uploaded_zip and st.button("🚀 분석 시작", use_container_width=True):

    progress = st.progress(0)
    status_text = st.empty()

    documents = []
    file_names = []

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
            st.warning("최소 2개 이상의 PDF가 필요합니다.")
            st.stop()

        raw_texts = []

        # 1단계
        status_text.text("① PDF 텍스트 추출 중...")
        for path in pdf_files:
            raw_texts.append(extract_text(path))
            file_names.append(os.path.basename(path))
        progress.progress(20)

        # 2단계
        status_text.text("② 레이아웃 제거 적용 중...")
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

        progress.progress(50)

        # 3단계
        status_text.text("③ 유사도 분석 중...")
        vectorizer = TfidfVectorizer()
        tfidf = vectorizer.fit_transform(processed_docs)
        sim = cosine_similarity(tfidf)
        progress.progress(90)

    progress.progress(100)
    status_text.text("✅ 분석 완료")

    # -----------------------------
    # 📊 대시보드 요약
    # -----------------------------
    st.markdown("## 📊 채점 요약")

    avg_chars = int(sum(after_counts) / len(after_counts))
    short_count = sum([1 for c in after_counts if c < min_char_limit])

    suspect_pairs = []
    for i in range(len(file_names)):
        for j in range(i + 1, len(file_names)):
            score = sim[i][j] * 100
            if score >= threshold:
                suspect_pairs.append((file_names[i], file_names[j], score))

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("제출 파일 수", len(file_names))
    col2.metric("평균 글자 수", avg_chars)
    col3.metric("글자 수 미달 인원", short_count)
    col4.metric("의심 파일 쌍", len(suspect_pairs))

    # -----------------------------
    # 📏 글자 수 표
    # -----------------------------
    st.markdown("## 📏 글자 수 분석")

    df = pd.DataFrame({
        "파일명": file_names,
        "원본 글자 수": before_counts,
        "레이아웃 제거 후": after_counts
    })

    def highlight_short(val):
        if val < min_char_limit:
            return "background-color:#ffcccc;"
        return ""

    st.dataframe(
        df.style.applymap(highlight_short, subset=["레이아웃 제거 후"]),
        use_container_width=True
    )

    # -----------------------------
    # 🚨 표절 의심 결과
    # -----------------------------
    st.markdown("## 🚨 표절 의심 분석")

    if not suspect_pairs:
        st.success("의심 파일 없음 🎉")
    else:
        for f1, f2, score in suspect_pairs:

            if score >= 90:
                st.error(f"🔴 {f1} ↔ {f2} → {round(score,2)}%")
            elif score >= 75:
                st.warning(f"🟠 {f1} ↔ {f2} → {round(score,2)}%")
            else:
                st.info(f"🟡 {f1} ↔ {f2} → {round(score,2)}%")
