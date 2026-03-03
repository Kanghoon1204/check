import streamlit as st
import zipfile
import fitz  # PyMuPDF
import re
import tempfile
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------------
# 기본 설정
# -----------------------------------
st.set_page_config(
    page_title="한동인성교육 채점 시스템",
    layout="wide",
)

# -----------------------------------
# 상단 UI
# -----------------------------------
st.markdown(
    """
    <h1 style='text-align: center; color: #1f4e79;'>
    📘 한동인성교육 채점 시스템
    </h1>
    <h4 style='text-align: center; color: gray;'>
    for 예림 · 하영
    </h4>
    <hr>
    """,
    unsafe_allow_html=True
)

st.markdown("### 📦 PDF 폴더 ZIP 업로드")

uploaded_zip = st.file_uploader(
    "PDF 파일들이 들어있는 ZIP 파일을 업로드하세요",
    type="zip"
)

st.markdown("### ❌ 제거할 문구 입력 (줄바꿈으로 구분)")
remove_patterns = st.text_area(
    "정규식 사용 가능",
    height=120
)

threshold = st.slider(
    "🚨 표절 의심 기준 (%)",
    min_value=10,
    max_value=100,
    value=70
)

# -----------------------------------
# PDF 텍스트 추출 함수
# -----------------------------------
def extract_text_from_pdf(file_bytes):
    text = ""
    try:
        pdf = fitz.open(stream=file_bytes, filetype="pdf")
        for page in pdf:
            text += page.get_text()
        pdf.close()
    except Exception:
        pass
    return text


# -----------------------------------
# 분석 실행
# -----------------------------------
if uploaded_zip and st.button("🔍 분석 시작"):

    documents = []
    file_names = []
    char_counts = []
    removal_stats = {}

    patterns = [p.strip() for p in remove_patterns.split("\n") if p.strip()]

    with tempfile.TemporaryDirectory() as tmpdirname:
        zip_path = os.path.join(tmpdirname, "uploaded.zip")

        with open(zip_path, "wb") as f:
            f.write(uploaded_zip.read())

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(tmpdirname)

        pdf_files = []
        for root, _, files in os.walk(tmpdirname):
            for file in files:
                if file.lower().endswith(".pdf"):
                    pdf_files.append(os.path.join(root, file))

        if len(pdf_files) < 2:
            st.warning("최소 2개 이상의 PDF가 필요합니다.")
            st.stop()

        progress = st.progress(0)

        for idx, path in enumerate(pdf_files):
            with open(path, "rb") as f:
                text = extract_text_from_pdf(f.read())

            # 제거 문구 처리
            for pattern in patterns:
                count = len(re.findall(pattern, text))
                removal_stats[pattern] = removal_stats.get(pattern, 0) + count
                text = re.sub(pattern, "", text)

            cleaned_text = re.sub(r"\s+", "", text)
            char_count = len(cleaned_text)

            documents.append(text)
            char_counts.append(char_count)
            file_names.append(os.path.basename(path))

            progress.progress((idx + 1) / len(pdf_files))

    # -----------------------------------
    # 결과 출력
    # -----------------------------------
    st.markdown("---")
    st.header("📊 파일별 글자 수")

    df_counts = pd.DataFrame({
        "파일명": file_names,
        "글자 수": char_counts
    }).sort_values("글자 수", ascending=False)

    st.dataframe(df_counts, use_container_width=True)

    # TF-IDF 유사도 계산
    vectorizer = TfidfVectorizer(
        analyzer="word",
        token_pattern=r"(?u)\b\w+\b"
    )

    tfidf_matrix = vectorizer.fit_transform(documents)
    similarity_matrix = cosine_similarity(tfidf_matrix)

    st.header("📊 파일 간 유사도 (%)")

    sim_df = pd.DataFrame(
        similarity_matrix * 100,
        index=file_names,
        columns=file_names
    )

    def highlight(val):
        if val >= threshold and val < 100:
            return "background-color: #d9534f; color: white;"
        return ""

    styled = sim_df.style.applymap(highlight).format("{:.2f}")
    st.dataframe(styled, use_container_width=True)

    # 의심 쌍 출력
    st.header("🚨 표절 의심 쌍")

    suspicious = []
    for i in range(len(file_names)):
        for j in range(i + 1, len(file_names)):
            score = similarity_matrix[i][j] * 100
            if score >= threshold:
                suspicious.append((file_names[i], file_names[j], round(score, 2)))

    if suspicious:
        for s in suspicious:
            st.error(f"{s[0]}  ↔  {s[1]}  →  {s[2]}%")
    else:
        st.success("의심 파일 없음")

    # 제거 통계
    if removal_stats:
        st.header("❌ 제거된 문구 통계")
        st.table(pd.DataFrame(
            removal_stats.items(),
            columns=["문구", "제거 횟수"]
        ))
