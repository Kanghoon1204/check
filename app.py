import streamlit as st
import os
import pdfplumber
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from io import BytesIO

st.set_page_config(page_title="PDF 표절 분석기", layout="wide")

st.title("📄 PDF 글자 수 & 내부 표절 분석기")

st.markdown("### 1️⃣ 분석 방식 선택")
mode = st.radio("분석 방법 선택", ["📁 폴더 경로 입력", "📤 PDF 파일 업로드"])

st.markdown("### 2️⃣ 제거할 문구 입력")
remove_patterns = st.text_area(
    "줄바꿈으로 구분 (정규식 가능)",
    height=120
)

st.markdown("### 3️⃣ 표절 의심 기준 설정")
threshold = st.slider(
    "유사도 기준 (%)",
    min_value=10,
    max_value=100,
    value=70
)

documents = []
char_counts = []
file_names = []
removal_stats = {}

# ----------------------------
# 📁 폴더 방식
# ----------------------------
if mode == "📁 폴더 경로 입력":
    folder_path = st.text_input("PDF 폴더 경로 입력")

    if st.button("🔍 분석 시작"):
        if not os.path.exists(folder_path):
            st.error("폴더 경로가 올바르지 않습니다.")
            st.stop()

        pdf_files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]

        if not pdf_files:
            st.warning("PDF 파일이 없습니다.")
            st.stop()

        progress = st.progress(0)

        patterns = [p.strip() for p in remove_patterns.split("\n") if p.strip()]

        for idx, file in enumerate(pdf_files):
            full_path = os.path.join(folder_path, file)
            text = ""

            with pdfplumber.open(full_path) as pdf:
                for page in pdf.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted + "\n"

            # 제거 통계 계산
            for pattern in patterns:
                count = len(re.findall(pattern, text))
                removal_stats[pattern] = removal_stats.get(pattern, 0) + count
                text = re.sub(pattern, "", text)

            cleaned_text = re.sub(r"\s+", "", text)
            char_count = len(cleaned_text)

            documents.append(text)
            char_counts.append(char_count)
            file_names.append(file)

            progress.progress((idx + 1) / len(pdf_files))

# ----------------------------
# 📤 업로드 방식 (웹 테스트용)
# ----------------------------
if mode == "📤 PDF 파일 업로드":
    uploaded_files = st.file_uploader(
        "PDF 여러 개 업로드",
        type="pdf",
        accept_multiple_files=True
    )

    if uploaded_files and st.button("🔍 분석 시작"):
        progress = st.progress(0)
        patterns = [p.strip() for p in remove_patterns.split("\n") if p.strip()]

        for idx, file in enumerate(uploaded_files):
            text = ""

            with pdfplumber.open(BytesIO(file.read())) as pdf:
                for page in pdf.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted + "\n"

            for pattern in patterns:
                count = len(re.findall(pattern, text))
                removal_stats[pattern] = removal_stats.get(pattern, 0) + count
                text = re.sub(pattern, "", text)

            cleaned_text = re.sub(r"\s+", "", text)
            char_count = len(cleaned_text)

            documents.append(text)
            char_counts.append(char_count)
            file_names.append(file.name)

            progress.progress((idx + 1) / len(uploaded_files))

# ----------------------------
# 📊 분석 결과 출력
# ----------------------------
if len(documents) > 1:

    st.markdown("---")
    st.header("📊 파일별 글자 수")

    df_counts = pd.DataFrame({
        "파일명": file_names,
        "글자 수": char_counts
    }).sort_values("글자 수", ascending=False)

    st.dataframe(df_counts, use_container_width=True)

    # 유사도 계산 (한국어 최적화)
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
            return "background-color: #ff4b4b; color: white;"
        return ""

    styled_df = sim_df.style.applymap(highlight).format("{:.2f}")

    st.dataframe(styled_df, use_container_width=True)

    # 의심 파일 리스트
    st.header("🚨 표절 의심 쌍")

    suspicious_pairs = []
    for i in range(len(file_names)):
        for j in range(i + 1, len(file_names)):
            if similarity_matrix[i][j] * 100 >= threshold:
                suspicious_pairs.append(
                    (file_names[i], file_names[j],
                     round(similarity_matrix[i][j] * 100, 2))
                )

    if suspicious_pairs:
        for pair in suspicious_pairs:
            st.error(f"{pair[0]}  ↔  {pair[1]}  →  {pair[2]}%")
    else:
        st.success("의심 파일 없음")

    # 제거 통계
    if removal_stats:
        st.header("❌ 제거된 문구 통계")
        st.table(pd.DataFrame(
            removal_stats.items(),
            columns=["문구", "제거 횟수"]
        ))

elif len(documents) == 1:
    st.warning("비교하려면 최소 2개 이상의 PDF가 필요합니다.")
