import streamlit as st
import zipfile
import pdfplumber
import re
import tempfile
import os
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="한동인성교육 채점 시스템", layout="wide")

st.markdown("""
<h1 style='text-align:center; color:#1f4e79;'>📘 한동인성교육 채점 시스템</h1>
<h4 style='text-align:center; color:gray;'>for 예림 · 하영</h4>
<hr>
""", unsafe_allow_html=True)

uploaded_zip = st.file_uploader("📦 PDF ZIP 업로드", type="zip")

threshold = st.slider("🚨 표절 의심 기준 (%)", 10, 100, 70)
min_char_limit = st.number_input("📏 최소 글자 수 기준", 0, 10000, 1000)
common_ratio = st.slider("📌 공통 문장 감지 비율 (%)", 50, 100, 70)

toggle_remove_common = st.checkbox("공통 문장 자동 제거 적용")

# --------------------------
def extract_text(path):
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
    return text

# --------------------------
if uploaded_zip and st.button("🔍 분석 시작"):

    documents_raw = []
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
                if file.endswith(".pdf"):
                    pdf_files.append(os.path.join(root, file))

        if len(pdf_files) < 2:
            st.warning("최소 2개 이상 필요")
            st.stop()

        for path in pdf_files:
            text = extract_text(path)
            documents_raw.append(text)
            file_names.append(os.path.basename(path))

    # --------------------------
    # 공통 문장 탐지
    # --------------------------
    sentence_lists = []
    for doc in documents_raw:
        sentences = re.split(r"[.!?\n]", doc)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        sentence_lists.append(sentences)

    all_sentences = Counter()
    for sentences in sentence_lists:
        unique = set(sentences)
        for s in unique:
            all_sentences[s] += 1

    common_threshold = len(documents_raw) * (common_ratio / 100)
    common_sentences = {s for s, c in all_sentences.items() if c >= common_threshold}

    # --------------------------
    # 제거 적용 여부
    # --------------------------
    documents = []

    for sentences in sentence_lists:
        if toggle_remove_common:
            filtered = [s for s in sentences if s not in common_sentences]
        else:
            filtered = sentences
        documents.append(" ".join(filtered))

    # --------------------------
    # 글자 수 비교
    # --------------------------
    char_counts = [len(re.sub(r"\s+", "", d)) for d in documents]

    df_counts = pd.DataFrame({
        "파일명": file_names,
        "글자 수": char_counts
    })

    def highlight_short(val):
        if val < min_char_limit:
            return "background-color:#ffcccc;"
        return ""

    st.header("📊 글자 수 (공통 제거 토글 반영)")
    st.dataframe(df_counts.style.applymap(highlight_short, subset=["글자 수"]))

    # --------------------------
    # 유사도 계산 (자기 자신 제외)
    # --------------------------
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(documents)
    sim = cosine_similarity(tfidf)

    st.header("🚨 의심 파일 쌍")

    found = False
    for i in range(len(file_names)):
        for j in range(i + 1, len(file_names)):
            score = sim[i][j] * 100
            if score >= threshold:
                found = True
                st.error(f"{file_names[i]} ↔ {file_names[j]} → {round(score,2)}%")

    if not found:
        st.success("의심 쌍 없음")
