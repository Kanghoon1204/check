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

def extract_text(path):
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
    return text

if uploaded_zip and st.button("🔍 분석 시작"):

    with st.status("🔄 분석 진행 중...", expanded=True) as status:

        documents_raw = []
        file_names = []

        status.write("📄 PDF 텍스트 추출 중...")

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

            for path in pdf_files:
                documents_raw.append(extract_text(path))
                file_names.append(os.path.basename(path))

        status.write("🧠 공통 문장 분석 중...")

        sentence_lists = []
        for doc in documents_raw:
            sentences = re.split(r"[.!?\n]", doc)
            sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
            sentence_lists.append(sentences)

        counter = Counter()
        for sentences in sentence_lists:
            for s in set(sentences):
                counter[s] += 1

        threshold_count = len(documents_raw) * (common_ratio / 100)
        detected_common = [s for s, c in counter.items() if c >= threshold_count]

        status.update(label="✏️ 공통 문장 편집 단계", state="complete")

    # -------------------------
    # 공통 문장 UI 표시
    # -------------------------
    st.header("📌 자동 감지된 공통 문장 (편집 가능)")

    edited_common = []
    for sentence in detected_common:
        if st.checkbox(sentence, value=True):
            edited_common.append(sentence)

    custom_add = st.text_area("➕ 추가로 제거할 문장 직접 입력 (줄바꿈 구분)")
    if custom_add:
        edited_common.extend([s.strip() for s in custom_add.split("\n") if s.strip()])

    # -------------------------
    # 제거 전/후 글자 수 비교
    # -------------------------
    st.header("📊 글자 수 비교")

    before_counts = []
    after_counts = []
    documents_filtered = []

    for sentences in sentence_lists:
        original_text = " ".join(sentences)
        filtered = [s for s in sentences if s not in edited_common]

        before_counts.append(len(re.sub(r"\s+", "", original_text)))
        after_counts.append(len(re.sub(r"\s+", "", " ".join(filtered))))
        documents_filtered.append(" ".join(filtered))

    df = pd.DataFrame({
        "파일명": file_names,
        "제거 전": before_counts,
        "제거 후": after_counts,
        "차이": [b - a for b, a in zip(before_counts, after_counts)]
    })

    def highlight_short(val):
        if val < min_char_limit:
            return "background-color:#ffcccc;"
        return ""

    st.dataframe(df.style.applymap(highlight_short, subset=["제거 후"]))

    # -------------------------
    # 유사도 계산
    # -------------------------
    st.header("🚨 의심 파일 쌍")

    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(documents_filtered)
    sim = cosine_similarity(tfidf)

    found = False
    for i in range(len(file_names)):
        for j in range(i + 1, len(file_names)):
            score = sim[i][j] * 100
            if score >= threshold:
                found = True
                st.error(f"{file_names[i]} ↔ {file_names[j]} → {round(score,2)}%")

    if not found:
        st.success("의심 쌍 없음")
