import streamlit as st
import zipfile
import os
import re
import io
import pdfplumber
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations
from collections import Counter

# --------------------------------------------------
# 페이지 설정
# --------------------------------------------------
st.set_page_config(
    page_title="한동인성교육 채점 시스템",
    layout="wide"
)

st.title("📘 한동인성교육 채점 시스템 (for 예림 · 하영)")
st.divider()

# --------------------------------------------------
# 사용 안내
# --------------------------------------------------
with st.expander("📖 사용 방법 안내", expanded=False):
    st.markdown("""
### 🔹 사용 방법
1️⃣ 학생 PDF를 ZIP으로 압축 후 업로드  
2️⃣ 분석 기준 설정  
3️⃣ 분석 시작 클릭  

### 🔹 분석 항목
- 📏 글자 수 (레이아웃 제거 후 기준)
- 🚨 표절 의심도 (레이아웃 제거 후 계산)
- 🎓 강의 주제 적합도 (너무 낮은 경우만 표시)

※ AI가 판정하지 않습니다.  
※ 표시된 항목만 직접 검토하시면 됩니다.
""")

# --------------------------------------------------
# 레이아웃 제거 설정
# --------------------------------------------------
st.subheader("✂ 레이아웃 제거 설정")

default_layout = """이름과 학번:
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

layout_text = st.text_area(
    "제외할 공통 문구 (편집 가능)",
    value=default_layout,
    height=200
)

st.divider()

# --------------------------------------------------
# 분석 기준 설정
# --------------------------------------------------
st.subheader("⚙ 분석 기준 설정")

col1, col2, col3 = st.columns(3)

with col1:
    similarity_threshold = st.slider("🚨 표절 기준 (%)", 50, 100, 75)

with col2:
    min_char_threshold = st.number_input("📏 최소 글자 수", min_value=0, value=800)

with col3:
    char_option = st.radio("글자 수 기준",
                           ["공백 포함", "공백 제외"])

lecture_topic = st.text_input("🎓 강의 주제 입력 (선택)")
topic_threshold = st.slider(
    "📚 주제 적합도 하한선 (너무 낮은 경우만 표시)",
    0.00, 0.20, 0.05, 0.01
)

st.divider()

uploaded_zip = st.file_uploader("📂 ZIP 파일 업로드", type=["zip"])

# --------------------------------------------------
# 함수
# --------------------------------------------------

def extract_name(filename):
    base = os.path.splitext(filename)[0]
    first = base.split("_")[0]
    match = re.match(r"[가-힣]+", first)
    return match.group() if match else first

def extract_text_from_pdf(file_bytes):
    text = ""
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text += t + " "
    return text

def clean_text(text):
    return re.sub(r"\s+", " ", text).strip()

def remove_layout(text, layout):
    for line in layout.split("\n"):
        line = line.strip()
        if line:
            text = text.replace(line, "")
    return text

def count_chars(text, include_space=True):
    if include_space:
        return len(text)
    return len(text.replace(" ", ""))

def get_common_terms(text1, text2):
    words1 = text1.split()
    words2 = text2.split()

    bigrams1 = [" ".join(words1[i:i+2]) for i in range(len(words1)-1)]
    bigrams2 = [" ".join(words2[i:i+2]) for i in range(len(words2)-1)]

    c1, c2 = Counter(bigrams1), Counter(bigrams2)

    common = []
    for term in c1:
        if term in c2:
            freq = min(c1[term], c2[term])
            if freq >= 3:
                common.append((term, freq))

    common.sort(key=lambda x: x[1], reverse=True)
    return common[:15]

def calculate_topic_similarity(topic, student_texts):
    results = {}
    if not topic.strip():
        return results

    documents = [topic] + list(student_texts.values())
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(documents)
    sim_matrix = cosine_similarity(tfidf)

    for i, name in enumerate(student_texts.keys()):
        results[name] = sim_matrix[0][i+1]

    return results

# --------------------------------------------------
# 분석 시작
# --------------------------------------------------

if uploaded_zip and st.button("🚀 채점 분석 시작"):

    with st.spinner("🔍 분석 중... 잠시만 기다려주세요..."):

        zip_bytes = uploaded_zip.read()
        zip_file = zipfile.ZipFile(io.BytesIO(zip_bytes))

        students = []
        texts = {}
        original_inc, original_exc = {}, {}
        cleaned_inc, cleaned_exc = {}, {}

        for file in zip_file.namelist():
            if file.endswith(".pdf"):
                name = extract_name(file)

                raw = clean_text(
                    extract_text_from_pdf(zip_file.read(file))
                )

                cleaned = clean_text(
                    remove_layout(raw, layout_text)
                )

                students.append(name)
                texts[name] = cleaned

                original_inc[name] = count_chars(raw, True)
                original_exc[name] = count_chars(raw, False)
                cleaned_inc[name] = count_chars(cleaned, True)
                cleaned_exc[name] = count_chars(cleaned, False)

        # --------------------------------------------------
        # 표절 분석
        # --------------------------------------------------
        plagiarism_flag = {name: "정상" for name in students}

        if len(texts) > 1:
            vectorizer = TfidfVectorizer()
            tfidf = vectorizer.fit_transform(texts.values())
            sim_matrix = cosine_similarity(tfidf)

            for i, j in combinations(range(len(students)), 2):
                score = sim_matrix[i][j] * 100
                if score >= similarity_threshold:
                    n1, n2 = students[i], students[j]
                    plagiarism_flag[n1] = "🚨 의심"
                    plagiarism_flag[n2] = "🚨 의심"

        # --------------------------------------------------
        # 주제 적합도
        # --------------------------------------------------
        topic_scores = calculate_topic_similarity(lecture_topic, texts)

        # --------------------------------------------------
        # 전체 요약 테이블
        # --------------------------------------------------
        summary = []

        for name in students:

            length_val = cleaned_inc[name] if char_option == "공백 포함" else cleaned_exc[name]
            length_status = "🔴 미달" if length_val < min_char_threshold else "✅"

            if lecture_topic.strip():
                score = topic_scores.get(name, 0)
                topic_status = "⚠ 낮음" if score < topic_threshold else "✅"
            else:
                topic_status = "-"

            final_flag = "검토 필요" if (
                length_status != "✅"
                or plagiarism_flag[name] != "정상"
                or topic_status != "✅"
            ) else "정상"

            summary.append([
                name,
                length_status,
                plagiarism_flag[name],
                topic_status,
                final_flag
            ])

        st.subheader("📊 전체 분석 요약")
        df_summary = pd.DataFrame(
            summary,
            columns=["이름", "글자수", "표절", "주제 적합도", "최종 상태"]
        )
        st.dataframe(df_summary, use_container_width=True)
