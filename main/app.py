import streamlit as st
import zipfile
import os
import re
import io
import fitz  # PyMuPDF
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations
from collections import Counter

# ----------------------------
# 기본 설정
# ----------------------------
st.set_page_config(
    page_title="한동인성교육 채점 시스템",
    layout="wide"
)

st.title("📘 한동인성교육 채점 시스템 (for 예림 · 하영)")

# ----------------------------
# 사용 설명
# ----------------------------
with st.expander("📖 사용 방법 안내", expanded=False):
    st.markdown("""
### 사용 방법

1️⃣ 학생 PDF 파일들을 ZIP으로 압축하여 업로드  
2️⃣ 표절 의심 기준 설정 (기본 75%)  
3️⃣ 최소 글자 수 설정  
4️⃣ 분석 시작 클릭  

### 분석 내용

- 📊 레이아웃 제거 전/후 글자 수 비교
- 🔴 기준 미달 학생 표시
- 🚨 기준 이상 유사 조합 표시
- 📌 왜 유사한지 공통 표현 확인 가능

※ 표절도는 **레이아웃 제거 후 텍스트** 기준으로 계산됩니다.
""")

# ----------------------------
# 기본 레이아웃 제거 텍스트
# ----------------------------
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

st.subheader("✂ 레이아웃 제거 설정")
layout_text = st.text_area(
    "글자 수 계산 및 표절 분석에서 제외할 공통 문구 (편집 가능)",
    value=default_layout,
    height=220
)

# ----------------------------
# 설정 값
# ----------------------------
col1, col2 = st.columns(2)

with col1:
    similarity_threshold = st.slider(
        "🚨 표절 의심 기준 (%)",
        min_value=50,
        max_value=100,
        value=75
    )

with col2:
    min_char_threshold = st.number_input(
        "📉 최소 글자 수 기준",
        min_value=0,
        value=800
    )

uploaded_zip = st.file_uploader("📂 ZIP 파일 업로드", type=["zip"])

# ----------------------------
# 함수
# ----------------------------
def extract_name(filename):
    base = os.path.splitext(filename)[0]
    parts = base.split("_")
    first_part = parts[0]
    name = re.match(r"[가-힣]+", first_part)
    return name.group() if name else first_part

def extract_text_from_pdf(file_bytes):
    text = ""
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def remove_layout(text, layout):
    for line in layout.split("\n"):
        line = line.strip()
        if line:
            text = text.replace(line, "")
    return text

def get_common_terms(text1, text2):
    words1 = text1.split()
    words2 = text2.split()

    bigrams1 = [" ".join(words1[i:i+2]) for i in range(len(words1)-1)]
    bigrams2 = [" ".join(words2[i:i+2]) for i in range(len(words2)-1)]

    counter1 = Counter(bigrams1)
    counter2 = Counter(bigrams2)

    common = []
    for term in counter1:
        if term in counter2:
            freq = min(counter1[term], counter2[term])
            if freq >= 3:
                common.append((term, freq))

    common.sort(key=lambda x: x[1], reverse=True)
    return common[:15]

# ----------------------------
# 분석 시작
# ----------------------------
if uploaded_zip:

    if st.button("🚀 채점 분석 시작"):

        with st.spinner("🔍 한인교 채점 분석 중... 잠시만 기다려주세요..."):

            zip_bytes = uploaded_zip.read()
            zip_file = zipfile.ZipFile(io.BytesIO(zip_bytes))

            students = []
            original_lengths = {}
            cleaned_lengths = {}
            texts = {}

            for file in zip_file.namelist():
                if file.endswith(".pdf"):
                    name = extract_name(file)
                    file_bytes = zip_file.read(file)

                    raw_text = extract_text_from_pdf(file_bytes)
                    raw_text = clean_text(raw_text)

                    cleaned = remove_layout(raw_text, layout_text)
                    cleaned = clean_text(cleaned)

                    students.append(name)
                    original_lengths[name] = len(raw_text)
                    cleaned_lengths[name] = len(cleaned)
                    texts[name] = cleaned

            # ----------------------------
            # 글자 수 결과 표
            # ----------------------------
            st.subheader("📊 레이아웃 제거 전/후 글자 수")

            df_lengths = pd.DataFrame({
                "이름": students,
                "제거 전 글자 수": [original_lengths[n] for n in students],
                "제거 후 글자 수": [cleaned_lengths[n] for n in students],
            })

            st.dataframe(df_lengths, use_container_width=True)

            # 기준 미달 표시
            st.subheader("📉 최소 글자 수 기준 미달")

            below = [
                n for n in students
                if cleaned_lengths[n] < min_char_threshold
            ]

            if below:
                for n in below:
                    st.error(f"🔴 {n} — {cleaned_lengths[n]}자")
            else:
                st.success("✅ 기준 미달 학생 없음")

            # ----------------------------
            # 표절 분석
            # ----------------------------
            st.subheader("🚨 표절 의심 분석 결과")

            if len(texts) > 1:

                vectorizer = TfidfVectorizer()
                tfidf = vectorizer.fit_transform(texts.values())
                similarity_matrix = cosine_similarity(tfidf)

                suspicious_found = False

                for i, j in combinations(range(len(students)), 2):
                    sim_score = similarity_matrix[i][j] * 100

                    if sim_score >= similarity_threshold:

                        suspicious_found = True
                        name1 = students[i]
                        name2 = students[j]

                        st.warning(
                            f"🚨 {name1} ↔ {name2} — {sim_score:.1f}%"
                        )

                        with st.expander("📌 왜 유사한지 보기"):
                            common_terms = get_common_terms(
                                texts[name1],
                                texts[name2]
                            )

                            if common_terms:
                                df_common = pd.DataFrame(
                                    common_terms,
                                    columns=["공통 표현", "겹친 횟수"]
                                )
                                st.dataframe(df_common,
                                             use_container_width=True)
                            else:
                                st.write("특이하게 많이 겹친 표현은 없음.")

                if not suspicious_found:
                    st.success("✅ 기준 이상 유사 조합 없음")

            else:
                st.info("학생 수가 2명 이상이어야 표절 분석 가능")
