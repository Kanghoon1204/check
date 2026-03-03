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
# 기본 공통 레이아웃 텍스트
# ----------------------------
DEFAULT_LAYOUT_TEXT = """이름과 학번:
작성일자:
강의주제:
1. 강의 내용 요약
2. 질문과 논의사항
3. 강의를 통한 자기 성찰

[작성안내]
위 1번, 2번, 3번 항목에 대하여 총 한글 300단어 이상 혹은 800자 이상으로 작성하시기 바랍니다.
참고를 위하여, 제출한 강의노트는 해당 강의를 하셨던 교수님께 전달될 수 있습니다.
"""

# ----------------------------
# UI
# ----------------------------
st.markdown("## 📘 한동인성교육 채점 시스템 (빠른 의심 탐지 버전)")

with st.expander("📖 사용 방법"):
    st.markdown("""
1. 학생 PDF 파일을 ZIP으로 압축하여 업로드
2. 분석 시작 클릭
3. 글자 수 및 75% 이상 유사 조합 확인
※ 실제 판정은 직접 PDF 확인 권장
""")

# ----------------------------
# 사이드바 설정
# ----------------------------
st.sidebar.header("⚙️ 설정")

uploaded_zip = st.sidebar.file_uploader("📦 PDF ZIP 업로드", type="zip")

similarity_threshold = st.sidebar.slider(
    "표절 의심 기준 (%)",
    min_value=30,
    max_value=100,
    value=75  # 🔥 기본값 75%
)

min_char_limit = st.sidebar.number_input(
    "최소 글자 수 기준",
    min_value=0,
    max_value=10000,
    value=800
)

top_n = st.sidebar.slider(
    "최대 표시 조합 수",
    min_value=3,
    max_value=30,
    value=10
)

st.sidebar.subheader("🧹 공통 레이아웃 제거 설정")

layout_text_input = st.sidebar.text_area(
    "글자 수 및 표절 분석에서 제외할 공통 텍스트",
    value=DEFAULT_LAYOUT_TEXT,
    height=220
)

# ----------------------------
# 이름 추출
# ----------------------------
def extract_name(filename):
    name_only = os.path.splitext(filename)[0]
    match = re.match(r"([^\d]+)", name_only)
    if match:
        return match.group(1).replace(" ", "").strip()
    return name_only

# ----------------------------
# PDF 텍스트 추출
# ----------------------------
def extract_text(path):
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
    return text

# ----------------------------
# 공통 레이아웃 제거
# ----------------------------
def remove_common_layout(text, layout_text):
    layout_lines = [line.strip() for line in layout_text.split("\n") if line.strip()]
    for line in layout_lines:
        text = text.replace(line, "")
    return text

# ----------------------------
# 분석 시작
# ----------------------------
if uploaded_zip and st.button("🚀 분석 시작", use_container_width=True):

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
            st.warning("최소 2개 이상 필요")
            st.stop()

        texts = []
        names = []
        char_counts = []

        for path in pdf_files:

            text = extract_text(path)
            text = remove_common_layout(text, layout_text_input)
            text = re.sub(r'\s+', ' ', text).strip()

            texts.append(text)
            names.append(extract_name(os.path.basename(path)))
            char_counts.append(len(text.replace(" ", "")))

        # ----------------------------
        # 글자 수 출력
        # ----------------------------
        st.subheader("📊 글자 수 분석")

        df = pd.DataFrame({
            "이름": names,
            "글자 수": char_counts
        })

        def highlight_short(val):
            if val < min_char_limit:
                return "background-color:#ffcccc;"
            return ""

        st.dataframe(
            df.style.applymap(highlight_short, subset=["글자 수"]),
            use_container_width=True
        )

        # ----------------------------
        # 표절 의심 분석 (문서 전체)
        # ----------------------------
        st.subheader(f"🚨 {similarity_threshold}% 이상 유사 조합")

        vectorizer = TfidfVectorizer(
            ngram_range=(1,2),
            min_df=1,
            max_df=0.9
        )

        tfidf = vectorizer.fit_transform(texts)
        sim_matrix = cosine_similarity(tfidf)

        results = []

        for i in range(len(names)):
            for j in range(i+1, len(names)):
                score = sim_matrix[i][j] * 100
                if score >= similarity_threshold:
                    results.append((names[i], names[j], round(score,1)))

        results.sort(key=lambda x: x[2], reverse=True)

        if not results:
            st.success("기준 이상 유사 조합 없음 🎉")
        else:
            for idx, (n1, n2, score) in enumerate(results[:top_n], start=1):

                if score >= 85:
                    color = "🔴"
                elif score >= 75:
                    color = "🟠"
                else:
                    color = "🟡"

                st.markdown(f"""
**{idx}. {color} {n1} ↔ {n2} — {score}%**
""")
