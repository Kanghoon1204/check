import streamlit as st
import zipfile
import re
import tempfile
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------
# 기본 설정
# ----------------------------
st.set_page_config(
    page_title="한동인성교육 채점 시스템",
    page_icon="📘",
    layout="wide"
)

# ----------------------------
# 상단 타이틀 복구
# ----------------------------
st.markdown("""
<h1 style='text-align:center;'>📘 한동인성교육 채점 시스템</h1>
<h4 style='text-align:center; color:gray;'>for 예림 · 하영</h4>
""", unsafe_allow_html=True)

st.markdown("---")

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
# 사이드바 설정
# ----------------------------
st.sidebar.header("⚙️ 설정")

uploaded_zip = st.sidebar.file_uploader("📦 PDF ZIP 업로드", type="zip")

similarity_threshold = st.sidebar.slider(
    "표절 의심 기준 (%)",
    30, 100, 75
)

min_char_limit = st.sidebar.number_input(
    "최소 글자 수 기준",
    0, 10000, 800
)

top_n = st.sidebar.slider(
    "최대 표시 조합 수",
    3, 30, 10
)

st.sidebar.subheader("🧹 공통 레이아웃 제거")

layout_text_input = st.sidebar.text_area(
    "제외할 공통 텍스트",
    value=DEFAULT_LAYOUT_TEXT,
    height=200
)

# ----------------------------
# PDF 텍스트 추출
# ----------------------------
def extract_text_from_pdf(path):
    try:
        import pdfplumber
        text = ""
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text += t + "\n"
        return text
    except:
        return ""

def extract_name(filename):
    name_only = os.path.splitext(filename)[0]
    match = re.match(r"([^\d]+)", name_only)
    if match:
        return match.group(1).strip()
    return name_only

def remove_common_layout(text, layout_text):
    lines = [line.strip() for line in layout_text.split("\n") if line.strip()]
    for line in lines:
        text = text.replace(line, "")
    return text

# ----------------------------
# 분석 시작
# ----------------------------
if uploaded_zip and st.button("🚀 채점 분석 시작", use_container_width=True):

    st.markdown("## 🔍 한인교 채점 분석 중입니다...")
    progress = st.progress(0)

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
            st.warning("PDF 파일이 2개 이상 필요합니다.")
            st.stop()

        texts = []
        names = []
        before_counts = []
        after_counts = []

        progress.progress(20)

        for path in pdf_files:
            text = extract_text_from_pdf(path)
            if not text.strip():
                continue

            before = len(re.sub(r"\s+", "", text))
            cleaned = remove_common_layout(text, layout_text_input)
            cleaned = re.sub(r"\s+", " ", cleaned).strip()
            after = len(cleaned.replace(" ", ""))

            if after < 50:
                continue

            texts.append(cleaned)
            names.append(extract_name(os.path.basename(path)))
            before_counts.append(before)
            after_counts.append(after)

        progress.progress(50)

        # ----------------------------
        # 글자 수 분석
        # ----------------------------
        st.subheader("📊 글자 수 분석 (레이아웃 제거 전/후 비교)")

        df = pd.DataFrame({
            "이름": names,
            "제거 전 글자 수": before_counts,
            "제거 후 글자 수": after_counts,
        })

        df["제거된 분량"] = df["제거 전 글자 수"] - df["제거 후 글자 수"]
        df["기준 충족 여부"] = df["제거 후 글자 수"].apply(
            lambda x: "❌ 기준 미달" if x < min_char_limit else "✅ 충족"
        )

        def highlight_row(row):
            if row["제거 후 글자 수"] < min_char_limit:
                return ["background-color:#ffcccc"] * len(row)
            return [""] * len(row)

        st.dataframe(
            df.style.apply(highlight_row, axis=1),
            use_container_width=True
        )

        progress.progress(70)

        # ----------------------------
        # 표절 분석
        # ----------------------------
        st.subheader(f"🚨 {similarity_threshold}% 이상 유사 조합")

        vectorizer = TfidfVectorizer(
            ngram_range=(1,2),
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

        progress.progress(100)

        if not results:
            st.success("기준 이상 유사 조합 없음 🎉")
        else:
            for idx, (n1, n2, score) in enumerate(results[:top_n], 1):

                if score >= 90:
                    color = "🔴"
                elif score >= 80:
                    color = "🟠"
                else:
                    color = "🟡"

                st.markdown(
                    f"### {idx}. {color} {n1} ↔ {n2} — {score}%"
                )

    st.success("✅ 분석 완료")
