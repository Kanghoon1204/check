import streamlit as st
import zipfile
import tempfile
import os
import re
import itertools
import pandas as pd
from difflib import SequenceMatcher
import pdfplumber

st.set_page_config(page_title="한동인성교육 채점 시스템", layout="wide")

st.title("📘 한동인성교육 채점 시스템 (for 예림 하영)")

# -----------------------------
# 사용 설명
# -----------------------------
with st.expander("📖 사용 설명"):
    st.markdown("""
    1. ZIP 파일 업로드 (PDF 포함)
    2. 레이아웃 제거 문구 확인
    3. 글자수/표절/주제 기준 설정
    4. 분석 실행

    ✔ 글자수는 레이아웃 제거 후 기준  
    ✔ 표절도는 레이아웃 제거 후 텍스트 기준  
    ✔ 주제 적합도는 매우 낮은 경우만 경고
    """)

# -----------------------------
# 설정 영역
# -----------------------------
col1, col2, col3 = st.columns(3)

with col1:
    min_length = st.number_input("최소 글자수 기준", value=800)

with col2:
    count_space = st.radio("글자수 계산 방식", ["공백 포함", "공백 제외"])

with col3:
    plag_threshold = st.slider("표절 기준 (%)", 50, 100, 75)

topic_input = st.text_input("강의 주제 입력 (선택)")

topic_threshold = st.number_input(
    "주제 적합도 경고 기준 (0~1)",
    min_value=0.0,
    max_value=1.0,
    value=0.02,
    step=0.01
)

st.markdown("---")

# -----------------------------
# 레이아웃 제거 기본값
# -----------------------------
default_layout = """이름과 학번:
작성일자:
강의주제:
1. 강의 내용 요약
2. 질문과 논의사항
3. 강의를 통한 자기 성찰
"""

layout_text = st.text_area("레이아웃 제거 문구", default_layout, height=150)

# -----------------------------
# ZIP 업로드
# -----------------------------
uploaded_zip = st.file_uploader("ZIP 파일 업로드", type="zip")

# -----------------------------
# 함수 정의
# -----------------------------
def extract_name_from_filename(filename):
    base = os.path.splitext(filename)[0]
    return base.split("_")[0][:3]

def extract_name_from_content(text):
    head = text[:300]
    match = re.search(r"이름과\s*학번[:\s]*([가-힣]{2,4})", head)
    if match:
        return match.group(1)
    return None

def clean_layout(text, layout_patterns):
    cleaned = text
    for line in layout_patterns.split("\n"):
        if line.strip():
            cleaned = cleaned.replace(line.strip(), "")
    return cleaned

def calc_lengths(text):
    with_space = len(text)
    without_space = len(text.replace(" ", "").replace("\n", ""))
    return with_space, without_space

def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

def topic_similarity(text, topic):
    if not topic.strip():
        return None
    return SequenceMatcher(None, text[:2000], topic).ratio()

# -----------------------------
# 분석 실행
# -----------------------------
if uploaded_zip and st.button("🚀 분석 실행"):

    with st.spinner("📊 ZIP 내부 PDF 분석 중입니다..."):

        texts = {}
        lengths = {}
        topic_scores = {}
        name_mismatch = {}

        with tempfile.TemporaryDirectory() as tmpdir:

            zip_path = os.path.join(tmpdir, "upload.zip")

            with open(zip_path, "wb") as f:
                f.write(uploaded_zip.read())

            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(tmpdir)

            for root, _, files in os.walk(tmpdir):
                for file in files:
                    if file.lower().endswith(".pdf"):

                        file_path = os.path.join(root, file)

                        raw_text = ""
                        with pdfplumber.open(file_path) as pdf:
                            for page in pdf.pages:
                                raw_text += page.extract_text() or ""

                        name = extract_name_from_filename(file)

                        content_name = extract_name_from_content(raw_text)
                        if content_name and content_name != name:
                            name_mismatch[name] = content_name

                        cleaned = clean_layout(raw_text, layout_text)

                        bw, bwo = calc_lengths(raw_text)
                        aw, awo = calc_lengths(cleaned)

                        lengths[name] = {
                            "before_with": bw,
                            "before_without": bwo,
                            "after_with": aw,
                            "after_without": awo
                        }

                        texts[name] = cleaned

                        ts = topic_similarity(cleaned, topic_input)
                        if ts is not None:
                            topic_scores[name] = ts

        # -----------------------------
        # 표절 검사
        # -----------------------------
        suspicious = []
        plag_details = []
        names = list(texts.keys())

        for a, b in itertools.combinations(names, 2):
            score = similarity(texts[a], texts[b])
            if score * 100 >= plag_threshold:
                suspicious.extend([a, b])
                plag_details.append((a, b, round(score*100,1)))

        suspicious = list(set(suspicious))

        # -----------------------------
        # 요약 테이블
        # -----------------------------
        summary_rows = []

        for name in names:

            actual = lengths[name]["after_with"] if count_space == "공백 포함" else lengths[name]["after_without"]
            meets = actual >= min_length

            length_display = f"{actual}자 {'✅' if meets else '🔴'}"
            plag_status = "⚠ 의심" if name in suspicious else "정상"

            summary_rows.append([name, length_display, plag_status])

        df_summary = pd.DataFrame(summary_rows, columns=["이름", "글자수", "표절"])

        st.subheader("📊 전체 분석 요약")
        st.dataframe(df_summary, use_container_width=True)

        # -----------------------------
        # 글자수 상세
        # -----------------------------
        with st.expander("🔎 글자수 상세 보기"):
            for name in names:
                st.markdown(f"### {name}")
                st.write(f"""
제거전: {lengths[name]["before_with"]} (공백포함) /
        {lengths[name]["before_without"]} (공백제외)

제거후: {lengths[name]["after_with"]} (공백포함) /
        {lengths[name]["after_without"]} (공백제외)
                """)

        # -----------------------------
        # 표절 상세
        # -----------------------------
        if plag_details:
            with st.expander("⚠ 표절 의심 상세 보기"):
                df_plag = pd.DataFrame(plag_details, columns=["학생 A", "학생 B", "유사도 (%)"])
                st.dataframe(df_plag, use_container_width=True)

        # -----------------------------
        # 주제 적합도
        # -----------------------------
        if topic_scores:
            st.subheader("📚 주제 적합도 분석")

            topic_rows = []
            for name, score in topic_scores.items():
                percent = round(score * 100, 1)
                status = "⚠ 매우 낮음" if score < topic_threshold else "정상"
                topic_rows.append([name, f"{percent}%", status])

            df_topic = pd.DataFrame(topic_rows, columns=["이름", "주제 유사도", "판정"])
            st.dataframe(df_topic, use_container_width=True)

        # -----------------------------
        # 이름 불일치
        # -----------------------------
        if name_mismatch:
            st.subheader("⚠ 파일명-본문 이름 불일치 감지")
            for f, c in name_mismatch.items():
                st.warning(f"파일명: {f} / 본문 이름: {c}")
