import streamlit as st
import re
import os
import tempfile
import itertools
import pandas as pd
from difflib import SequenceMatcher
from PyPDF2 import PdfReader

st.set_page_config(page_title="한동인성교육 채점 시스템", layout="wide")

# -----------------------------
# 제목
# -----------------------------
st.title("📘 한동인성교육 채점 시스템 (for 예림 하영)")

# -----------------------------
# 사용 설명
# -----------------------------
with st.expander("📖 사용 설명", expanded=False):
    st.markdown("""
    1. PDF 파일 여러 개 업로드
    2. 레이아웃 제거 문구 확인/수정
    3. 글자수 기준 및 공백 포함 여부 선택
    4. 표절 기준 설정
    5. 강의 주제 입력 (선택)
    6. 분석 실행

    ✔ 글자수는 레이아웃 제거 후 기준으로 판정  
    ✔ 표절도는 레이아웃 제거 후 텍스트 기준  
    ✔ 주제 적합도는 매우 낮은 경우만 검토 권장
    """)

# -----------------------------
# 설정 영역
# -----------------------------
st.subheader("⚙ 분석 설정")

col1, col2, col3 = st.columns(3)

with col1:
    min_length = st.number_input("최소 글자수 기준", value=800)

with col2:
    count_space = st.radio("글자수 계산 방식", ["공백 포함", "공백 제외"])

with col3:
    plag_threshold = st.slider("표절 의심 기준 (%)", 50, 100, 75)

topic_input = st.text_input("강의 주제 입력 (선택)")

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

[작성안내]
위 1번, 2번, 3번 항목에 대하여 총 한글 300단어 이상 혹은 800자 이상으로
작성하시기 바랍니다. 참고를 위하여, 제출한 강의노트는 해당 강의를 하셨던
교수님께 전달될 수 있습니다.
"""

layout_text = st.text_area("레이아웃 제거 문구 (편집 가능)", default_layout, height=200)

# -----------------------------
# 파일 업로드
# -----------------------------
uploaded_files = st.file_uploader("PDF 파일 업로드", type="pdf", accept_multiple_files=True)

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
if uploaded_files and st.button("🚀 분석 실행"):

    with st.spinner("📊 분석 중입니다... 잠시만 기다려주세요."):
        
        texts = {}
        lengths = {}
        name_mismatch = {}
        topic_scores = {}

        # PDF 처리
        for file in uploaded_files:
            reader = PdfReader(file)
            raw_text = ""
            for page in reader.pages:
                raw_text += page.extract_text() or ""

            name = extract_name_from_filename(file.name)

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
        # 표절 검사 (제거 후 기준)
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
        # 메인 요약 테이블
        # -----------------------------
        summary_rows = []

        for name in names:

            if count_space == "공백 포함":
                actual = lengths[name]["after_with"]
            else:
                actual = lengths[name]["after_without"]

            meets = actual >= min_length

            length_display = f"{actual}자"
            if meets:
                length_display += " ✅"
            else:
                length_display += " 🔴"

            plag_status = "⚠ 의심" if name in suspicious else "정상"

            summary_rows.append([
                name,
                length_display,
                plag_status
            ])

        df_summary = pd.DataFrame(
            summary_rows,
            columns=["이름", "글자수", "표절"]
        )

        st.subheader("📊 전체 분석 요약")
        st.dataframe(df_summary, use_container_width=True)

        # -----------------------------
        # 글자수 상세 보기
        # -----------------------------
        with st.expander("🔎 글자수 상세 보기"):
            for name in names:
                st.markdown(f"### {name}")
                st.write(
                    f"""
제거전: {lengths[name]["before_with"]}자(공백포함) /
        {lengths[name]["before_without"]}자(공백제외)

제거후: {lengths[name]["after_with"]}자(공백포함) /
        {lengths[name]["after_without"]}자(공백제외)
                    """
                )

        # -----------------------------
        # 표절 상세 보기
        # -----------------------------
        if plag_details:
            with st.expander("⚠ 표절 의심 상세 보기"):
                df_plag = pd.DataFrame(
                    plag_details,
                    columns=["학생 A", "학생 B", "유사도 (%)"]
                )
                st.dataframe(df_plag, use_container_width=True)

        # -----------------------------
        # 주제 적합도 섹션
        # -----------------------------
        if topic_scores:
            st.subheader("📚 주제 적합도 분석")

            topic_rows = []
            for name, score in topic_scores.items():
                percent = round(score * 100, 1)
                if percent < 10:
                    status = "⚠ 매우 낮음"
                    action = "직접 검토 권장"
                else:
                    status = "정상"
                    action = "-"

                topic_rows.append([
                    name,
                    f"{percent}%",
                    status,
                    action
                ])

            df_topic = pd.DataFrame(
                topic_rows,
                columns=["이름", "주제 유사도", "판정", "권장 조치"]
            )

            st.dataframe(df_topic, use_container_width=True)

        # -----------------------------
        # 이름 불일치 경고
        # -----------------------------
        if name_mismatch:
            st.subheader("⚠ 파일명-본문 이름 불일치 감지")
            for fname, cname in name_mismatch.items():
                st.warning(f"파일명: {fname} / 본문 추출 이름: {cname}")
