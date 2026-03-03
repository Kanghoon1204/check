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
# UI 스타일
# ----------------------------
st.markdown("""
<style>
header {visibility: hidden;}
.block-container {padding-top: 1rem !important;}
.big-title {
    text-align:center;
    font-size:28px;
    font-weight:700;
    color:#1f4e79;
}
.sub-title {
    text-align:center;
    color:gray;
    margin-bottom:20px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='big-title'>📘 한동인성교육 채점 시스템</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>for 예림 · 하영</div>", unsafe_allow_html=True)

# ----------------------------
# 사용 안내
# ----------------------------
with st.expander("📖 사용 방법 안내"):
    st.markdown("""
- PDF 파일들을 ZIP으로 압축 후 업로드
- 이름+학번 형식 파일명 사용
- 분석 시작 클릭
- 전체 유사도 + 문장 단위 정밀 분석 수행
""")

# ----------------------------
# 사이드바 설정
# ----------------------------
st.sidebar.header("⚙️ 채점 설정")

uploaded_zip = st.sidebar.file_uploader("📦 PDF ZIP 업로드", type="zip")
doc_threshold = st.sidebar.slider("문서 전체 유사도 기준 (%)", 10, 100, 60)
sentence_threshold = st.sidebar.slider("문장 유사도 기준 (%)", 50, 100, 80)
min_char_limit = st.sidebar.number_input("최소 글자 수 기준", 0, 10000, 800)

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
# PDF 추출
# ----------------------------
def extract_text(path):
    text = ""
    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text += t + "\n"
    except:
        pass
    return text

# ----------------------------
# 문장 분리
# ----------------------------
def split_sentences(text):
    sentences = re.split(r'[.\n]', text)
    return [s.strip() for s in sentences if len(s.strip()) > 20]

# ----------------------------
# 문장 단위 비교
# ----------------------------
def compare_sentences(text1, text2, threshold):
    sents1 = split_sentences(text1)
    sents2 = split_sentences(text2)

    matched = []

    for s1 in sents1:
        for s2 in sents2:
            vect = TfidfVectorizer().fit([s1, s2])
            tfidf = vect.transform([s1, s2])
            sim = cosine_similarity(tfidf)[0][1] * 100
            if sim >= threshold:
                matched.append((s1, s2, round(sim,1)))

    return matched

# ----------------------------
# 분석 시작
# ----------------------------
if uploaded_zip and st.button("🚀 분석 시작", use_container_width=True):

    progress = st.progress(0)
    status = st.empty()

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

        raw_texts = []
        names = []

        status.text("① 텍스트 추출 중...")
        for path in pdf_files:
            text = extract_text(path)
            raw_texts.append(text)
            names.append(extract_name(os.path.basename(path)))
        progress.progress(30)

        # 글자수 계산
        char_counts = [len(re.sub(r"\s+","", t)) for t in raw_texts]

        progress.progress(50)
        status.text("② 문서 전체 유사도 계산 중...")

        vectorizer = TfidfVectorizer()
        tfidf = vectorizer.fit_transform(raw_texts)
        doc_sim = cosine_similarity(tfidf)

        progress.progress(70)
        status.text("③ 문장 단위 정밀 분석 중...")

        suspect_results = []

        for i in range(len(names)):
            for j in range(i+1, len(names)):

                overall_score = doc_sim[i][j] * 100

                matched_sentences = compare_sentences(
                    raw_texts[i],
                    raw_texts[j],
                    sentence_threshold
                )

                if overall_score >= doc_threshold or len(matched_sentences) >= 3:

                    suspect_results.append({
                        "name1": names[i],
                        "name2": names[j],
                        "overall": round(overall_score,1),
                        "matched": matched_sentences
                    })

        progress.progress(100)
        status.text("✅ 분석 완료")

    # ----------------------------
    # 글자수 출력
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
    # 표절 의심 출력
    # ----------------------------
    st.subheader("🚨 표절 의심 분석 (정밀)")

    if not suspect_results:
        st.success("의심 파일 없음 🎉")
    else:
        for result in suspect_results:

            st.markdown(f"""
### 🔎 {result['name1']} ↔ {result['name2']}
- 문서 전체 유사도: **{result['overall']}%**
- 유사 문장 개수: **{len(result['matched'])}개**
""")

            with st.expander("📌 겹친 문장 보기"):

                for s1, s2, score in result['matched']:
                    st.markdown(f"""
**유사도 {score}%**

- A: {s1}
- B: {s2}

---
""")
