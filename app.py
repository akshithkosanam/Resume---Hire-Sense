import io, re, pickle, pandas as pd, streamlit as st, docx2txt, PyPDF2
from sklearn.metrics.pairwise import cosine_similarity
from interview_questions import generate_interview_questions

# ── ML artefacts ───────────────────────────────────────
tfidf = pickle.load(open("tfidf.pkl", "rb"))
clf   = pickle.load(open("clf.pkl",   "rb"))
df    = pd.read_csv("final_dataset.csv")

label_encoding = [4,3,9,0,2,1,5,6,7,8,10,11,12]
categories     = [
    "HR","DotNet Developer","Quality Assurance","Business Analyst","DevOps Engineer",
    "Database Administrator","Java Developer","Network Administrator","Project Manager",
    "Python Developer","Security Analyst","Systems Administrator","Web Developer"
]
CATEGORY_MAP = dict(zip(label_encoding, categories))

# ── helper to normalise prediction → role string ───────
def to_role(pred):
    return pred if isinstance(pred, str) else CATEGORY_MAP.get(pred, "Unknown")

# ── text utils ─────────────────────────────────────────
CLEAN_RE = re.compile(
    r"http\S+|RT|cc|#\S+|@\S+|[%s]|[^\x00-\x7F]|\s+" %
    re.escape(r"""!"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~""")
)
clean_text = lambda t: CLEAN_RE.sub(" ", t).strip()
pdf_to_text  = lambda f: "".join(p.extract_text() or "" for p in PyPDF2.PdfReader(f).pages)
docx_to_text = lambda f: docx2txt.process(f)

@st.cache_data
def df_to_excel(_df: pd.DataFrame)->bytes:
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as w: _df.to_excel(w, index=False)
    return bio.getvalue()

def render_pretty_table(df):
    st.markdown("""
    <style>
    .custom-table {
        width: 100%;
        border-collapse: collapse;
        background: #ffffff;
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        margin-top: 10px;
    }
    .custom-table th, .custom-table td {
        text-align: center;
        padding: 12px;
        border-bottom: 1px solid #f0f0f0;
    }
    .custom-table th {
        background-color: #f2f4f7;
        font-weight: 600;
        color: #333;
    }
    </style>
    """, unsafe_allow_html=True)

    html = "<table class='custom-table'><thead><tr>"
    for col in df.columns:
        html += f"<th>{col}</th>"
    html += "</tr></thead><tbody>"
    for _, row in df.iterrows():
        html += "<tr>" + "".join(f"<td>{v}</td>" for v in row) + "</tr>"
    html += "</tbody></table>"
    st.markdown(html, unsafe_allow_html=True)

# ── Page style ─────────────────────────────────────────
st.set_page_config(page_title="HireSense", layout="wide")
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #e8f0fe, #fefefe);
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
main > div {
    background: rgba(255, 255, 255, 0.9) !important;
    padding: 1.5rem !important;
    border-radius: 15px !important;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12);
}
</style>
""", unsafe_allow_html=True)

# ── UI ─────────────────────────────────────────────────
st.title("🎯 HireSense")
tab_resume, tab_job = st.tabs(["📄 Upload Resume", "📝 Job-Description Search"])

# résumé tab
with tab_resume:
    up = st.file_uploader("Upload résumé (PDF or DOCX)", type=["pdf","docx"])
    if up:
        txt  = pdf_to_text(up) if up.name.lower().endswith(".pdf") else docx_to_text(up)
        st.subheader("📋 Extracted Text")
        st.text_area("", txt, height=240)

        role = to_role(clf.predict(tfidf.transform([clean_text(txt)]))[0])
        st.success(f"🔍 Predicted Role → **{role}**")

        st.subheader(" AI-generated Questions")
        for q in generate_interview_questions(txt):
            st.markdown(f"- {q}")

# JD tab
with tab_job:
    jd    = st.text_area("Paste job description", height=180)
    top_k = st.slider("Number of candidates", 1, 10, 3)
    if st.button("🔍 Find & Generate") and jd.strip():
        jd_vec = tfidf.transform([clean_text(jd)])
        role   = to_role(clf.predict(jd_vec)[0])
        st.info(f"📌 **Predicted JD Role → {role}**")

        pool = df[df["Category"].str.lower()==role.lower()].copy()
        if pool.empty:
            st.error("❌ No matching résumés found.")
            st.stop()

        pool["Similarity"] = cosine_similarity(jd_vec, tfidf.transform(pool["Resume"]))[0]
        top = pool.nlargest(top_k, "Similarity")

        st.subheader(" Top Candidates")
        table = top[["S.No", "Similarity"]].copy()
        table["Similarity"] = table["Similarity"].map("{:.2%}".format)
        render_pretty_table(table)

        for _, row in top.iterrows():
            with st.expander(f"Candidate {int(row['S.No'])} · {row['Similarity']:.1%} match"):
                qs = generate_interview_questions(row["Resume"], n=3)
                st.markdown("\n".join(f"{i+1}. {q}" for i,q in enumerate(qs)))

        st.download_button(
            "📥 Download Top-K (.xlsx)",
            data=df_to_excel(top[["S.No","Category","Similarity"]]),
            file_name="top_candidates.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
