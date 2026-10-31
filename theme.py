 
import streamlit as st

def apply_theme():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
    :root { --bg:#080D1B; --panel:#111D31; --panel2:#14243B; --line:#243B55; --cyan:#21D4FD; --teal:#00C9A7; --muted:#9BAAC3; }
    .stApp { background:radial-gradient(circle at 85% 0%, #102b43 0%, #080D1B 38%); color:#F4F7FF; font-family:'DM Sans',sans-serif; }
    .block-container { max-width:1500px; padding-top:2rem; padding-bottom:3rem; }
    h1,h2,h3 { font-family:'Space Grotesk',sans-serif !important; color:#F4F7FF !important; letter-spacing:-.4px; }
    p, label, .stMarkdown { color:#C3CEE0; }
    [data-testid="stSidebar"] { background:linear-gradient(180deg,#0D192B,#091321); border-right:1px solid var(--line); }
    [data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3 { color:#F4F7FF !important; }
    [data-testid="stMetric"] { background:linear-gradient(145deg,#14243B,#0F1B2E); border:1px solid #29445F; padding:18px 20px; border-radius:16px; box-shadow:0 8px 26px #00000022; }
    [data-testid="stMetricValue"] { color:var(--cyan) !important; font-weight:700; }
    [data-testid="stMetricLabel"] { color:var(--muted) !important; }
    div[data-testid="stVerticalBlockBorderWrapper"] { background:#111D31; border:1px solid #243B55; border-radius:16px; padding:14px; }
    .stButton>button,.stDownloadButton>button { background:linear-gradient(90deg,#087FA5,#21D4FD); color:#06111E; border:0; border-radius:10px; font-weight:700; padding:.55rem 1rem; }
    .stButton>button:hover,.stDownloadButton>button:hover { box-shadow:0 0 20px #21D4FD55; color:#06111E; }
    input, textarea, [data-baseweb="select"] > div { background:#14243B !important; color:#F4F7FF !important; border-color:#29445F !important; }
    [data-testid="stDataFrame"] { border:1px solid #29445F; border-radius:12px; overflow:hidden; }
    hr { border-color:#243B55; }
    .eyebrow { color:#21D4FD; font-size:.76rem; font-weight:700; letter-spacing:2px; text-transform:uppercase; margin-bottom:.35rem; }
    .hero { background:linear-gradient(115deg,#142943 0%,#102136 58%,#12384A 100%); border:1px solid #2A526B; border-radius:22px; padding:28px 30px; margin:0 0 24px 0; box-shadow:0 18px 45px #00000030; }
    .hero h1 { font-size:2.25rem; margin:0 0 .45rem 0; }
    .hero p { color:#B8C8DC; font-size:1rem; margin:0; max-width:850px; }
    .section-head { margin-top:1.35rem; margin-bottom:.2rem; }
    .small-muted { color:#91A4BE; font-size:.86rem; }
    .pill { display:inline-block; border:1px solid #2A6378; color:#7CE8F9; background:#123247; border-radius:999px; padding:4px 10px; font-size:.75rem; margin-right:6px; }
    </style>
    """, unsafe_allow_html=True)

def page_header(kicker, title, subtitle):
    st.markdown(f"""<div class="hero"><div class="eyebrow">{kicker}</div>
    <h1>{title}</h1><p>{subtitle}</p></div>""", unsafe_allow_html=True)

def sidebar():
    with st.sidebar:
        st.markdown("## ✦ SPARKERS")
        st.caption("CLINICAL INTELLIGENCE PLATFORM")
        st.markdown("---")
        st.markdown("### PLATFORM NAVIGATION")
        st.caption("Use the page links above to move between workspaces.")
        st.markdown("### CLINICAL WORKSPACE")
        st.markdown("🩺 Patient Assessment")
        st.markdown("📊 Data Intelligence")
        st.markdown("### MODEL INTELLIGENCE")
        st.markdown("🧪 Model Laboratory")
        st.markdown("🔍 Prediction Explainability")
        st.markdown("📈 Model Performance")
        st.markdown("### DATA & GOVERNANCE")
        st.markdown("🗂️ Dataset Explorer")
        st.markdown("ℹ️ About SPARKERS")
        st.markdown("---")
        st.markdown("**SYSTEM STATUS**")
        st.success("Local workspace ready")
        st.caption("Educational prototype • Not for clinical diagnosis")

def setup(kicker, title, subtitle):
    st.set_page_config(page_title=f"{title} | SPARKERS", page_icon="✦", layout="wide")
    apply_theme()
    sidebar()
    page_header(kicker, title, subtitle)
