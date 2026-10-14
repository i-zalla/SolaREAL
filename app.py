import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="SolaREAL",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CSS = Path(__file__).parent / "assets" / "style.css"
st.markdown(f"<style>{CSS.read_text()}</style>", unsafe_allow_html=True)

st.markdown("""
<div class="topbar">
  <div class="brand">
    <span class="brand-mark">✦</span>
    <span class="brand-name">SolaREAL</span>
  </div>
  <div class="nav-links">
    <a href="/?page=detector">Detector</a>
    <a href="/?page=pricing">Pricing</a>
    <a href="/?page=technology">Technology</a>
    <a href="/?page=Login">Login</a>
  </div>
</div>
""", unsafe_allow_html=True)

page = st.query_params.get("page", "home")

if page == "detector":
    st.switch_page("pages/1_Detector.py")
elif page == "pricing":
    st.switch_page("pages/2_Pricing.py")
elif page == "technology":
    st.switch_page("pages/3_Technology.py")
elif page == "Login":
    st.switch_page("pages/Login.py")

st.markdown("""
<section class="hero">
  <div class="hero-content">
    <div class="eyebrow">AI CONTENT FORENSICS</div>
    <h1>LIPSYNC DEEPFAKE<br>DETECTION TECHNOLOGY</h1>
    <p>THE WAY TO UNFOLD <span>THE TRUTH</span></p>
  </div>
</section>
""", unsafe_allow_html=True)

a, b = st.columns([1.25, 1])
with a:
    st.markdown("""
    <div class="copy-block">
      <h2>Verify whether a video is authentic or AI-manipulated.</h2>
      <p>Upload a video and run the provided audio-visual deepfake detection pipeline.</p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("TRY FOR FREE", type="primary"):
        st.switch_page("pages/1_Detector.py")

with b:
    st.markdown("""
    <div class="metric-card">
      <div class="metric-title">Detection pipeline</div>
      <div class="pipeline"><span>UPLOAD</span><b>→</b><span>ANALYZE</span><b>→</b><span>VERIFY</span></div>
      <div class="metric-sub">ViT + XLS-R + Attention Fusion + AV-MSTCN</div>
    </div>
    """, unsafe_allow_html=True)
