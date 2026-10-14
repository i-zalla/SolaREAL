import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Technology — SolaREAL", page_icon="✦", layout="wide")
CSS = Path(__file__).parents[1] / "assets" / "style.css"
st.markdown(f"<style>{CSS.read_text()}</style>", unsafe_allow_html=True)

st.markdown("""
<div class="topbar">
  <div class="brand"><span class="brand-mark">✦</span><span class="brand-name">SolaREAL</span></div>
  <div class="nav-links">
    <a href="/?page=detector">Detector</a>
    <a href="/?page=pricing">Pricing</a>
    <a href="/?page=technology">Technology</a>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<h1 class="page-title">Technology</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="page-subtitle">Overview of the provided audio-visual deepfake detection pipeline.</p>',
    unsafe_allow_html=True,
)

steps = [
    ("01", "Face detection", "MediaPipe Face Mesh is used first, with the provided YOLO face detector as fallback."),
    ("02", "Visual features", "Detected face crops are normalized to 224×224 and encoded with ViT."),
    ("03", "Audio features", "Audio is extracted at 16 kHz and encoded with XLS-R."),
    ("04", "Attention fusion", "Visual and audio representations are combined into 256-dimensional features."),
    ("05", "Temporal classification", "The scaled per-frame feature sequence is classified by the trained AV-MSTCN."),
]

for number, title, description in steps:
    st.markdown(f"""
    <div class="tech-row">
      <div class="tech-number">{number}</div>
      <div><h3>{title}</h3><p>{description}</p></div>
    </div>
    """, unsafe_allow_html=True)
