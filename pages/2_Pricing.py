import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Pricing — SolaREAL", page_icon="✦", layout="wide")
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

st.markdown('<h1 class="page-title center">Plans for every level of verification</h1>', unsafe_allow_html=True)
st.markdown('<p class="page-subtitle center">Choose the plan that fits your needs</p>', unsafe_allow_html=True)

plans = [
    {
        "name": "SILVER",
        "price": "6,500 DA",
        "features": ["100 free seconds", "3,800 seconds total"],
        "color": "#808080",  
        "period": ""  
    },
    {
        "name": "GOLDEN",
        "price": "26,000 DA",
        "features": ["Free week trial", "7,200 seconds total"],
        "color": "#D4AF37",  
        "period": ""  
    },
    {
        "name": "PLATINUM",
        "price": "66,000 DA",
        "features": ["Free week trial", "30,000 seconds total"],
        "color": "#A8A8A8",  
        "period": ""  
    },
    {
        "name": "ENTERPRISE",
        "price": "800,000 DA",
        "features": ["Application integration", "Annual subscription", "Unlimited seconds"],
        "color": "#8B0000",  
        "period": "/year"  
    }
]

for i in range(0, len(plans), 2):
    cols = st.columns(2)
    for col, plan in zip(cols, plans[i:i+2]):
        with col:
            st.markdown(f"""
            <div class="pricing-card">
              <div class="plan-name" style="color: {plan['color']};">{plan['name']}</div>
              <div class="plan-price">{plan['price']}<span style="font-size: 14px; color: #888;">{plan['period']}</span></div>
              <div class="plan-list">{''.join(f'<div style="padding: 6px 0;">✓ {x}</div>' for x in plan['features'])}</div>
            </div>
            """, unsafe_allow_html=True)
            st.button("CHOOSE PLAN", key=plan['name'], use_container_width=True)