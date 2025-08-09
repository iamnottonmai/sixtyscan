import streamlit as st
import base64
import os

# =============================
# Logo Loading Function
# =============================
@st.cache_data
def load_logo():
    """Load logo with fallback options for reliability"""
    logo_paths = [
        "logo.png",           # Same directory
        "./logo.png",         # Explicit relative path
        "assets/logo.png",    # If in assets folder
        "images/logo.png"     # If in images folder
    ]
    
    for path in logo_paths:
        try:
            if os.path.exists(path):
                with open(path, "rb") as f:
                    return base64.b64encode(f.read()).decode()
        except Exception as e:
            continue
    
    # If no logo found, return None
    return None

def display_logo():
    """Display logo if available"""
    logo_b64 = load_logo()
    if logo_b64:
        st.markdown(f"""
        <img src="data:image/png;base64,{logo_b64}" class="logo" alt="SixtyScan Logo">
        """, unsafe_allow_html=True)

# =============================
# Page Config & Styles
# =============================
st.set_page_config(
    page_title="SixtyScan - เริ่มต้น", 
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Thai:wght@400;600&family=Lexend+Deca:wght@700&display=swap');
        
        /* Global */
        html, body {
            background-color: #f2f4f8;
            font-family: 'Noto Sans Thai', sans-serif;
            font-weight: 400;
        }
        
        /* Hide Streamlit elements */
        .stDeployButton {display:none;}
        footer {visibility: hidden;}
        .stApp > header {visibility: hidden;}
        
        /* Centered logo */
        .logo {
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 200px;
            margin-bottom: 40px;
        }
        
        /* Main Title */
        h1.title {
            text-align: center;
            font-family: 'Lexend Deca', sans-serif;
            font-size: 84px;
            color: #4A148C;
            font-weight: 700;
            margin-bottom: 10px;
            line-height: 1.1;
        }
        
        /* Subtitle/Description */
        .description {
            text-align: center;
            font-family: 'Noto Sans Thai', sans-serif;
            font-weight: 400;
            font-size: 32px;
            color: #333;
            margin-bottom: 60px;
            line-height: 1.3;
        }
        
        /* Start Button Container */
        .start-button-container {
            display: flex;
            justify-content: center;
            margin: 60px 0;
        }
        
        /* About Us Section */
        .about-section {
            background-color: #ffffff;
            border-radius: 16px;
            padding: 40px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.06);
            margin: 40px auto;
            max-width: 800px;
        }
        
        .about-title {
            font-size: 36px;
            color: #4A148C;
            font-weight: 600;
            font-family: 'Noto Sans Thai', sans-serif;
            margin-bottom: 20px;
            text-align: center;
        }
        
        .about-content {
            font-size: 20px;
            color: #333;
            font-weight: 400;
            font-family: 'Noto Sans Thai', sans-serif;
            line-height: 1.6;
            text-align: justify;
        }
        
        /* Custom button styling */
        .stButton > button {
            font-size: 28px !important;
            padding: 1.2em 3em !important;
            border-radius: 50px !important;
            font-weight: bold !important;
            background: linear-gradient(135deg, #009688, #00bcd4) !important;
            color: white !important;
            border: none !important;
            box-shadow: 0 4px 15px rgba(0, 150, 136, 0.3) !important;
            transition: all 0.3s ease !important;
            font-family: 'Noto Sans Thai', sans-serif !important;
            min-width: 300px !important;
        }
        
        .stButton > button:hover {
            background: linear-gradient(135deg, #00796b, #0097a7) !important;
            box-shadow: 0 6px 20px rgba(0, 150, 136, 0.4) !important;
            transform: translateY(-2px) !important;
        }
        
        .stButton > button:active {
            transform: translateY(0px) !important;
        }
    </style>
""", unsafe_allow_html=True)

# =============================
# Main Content
# =============================

# Logo
display_logo()

# Title
st.markdown("<h1 class='title'>SixtyScan</h1>", unsafe_allow_html=True)

# Description
st.markdown("""
    <div class='description'>
        ตรวจโรคพาร์กินสันจากเสียงด้วยปัญญาประดิษฐ์<br>
        เทคโนโลยีที่ทันสมัยเพื่อการตรวจคัดกรองเบื้องต้น
    </div>
""", unsafe_allow_html=True)

# Start Button (centered) - CHANGED THIS LINE
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    if st.button("เริ่มการวิเคราะห์", key="start_analysis"):
        st.switch_page("analysis.py")  # CHANGED from "main.py" to "analysis.py"

# About Us Section
st.markdown("""
    <div class='about-section'>
        <h2 class='about-title'>เกี่ยวกับเรา</h2>
        <div class='about-content'>
            นวัตกรรมนี้ได้รับการพัฒนาขึ้นเพื่อการตรวจคัดกรองโรคพาร์กินสันในระยะเริ่มต้นแบบไม่รุกรานโดยใช้การวิเคราะห์เสียง 
            โมเดลที่ใช้เทคโนโลยี ResNet18 ได้รับการฝึกฝนด้วยข้อมูล Mel Spectrogram จากตัวอย่างเสียงภาษาไทย 
            และสามารถบรรลุความแม่นยำ 100% ในชุดข้อมูลทดสอบ แสดงให้เห็นถึงศักยภาพที่แข็งแกร่งในการสนับสนุน
            การวินิจฉัยในโลกแห่งความเป็นจริง ระบบนี้ออกแบบมาเพื่อเป็นเครื่องมือสนับสนุนการตัดสินใจทางการแพทย์
            และไม่ใช่การทดแทนการวินิจฉัยโดยแพทย์ผู้เชี่ยวชาญ
        </div>
    </div>
""", unsafe_allow_html=True)

# Footer spacing
st.markdown("<br><br>", unsafe_allow_html=True)
