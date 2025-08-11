import streamlit as st
import datetime
import pytz
from PIL import Image

# Set page config
st.set_page_config(
    page_title="SixtyScan",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for Thai font and styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;500;600;700&display=swap');
    
    .main {
        padding: 0;
    }
    
    .block-container {
        padding: 0;
        max-width: 100%;
    }
    
    .header-container {
        background: linear-gradient(135deg, #6b46c1 0%, #9333ea 100%);
        padding: 20px 40px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        color: white;
    }
    
    .logo-section {
        display: flex;
        align-items: center;
        gap: 15px;
    }
    
    .logo-text {
        font-family: 'Prompt', sans-serif;
        font-size: 32px;
        font-weight: 700;
        color: white;
    }
    
    .tagline {
        font-family: 'Prompt', sans-serif;
        font-size: 18px;
        font-weight: 400;
        color: white;
        margin-left: 20px;
    }
    
    .datetime-display {
        font-family: 'Prompt', sans-serif;
        font-size: 16px;
        font-weight: 400;
        background: rgba(255, 255, 255, 0.2);
        padding: 10px 20px;
        border-radius: 25px;
        backdrop-filter: blur(10px);
    }
    
    .main-content {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        min-height: calc(100vh - 100px);
        padding: 60px 40px;
    }
    
    .content-wrapper {
        display: flex;
        align-items: center;
        gap: 60px;
        max-width: 1200px;
        margin: 0 auto;
    }
    
    .text-section {
        flex: 1;
    }
    
    .main-title {
        font-family: 'Prompt', sans-serif;
        font-size: 48px;
        font-weight: 600;
        color: #1e293b;
        line-height: 1.2;
        margin-bottom: 40px;
    }
    
    .highlight {
        color: #8b5cf6;
    }
    
    .button-container {
        display: flex;
        flex-direction: column;
        gap: 20px;
        width: 300px;
    }
    
    .custom-button {
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
        color: white;
        border: none;
        padding: 18px 30px;
        border-radius: 15px;
        font-family: 'Prompt', sans-serif;
        font-size: 20px;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(139, 92, 246, 0.3);
    }
    
    .custom-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(139, 92, 246, 0.4);
    }
    
    .image-section {
        flex: 1;
        display: flex;
        justify-content: center;
    }
    
    .main-image {
        max-width: 100%;
        height: auto;
        border-radius: 20px;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .stApp > div:first-child {
        padding-top: 0;
    }
</style>
""", unsafe_allow_html=True)

# Get current Thai time
def get_thai_time():
    thai_tz = pytz.timezone('Asia/Bangkok')
    now = datetime.datetime.now(thai_tz)
    return now.strftime("%d/%m/%Y %H:%M:%S")

# Create header
st.markdown(f"""
<div class="header-container">
    <div class="logo-section">
        <div class="logo-text">SixtyScan</div>
        <div class="tagline">นวัตกรรมคัดกรองโรคพาร์กินสันจากเสียง</div>
    </div>
    <div class="datetime-display">
        {get_thai_time()}
    </div>
</div>
""", unsafe_allow_html=True)

# Main content
st.markdown("""
<div class="main-content">
    <div class="content-wrapper">
        <div class="text-section">
            <h1 class="main-title">
                ตรวจเช็คโรคพาร์กินสัน<br>
                ที่บ้านด้วย <span class="highlight">SixtyScan</span>
            </h1>
            <div class="button-container">
                <button class="custom-button">เริ่มใช้งาน</button>
                <button class="custom-button">คู่มือ</button>
            </div>
        </div>
        <div class="image-section">
            <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==" alt="Woman using phone" class="main-image">
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Auto-refresh for time update
import time
time.sleep(1)
st.rerun()
