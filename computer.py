import streamlit as st
import base64
import os
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
import librosa
import librosa.display
import matplotlib.pyplot as plt
from PIL import Image
import io
import tempfile
import gdown
from datetime import datetime

# Import your model class
try:
    from model import ResNet18Classifier
except ImportError:
    st.error("Could not import ResNet18Classifier from model.py. Make sure the file exists.")
    st.stop()

def run_desktop_app():
    """Main function to run the desktop version"""
    # =============================
    # Initialize Session State
    # =============================
    if 'page' not in st.session_state:
        st.session_state.page = 'home'

    # =============================
    # Logo and Image Loading Functions
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
        
        return None

    @st.cache_data
    def load_woman_image():
        """Load the woman image"""
        image_paths = [
            "insert.png",           # Same directory
            "./insert.png",         # Explicit relative path
            "assets/insert.png",    # If in assets folder
            "images/insert.png"     # If in images folder
        ]
        
        for path in image_paths:
            try:
                if os.path.exists(path):
                    with open(path, "rb") as f:
                        return base64.b64encode(f.read()).decode()
            except Exception as e:
                continue
        
        return None

    # =============================
    # Global Styles
    # =============================
    def load_styles():
        st.markdown("""
            <style>
                @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Thai:wght@400;500;600;700&display=swap');
                
                /* Global Reset */
                .stApp {
                    background-color: #f5f5f5 !important;
                    margin: 0 !important;
                    padding: 0 !important;
                }
                
                /* Hide Streamlit elements */
                .stDeployButton {display:none;}
                footer {visibility: hidden;}
                .stApp > header {visibility: hidden;}
                #MainMenu {visibility: hidden;}
                .stButton > button:first-child {
                    all: unset;
                }
                
                /* Header Styles */
                .header {
                    background: linear-gradient(135deg, #4A148C 0%, #6A1B9A 50%, #8E24AA 100%);
                    padding: 16px 40px;
                    display: flex;
                    align-items: center;
                    justify-content: space-between;
                    margin: 0;
                    width: 100%;
                    box-sizing: border-box;
                }
                
                .header-logo {
                    height: 48px;
                    width: auto;
                }
                
                .header-title {
                    color: white;
                    font-family: 'Noto Sans Thai', sans-serif;
                    font-size: 24px;
                    font-weight: 500;
                    margin: 0;
                    flex: 1;
                    text-align: center;
                }
                
                .header-datetime {
                    background: rgba(255, 255, 255, 0.2);
                    color: white;
                    padding: 8px 16px;
                    border-radius: 20px;
                    font-family: 'Noto Sans Thai', sans-serif;
                    font-size: 14px;
                    font-weight: 400;
                }
                
                /* Main Content Area */
                .main-content {
                    padding: 80px 60px;
                    display: flex;
                    align-items: center;
                    justify-content: space-between;
                    max-width: 1400px;
                    margin: 0 auto;
                    min-height: 70vh;
                }
                
                .content-left {
                    flex: 1;
                    padding-right: 60px;
                }
                
                .content-right {
                    flex: 1;
                    text-align: center;
                }
                
                /* Main Title */
                .main-title {
                    font-family: 'Noto Sans Thai', sans-serif;
                    font-size: 64px;
                    font-weight: 700;
                    color: #2d2d2d;
                    line-height: 1.2;
                    margin-bottom: 40px;
                }
                
                .title-highlight {
                    color: #4A148C;
                }
                
                /* Buttons */
                .custom-button {
                    display: inline-block;
                    background: linear-gradient(135deg, #1976D2 0%, #9C27B0 100%);
                    color: white;
                    padding: 20px 40px;
                    border-radius: 50px;
                    text-decoration: none;
                    font-family: 'Noto Sans Thai', sans-serif;
                    font-size: 24px;
                    font-weight: 600;
                    margin: 10px 0;
                    border: none;
                    cursor: pointer;
                    transition: all 0.3s ease;
                    box-shadow: 0 4px 15px rgba(74, 20, 140, 0.3);
                    width: 100%;
                    max-width: 300px;
                    text-align: center;
                }
                
                .custom-button:hover {
                    transform: translateY(-2px);
                    box-shadow: 0 6px 20px rgba(74, 20, 140, 0.4);
                }
                
                .custom-button:active {
                    transform: translateY(0px);
                }
                
                /* Woman Image */
                .woman-image {
                    width: 100%;
                    max-width: 500px;
                    height: auto;
                    border-radius: 20px;
                    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
                }
                
                /* Button Container */
                .button-container {
                    display: flex;
                    flex-direction: column;
                    gap: 20px;
                    align-items: flex-start;
                }
                
                /* Responsive adjustments */
                @media (max-width: 1200px) {
                    .main-content {
                        flex-direction: column;
                        text-align: center;
                        padding: 40px 30px;
                    }
                    
                    .content-left {
                        padding-right: 0;
                        margin-bottom: 40px;
                    }
                    
                    .main-title {
                        font-size: 48px;
                    }
                    
                    .button-container {
                        align-items: center;
                    }
                }
            </style>
        """, unsafe_allow_html=True)

    # =============================
    # Page Functions
    # =============================
    def show_header():
        """Display the header with logo, title, and datetime"""
        logo_b64 = load_logo()
        current_time = datetime.now()
        thai_datetime = current_time.strftime("%d/%m/%Y %H:%M")
        
        logo_html = ""
        if logo_b64:
            logo_html = f'<img src="data:image/png;base64,{logo_b64}" class="header-logo" alt="SixtyScan Logo">'
        
        st.markdown(f"""
            <div class="header">
                <div>{logo_html}</div>
                <div class="header-title">นวัตกรรมคัดกรองโรคพาร์กินสันจากเสียง</div>
                <div class="header-datetime">{thai_datetime}</div>
            </div>
        """, unsafe_allow_html=True)

    def show_home_page():
        """Display the home page matching the design"""
        load_styles()
        show_header()
        
        woman_image_b64 = load_woman_image()
        
        # Main content area
        st.markdown("""
            <div class="main-content">
                <div class="content-left">
                    <h1 class="main-title">
                        ตรวจเช็คโรคพาร์กินสัน<br>
                        ที่บ้านด้วย <span class="title-highlight">SixtyScan</span>
                    </h1>
                    <div class="button-container">
                        <div id="start-button" class="custom-button">เริ่มใช้งาน</div>
                        <div id="guide-button" class="custom-button">คู่มือ</div>
                    </div>
                </div>
                <div class="content-right">
        """, unsafe_allow_html=True)
        
        # Display woman image
        if woman_image_b64:
            st.markdown(f"""
                <img src="data:image/png;base64,{woman_image_b64}" class="woman-image" alt="Woman using phone">
            """, unsafe_allow_html=True)
        else:
            # Fallback placeholder
            st.markdown("""
                <div style="width: 100%; max-width: 500px; height: 400px; background: linear-gradient(135deg, #e3f2fd, #f3e5f5); 
                           border-radius: 20px; display: flex; align-items: center; justify-content: center; 
                           box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);">
                    <div style="text-align: center; color: #666;">
                        <div style="font-size: 48px; margin-bottom: 10px;">📱</div>
                        <div style="font-size: 18px;">insert.png<br>not found</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Add JavaScript for button functionality
        st.markdown("""
            <script>
                document.getElementById('start-button').onclick = function() {
                    window.parent.postMessage({type: 'streamlit:setComponentValue', value: 'start_analysis'}, '*');
                };
                
                document.getElementById('guide-button').onclick = function() {
                    window.parent.postMessage({type: 'streamlit:setComponentValue', value: 'guide'}, '*');
                };
            </script>
        """, unsafe_allow_html=True)
        
        # Handle button clicks using Streamlit components
        if st.button("เริ่มใช้งาน", key="start_hidden", type="primary"):
            st.session_state.page = 'analysis'
            st.rerun()
            
        if st.button("คู่มือ", key="guide_hidden", type="secondary"):
            st.session_state.page = 'guide'
            st.rerun()
        
        # Hide the actual Streamlit buttons
        st.markdown("""
            <style>
                div[data-testid="stButton"]:has(button[key="start_hidden"]),
                div[data-testid="stButton"]:has(button[key="guide_hidden"]) {
                    display: none !important;
                }
            </style>
        """, unsafe_allow_html=True)

    def show_guide_page():
        """Display the guide/manual page"""
        load_styles()
        show_header()
        
        # Back button
        if st.button("← กลับหน้าแรก", key="back_to_home_from_guide"):
            st.session_state.page = 'home'
            st.rerun()
        
        st.markdown("""
            <div style="max-width: 1000px; margin: 40px auto; padding: 0 40px;">
                <h1 style="text-align: center; color: #4A148C; font-size: 48px; margin-bottom: 40px;">คู่มือการใช้งาน SixtyScan</h1>
                
                <div style="background: white; padding: 40px; border-radius: 16px; box-shadow: 0 4px 20px rgba(0,0,0,0.06); margin-bottom: 30px;">
                    <h2 style="color: #4A148C; font-size: 32px; margin-bottom: 20px;">การเตรียมตัวก่อนการตรวจ</h2>
                    <ul style="font-size: 20px; line-height: 1.6;">
                        <li>หาสถานที่เงียบ ปราศจากเสียงรบกวน</li>
                        <li>ใช้ไมโครโฟนหรืออุปกรณ์บันทึกเสียงที่มีคุณภาพ</li>
                        <li>นั่งหรือยืนในท่าที่สบาย</li>
                        <li>พักผ่อนเพียงพอก่อนการตรวจ</li>
                    </ul>
                </div>
                
                <div style="background: white; padding: 40px; border-radius: 16px; box-shadow: 0 4px 20px rgba(0,0,0,0.06); margin-bottom: 30px;">
                    <h2 style="color: #4A148C; font-size: 32px; margin-bottom: 20px;">ขั้นตอนการตรวจ</h2>
                    <div style="font-size: 20px; line-height: 1.6;">
                        <h3 style="color: #666; font-size: 24px;">1. การออกเสียงสระ</h3>
                        <ul>
                            <li>ออกเสียงสระแต่ละตัว 5-8 วินาที</li>
                            <li>ออกเสียงให้ชัดเจนและคงที่</li>
                            <li>ไม่ต้องออกเสียงดังเกินไป</li>
                        </ul>
                        
                        <h3 style="color: #666; font-size: 24px;">2. การออกเสียงพยางค์</h3>
                        <ul>
                            <li>ออกเสียง "พา-ทา-คา" ซ้ำๆ</li>
                            <li>ใช้เวลาประมาณ 6 วินาที</li>
                            <li>พยายามออกเสียงให้เร็วและชัดเจน</li>
                        </ul>
                        
                        <h3 style="color: #666; font-size: 24px;">3. การอ่านประโยค</h3>
                        <ul>
                            <li>อ่านประโยคที่กำหนดให้อย่างเป็นธรรมชาติ</li>
                            <li>ไม่ต้องรีบร้อน</li>
                            <li>ออกเสียงให้ชัดเจน</li>
                        </ul>
                    </div>
                </div>
                
                <div style="background: white; padding: 40px; border-radius: 16px; box-shadow: 0 4px 20px rgba(0,0,0,0.06);">
                    <h2 style="color: #4A148C; font-size: 32px; margin-bottom: 20px;">ข้อควรระวัง</h2>
                    <ul style="font-size: 20px; line-height: 1.6; color: #d32f2f;">
                        <li><strong>ระบบนี้เป็นเพียงการตรวจคัดกรองเบื้องต้น</strong></li>
                        <li><strong>ไม่สามารถทดแทนการวินิจฉัยโดยแพทย์ได้</strong></li>
                        <li><strong>หากมีข้อสงสัยควรปรึกษาแพทย์เฉพาะทาง</strong></li>
                    </ul>
                </div>
            </div>
        """, unsafe_allow_html=True)

    def show_analysis_page():
        """Display the analysis page - desktop version with full features"""
        load_styles()
        initialize_analysis_session_state()
        
        # Back button
        if st.button("← กลับหน้าแรก", key="back_to_home"):
            st.session_state.page = 'home'
            st.rerun()
        
        # Load model
        model = load_model()
        
        # Header (same as home page)
        show_header()
        
        st.markdown("<h1 style='text-align: center; font-size: 48px; color: #4A148C; margin: 40px 0;'>การวิเคราะห์เสียง</h1>", unsafe_allow_html=True)

        # Clear button logic
        if 'clear_button_clicked' in st.session_state and st.session_state.clear_button_clicked:
            cleanup_temp_files()
            st.session_state.vowel_files = []
            st.session_state.pataka_file = None
            st.session_state.sentence_file = None
            st.session_state.clear_clicked = True
            st.session_state.clear_button_clicked = False
            st.success("ลบข้อมูลทั้งหมดเรียบร้อยแล้ว", icon="🗑️")
            st.rerun()

        # Vowel recordings
        st.markdown("""
        <div style='background: white; padding: 40px; border-radius: 16px; box-shadow: 0 4px 20px rgba(0,0,0,0.06); margin-bottom: 30px;'>
            <h2 style='font-size: 36px; color: #4A148C; margin-bottom: 20px;'>1. สระ</h2>
            <p style='font-size: 20px; color: #333; margin-bottom: 24px;'>กรุณาออกเสียงแต่ละสระ 5-8 วินาทีอย่างชัดเจน โดยกดปุ่มบันทึกทีละสระ</p>
        </div>
        """, unsafe_allow_html=True)

        vowel_sounds = ["อา", "อี", "อือ", "อู", "ไอ", "อำ", "เอา"]

        for i, sound in enumerate(vowel_sounds):
            st.markdown(f"<p style='font-size: 24px; color: #000; margin: 20px 0;'>ออกเสียง <b>\"{sound}\"</b></p>", unsafe_allow_html=True)
            
            if not st.session_state.clear_clicked:
                audio_bytes = st.audio_input(f"🎤 บันทึกเสียง {sound}", key=f"vowel_{i}")
                if audio_bytes:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                        tmp.write(audio_bytes.read())
                        while len(st.session_state.vowel_files) <= i:
                            st.session_state.vowel_files.append(None)
                        if st.session_state.vowel_files[i] and os.path.exists(st.session_state.vowel_files[i]):
                            os.unlink(st.session_state.vowel_files[i])
                        st.session_state.vowel_files[i] = tmp.name
                    st.success(f"บันทึกเสียง \"{sound}\" สำเร็จ", icon="✅")
            else:
                st.audio_input(f"🎤 บันทึกเสียง {sound}", key=f"vowel_{i}_new")

        # Continue with rest of analysis page (pataka, sentence, etc.)
        # [Rest of the analysis logic would go here - keeping the existing functionality]

    # =============================
    # Analysis Functions (keeping existing)
    # =============================
    def cleanup_temp_files():
        """Clean up all temporary files stored in session state"""
        files_to_clean = []
        
        if 'vowel_files' in st.session_state:
            files_to_clean.extend(st.session_state.vowel_files)
        if 'pataka_file' in st.session_state:
            files_to_clean.append(st.session_state.pataka_file)
        if 'sentence_file' in st.session_state:
            files_to_clean.append(st.session_state.sentence_file)
        
        for file_path in files_to_clean:
            try:
                if file_path and os.path.exists(file_path):
                    os.unlink(file_path)
            except Exception as e:
                pass

    def initialize_analysis_session_state():
        """Initialize session state variables for analysis"""
        if 'vowel_files' not in st.session_state:
            st.session_state.vowel_files = []
        if 'pataka_file' not in st.session_state:
            st.session_state.pataka_file = None
        if 'sentence_file' not in st.session_state:
            st.session_state.sentence_file = None
        if 'clear_clicked' not in st.session_state:
            st.session_state.clear_clicked = False

    @st.cache_resource
    def load_model():
        """Load the ResNet18 model"""
        MODEL_PATH = "best_resnet18.pth"
        if not os.path.exists(MODEL_PATH):
            with st.spinner("Downloading model..."):
                gdown.download(
                    "https://drive.google.com/uc?id=1_oHE9B-2PgSqpTQCC9HrG7yO0rsnZtqs",
                    MODEL_PATH,
                    quiet=False
                )
        
        model = ResNet18Classifier()
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
        model.eval()
        return model

    # =============================
    # Main App Logic
    # =============================
    if st.session_state.page == 'home':
        show_home_page()
    elif st.session_state.page == 'guide':
        show_guide_page()
    elif st.session_state.page == 'analysis':
        show_analysis_page()
