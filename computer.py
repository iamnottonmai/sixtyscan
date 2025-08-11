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

st.set_page_config(
    page_title="SixtyScan - Parkinson Detection",
    page_icon="🎤",
    initial_sidebar_state="collapsed"
)
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
            "insert.jpg",           # Same directory
            "./insert.jpg",         # Explicit relative path
            "assets/insert.jpg",    # If in assets folder
            "images/insert.jpg"     # If in images folder
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
    # Global Styles - Fixed CSS
    # =============================
    def load_styles():
        css_content = """
            <style>
                /* Import fonts */
                @import url('https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;500;600;700;800;900&display=swap');
                
                /* Global Reset */
                .stApp {
                    background: linear-gradient(135deg, #f8f4ff 0%, #e8f4fd 100%) !important;
                    margin: 0 !important;
                    padding: 0 !important;
                    font-family: 'Prompt', sans-serif !important;
                    min-height: 100vh;
                }
                
                /* Hide Streamlit elements */
                .stDeployButton {display:none;}
                footer {visibility: hidden;}
                .stApp > header {visibility: hidden;}
                #MainMenu {visibility: hidden;}
                
                /* Header Styles - Redesigned */
                .header {
                    background: linear-gradient(135deg, #4A148C 0%, #6A1B9A 50%, #8E24AA 100%);
                    padding: 20px 60px;
                    display: flex;
                    align-items: center;
                    justify-content: space-between;
                    margin: 0;
                    width: 100%;
                    box-sizing: border-box;
                    box-shadow: 0 4px 20px rgba(0,0,0,0.1);
                }
                
                .header-left {
                    display: flex;
                    align-items: center;
                }
                
                .header-logo {
                    height: 56px;
                    width: auto;
                    margin-right: 24px;
                }
                
                .header-content {
                    display: flex;
                    align-items: center;
                }
                
                .header-divider {
                    width: 2px;
                    height: 40px;
                    background-color: rgba(255, 255, 255, 0.3);
                    margin: 0 24px;
                }
                
                .header-title {
                    color: white;
                    font-family: 'Prompt', sans-serif;
                    font-size: 28px;
                    font-weight: 500;
                    margin: 0;
                    letter-spacing: -0.5px;
                }
                
                .header-datetime {
                    background: rgba(255, 255, 255, 0.15);
                    color: white;
                    padding: 12px 24px;
                    border-radius: 25px;
                    font-family: 'Prompt', sans-serif;
                    font-size: 16px;
                    font-weight: 400;
                    backdrop-filter: blur(10px);
                    border: 1px solid rgba(255, 255, 255, 0.2);
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
                    gap: 80px;
                }
                
                .content-left {
                    flex: 1;
                    max-width: 600px;
                }
                
                .content-right {
                    flex: 1;
                    text-align: center;
                    max-width: 600px;
                }
                
                /* Main Title - Updated */
                .main-title {
                    font-family: 'Prompt', sans-serif;
                    font-size: 72px;
                    font-weight: 700;
                    color: #2d2d2d;
                    line-height: 1.1;
                    margin-bottom: 60px;
                    margin-top: 0;
                    letter-spacing: -1px;
                }
                
                .title-highlight {
                    color: #6A1B9A;
                    font-weight: 800;
                }
                
                /* Button Container */
                .button-container {
                    display: flex;
                    flex-direction: column;
                    gap: 24px;
                    align-items: flex-start;
                }
                
                /* Enhanced Button Styles */
                .stButton > button {
                    font-size: 28px !important;
                    padding: 20px 60px !important;
                    border-radius: 60px !important;
                    font-weight: 700 !important;
                    font-family: 'Prompt', sans-serif !important;
                    min-width: 320px !important;
                    height: 75px !important;
                    margin: 0 !important;
                    border: none !important;
                    cursor: pointer !important;
                    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
                    text-align: center !important;
                    display: flex !important;
                    align-items: center !important;
                    justify-content: center !important;
                    letter-spacing: 0.5px !important;
                    position: relative !important;
                    overflow: hidden !important;
                }
                
                /* Primary Button */
                .stButton:first-child > button {
                    background: linear-gradient(135deg, #1976D2 0%, #42A5F5 50%, #64B5F6 100%) !important;
                    color: white !important;
                    box-shadow: 0 6px 25px rgba(25, 118, 210, 0.4) !important;
                }
                
                .stButton:first-child > button:hover {
                    transform: translateY(-4px) !important;
                    box-shadow: 0 12px 35px rgba(25, 118, 210, 0.5) !important;
                    background: linear-gradient(135deg, #1565C0 0%, #1976D2 50%, #42A5F5 100%) !important;
                }
                
                /* Secondary Button */
                .stButton:nth-child(2) > button {
                    background: linear-gradient(135deg, #4A148C 0%, #6A1B9A 50%, #8E24AA 100%) !important;
                    color: white !important;
                    box-shadow: 0 6px 25px rgba(74, 20, 140, 0.4) !important;
                }
                
                .stButton:nth-child(2) > button:hover {
                    transform: translateY(-4px) !important;
                    box-shadow: 0 12px 35px rgba(74, 20, 140, 0.5) !important;
                    background: linear-gradient(135deg, #38006b 0%, #4A148C 50%, #6A1B9A 100%) !important;
                }
                
                .stButton > button:active {
                    transform: translateY(-1px) !important;
                }
                
                /* Woman Image - Enhanced */
                .woman-image {
                    width: 100%;
                    max-width: 520px;
                    height: auto;
                    border-radius: 24px;
                    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.15);
                    transition: transform 0.3s ease;
                }
                
                .woman-image:hover {
                    transform: translateY(-5px);
                }
                
                /* Analysis page styles */
                .card {
                    background-color: #ffffff;
                    border-radius: 20px;
                    padding: 40px;
                    box-shadow: 0 8px 32px rgba(0,0,0,0.08);
                    margin-bottom: 32px;
                    border: 1px solid rgba(255, 255, 255, 0.2);
                    backdrop-filter: blur(10px);
                }
                
                .card h2 {
                    font-size: 48px;
                    margin-bottom: 20px;
                    color: #222;
                    font-weight: 600;
                    font-family: 'Prompt', sans-serif;
                }
                
                .instructions {
                    font-size: 26px !important;
                    color: #444;
                    margin-bottom: 24px;
                    font-weight: 400;
                    font-family: 'Prompt', sans-serif;
                    line-height: 1.5;
                }
                
                .pronounce {
                    font-size: 24px !important;
                    color: #000;
                    font-weight: 500;
                    margin-top: 0;
                    margin-bottom: 20px;
                    font-family: 'Prompt', sans-serif;
                }
                
                .sentence-instruction {
                    font-size: 26px !important;
                    font-weight: 400 !important;
                    color: #444 !important;
                    margin-bottom: 24px !important;
                    font-family: 'Prompt', sans-serif !important;
                    display: block !important;
                    line-height: 1.5 !important;
                }
                
                /* Back button styling */
                .stButton[data-testid="stButton"]:has(button:contains("←")) > button {
                    background: linear-gradient(135deg, #6c757d 0%, #5a6268 100%) !important;
                    color: white !important;
                    font-size: 18px !important;
                    padding: 12px 24px !important;
                    min-width: auto !important;
                    height: 45px !important;
                    border-radius: 25px !important;
                    margin-bottom: 20px !important;
                }
                
                /* Responsive adjustments */
                @media (max-width: 1200px) {
                    .main-content {
                        flex-direction: column;
                        text-align: center;
                        padding: 60px 40px;
                        gap: 60px;
                    }
                    
                    .content-left {
                        max-width: none;
                    }
                    
                    .content-right {
                        max-width: none;
                    }
                    
                    .main-title {
                        font-size: 56px;
                    }
                    
                    .button-container {
                        align-items: center;
                    }
                    
                    .header {
                        padding: 16px 40px;
                    }
                    
                    .header-title {
                        font-size: 24px;
                    }
                }
                
                @media (max-width: 768px) {
                    .main-title {
                        font-size: 48px;
                    }
                    
                    .header {
                        flex-direction: column;
                        gap: 16px;
                        text-align: center;
                    }
                    
                    .header-content {
                        flex-direction: column;
                        gap: 16px;
                    }
                    
                    .header-divider {
                        display: none;
                    }
                    
                    .stButton > button {
                        min-width: 280px !important;
                        font-size: 24px !important;
                    }
                }
                
                /* Ensure all text elements use Prompt font */
                * {
                    font-family: 'Prompt', sans-serif !important;
                }
                
                h1, h2, h3, h4, h5, h6 {
                    font-family: 'Prompt', sans-serif !important;
                }
                
                p, div, span, label {
                    font-family: 'Prompt', sans-serif !important;
                }
                
                /* Enhanced placeholder styling */
                .image-placeholder {
                    width: 100%;
                    max-width: 520px;
                    height: 400px;
                    background: linear-gradient(135deg, #e3f2fd 0%, #f3e5f5 50%, #e8f5e8 100%);
                    border-radius: 24px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.15);
                    transition: transform 0.3s ease;
                    border: 2px dashed rgba(106, 27, 154, 0.3);
                }
                
                .image-placeholder:hover {
                    transform: translateY(-5px);
                }
                
                .placeholder-content {
                    text-align: center;
                    color: #666;
                }
                
                .placeholder-icon {
                    font-size: 64px;
                    margin-bottom: 16px;
                    opacity: 0.7;
                }
                
                .placeholder-text {
                    font-size: 20px;
                    font-weight: 500;
                    font-family: 'Prompt', sans-serif;
                }
            </style>
        """
        st.markdown(css_content, unsafe_allow_html=True)

    # =============================
    # Analysis Functions
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

    def audio_to_mel_tensor(file_path):
        """Convert audio file to mel spectrogram tensor"""
        try:
            from pydub import AudioSegment
        except ImportError:
            st.error("pydub library is required. Please install it with: pip install pydub")
            return None
        
        # Convert to WAV if necessary
        if not file_path.lower().endswith(".wav"):
            audio = AudioSegment.from_file(file_path)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                audio.export(tmp.name, format="wav")
                file_path = tmp.name

        y, sr = librosa.load(file_path, sr=22050)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_db = librosa.power_to_db(mel, ref=np.max)

        fig, ax = plt.subplots(figsize=(2.24, 2.24), dpi=100)
        ax.axis('off')
        librosa.display.specshow(mel_db, sr=sr, ax=ax)

        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)

        buf.seek(0)
        image = Image.open(buf).convert('RGB')

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        return transform(image).unsqueeze(0)

    def create_mel_spectrogram_display(file_path, title="Mel Spectrogram"):
        """Create a mel spectrogram for display purposes"""
        try:
            from pydub import AudioSegment
            
            # Convert to WAV if necessary
            if not file_path.lower().endswith(".wav"):
                audio = AudioSegment.from_file(file_path)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    audio.export(tmp.name, format="wav")
                    file_path = tmp.name

            y, sr = librosa.load(file_path, sr=22050)
            mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
            mel_db = librosa.power_to_db(mel, ref=np.max)

            fig, ax = plt.subplots(figsize=(8, 4), dpi=100, facecolor='white')
            
            img = librosa.display.specshow(mel_db, sr=sr, ax=ax, x_axis='time', y_axis='mel', 
                                          cmap='plasma', fmax=8000)
            
            ax.set_xlabel('Time (s)', fontsize=12)
            ax.set_ylabel('Mel Frequency', fontsize=12)
            
            cbar = plt.colorbar(img, ax=ax, format='%+2.0f dB')
            cbar.set_label('Power (dB)', fontsize=10)
            
            ax.grid(True, alpha=0.3)
            plt.tight_layout()

            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=150, facecolor='white')
            plt.close(fig)
            
            buf.seek(0)
            return Image.open(buf)
            
        except Exception as e:
            st.error(f"Error creating spectrogram: {str(e)}")
            return None

    def predict_from_model(vowel_paths, pataka_path, sentence_path, model):
        """Make predictions from the model"""
        inputs = [audio_to_mel_tensor(p) for p in vowel_paths]
        inputs.append(audio_to_mel_tensor(pataka_path))
        inputs.append(audio_to_mel_tensor(sentence_path))
        with torch.no_grad():
            return [F.softmax(model(x), dim=1)[0][1].item() for x in inputs]

    # =============================
    # Page Functions
    # =============================
    def show_header():
        """Display the enhanced header matching the image design"""
        logo_b64 = load_logo()
        current_time = datetime.now()
        thai_datetime = current_time.strftime("%d/%m/%Y %H:%M")
        
        logo_html = ""
        if logo_b64:
            logo_html = f'<img src="data:image/png;base64,{logo_b64}" class="header-logo" alt="SixtyScan Logo">'
        
        header_html = f"""
            <div class="header">
                <div class="header-left">
                    <div class="header-content">
                        {logo_html}
                        <div class="header-title">SixtyScan</div>
                        <div class="header-divider"></div>
                        <div class="header-title">นวัตกรรมคัดกรองโรคพาร์กินสันจากเสียง</div>
                    </div>
                </div>
                <div class="header-datetime">{thai_datetime}</div>
            </div>
        """
        st.markdown(header_html, unsafe_allow_html=True)

    def show_home_page():
        """Display the enhanced home page matching the design image"""
        load_styles()
        show_header()
        
        woman_image_b64 = load_woman_image()
        
        # Main content area
        main_content_start = """
            <div class="main-content">
                <div class="content-left">
                    <h1 class="main-title">
                        ตรวจเช็คโรคพาร์กินสัน<br>
                        ทันทีด้วย <span class="title-highlight">SixtyScan</span>
                    </h1>
        """
        st.markdown(main_content_start, unsafe_allow_html=True)
        
        # Button container with enhanced styling
        col1, col2 = st.columns([1, 1], gap="medium")
        
        with col1:
            if st.button("เริ่มใช้งาน", key="start_analysis", use_container_width=True):
                st.session_state.page = 'analysis'
                st.rerun()
        
        with col2:
            if st.button("คู่มือ", key="guide_manual", use_container_width=True):
                st.session_state.page = 'guide'
                st.rerun()
        
        content_right_start = """
                </div>
                <div class="content-right">
        """
        st.markdown(content_right_start, unsafe_allow_html=True)
        
        # Display woman image with enhanced styling
        if woman_image_b64:
            image_html = f'<img src="data:image/jpg;base64,{woman_image_b64}" class="woman-image" alt="Woman using phone">'
            st.markdown(image_html, unsafe_allow_html=True)
        else:
            # Enhanced placeholder
            placeholder_html = """
                <div class="image-placeholder">
                    <div class="placeholder-content">
                        <div class="placeholder-icon">📱</div>
                        <div class="placeholder-text">insert.jpg<br>not found</div>
                    </div>
                </div>
            """
            st.markdown(placeholder_html, unsafe_allow_html=True)
        
        main_content_end = """
                </div>
            </div>
        """
        st.markdown(main_content_end, unsafe_allow_html=True)

    def show_guide_page():
        """Display the guide/manual page with enhanced styling"""
        load_styles()
        show_header()
        
        # Back button
        if st.button("← กลับหน้าแรก", key="back_to_home_from_guide"):
            st.session_state.page = 'home'
            st.rerun()
        
        guide_content = """
            <div style="max-width: 1000px; margin: 40px auto; padding: 0 40px;">
                <h1 style="text-align: center; color: #4A148C; font-size: 56px; margin-bottom: 50px; font-family: 'Prompt', sans-serif; font-weight: 700;">คู่มือการใช้งาน SixtyScan</h1>
                
                <div style="background: white; padding: 50px; border-radius: 20px; box-shadow: 0 8px 32px rgba(0,0,0,0.08); margin-bottom: 40px;">
                    <h2 style="color: #4A148C; font-size: 36px; margin-bottom: 30px; font-family: 'Prompt', sans-serif;">การเตรียมตัวก่อนการตรวจ</h2>
                    <ul style="font-size: 22px; line-height: 1.7; font-family: 'Prompt', sans-serif;">
                        <li>หาสถานที่เงียบ ปราศจากเสียงรบกวน</li>
                        <li>ใช้ไมโครโฟนหรืออุปกรณ์บันทึกเสียงที่มีคุณภาพ</li>
                        <li>นั่งหรือยืนในท่าที่สบาย</li>
                        <li>พักผ่อนเพียงพอก่อนการตรวจ</li>
                    </ul>
                </div>
                
                <div style="background: white; padding: 50px; border-radius: 20px; box-shadow: 0 8px 32px rgba(0,0,0,0.08); margin-bottom: 40px;">
                    <h2 style="color: #4A148C; font-size: 36px; margin-bottom: 30px; font-family: 'Prompt', sans-serif;">ขั้นตอนการตรวจ</h2>
                    <div style="font-size: 22px; line-height: 1.7; font-family: 'Prompt', sans-serif;">
                        <h3 style="color: #666; font-size: 28px; margin-top: 30px;">1. การออกเสียงสระ</h3>
                        <ul>
                            <li>ออกเสียงสระแต่ละตัว 5-8 วินาที</li>
                            <li>ออกเสียงให้ชัดเจนและคงที่</li>
                            <li>ไม่ต้องออกเสียงดังเกินไป</li>
                        </ul>
                        
                        <h3 style="color: #666; font-size: 28px; margin-top: 30px;">2. การออกเสียงพยางค์</h3>
                        <ul>
                            <li>ออกเสียง "พา-ทา-คา" ซ้ำๆ</li>
                            <li>ใช้เวลาประมาณ 6 วินาที</li>
                            <li>พยายามออกเสียงให้เร็วและชัดเจน</li>
                        </ul>
                        
                        <h3 style="color: #666; font-size: 28px; margin-top: 30px;">3. การอ่านประโยค</h3>
                        <ul>
                            <li>อ่านประโยคที่กำหนดให้อย่างเป็นธรรมชาติ</li>
                            <li>ไม่ต้องรีบร้อน</li>
                            <li>ออกเสียงให้ชัดเจน</li>
                        </ul>
                    </div>
                </div>
                
                <div style="background: white; padding: 50px; border-radius: 20px; box-shadow: 0 8px 32px rgba(0,0,0,0.08);">
                    <h2 style="color: #4A148C; font-size: 36px; margin-bottom: 30px; font-family: 'Prompt', sans-serif;">ข้อควรระวัง</h2>
                    <ul style="font-size: 22px; line-height: 1.7; color: #d32f2f; font-family: 'Prompt', sans-serif;">
                        <li><strong>ระบบนี้เป็นเพียงการตรวจคัดกรองเบื้องต้น</strong></li>
                        <li><strong>ไม่สามารถทดแทนการวินิจฉัยโดยแพทย์ได้</strong></li>
                        <li><strong>หากมีข้อสงสัยควรปรึกษาแพทย์เฉพาะทาง</strong></li>
                    </ul>
                </div>
            </div>
        """
        st.markdown(guide_content, unsafe_allow_html=True)

    def show_analysis_page():
        """Display the analysis page with consistent styling"""
        load_styles()
        initialize_analysis_session_state()
        
        # Header
        show_header()
        
        # Back button
        if st.button("← กลับหน้าแรก", key="back_to_home"):
            st.session_state.page = 'home'
            st.rerun()
        
        # Load model
        model = load_model()
        
        st.markdown("<h1 style='text-align: center; font-size: 56px; color: #4A148C; margin: 30px 0; font-family: \"Prompt\", sans-serif; font-weight: 700;'>การวิเคราะห์เสียง</h1>", unsafe_allow_html=True)

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
        vowel_card_html = """
        <div class='card'>
            <h2>1. สระ</h2>
            <p class='instructions'>กรุณาออกเสียงแต่ละสระ 5-8 วินาทีอย่างชัดเจน โดยกดปุ่มบันทึกทีละสระ</p>
        </div>
        """
        st.markdown(vowel_card_html, unsafe_allow_html=True)

        vowel_sounds = ["อา", "อี", "อือ", "อู", "ไอ", "อำ", "เอา"]

        for i, sound in enumerate(vowel_sounds):
            st.markdown(f"<p class='pronounce'>ออกเสียง <b>\"{sound}\"</b></p>", unsafe_allow_html=True)
            
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
            
            if i < len(st.session_state.vowel_files) and st.session_state.vowel_files[i]:
                spec_image = create_mel_spectrogram_display(st.session_state.vowel_files[i], f"สระ \"{sound}\"")
                if spec_image:
                    st.markdown(f"<div style='color: black; font-size: 18px; margin-bottom: 12px; text-align: center; font-family: \"Prompt\", sans-serif; font-weight: 500;'>Mel Spectrogram: <b>\"{sound}\"</b></div>", unsafe_allow_html=True)
                    st.image(spec_image, use_container_width=True)

        # File uploader for vowels
        uploaded_vowels = st.file_uploader("อัปโหลดไฟล์เสียงสระ (7 ไฟล์)", type=["wav", "mp3", "m4a"], accept_multiple_files=True)
        if uploaded_vowels and len([f for f in st.session_state.vowel_files if f is not None]) < 7:
            cleanup_temp_files()
            st.session_state.vowel_files = []
            for file in uploaded_vowels[:7]:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    tmp.write(file.read())
                    st.session_state.vowel_files.append(tmp.name)

        # Pataka recording
        pataka_card_html = """
        <div class='card'>
            <h2>2. พยางค์</h2>
            <p class='instructions'>กรุณาออกเสียงคำว่า <b>"พา - ทา - คา"</b> ให้จบภายใน 6 วินาที</p>
        </div>
        """
        st.markdown(pataka_card_html, unsafe_allow_html=True)

        if not st.session_state.clear_clicked:
            pataka_bytes = st.audio_input("🎤 บันทึกเสียงพยางค์", key="pataka")
            if pataka_bytes:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    tmp.write(pataka_bytes.read())
                    if st.session_state.pataka_file and os.path.exists(st.session_state.pataka_file):
                        os.unlink(st.session_state.pataka_file)
                    st.session_state.pataka_file = tmp.name
                st.success("บันทึกพยางค์สำเร็จ", icon="✅")
        else:
            pataka_bytes = st.audio_input("🎤 บันทึกเสียงพยางค์", key="pataka_new")

        if st.session_state.pataka_file:
            spec_image = create_mel_spectrogram_display(st.session_state.pataka_file, "พยางค์")
            if spec_image:
                st.markdown("<div style='color: black; font-size: 18px; margin-bottom: 12px; text-align: center; font-family: \"Prompt\", sans-serif; font-weight: 500;'>Mel Spectrogram: <b>\"พา-ทา-คา\"</b></div>", unsafe_allow_html=True)
                st.image(spec_image, use_container_width=True)

        # File uploader for pataka
        uploaded_pataka = st.file_uploader("อัปโหลดไฟล์เสียงพยางค์", type=["wav", "mp3", "m4a"], accept_multiple_files=False)
        if uploaded_pataka and not st.session_state.pataka_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(uploaded_pataka.read())
                st.session_state.pataka_file = tmp.name

        # Sentence recording
        sentence_card_html = """
        <div class='card'>
            <h2>3. ประโยค</h2>
            <p class='sentence-instruction'>กรุณาอ่านประโยค <b>"วันนี้อากาศแจ่มใสนกร้องเสียงดังเป็นจังหวะ"</b></p>
        </div>
        """
        st.markdown(sentence_card_html, unsafe_allow_html=True)

        if not st.session_state.clear_clicked:
            sentence_bytes = st.audio_input("🎤 บันทึกการอ่านประโยค", key="sentence")
            if sentence_bytes:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    tmp.write(sentence_bytes.read())
                    if st.session_state.sentence_file and os.path.exists(st.session_state.sentence_file):
                        os.unlink(st.session_state.sentence_file)
                    st.session_state.sentence_file = tmp.name
                st.success("บันทึกประโยคสำเร็จ", icon="✅")
        else:
            sentence_bytes = st.audio_input("🎤 บันทึกการอ่านประโยค", key="sentence_new")

        if st.session_state.sentence_file:
            spec_image = create_mel_spectrogram_display(st.session_state.sentence_file, "ประโยค")
            if spec_image:
                st.markdown("<div style='color: black; font-size: 18px; margin-bottom: 12px; text-align: center; font-family: \"Prompt\", sans-serif; font-weight: 500;'>Mel Spectrogram: <b>\"วันนี้อากาศแจ่มใสนกร้องเสียงดังเป็นจังหวะ\"</b></div>", unsafe_allow_html=True)
                st.image(spec_image, use_container_width=True)

        # File uploader for sentence
        uploaded_sentence = st.file_uploader("อัปโหลดไฟล์เสียงประโยค", type=["wav", "mp3", "m4a"], accept_multiple_files=False)
        if uploaded_sentence and not st.session_state.sentence_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(uploaded_sentence.read())
                st.session_state.sentence_file = tmp.name

        # Action buttons with enhanced styling
        col1, col2 = st.columns([1, 1])
        with col1:
            predict_btn = st.button("🔍 วิเคราะห์", key="predict", type="primary", use_container_width=True)
        with col2:
            if st.button("🗑️ ลบข้อมูล", key="clear", type="secondary", use_container_width=True):
                st.session_state.clear_button_clicked = True
                st.rerun()

        # Reset clear_clicked flag
        if st.session_state.clear_clicked:
            st.session_state.clear_clicked = False

        # Prediction logic
        if predict_btn:
            valid_vowel_files = [f for f in st.session_state.vowel_files if f is not None]
            
            if len(valid_vowel_files) == 7 and st.session_state.pataka_file and st.session_state.sentence_file:
                with st.spinner("กำลังวิเคราะห์..."):
                    try:
                        all_probs = predict_from_model(valid_vowel_files, st.session_state.pataka_file, st.session_state.sentence_file, model)
                        final_prob = np.mean(all_probs)
                        percent = int(final_prob * 100)

                        if percent <= 50:
                            level = "ระดับต่ำ (Low)"
                            label = "Non Parkinson"
                            diagnosis = "ไม่เป็นพาร์กินสัน"
                            box_color = "#e8f5e9"
                            border_color = "#4caf50"
                            advice_html = """
                            <ul style='font-size:26px; font-family: "Prompt", sans-serif; line-height: 1.6;'>
                                <li>ถ้าไม่มีอาการ: ควรตรวจปีละครั้ง(ไม่บังคับ)</li>
                                <li>ถ้ามีอาการเล็กน้อย: ตรวจปีละ 2 ครั้ง</li>
                                <li>ถ้ามีอาการเตือน: ตรวจ 2–4 ครั้งต่อปี</li>
                            </ul>
                            """
                        elif percent <= 75:
                            level = "ปานกลาง (Moderate)"
                            label = "Parkinson"
                            diagnosis = "เป็นพาร์กินสัน"
                            box_color = "#fff8e1"
                            border_color = "#ff9800"
                            advice_html = """
                            <ul style='font-size:26px; font-family: "Prompt", sans-serif; line-height: 1.6;'>
                                <li>พบแพทย์เฉพาะทางระบบประสาท</li>
                                <li>บันทึกอาการประจำวัน</li>
                                <li>หากได้รับยา: บันทึกผลข้างเคียง</li>
                            </ul>
                            """
                        else:
                            level = "สูง (High)"
                            label = "Parkinson"
                            diagnosis = "เป็นพาร์กินสัน"
                            box_color = "#ffebee"
                            border_color = "#f44336"
                            advice_html = """
                            <ul style='font-size:26px; font-family: "Prompt", sans-serif; line-height: 1.6;'>
                                <li>พบแพทย์เฉพาะทางโดยเร็วที่สุด</li>
                                <li>บันทึกอาการทุกวัน</li>
                                <li>หากได้รับยา: ติดตามผลอย่างละเอียด</li>
                            </ul>
                            """

                        results_html = f"""
                            <div style='background-color:{box_color}; padding: 40px; border-radius: 20px; font-size: 28px; color: #000000; font-family: "Prompt", sans-serif; border-left: 8px solid {border_color}; box-shadow: 0 8px 32px rgba(0,0,0,0.08); margin: 30px 0;'>
                                <div style='text-align: center; font-size: 48px; font-weight: 700; margin-bottom: 30px; color: {border_color};'>{label}</div>
                                <p style='margin-bottom: 20px;'><b>ระดับความน่าจะเป็น:</b> {level}</p>
                                <p style='margin-bottom: 20px;'><b>ความน่าจะเป็นของพาร์กินสัน:</b> {percent}%</p>
                                <div style='height: 40px; background: linear-gradient(to right, #4caf50, #ff9800, #f44336); border-radius: 20px; margin-bottom: 25px; position: relative; box-shadow: inset 0 2px 4px rgba(0,0,0,0.1);'>
                                    <div style='position: absolute; left: {percent}%; top: -5px; bottom: -5px; width: 6px; background-color: #333; border-radius: 3px; box-shadow: 0 2px 8px rgba(0,0,0,0.3);'></div>
                                </div>
                                <p style='margin-bottom: 20px;'><b>ผลการวิเคราะห์:</b> {diagnosis}</p>
                                <p style='margin-bottom: 15px; font-size: 30px; font-weight: 600;'><b>คำแนะนำ</b></p>
                                {advice_html}
                            </div>
                        """
                        st.markdown(results_html, unsafe_allow_html=True)
                        
                        # Display all spectrograms in the results section
                        st.markdown("### 📊 การวิเคราะห์ Mel Spectrogram ทั้งหมด")
                        
                        # Create a grid layout for all spectrograms
                        spec_cols = st.columns(3)
                        
                        # Display vowel spectrograms
                        for i, (sound, file_path) in enumerate(zip(vowel_sounds, valid_vowel_files)):
                            with spec_cols[i % 3]:
                                spec_image = create_mel_spectrogram_display(file_path, f"สระ \"{sound}\"")
                                if spec_image:
                                    st.markdown(f"<div style='color: black; font-size: 16px; margin-bottom: 10px; text-align: center; font-family: \"Prompt\", sans-serif; font-weight: 500;'>Mel Spectrogram: <b>\"{sound}\"</b></div>", unsafe_allow_html=True)
                                    st.image(spec_image, use_container_width=True)
                        
                        # Display pataka spectrogram
                        col_idx = len(vowel_sounds) % 3
                        with spec_cols[col_idx]:
                            spec_image = create_mel_spectrogram_display(st.session_state.pataka_file, "พยางค์")
                            if spec_image:
                                st.markdown("<div style='color: black; font-size: 16px; margin-bottom: 10px; text-align: center; font-family: \"Prompt\", sans-serif; font-weight: 500;'>Mel Spectrogram: <b>\"พา-ทา-คา\"</b></div>", unsafe_allow_html=True)
                                st.image(spec_image, use_container_width=True)
                        
                        # Display sentence spectrogram
                        col_idx = (len(vowel_sounds) + 1) % 3
                        with spec_cols[col_idx]:
                            spec_image = create_mel_spectrogram_display(st.session_state.sentence_file, "ประโยค")
                            if spec_image:
                                st.markdown("<div style='color: black; font-size: 16px; margin-bottom: 10px; text-align: center; font-family: \"Prompt\", sans-serif; font-weight: 500;'>Mel Spectrogram: <b>\"ประโยค\"</b></div>", unsafe_allow_html=True)
                                st.image(spec_image, use_container_width=True)
                        
                        info_html = """
                        <div style='margin-top: 30px; padding: 30px; background-color: #f8f9fa; border-radius: 16px; border-left: 6px solid #6A1B9A;'>
                            <h4 style='color: #4A148C; margin-bottom: 20px; font-family: "Prompt", sans-serif; font-size: 24px; font-weight: 600;'>💡 เกี่ยวกับ Mel Spectrogram</h4>
                            <p style='font-size: 18px; margin-bottom: 12px; font-family: "Prompt", sans-serif; line-height: 1.6;'>• <b>สีเข้ม (น้ำเงิน/ม่วง):</b> ความถี่ที่มีพลังงานต่ำ</p>
                            <p style='font-size: 18px; margin-bottom: 12px; font-family: "Prompt", sans-serif; line-height: 1.6;'>• <b>สีอ่อน (เหลือง/แดง):</b> ความถี่ที่มีพลังงานสูง</p>
                            <p style='font-size: 18px; margin-bottom: 12px; font-family: "Prompt", sans-serif; line-height: 1.6;'>• <b>แกน X:</b> เวลา (วินาที)</p>
                            <p style='font-size: 18px; margin-bottom: 12px; font-family: "Prompt", sans-serif; line-height: 1.6;'>• <b>แกน Y:</b> ความถี่ Mel</p>
                            <p style='font-size: 18px; font-family: "Prompt", sans-serif; line-height: 1.6;'>• รูปแบบของ Spectrogram สามารถช่วยระบุความผิดปกติของการออกเสียงได้</p>
                        </div>
                        """
                        st.markdown(info_html, unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"เกิดข้อผิดพลาดในการวิเคราะห์: {str(e)}")
            else:
                st.warning("กรุณาอัดเสียงหรืออัปโหลดให้ครบทั้ง 7 สระ พยางค์ และประโยค", icon="⚠")

    # =============================
    # Main App Logic
    # =============================
    if st.session_state.page == 'home':
        show_home_page()
    elif st.session_state.page == 'guide':
        show_guide_page()
    elif st.session_state.page == 'analysis':
        show_analysis_page()

# Run the app
if __name__ == "__main__":
    run_desktop_app()
