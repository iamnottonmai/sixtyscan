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
import pytz
import atexit
from pathlib import Path

# Configuration
CONFIG = {
    'MODEL_PATH': "best_resnet18.pth",
    'MODEL_URL': "https://drive.google.com/uc?id=1_oHE9B-2PgSqpTQCC9HrG7yO0rsnZtqs",
    'CSS_FILE': "deskstyle.css",
    'LOGO_PATHS': ["logo.png", "./logo.png", "assets/logo.png", "images/logo.png"],
    'IMAGE_PATHS': ["insert.jpg", "./insert.jpg", "assets/insert.jpg", "images/insert.jpg"],
    'THAI_TIMEZONE': 'Asia/Bangkok'
}

st.set_page_config(
    page_title="SixtyScan - Parkinson Detection",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Import your model class
try:
    from model import ResNet18Classifier
except ImportError:
    st.error("Could not import ResNet18Classifier from model.py. Make sure the file exists.")
    st.stop()

# =============================
# Utility Functions
# =============================
def initialize_session_state():
    """Initialize all session state variables"""
    defaults = {
        'page': 'home',
        'vowel_files': [],
        'pataka_file': None,
        'sentence_file': None,
        'clear_clicked': False,
        'temp_files': []  # Track all temp files for cleanup
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

def cleanup_temp_files(file_list):
    """Clean up specific temporary files"""
    for file_path in file_list:
        try:
            if file_path and os.path.exists(file_path):
                os.unlink(file_path)
        except Exception as e:
            st.warning(f"Failed to delete temp file {file_path}: {str(e)}")

def cleanup_all_temp_files():
    """Clean up all temporary files stored in session state"""
    if 'temp_files' in st.session_state:
        cleanup_temp_files(st.session_state.temp_files)
        st.session_state.temp_files = []

def add_temp_file(file_path):
    """Add a file to the temp files tracking list"""
    if 'temp_files' not in st.session_state:
        st.session_state.temp_files = []
    st.session_state.temp_files.append(file_path)

def run_desktop_app():
    """Main function to run the desktop version"""
    # Initialize Session State
    initialize_session_state()
    
    # Register cleanup function
    atexit.register(cleanup_all_temp_files)
    
    # =============================
    # Page-specific Functions  
    # =============================
    def load_css():
        """Load external CSS file"""
        css_file = Path(CONFIG['CSS_FILE'])
        if css_file.exists():
            with open(css_file, 'r', encoding='utf-8') as f:
                st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
        else:
            st.warning(f"CSS file '{CONFIG['CSS_FILE']}' not found. Using minimal styling.")
            # Fallback minimal CSS
            st.markdown("""
                <style>
                @import url('https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;500;600;700;800;900&display=swap');
                * { font-family: 'Prompt', sans-serif !important; }
                .stApp { background: linear-gradient(135deg, #f8f4ff 0%, #e8f4fd 100%) !important; }
                </style>
            """, unsafe_allow_html=True)

    @st.cache_data
    def load_image_file(image_paths, alt_text="Image"):
        """Generic function to load image files with fallback options"""
        for path in image_paths:
            try:
                if os.path.exists(path):
                    with open(path, "rb") as f:
                        return base64.b64encode(f.read()).decode()
            except Exception as e:
                st.warning(f"Failed to load {path}: {str(e)}")
                continue
        return None

    def get_thai_time():
        """Get current Thai time formatted for display"""
        try:
            thai_tz = pytz.timezone(CONFIG['THAI_TIMEZONE'])
            now = datetime.now(thai_tz)
            return now.strftime("%d/%m/%Y %H:%M:%S")
        except Exception as e:
            st.error(f"Error getting Thai time: {str(e)}")
            return datetime.now().strftime("%d/%m/%Y %H:%M:%S")

    def save_uploaded_file(uploaded_file):
        """Save uploaded file to temporary location"""
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
                tmp.write(uploaded_file.read())
                add_temp_file(tmp.name)
                return tmp.name
        except Exception as e:
            st.error(f"Error saving uploaded file: {str(e)}")
            return None

    # =============================
    # Model and Analysis Functions
    # =============================
    @st.cache_resource
    def load_model():
        """Load the ResNet18 model with error handling"""
        try:
            if not os.path.exists(CONFIG['MODEL_PATH']):
                with st.spinner("Downloading model..."):
                    gdown.download(
                        CONFIG['MODEL_URL'],
                        CONFIG['MODEL_PATH'],
                        quiet=False
                    )
            
            model = ResNet18Classifier()
            model.load_state_dict(torch.load(CONFIG['MODEL_PATH'], map_location=torch.device("cpu")))
            model.eval()
            return model
        except Exception as e:
            st.error(f"Failed to load model: {str(e)}")
            return None

    def convert_to_wav_if_needed(file_path):
        """Convert audio file to WAV format if necessary"""
        try:
            from pydub import AudioSegment
            
            if not file_path.lower().endswith(".wav"):
                audio = AudioSegment.from_file(file_path)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    audio.export(tmp.name, format="wav")
                    add_temp_file(tmp.name)
                    return tmp.name
            return file_path
        except ImportError:
            st.error("pydub library is required. Please install it with: pip install pydub")
            return None
        except Exception as e:
            st.error(f"Error converting audio file: {str(e)}")
            return None

    def audio_to_mel_tensor(file_path):
        """Convert audio file to mel spectrogram tensor"""
        try:
            # Convert to WAV if necessary
            wav_file = convert_to_wav_if_needed(file_path)
            if not wav_file:
                return None

            y, sr = librosa.load(wav_file, sr=22050)
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
        except Exception as e:
            st.error(f"Error creating mel tensor: {str(e)}")
            return None

    def create_mel_spectrogram_display(file_path, title="Mel Spectrogram"):
        """Create a mel spectrogram for display purposes"""
        try:
            # Convert to WAV if necessary
            wav_file = convert_to_wav_if_needed(file_path)
            if not wav_file:
                return None

            y, sr = librosa.load(wav_file, sr=22050)
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
        """Make predictions from the model with error handling"""
        try:
            inputs = []
            
            # Process vowel files
            for path in vowel_paths:
                tensor = audio_to_mel_tensor(path)
                if tensor is not None:
                    inputs.append(tensor)
                else:
                    st.error(f"Failed to process vowel file: {path}")
                    return None
            
            # Process pataka and sentence
            for path in [pataka_path, sentence_path]:
                tensor = audio_to_mel_tensor(path)
                if tensor is not None:
                    inputs.append(tensor)
                else:
                    st.error(f"Failed to process file: {path}")
                    return None
            
            # Make predictions
            with torch.no_grad():
                predictions = []
                for tensor in inputs:
                    output = model(tensor)
                    prob = F.softmax(output, dim=1)[0][1].item()
                    predictions.append(prob)
                
                return predictions
                
        except Exception as e:
            st.error(f"Error making predictions: {str(e)}")
            return None

    # =============================
    # Page Functions
    # =============================
    def get_header_html():
        """Get the header HTML without rendering it"""
        logo_b64 = load_image_file(CONFIG['LOGO_PATHS'], "SixtyScan Logo")
        current_time = get_thai_time()
        
        logo_html = ""
        if logo_b64:
            logo_html = f'<img src="data:image/png;base64,{logo_b64}" class="header-logo" style="height: 56px; width: auto; margin-right: 24px;" alt="SixtyScan Logo">'
        
        return f"""
            <div class="header-container">
                <div class="logo-section">
                    {logo_html}
                    <div class="logo-text">SixtyScan</div>
                    <div class="header-divider"></div>
                    <div class="tagline">นวัตกรรมคัดกรองโรคพาร์กินสันจากเสียง</div>
                </div>
                <div class="datetime-display">{current_time}</div>
            </div>
        """

    def show_home_page():
        """Display the home page - FIXED VERSION"""
        load_css()
        
        woman_image_b64 = load_image_file(CONFIG['IMAGE_PATHS'], "Woman using phone")
        
        # SOLUTION: Combine header and main content in ONE st.markdown call
        combined_html = f"""
            {get_header_html()}
            <div class="main-content">
                <div class="content-wrapper">
                    <div class="text-section">
                        <h1 class="main-title">
                            ตรวจเช็คโรคพาร์กินสัน<br>ทันทีด้วย <span class="highlight">SixtyScan</span>
                        </h1>
                    </div>
                    <div class="image-section">
                        {f'<img src="data:image/jpg;base64,{woman_image_b64}" alt="Woman using phone" class="main-image">' if woman_image_b64 else '''
                        <div class="image-placeholder">
                            <div class="placeholder-content">
                                <div class="placeholder-icon">📱</div>
                                <div class="placeholder-text">
                                    insert.jpg<br>not found
                                </div>
                            </div>
                        </div>
                        '''}
                    </div>
                </div>
            </div>
        """
        
        # Render the combined HTML (no gap between header and content!)
        st.markdown(combined_html, unsafe_allow_html=True)
        
        # Now add the interactive buttons using Streamlit columns
        st.markdown('<div class="button-container" style="margin: -50px auto 0 auto; max-width: 600px;">', unsafe_allow_html=True)
        
        btn_col1, btn_col2 = st.columns([1, 1], gap="medium")
        
        with btn_col1:
            if st.button("เริ่มใช้งาน", key="start_analysis", use_container_width=True):
                st.session_state.page = 'analysis'
                st.rerun()
        
        with btn_col2:
            if st.button("คู่มือ", key="guide_manual", use_container_width=True):
                st.session_state.page = 'guide'
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)

    def show_guide_page():
        """Display the guide/manual page with proper styling"""
        load_css()
        
        # FIXED: Combine header with back button and title
        guide_html = f"""
            {get_header_html()}
            <div class="guide-container">
                <h1 class="guide-title">คู่มือการใช้งาน SixtyScan</h1>
            </div>
        """
        
        st.markdown(guide_html, unsafe_allow_html=True)
        
        # Back button
        if st.button("← กลับหน้าแรก", key="back_to_home_from_guide"):
            st.session_state.page = 'home'
            st.rerun()
    
        # Guide content - Fixed version with proper HTML structure
        st.markdown("""
            <div style="max-width: 1000px; margin: 0 auto; padding: 0 40px;">
                <div style="background: white; padding: 40px; border-radius: 20px; box-shadow: 0 8px 32px rgba(0,0,0,0.08); margin-bottom: 32px;">
                    <h2 style="color: #4A148C; font-size: 36px; margin-bottom: 24px; margin-top: 0; font-family: 'Prompt', sans-serif;">การเตรียมตัวก่อนการตรวจ</h2>
                    <ul style="font-size: 22px; line-height: 1.7; font-family: 'Prompt', sans-serif; margin-top: 0; padding-left: 24px;">
                        <li style="margin-bottom: 8px;">หาสถานที่เงียบ ปราศจากเสียงรบกวน</li>
                        <li style="margin-bottom: 8px;">ใช้ไมโครโฟนหรืออุปกรณ์บันทึกเสียงที่มีคุณภาพ</li>
                        <li style="margin-bottom: 8px;">นั่งหรือยืนในท่าที่สบาย</li>
                        <li style="margin-bottom: 8px;">พักผ่อนเพียงพอก่อนการตรวจ</li>
                    </ul>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <div style="max-width: 1000px; margin: 0 auto; padding: 0 40px;">
                <div style="background: white; padding: 40px; border-radius: 20px; box-shadow: 0 8px 32px rgba(0,0,0,0.08); margin-bottom: 32px;">
                    <h2 style="color: #4A148C; font-size: 36px; margin-bottom: 24px; margin-top: 0; font-family: 'Prompt', sans-serif;">ขั้นตอนการตรวจ</h2>
                    <ul style="font-size: 22px; line-height: 1.7; font-family: 'Prompt', sans-serif; margin-top: 0; padding-left: 24px;">
                        <li style="margin-bottom: 16px;"><strong>1. การออกเสียงสระ:</strong> ออกเสียงสระแต่ละตัว 5-8 วินาที ให้ชัดเจนและคงที่</li>
                        <li style="margin-bottom: 16px;"><strong>2. การออกเสียงพยางค์:</strong> ออกเสียง "พา-ทา-คา" ซ้ำๆ ประมาณ 6 วินาที</li>
                        <li style="margin-bottom: 16px;"><strong>3. การอ่านประโยค:</strong> อ่านประโยคที่กำหนดให้อย่างเป็นธรรมชาติและชัดเจน</li>
                    </ul>
                </div>
            </div>
        """, unsafe_allow_html=True)
        

        
        st.markdown("""
            <div style="max-width: 1000px; margin: 0 auto; padding: 0 40px;">
                <div style="background: white; padding: 40px; border-radius: 20px; box-shadow: 0 8px 32px rgba(0,0,0,0.08);">
                    <h2 style="color: #4A148C; font-size: 36px; margin-bottom: 24px; margin-top: 0; font-family: 'Prompt', sans-serif;">ข้อควรระวัง</h2>
                    <ul style="font-size: 22px; line-height: 1.7; color: #d32f2f; font-family: 'Prompt', sans-serif; margin-top: 0; padding-left: 24px;">
                        <li style="margin-bottom: 12px;"><strong style="font-weight: 600;">ระบบนี้เป็นเพียงการตรวจคัดกรองเบื้องต้น</strong></li>
                        <li style="margin-bottom: 12px;"><strong style="font-weight: 600;">ไม่สามารถทดแทนการวินิจฉัยโดยแพทย์ได้</strong></li>
                        <li style="margin-bottom: 12px;"><strong style="font-weight: 600;">หากมีข้อสงสัยควรปรึกษาแพทย์เฉพาะทาง</strong></li>
                    </ul>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Sample audio section
        st.markdown("""
            <div style="max-width: 1000px; margin: 0 auto; padding: 0 40px;">
                <div style="background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); border-radius: 20px; padding: 30px; margin: 30px 0; box-shadow: 0 8px 32px rgba(0,0,0,0.1);">
                    <h3 style="color: #495057; margin-bottom: 25px; font-family: 'Prompt', sans-serif; font-size: 24px; font-weight: 600; text-align: center;">🎵 ตัวอย่างเสียงที่ถูกต้อง</h3>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Sample audio files in order according to analysis page
        sample_audio_files = [
            ("อา", "sampleaudio/no/อา 1(1) pd.m4a"),
            ("อี", "sampleaudio/no/E 1(1) pd.m4a"),
            ("อือ", "sampleaudio/no/อือ 1(1) pd.m4a"),
            ("อู", "sampleaudio/no/อู 1(1) pd.m4a"),
            ("ไอ", "sampleaudio/no/ไอ 1(1) pd.m4a"),
            ("อำ", "sampleaudio/no/อำ 1(1) pd.m4a"),
            ("เอา", "sampleaudio/no/เอา 1(1) pd.m4a"),
            ("พยางค์ (พา-ทา-คา)", "sampleaudio/no/Pa-ta-ka 1(1) pd.m4a"),
            ("ประโยค", "sampleaudio/no/Sentence 1(1) pd.m4a")
        ]
        
        # Create columns for audio display
        audio_cols = st.columns(3)
        
        for i, (title, file_path) in enumerate(sample_audio_files):
            with audio_cols[i % 3]:
                try:
                    if os.path.exists(file_path):
                        with open(file_path, "rb") as audio_file:
                            audio_bytes = audio_file.read()
                            st.markdown(f"""
                                <div style="background: white; border-radius: 15px; padding: 20px; margin: 10px 0; box-shadow: 0 4px 16px rgba(0,0,0,0.1); border-left: 4px solid #6A1B9A;">
                                    <h4 style="color: #4A148C; margin-bottom: 15px; font-family: 'Prompt', sans-serif; font-size: 18px; font-weight: 600; text-align: center;">{title}</h4>
                                </div>
                            """, unsafe_allow_html=True)
                            st.audio(audio_bytes, format="audio/m4a")
                    else:
                        st.markdown(f"""
                            <div style="background: #fff3cd; border-radius: 15px; padding: 20px; margin: 10px 0; box-shadow: 0 4px 16px rgba(0,0,0,0.1); border-left: 4px solid #ffc107;">
                                <h4 style="color: #856404; margin-bottom: 15px; font-family: 'Prompt', sans-serif; font-size: 18px; font-weight: 600; text-align: center;">{title}</h4>
                                <p style="color: #856404; text-align: center; font-size: 14px;">ไฟล์เสียงไม่พบ</p>
                            </div>
                        """, unsafe_allow_html=True)
                except Exception as e:
                    st.markdown(f"""
                        <div style="background: #f8d7da; border-radius: 15px; padding: 20px; margin: 10px 0; box-shadow: 0 4px 16px rgba(0,0,0,0.1); border-left: 4px solid #dc3545;">
                            <h4 style="color: #721c24; margin-bottom: 15px; font-family: 'Prompt', sans-serif; font-size: 18px; font-weight: 600; text-align: center;">{title}</h4>
                            <p style="color: #721c24; text-align: center; font-size: 14px;">เกิดข้อผิดพลาดในการโหลดไฟล์</p>
                        </div>
                    """, unsafe_allow_html=True)
        
        # Additional information about sample audio
        st.markdown("""
            <div style="max-width: 1000px; margin: 0 auto; padding: 0 40px;">
                <div style="background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%); border-radius: 20px; padding: 25px; margin: 30px 0; box-shadow: 0 8px 32px rgba(0,0,0,0.1); border-left: 6px solid #1976d2;">
                    <h4 style="color: #1565c0; margin-bottom: 15px; font-family: 'Prompt', sans-serif; font-size: 20px; font-weight: 600;">💡 คำแนะนำเพิ่มเติม</h4>
                    <ul style="font-size: 16px; font-family: 'Prompt', sans-serif; line-height: 1.6; color: #2e7d32;">
                        <li>ฟังตัวอย่างเสียงก่อนเริ่มการตรวจเพื่อเข้าใจรูปแบบการออกเสียงที่ถูกต้อง</li>
                        <li>พยายามออกเสียงให้เหมือนกับตัวอย่างให้มากที่สุด</li>
                        <li>หากไม่แน่ใจ สามารถฟังตัวอย่างซ้ำได้หลายครั้ง</li>
                        <li>ตัวอย่างเสียงเหล่านี้เป็นเสียงจากผู้ที่ไม่เป็นโรคพาร์กินสัน</li>
                    </ul>
                </div>
            </div>
        """, unsafe_allow_html=True)

    def show_analysis_page():
        """Display the analysis page"""
        load_css()
        
        # FIXED: Combine header with analysis content
        analysis_html = f"""
            {get_header_html()}
        """
        
        st.markdown(analysis_html, unsafe_allow_html=True)
        
        # Back button
        if st.button("← กลับหน้าแรก", key="back_to_home"):
            st.session_state.page = 'home'
            st.rerun()
        
        # Load model
        model = load_model()
        if not model:
            st.error("Cannot proceed without model. Please check your internet connection and try again.")
            return

        # Clear button logic
        if 'clear_button_clicked' in st.session_state and st.session_state.clear_button_clicked:
            cleanup_all_temp_files()
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
                        add_temp_file(tmp.name)
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
            cleanup_all_temp_files()
            st.session_state.vowel_files = []
            for file in uploaded_vowels[:7]:
                saved_path = save_uploaded_file(file)
                if saved_path:
                    st.session_state.vowel_files.append(saved_path)

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
                    add_temp_file(tmp.name)
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
            saved_path = save_uploaded_file(uploaded_pataka)
            if saved_path:
                st.session_state.pataka_file = saved_path

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
                    add_temp_file(tmp.name)
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
            saved_path = save_uploaded_file(uploaded_sentence)
            if saved_path:
                st.session_state.sentence_file = saved_path

        # Action buttons
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
                        all_probs = predict_from_model(
                            valid_vowel_files, 
                            st.session_state.pataka_file, 
                            st.session_state.sentence_file, 
                            model
                        )
                        
                        if all_probs is None:
                            st.error("การวิเคราะห์ล้มเหลว กรุณาลองใหม่อีกครั้ง")
                            return
                            
                        final_prob = np.mean(all_probs)
                        percent = int(final_prob * 100)

                        # Determine risk level and advice
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

                        # Display results
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
                        
                        # Information about spectrograms
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
