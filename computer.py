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
import requests
import time

# =============================
# Configuration
# =============================
CONFIG = {
    'MODEL_PATH': "best_model.pth",
    'MODEL_FILE_ID': "1CrvAqTrBGvTau3vTvgac8NAspv5xFKkA",
    'CSS_FILE': "deskstyle.css",
    'LOGO_PATHS': ["logo.png", "./logo.png", "assets/logo.png", "images/logo.png"],
    'IMAGE_PATHS': ["insert.jpg", "./insert.jpg", "assets/insert.jpg", "images/insert.jpg"],
    'THAI_TIMEZONE': 'Asia/Bangkok',
    'PD_INDEX': 1,
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
# Session State & File Management
# =============================
def initialize_session_state():
    """Initialize all session state variables"""
    defaults = {
        'page': 'home',
        'vowel_files': [],
        'pataka_file': None,
        'sentence_file': None,
        'clear_clicked': False,
        'temp_files': []
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
        except:
            pass

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

# =============================
# Enhanced Model Download Functions
# =============================
def validate_downloaded_file(file_path):
    """Validate that the downloaded file is actually a PyTorch model"""
    try:
        if not os.path.exists(file_path):
            return False
        
        file_size = os.path.getsize(file_path)
        
        # Check if file is too small (likely an error page)
        if file_size < 1024 * 1024:  # Less than 1MB
            return False
        
        # Try to load as PyTorch model
        try:
            checkpoint = torch.load(file_path, map_location='cpu')
            return True
        except:
            return False
            
    except:
        return False

def download_large_file_from_gdrive(file_id, destination):
    """Download large files from Google Drive handling confirmation"""
    
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768
        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:
                    f.write(chunk)

    session = requests.Session()
    
    response = session.get(
        f"https://docs.google.com/uc?export=download&id={file_id}",
        stream=True
    )
    
    token = get_confirm_token(response)
    
    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(
            "https://docs.google.com/uc?export=download",
            params=params,
            stream=True
        )
    
    save_response_content(response, destination)
    return True

def download_model_from_gdrive():
    """Enhanced Google Drive download with multiple methods"""
    model_path = CONFIG['MODEL_PATH']
    file_id = CONFIG['MODEL_FILE_ID']
    
    # Method 1: Standard gdown with corrected URL
    try:
        url = f"https://drive.google.com/uc?id={file_id}&export=download"
        success = gdown.download(url, model_path, quiet=True)
        
        if success and validate_downloaded_file(model_path):
            return model_path
    except:
        pass
    
    # Method 2: gdown with fuzzy matching
    try:
        if os.path.exists(model_path):
            os.remove(model_path)
        
        share_url = f"https://drive.google.com/file/d/{file_id}/view?usp=sharing"
        success = gdown.download(share_url, model_path, quiet=True, fuzzy=True)
        
        if success and validate_downloaded_file(model_path):
            return model_path
    except:
        pass
    
    # Method 3: Manual requests method with session handling
    try:
        if download_large_file_from_gdrive(file_id, model_path):
            if validate_downloaded_file(model_path):
                return model_path
    except:
        pass
    
    # Method 4: gdown CLI fallback
    try:
        import subprocess
        
        result = subprocess.run([
            'python', '-m', 'gdown', file_id, '-O', model_path
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0 and validate_downloaded_file(model_path):
            return model_path
    except:
        pass
    
    return None

def _clean_state_dict(state_dict: dict) -> dict:
    """Strip 'module.' prefixes etc. for DataParallel checkpoints."""
    cleaned = {}
    for k, v in state_dict.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module."):]
        cleaned[nk] = v
    return cleaned

@st.cache_resource
def load_model():
    """Load the ResNet18 model with enhanced error handling."""
    model_path = CONFIG['MODEL_PATH']
    
    try:
        # Check if model already exists and is valid
        if os.path.exists(model_path):
            if not validate_downloaded_file(model_path):
                os.remove(model_path)
        
        # Download if needed
        if not os.path.exists(model_path):
            downloaded_path = download_model_from_gdrive()
            if not downloaded_path:
                st.error("""
                ❌ **Failed to download model file**
                
                **Manual Solution:**
                1. Go to: https://drive.google.com/file/d/1CrvAqTrBGvTau3vTvgac8NAspv5xFKkA/view
                2. Click "Download" (you may need to click "Download anyway" if there's a virus warning)
                3. Save the file as `best_model.pth` in your app directory
                4. Restart the app
                """)
                return None
        
        # Load the model
        model = ResNet18Classifier()
        
        # Load the checkpoint
        checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif isinstance(checkpoint, dict):
            state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # Clean state dict (remove 'module.' prefixes if present)
        cleaned_state_dict = _clean_state_dict(state_dict)
        
        # Load into model
        try:
            model.load_state_dict(cleaned_state_dict, strict=True)
        except RuntimeError:
            model.load_state_dict(cleaned_state_dict, strict=False)
        
        model.eval()
        return model
        
    except Exception as e:
        st.error(f"❌ Critical error loading model: {str(e)}")
        return None

# =============================
# Utility Functions
# =============================
@st.cache_data
def load_image_file(image_paths, alt_text="Image"):
    """Generic function to load image files with fallback options"""
    for path in image_paths:
        try:
            if os.path.exists(path):
                with open(path, "rb") as f:
                    return base64.b64encode(f.read()).decode()
        except:
            continue
    return None

def get_thai_time():
    """Get current Thai time formatted for display"""
    try:
        thai_tz = pytz.timezone(CONFIG['THAI_TIMEZONE'])
        now = datetime.now(thai_tz)
        return now.strftime("%d/%m/%Y %H:%M:%S")
    except:
        return datetime.now().strftime("%d/%m/%Y %H:%M:%S")

def save_uploaded_file(uploaded_file):
    """Save uploaded file to temporary location"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
            tmp.write(uploaded_file.read())
            add_temp_file(tmp.name)
            return tmp.name
    except:
        return None

def inject_critical_css():
    """Inject critical CSS with maximum specificity to override Streamlit defaults"""
    critical_css = """
    <style>
    /* CRITICAL CSS INJECTION - Maximum Specificity Override */
    
    /* Reset all button styling completely */
    div[data-testid="stVerticalBlock"] div.element-container div.stButton button,
    div[data-testid="column"] div[data-testid="stVerticalBlock"] div.element-container div.stButton button,
    .stApp div.stButton button,
    .stApp button[kind],
    button[data-testid*="start_analysis"],
    button[data-testid*="guide_manual"],
    button[data-testid*="back_to_home"],
    button[data-testid*="back_to_home_from_guide"],
    button[data-testid*="predict"],
    button[data-testid*="clear"] {
        /* Nuclear reset */
        all: unset !important;
        
        /* Base properties */
        font-family: 'Prompt', sans-serif !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        cursor: pointer !important;
        text-align: center !important;
        box-sizing: border-box !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        border: none !important;
        outline: none !important;
        text-decoration: none !important;
    }
    
    /* HOMEPAGE BUTTONS - เริ่มใช้งาน */
    button[data-testid*="start_analysis"] {
        font-size: 72px !important;
        padding: 50px 100px !important;
        border-radius: 100px !important;
        font-weight: 900 !important;
        width: calc(100vw - 60px) !important;
        height: 150px !important;
        background: linear-gradient(135deg, #FF1744 0%, #FF5722 20%, #FF9800 40%, #FFC107 60%, #8BC34A 80%, #2196F3 100%) !important;
        background-size: 600% 600% !important;
        animation: gradientPulse 2s ease infinite, buttonGlow 1.5s ease-in-out infinite alternate !important;
        color: white !important;
        box-shadow: 0 20px 80px rgba(255, 23, 68, 0.8), inset 0 0 30px rgba(255,255,255,0.2) !important;
        text-shadow: 3px 3px 6px rgba(0,0,0,0.5) !important;
        border: 4px solid rgba(255,255,255,0.4) !important;
        transform: scale(1.05) !important;
        margin: 15px 0 !important;
    }
    
    button[data-testid*="start_analysis"]:hover {
        transform: translateY(-12px) scale(1.08) !important;
        box-shadow: 0 35px 100px rgba(255, 23, 68, 1), inset 0 0 40px rgba(255,255,255,0.3) !important;
    }
    
    /* HOMEPAGE BUTTONS - คู่มือ */
    button[data-testid*="guide_manual"] {
        font-size: 72px !important;
        padding: 50px 100px !important;
        border-radius: 100px !important;
        font-weight: 900 !important;
        width: calc(100vw - 60px) !important;
        height: 150px !important;
        background: linear-gradient(135deg, #4A148C 0%, #6A1B9A 30%, #8E24AA 60%, #AB47BC 100%) !important;
        color: white !important;
        box-shadow: 0 20px 70px rgba(74, 20, 140, 0.6), inset 0 0 20px rgba(255,255,255,0.1) !important;
        border: 3px solid rgba(255,255,255,0.3) !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3) !important;
        margin: 15px 0 !important;
    }
    
    button[data-testid*="guide_manual"]:hover {
        transform: translateY(-10px) scale(1.06) !important;
        box-shadow: 0 30px 90px rgba(74, 20, 140, 0.8), inset 0 0 30px rgba(255,255,255,0.2) !important;
        background: linear-gradient(135deg, #3A0E6B 0%, #5A1B7A 30%, #7E1F9A 60%, #9B37BC 100%) !important;
    }
    
    /* BACK BUTTONS */
    button[data-testid*="back_to_home"],
    button[data-testid*="back_to_home_from_guide"] {
        background: linear-gradient(135deg, #37474F 0%, #455A64 30%, #546E7A 60%, #607D8B 100%) !important;
        color: white !important;
        font-size: 36px !important;
        padding: 30px 60px !important;
        min-width: 400px !important;
        max-width: 500px !important;
        height: 90px !important;
        border-radius: 50px !important;
        margin-bottom: 40px !important;
        box-shadow: 0 15px 40px rgba(55, 71, 79, 0.5), inset 0 0 20px rgba(255,255,255,0.1) !important;
        font-weight: 700 !important;
        border: 3px solid rgba(255,255,255,0.2) !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3) !important;
    }
    
    button[data-testid*="back_to_home"]:hover,
    button[data-testid*="back_to_home_from_guide"]:hover {
        transform: translateY(-6px) scale(1.05) !important;
        box-shadow: 0 20px 50px rgba(55, 71, 79, 0.7), inset 0 0 30px rgba(255,255,255,0.2) !important;
        background: linear-gradient(135deg, #263238 0%, #37474F 30%, #455A64 60%, #546E7A 100%) !important;
    }
    
    /* ANALYSIS BUTTONS - วิเคราะห์ */
    button[data-testid*="predict"] {
        background: linear-gradient(135deg, #1B5E20 0%, #2E7D32 20%, #388E3C 40%, #4CAF50 60%, #66BB6A 80%, #81C784 100%) !important;
        color: white !important;
        box-shadow: 0 18px 60px rgba(27, 94, 32, 0.6), inset 0 0 25px rgba(255,255,255,0.15) !important;
        font-size: 44px !important;
        padding: 35px 80px !important;
        min-width: 500px !important;
        max-width: 600px !important;
        height: 110px !important;
        border-radius: 60px !important;
        font-weight: 800 !important;
        border: 4px solid rgba(255,255,255,0.25) !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3) !important;
        letter-spacing: 1px !important;
    }
    
    button[data-testid*="predict"]:hover {
        transform: translateY(-8px) scale(1.08) !important;
        box-shadow: 0 25px 80px rgba(27, 94, 32, 0.8), inset 0 0 35px rgba(255,255,255,0.2) !important;
        background: linear-gradient(135deg, #0D4E12 0%, #1B5E20 20%, #2E7D32 40%, #388E3C 60%, #4CAF50 80%, #66BB6A 100%) !important;
    }
    
    /* ANALYSIS BUTTONS - ลบข้อมูล */
    button[data-testid*="clear"] {
        background: linear-gradient(135deg, #BF360C 0%, #D84315 20%, #E65100 40%, #F57C00 60%, #FF9800 80%, #FFB74D 100%) !important;
        color: white !important;
        box-shadow: 0 18px 60px rgba(191, 54, 12, 0.6), inset 0 0 25px rgba(255,255,255,0.15) !important;
        font-size: 44px !important;
        padding: 35px 80px !important;
        min-width: 500px !important;
        max-width: 600px !important;
        height: 110px !important;
        border-radius: 60px !important;
        font-weight: 800 !important;
        border: 4px solid rgba(255,255,255,0.25) !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3) !important;
        letter-spacing: 1px !important;
    }
    
    button[data-testid*="clear"]:hover {
        transform: translateY(-8px) scale(1.08) !important;
        box-shadow: 0 25px 80px rgba(191, 54, 12, 0.8), inset 0 0 35px rgba(255,255,255,0.2) !important;
        background: linear-gradient(135deg, #9C2D08 0%, #BF360C 20%, #D84315 40%, #E65100 60%, #F57C00 80%, #FF9800 100%) !important;
    }
    
    /* Animations */
    @keyframes gradientPulse {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    @keyframes buttonGlow {
        0% { 
            box-shadow: 0 20px 80px rgba(255, 23, 68, 0.8), inset 0 0 30px rgba(255,255,255,0.2) !important;
        }
        100% { 
            box-shadow: 0 25px 100px rgba(255, 23, 68, 1), inset 0 0 50px rgba(255,255,255,0.4) !important;
        }
    }
    
    /* Responsive */
    @media (max-width: 1200px) {
        button[data-testid*="start_analysis"],
        button[data-testid*="guide_manual"] {
            font-size: 60px !important;
            height: 130px !important;
            padding: 40px 80px !important;
        }
        
        button[data-testid*="back_to_home"],
        button[data-testid*="back_to_home_from_guide"] {
            font-size: 32px !important;
            height: 85px !important;
            min-width: 350px !important;
            max-width: 450px !important;
        }
        
        button[data-testid*="predict"],
        button[data-testid*="clear"] {
            font-size: 40px !important;
            height: 100px !important;
            min-width: 450px !important;
            max-width: 550px !important;
        }
    }
    
    @media (max-width: 768px) {
        button[data-testid*="start_analysis"],
        button[data-testid*="guide_manual"] {
            width: calc(100vw - 40px) !important;
            font-size: 48px !important;
            height: 110px !important;
            padding: 35px 50px !important;
        }
        
        button[data-testid*="back_to_home"],
        button[data-testid*="back_to_home_from_guide"] {
            font-size: 28px !important;
            height: 80px !important;
            min-width: 300px !important;
            max-width: 400px !important;
        }
        
        button[data-testid*="predict"],
        button[data-testid*="clear"] {
            font-size: 36px !important;
            height: 95px !important;
            min-width: 350px !important;
            max-width: 450px !important;
            padding: 30px 60px !important;
        }
    }
    </style>
    """
    st.markdown(critical_css, unsafe_allow_html=True)

def load_css():
    """Load external CSS file"""
    css_file = Path(CONFIG['CSS_FILE'])
    if css_file.exists():
        with open(css_file, 'r', encoding='utf-8') as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        # Fallback minimal CSS
        st.markdown("""
            <style>
            @import url('https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;500;600;700;800;900&display=swap');
            * { font-family: 'Prompt', sans-serif !important; }
            .stApp { background: linear-gradient(135deg, #f8f4ff 0%, #e8f4fd 100%) !important; }
            
            .header-container {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 20px 40px;
                background: rgba(255, 255, 255, 0.95);
                backdrop-filter: blur(10px);
                border-bottom: 1px solid rgba(106, 27, 154, 0.1);
                margin-bottom: 0;
            }
            
            .logo-section {
                display: flex;
                align-items: center;
            }
            
            .logo-text {
                font-size: 32px;
                font-weight: 800;
                color: #4A148C;
                margin-right: 20px;
            }
            
            .header-divider {
                width: 2px;
                height: 40px;
                background: linear-gradient(180deg, #6A1B9A, #9C27B0);
                margin-right: 20px;
            }
            
            .tagline {
                font-size: 16px;
                color: #666;
                font-weight: 500;
            }
            
            .datetime-display {
                font-size: 14px;
                color: #666;
                background: rgba(106, 27, 154, 0.05);
                padding: 8px 16px;
                border-radius: 20px;
                border: 1px solid rgba(106, 27, 154, 0.1);
            }
            
            .main-content {
                padding: 0 40px;
            }
            
            .content-wrapper {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 60px;
                align-items: center;
                max-width: 1200px;
                margin: 60px auto;
            }
            
            .main-title {
                font-size: 48px;
                font-weight: 800;
                line-height: 1.2;
                color: #2E2E2E;
                margin-bottom: 40px;
            }
            
            .highlight {
                background: linear-gradient(135deg, #6A1B9A, #9C27B0);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }
            
            .main-image {
                width: 100%;
                height: auto;
                border-radius: 20px;
                box-shadow: 0 20px 60px rgba(106, 27, 154, 0.15);
            }
            
            .image-placeholder {
                width: 100%;
                height: 400px;
                background: linear-gradient(135deg, #f5f5f5, #e0e0e0);
                border-radius: 20px;
                display: flex;
                align-items: center;
                justify-content: center;
                border: 2px dashed #ccc;
            }
            
            .placeholder-content {
                text-align: center;
                color: #999;
            }
            
            .placeholder-icon {
                font-size: 48px;
                margin-bottom: 12px;
            }
            
            .homepage-buttons-wrapper {
                margin-top: 40px;
            }
            
            .stButton > button {
                width: 200px;
                height: 60px;
                font-size: 20px;
                font-weight: 600;
                border-radius: 30px;
                border: none;
                margin-right: 20px;
                margin-bottom: 20px;
                transition: all 0.3s ease;
            }
            
            .stButton > button:first-child {
                background: linear-gradient(135deg, #6A1B9A, #9C27B0);
                color: white;
            }
            
            .stButton > button:first-child:hover {
                transform: translateY(-2px);
                box-shadow: 0 10px 30px rgba(106, 27, 154, 0.3);
            }
            
            .card {
                background: white;
                padding: 30px;
                border-radius: 20px;
                box-shadow: 0 8px 32px rgba(0,0,0,0.08);
                margin: 30px 0;
                border-left: 6px solid #6A1B9A;
            }
            
            .card h2 {
                color: #4A148C;
                font-size: 32px;
                margin-bottom: 16px;
                font-weight: 700;
            }
            
            .instructions {
                font-size: 18px;
                color: #555;
                line-height: 1.6;
                margin-bottom: 0;
            }
            
            .pronounce {
                font-size: 24px;
                color: #4A148C;
                font-weight: 600;
                text-align: center;
                margin: 20px 0;
                padding: 15px;
                background: rgba(106, 27, 154, 0.05);
                border-radius: 12px;
                border: 1px solid rgba(106, 27, 154, 0.1);
            }
            
            .sentence-instruction {
                font-size: 20px;
                color: #555;
                line-height: 1.6;
                margin-bottom: 0;
                text-align: center;
                padding: 20px;
                background: rgba(106, 27, 154, 0.05);
                border-radius: 12px;
                margin-top: 10px;
            }
            
            .guide-container {
                text-align: center;
                margin: 40px 0;
            }
            
            .guide-title {
                font-size: 48px;
                font-weight: 800;
                color: #4A148C;
                margin-bottom: 20px;
            }
            </style>
        """, unsafe_allow_html=True)
        pass
    
    # CRITICAL: Inject override CSS AFTER main CSS
    inject_critical_css()

# =============================
# Audio Processing Functions
# =============================
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
        return file_path
    except:
        return None

def audio_to_mel_tensor(file_path):
    """Convert audio file to mel spectrogram tensor (training-aligned)."""
    try:
        wav_file = convert_to_wav_if_needed(file_path)
        if not wav_file:
            return None

        y, sr = librosa.load(wav_file, sr=None, mono=True)

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
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std =[0.229, 0.224, 0.225]),
        ])
        return transform(image).unsqueeze(0)
    except:
        return None

def create_mel_spectrogram_display(file_path, title="Mel Spectrogram"):
    """Create a mel spectrogram for display purposes"""
    try:
        wav_file = convert_to_wav_if_needed(file_path)
        if not wav_file:
            return None

        y, sr = librosa.load(wav_file, sr=None, mono=True)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_db = librosa.power_to_db(mel, ref=np.max)

        fig, ax = plt.subplots(figsize=(8, 4), dpi=100, facecolor='white')
        img = librosa.display.specshow(mel_db, sr=sr, ax=ax, x_axis='time', y_axis='mel')
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
    except:
        return None

def predict_from_model(vowel_paths, pataka_path, sentence_path, model):
    """Predict PD probability by averaging logits (more stable than averaging probs)."""
    try:
        tensors = []

        # Process vowel files
        for path in vowel_paths:
            t = audio_to_mel_tensor(path)
            if t is None:
                return None
            tensors.append(t)

        # Process pataka and sentence
        for path in [pataka_path, sentence_path]:
            t = audio_to_mel_tensor(path)
            if t is None:
                return None
            tensors.append(t)

        if not tensors:
            return None

        with torch.no_grad():
            logits_list = []
            for t in tensors:
                out = model(t)
                logits_list.append(out)

            logits_all = torch.cat(logits_list, dim=0)
            mean_logits = logits_all.mean(dim=0)
            probs = torch.softmax(mean_logits, dim=0)

            pd_idx = CONFIG.get('PD_INDEX', 1)
            final_prob_pd = float(probs[pd_idx].item())
            return final_prob_pd

    except:
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
    """Display the home page"""
    load_css()
    inject_critical_css()

    woman_image_b64 = load_image_file(CONFIG['IMAGE_PATHS'], "Woman using phone")

    # Combine header and main content in ONE st.markdown call
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

    # Render the combined HTML
    st.markdown(combined_html, unsafe_allow_html=True)

    # Add buttons positioned within the text section area
    st.markdown('<div class="homepage-buttons-wrapper">', unsafe_allow_html=True)

    # First button - เริ่มใช้งาน (Start Analysis)
    if st.button("เริ่มใช้งาน", key="start_analysis"):
        st.session_state.page = 'analysis'
        st.rerun()

    # Second button - คู่มือ (Guide)
    if st.button("คู่มือ", key="guide_manual"):
        st.session_state.page = 'guide'
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

def show_guide_page():
    """Display the guide/manual page with proper styling"""
    load_css()
    inject_critical_css()

    # Combine header with back button and title
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

    # Guide content
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

    # Sample audio files
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
            except:
                st.markdown(f"""
                    <div style="background: #f8d7da; border-radius: 15px; padding: 20px; margin: 10px 0; box-shadow: 0 4px 16px rgba(0,0,0,0.1); border-left: 4px solid #dc3545;">
                        <h4 style="color: #721c24; margin-bottom: 15px; font-family: 'Prompt', sans-serif; font-size: 18px; font-weight: 600; text-align: center;">{title}</h4>
                        <p style="color: #721c24; text-align: center; font-size: 14px;">เกิดข้อผิดพลาดในการโหลดไฟล์</p>
                    </div>
                """, unsafe_allow_html=True)

    # Additional information
    st.markdown("""
        <div style="max-width: 1000px; margin: 0 auto; padding: 0 40px;">
            <div style="background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%); border-radius: 20px; padding: 25px; margin: 30px 0; box-shadow: 0 8px 32px rgba(0,0,0,0.1); border-left: 6px solid #1976d2;">
                <h4 style="color: #1565c0; margin-bottom: 15px; font-family: 'Prompt', sans-serif; font-size: 20px; font-weight: 600;">💡 คำแนะนำเพิ่มเติม</h4>
                <ul style="font-size: 16px; font-family: 'Prompt', sans-serif; line-height: 1.6; color: #2e7d32;">
                    <li>ฟังตัวอย่างเสียงก่อนเริ่มการตรวจเพื่อเข้าใจรูปแบบการออกเสียงที่ถูกต้อง</li>
                    <li>พยายามออกเสียงให้เหมือนกับตัวอย่างให้มากที่สุด</li>
                    <li>หากไม่แน่ใจ สามารถฟังตัวอย่างซ้ำได้หลายครั้ง</li>
                    <li>ตัวอย่างเสียงเหล่านี้เป็นเสียงจากผู้ที่ไม่เป็นพาร์กินสัน</li>
                </ul>
            </div>
        </div>
    """, unsafe_allow_html=True)

def show_analysis_page():
    """Display the analysis page"""
    load_css()
    inject_critical_css()

    # Header
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
                    final_prob = predict_from_model(
                        valid_vowel_files,
                        st.session_state.pataka_file,
                        st.session_state.sentence_file,
                        model
                    )

                    if final_prob is None:
                        st.error("การวิเคราะห์ล้มเหลว กรุณาลองใหม่อีกครั้ง")
                        return

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
                except:
                    st.error("เกิดข้อผิดพลาดในการวิเคราะห์")
        else:
            st.warning("กรุณาอัดเสียงหรืออัปโหลดให้ครบทั้ง 7 สระ พยางค์ และประโยค", icon="⚠")

# =============================
# Main App Logic
# =============================
def run_desktop_app():
    """Main function to run the desktop version"""
    # Initialize Session State
    initialize_session_state()

    # Register cleanup function
    atexit.register(cleanup_all_temp_files)

    if st.session_state.page == 'home':
        show_home_page()
    elif st.session_state.page == 'guide':
        show_guide_page()
    elif st.session_state.page == 'analysis':
        show_analysis_page()

# =============================
# Entry Point
# =============================
if __name__ == "__main__":
    if 'page' not in st.session_state:
        st.session_state.page = 'home'
    
    # Run the main app
    run_desktop_app()
