import streamlit as st
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from model import ResNet18Classifier
import librosa
import librosa.display
import matplotlib.pyplot as plt
from PIL import Image
import io
import tempfile
import os
import gdown
import base64

# =============================
# Page Config & Session State
# =============================
st.set_page_config(page_title="SixtyScan", layout="centered")

# Initialize session state
if 'show_main_app' not in st.session_state:
    st.session_state.show_main_app = False
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []

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

def display_logo(size="180px"):
    """Display logo if available"""
    logo_b64 = load_logo()
    if logo_b64:
        st.markdown(f"""
        <img src="data:image/png;base64,{logo_b64}" class="logo" style="width: {size};" alt="SixtyScan Logo">
        """, unsafe_allow_html=True)
    else:
        # Fallback logo
        st.markdown(f"""
        <div class="logo-fallback" style="width: {size}; height: {size};">
            SS
        </div>
        """, unsafe_allow_html=True)

# =============================
# Download model from Google Drive
# =============================
MODEL_PATH = "best_resnet18.pth"
if not os.path.exists(MODEL_PATH):
    gdown.download(
        "https://drive.google.com/uc?id=1_oHE9B-2PgSqpTQCC9HrG7yO0rsnZtqs",
        MODEL_PATH,
        quiet=False
    )

# =============================
# Enhanced Styles
# =============================
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Thai:wght@400;600&family=Lexend+Deca:wght@700&display=swap');
        
        /* Global */
        html, body, .main {
            background-color: #f2f4f8;
            font-family: 'Noto Sans Thai', sans-serif;
            font-weight: 400;
        }
        
        /* Hide Streamlit header and menu */
        .stApp > header {visibility: hidden;}
        .stApp > div:first-child {padding-top: 0px;}
        
        /* Landing Page Styles */
        .landing-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            min-height: 100vh;
            text-align: center;
            padding: 40px 20px;
        }
        
        /* Centered logo */
        .logo {
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 180px;
            margin-bottom: 24px;
            border-radius: 20px;
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
        }
        
        /* Fallback logo */
        .logo-fallback {
            background: linear-gradient(45deg, #4A148C, #7B1FA2);
            border-radius: 20px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 48px;
            font-weight: bold;
            color: white;
            margin: 0 auto 24px auto;
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
            font-family: 'Lexend Deca', sans-serif;
        }
        
        /* Landing Page Title */
        .landing-title {
            text-align: center;
            font-family: 'Lexend Deca', sans-serif;
            font-size: 84px;
            color: #4A148C;
            font-weight: 700;
            margin-bottom: 10px;
        }
        
        /* Landing Page Subtitle */
        .landing-subtitle {
            text-align: center;
            font-family: 'Noto Sans Thai', sans-serif;
            font-weight: 400;
            font-size: 42px;
            color: #333;
            margin-bottom: 30px;
        }
        
        /* Landing Page Description */
        .landing-description {
            font-family: 'Noto Sans Thai', sans-serif;
            font-size: 24px;
            color: #555;
            margin-bottom: 50px;
            max-width: 700px;
            line-height: 1.5;
            font-weight: 400;
            text-align: center;
        }
        
        /* Main Title */
        h1.title {
            text-align: center;
            font-family: 'Lexend Deca', sans-serif;
            font-size: 84px;
            color: #4A148C;
            font-weight: 700;
            margin-bottom: 10px;
        }
        
        /* Subtitle */
        p.subtitle {
            text-align: center;
            font-family: 'Noto Sans Thai', sans-serif;
            font-weight: 400;
            font-size: 32px;
            color: #333;
            margin-bottom: 56px;
        }
        
        /* Card container */
        .card {
            background-color: #ffffff;
            border-radius: 16px;
            padding: 40px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.06);
            margin-bottom: 40px;
        }
        
        /* Section headers */
        .card h2 {
            font-size: 48px;
            margin-bottom: 20px;
            color: #222;
            font-weight: 600;
            font-family: 'Noto Sans Thai', sans-serif;
        }
        
        /* Instructions text */
        .instructions {
            font-size: 28px !important;
            color: #333;
            margin-bottom: 24px;
            font-weight: 400;
            font-family: 'Noto Sans Thai', sans-serif;
        }
        
        /* Pronunciation display */
        .pronounce {
            font-size: 24px !important;
            color: #000;
            font-weight: 400;
            margin-top: 0;
            margin-bottom: 24px;
            font-family: 'Noto Sans Thai', sans-serif;
        }
        
        /* Buttons */
        .predict-btn, .clear-btn {
            font-size: 38px !important;
            padding: 1.4em 2.7em;
            border-radius: 14px;
            font-weight: bold;
            width: 100%;
            max-width: 300px;
            display: block;
            margin: 10px auto;
            font-family: 'Noto Sans Thai', sans-serif;
        }
        
        .predict-btn {
            background-color: #009688;
            color: white;
            border: none;
            cursor: pointer;
        }
        
        .clear-btn {
            background-color: #cfd8dc;
            color: black;
            border: none;
            cursor: pointer;
        }
        
        /* All other text elements */
        .stMarkdown, .stText, .stSuccess, .stWarning, ul, li {
            font-family: 'Noto Sans Thai', sans-serif !important;
            font-weight: 400 !important;
        }
        
        /* Make bold inside .pronounce, .instructions, .sentence-instruction larger and more visible */
        .pronounce b, .instructions b, .sentence-instruction b {
            font-weight: 700 !important;
        }
        
        /* Sentence instruction styling */
        .card .sentence-instruction {
            font-size: 24px !important;
            font-weight: 400 !important;
            color: #333 !important;
            margin-bottom: 24px !important;
            font-family: 'Noto Sans Thai', sans-serif !important;
            display: block !important;
        }
        
        /* Enhanced Audio Input Styling */
        .stAudioInput > div {
            background-color: #f8f9ff;
            border-radius: 12px;
            border: 2px solid #e0e7ff;
            padding: 20px;
            margin: 15px 0;
        }
        
        /* Enhanced File Uploader */
        .stFileUploader > div {
            background-color: #f0f8ff;
            border-radius: 12px;
            border: 2px dashed #667eea;
            padding: 25px;
            text-align: center;
        }
        
        /* Back to Home Button */
        .back-button {
            position: fixed;
            top: 20px;
            left: 20px;
            background-color: #4A148C;
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 25px;
            cursor: pointer;
            transition: all 0.3s ease;
            font-weight: 600;
            z-index: 1000;
            font-family: 'Noto Sans Thai', sans-serif;
            font-size: 16px;
            text-decoration: none;
        }
        
        .back-button:hover {
            background-color: #6A1B9A;
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(74, 20, 140, 0.3);
            color: white;
        }
        
        /* Responsive Design */
        @media (max-width: 768px) {
            .landing-title {
                font-size: 48px;
            }
            
            .landing-subtitle {
                font-size: 28px;
            }
            
            .landing-description {
                font-size: 20px;
            }
            
            h1.title {
                font-size: 48px;
            }
            
            p.subtitle {
                font-size: 24px;
            }
            
            .card h2 {
                font-size: 32px;
            }
            
            .instructions {
                font-size: 20px !important;
            }
            
            .pronounce {
                font-size: 18px !important;
            }
        }
        
        /* Hide scrollbar but keep functionality */
        .main::-webkit-scrollbar {
            width: 8px;
        }
        
        .main::-webkit-scrollbar-track {
            background: #f1f1f1;
        }
        
        .main::-webkit-scrollbar-thumb {
            background: #c1c1c1;
            border-radius: 4px;
        }
        
        .main::-webkit-scrollbar-thumb:hover {
            background: #a8a8a8;
        }
    </style>
""", unsafe_allow_html=True)

# =============================
# Load Model
# =============================
@st.cache_resource
def load_model():
    model = ResNet18Classifier()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
    model.eval()
    return model

model = load_model()

# =============================
# Audio Preprocessing
# =============================
from pydub import AudioSegment
import soundfile as sf

def audio_to_mel_tensor(file_path):
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

# =============================
# Prediction
# =============================
def predict_from_model(vowel_paths, pataka_path, sentence_path):
    inputs = [audio_to_mel_tensor(p) for p in vowel_paths]
    inputs.append(audio_to_mel_tensor(pataka_path))
    inputs.append(audio_to_mel_tensor(sentence_path))
    with torch.no_grad():
        return [F.softmax(model(x), dim=1)[0][1].item() for x in inputs]

# =============================
# Clear Function (Enhanced)
# =============================
def clear_all_data():
    """Enhanced clear function that properly clears uploaded files"""
    # Clear all session state
    keys_to_clear = [key for key in st.session_state.keys() if not key.startswith('FormSubmitter')]
    for key in keys_to_clear:
        del st.session_state[key]
    
    # Reset to landing page
    st.session_state.show_main_app = False
    st.session_state.uploaded_files = []
    
    # Force rerun
    st.rerun()

# =============================
# Landing Page
# =============================
def show_landing_page():
    st.markdown('<div class="landing-container">', unsafe_allow_html=True)
    
    # Logo
    display_logo()
    
    # Title and description
    st.markdown('<h1 class="landing-title">SixtyScan</h1>', unsafe_allow_html=True)
    st.markdown('<p class="landing-subtitle">ตรวจโรคพาร์กินสันจากเสียง</p>', unsafe_allow_html=True)
    st.markdown("""
    <p class="landing-description">
        ระบบตรวจสอบโรคพาร์กินสันด้วยเทคนิค AI และการวิเคราะห์เสียง 
        ช่วยให้คุณสามารถตรวจสอบสัญญาณเบื้องต้นของโรคพาร์กินสันได้อย่างง่ายดาย 
        ผ่านการบันทึกเสียงและการวิเคราะห์ด้วยปัญญาประดิษฐ์
    </p>
    """, unsafe_allow_html=True)
    
    # Start button
    if st.button("เริ่มการตรวจสอบ", key="start_analysis", type="primary"):
        st.session_state.show_main_app = True
        st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

# =============================
# Main Application
# =============================
def show_main_app():
    # Back to home button
    if st.button("← กลับหน้าแรก", key="back_home"):
        st.session_state.show_main_app = False
        st.rerun()
    
    # Header with Logo
    display_logo("120px")
    st.markdown("<h1 class='title'>SixtyScan</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>ตรวจโรคพาร์กินสันจากเสียง</p>", unsafe_allow_html=True)

    # =============================
    # Vowel Recordings (7)
    # =============================
    st.markdown("""
    <div class='card'>
        <h2>1. สระ</h2>
        <p class='instructions'>กรุณาออกเสียงแต่ละสระ 5-8 วินาทีอย่างชัดเจน โดยกดปุ่มบันทึกทีละสระ</p>
    </div>
    """, unsafe_allow_html=True)

    vowel_sounds = ["อา", "อี", "อือ", "อู", "ไอ", "อำ", "เอา"]
    vowel_paths = []

    for sound in vowel_sounds:
        st.markdown(f"<p class='pronounce'>ออกเสียง <b>\"{sound}\"</b></p>", unsafe_allow_html=True)
        audio_bytes = st.audio_input(f"🎤 บันทึกเสียง {sound}")
        if audio_bytes:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(audio_bytes.read())
                vowel_paths.append(tmp.name)
            st.success(f"บันทึกเสียง \"{sound}\" สำเร็จ", icon="✅")

    uploaded_vowels = st.file_uploader("อัปโหลดไฟล์เสียงสระ (7 ไฟล์)", type=["wav", "mp3", "m4a"], accept_multiple_files=True)
    if uploaded_vowels and not vowel_paths:
        for file in uploaded_vowels[:7]:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(file.read())
                vowel_paths.append(tmp.name)

    # =============================
    # Pataka Recording
    # =============================
    st.markdown("""
    <div class='card'>
        <h2>2. พยางค์</h2>
        <p class='instructions'>กรุณาออกเสียงคำว่า <b>"พา - ทา - คา"</b> ให้จบภายใน 6 วินาที</p>
    </div>
    """, unsafe_allow_html=True)

    pataka_path = None
    pataka_bytes = st.audio_input("🎤 บันทึกเสียงพยางค์")
    if pataka_bytes:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(pataka_bytes.read())
            pataka_path = tmp.name
        st.success("บันทึกพยางค์สำเร็จ", icon="✅")

    uploaded_pataka = st.file_uploader("อัปโหลดไฟล์เสียงพยางค์", type=["wav", "mp3", "m4a"], accept_multiple_files=False)
    if uploaded_pataka and not pataka_path:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded_pataka.read())
            pataka_path = tmp.name

    # =============================
    # Sentence Recording
    # =============================
    st.markdown("""
    <div class='card'>
        <h2>3. ประโยค</h2>
        <p class='sentence-instruction'>กรุณาอ่านประโยค <b>"วันนี้อากาศแจ่มใสนกร้องเสียงดังเป็นจังหวะ"</b></p>
    </div>
    """, unsafe_allow_html=True)

    sentence_path = None
    sentence_bytes = st.audio_input("🎤 บันทึกการอ่านประโยค")
    if sentence_bytes:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(sentence_bytes.read())
            sentence_path = tmp.name
        st.success("บันทึกประโยคสำเร็จ", icon="✅")

    uploaded_sentence = st.file_uploader("อัปโหลดไฟล์เสียงประโยค", type=["wav", "mp3", "m4a"], accept_multiple_files=False)
    if uploaded_sentence and not sentence_path:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded_sentence.read())
            sentence_path = tmp.name

    # =============================
    # Enhanced Buttons Layout
    # =============================
    col1, col2 = st.columns([1, 0.18])
    with col1:
        button_col1, button_col2 = st.columns([1, 1])
        with button_col1:
            predict_btn = st.button("วิเคราะห์", key="predict", type="primary")
        with button_col2:
            loading_placeholder = st.empty()
    with col2:
        st.markdown("""
            <div style="display: flex; justify-content: flex-end;">
        """, unsafe_allow_html=True)
        # Enhanced clear button that actually clears uploaded files
        if st.button("ลบข้อมูล", key="clear", type="secondary"):
            clear_all_data()
        st.markdown("</div>", unsafe_allow_html=True)

    # =============================
    # Prediction Logic
    # =============================
    if predict_btn:
        if len(vowel_paths) == 7 and pataka_path and sentence_path:
            # Show loading indicator
            loading_placeholder.markdown("""
                <div style="display: flex; align-items: center; margin-top: 8px;">
                    <div style="width: 20px; height: 20px; border: 3px solid #f3f3f3; border-top: 3px solid #009688; border-radius: 50%; animation: spin 1s linear infinite;"></div>
                    <span style="margin-left: 10px; font-size: 16px; color: #009688;">กำลังวิเคราะห์...</span>
                </div>
                <style>
                    @keyframes spin {
                        0% { transform: rotate(0deg); }
                        100% { transform: rotate(360deg); }
                    }
                </style>
            """, unsafe_allow_html=True)
            
            all_probs = predict_from_model(vowel_paths, pataka_path, sentence_path)
            final_prob = np.mean(all_probs)
            percent = int(final_prob * 100)
            
            # Clear loading indicator
            loading_placeholder.empty()

            if percent <= 50:
                level = "ระดับต่ำ (Low)"
                label = "Non Parkinson"
                diagnosis = "ไม่เป็นพาร์กินสัน"
                box_color = "#e6f9e6"
                advice = """
                <ul style='font-size:28px;'>
                    <li>ถ้าไม่มีอาการ: ควรตรวจปีละครั้ง(ไม่บังคับ)</li>
                    <li>ถ้ามีอาการเล็กน้อย: ตรวจปีละ 2 ครั้ง</li>
                    <li>ถ้ามีอาการเตือน: ตรวจ 2–4 ครั้งต่อปี</li>
                </ul>
                """
            elif percent <= 75:
                level = "ปานกลาง (Moderate)"
                label = "Parkinson"
                diagnosis = "เป็นพาร์กินสัน"
                box_color = "#fff7e6"
                advice = """
                <ul style='font-size:28px;'>
                    <li>พบแพทย์เฉพาะทางระบบประสาท</li>
                    <li>บันทึกอาการประจำวัน</li>
                    <li>หากได้รับยา: บันทึกผลข้างเคียง</li>
                </ul>
                """
            else:
                level = "สูง (High)"
                label = "Parkinson"
                diagnosis = "เป็นพาร์กินสัน"
                box_color = "#ffe6e6"
                advice = """
                <ul style='font-size:28px;'>
                    <li>พบแพทย์เฉพาะทางโดยเร็วที่สุด</li>
                    <li>บันทึกอาการทุกวัน</li>
                    <li>หากได้รับยา: ติดตามผลอย่างละเอียด</li>
                </ul>
                """

            st.markdown(f"""
                <div style='background-color:{box_color}; padding: 32px; border-radius: 14px; font-size: 30px; color: #000000;'>
                    <div style='text-align: center; font-size: 42px; font-weight: bold; margin-bottom: 20px;'>{label}:</div>
                    <p><b>ระดับความน่าจะเป็น:</b> {level}</p>
                    <p><b>ความน่าจะเป็นของพาร์กินสัน:</b> {percent}%</p>
                    <div style='height: 36px; background: linear-gradient(to right, green, yellow, red); border-radius: 6px; margin-bottom: 16px; position: relative;'>
                        <div style='position: absolute; left: {percent}%; top: 0; bottom: 0; width: 4px; background-color: black;'></div>
                    </div>
                    <p><b>ผลการวิเคราะห์:</b> {diagnosis}</p>
                    <p><b>คำแนะนำ</b></p>
                    {advice}
                </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("กรุณาอัดเสียงหรืออัปโหลดให้ครบทั้ง 7 สระ พยางค์ และประโยค", icon="⚠️")

# =============================
# Main App Logic
# =============================
if not st.session_state.show_main_app:
    show_landing_page()
else:
    show_main_app()
