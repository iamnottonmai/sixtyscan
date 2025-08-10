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

# Import your model class
try:
    from model import ResNet18Classifier
except ImportError:
    st.error("Could not import ResNet18Classifier from model.py. Make sure the file exists.")
    st.stop()

# =============================
# Initialize Session State
# =============================
if 'page' not in st.session_state:
    st.session_state.page = 'home'

# =============================
# Page Config
# =============================
st.set_page_config(
    page_title="SixtyScan - เริ่มต้น", 
    layout="centered",
    initial_sidebar_state="collapsed"
)

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
    
    return None

def display_logo():
    """Display logo if available"""
    logo_b64 = load_logo()
    if logo_b64:
        st.markdown(f"""
        <img src="data:image/png;base64,{logo_b64}" class="logo" alt="SixtyScan Logo">
        """, unsafe_allow_html=True)

# =============================
# Global Styles
# =============================
def load_styles():
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
            
            .pronounce b, .instructions b, .sentence-instruction b {
                font-weight: 700 !important;
            }
            
            .sentence-instruction {
                font-size: 24px !important;
                font-weight: 400 !important;
                color: #333 !important;
                margin-bottom: 24px !important;
                font-family: 'Noto Sans Thai', sans-serif !important;
                display: block !important;
            }
            
            /* Custom button styling */
            .stButton > button {
                font-size: 40px !important;
                padding: 35px 48px !important;
                border-radius: 50px !important;
                font-weight: 900 !important;
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
def show_home_page():
    """Display the home page"""
    load_styles()
    display_logo()

    st.markdown("<h1 class='title'>SixtyScan</h1>", unsafe_allow_html=True)

    st.markdown("""
        <div class='description'>
            ตรวจโรคพาร์กินสันจากเสียงด้วยปัญญาประดิษฐ์<br>
            เทคโนโลยีที่ทันสมัยเพื่อการตรวจคัดกรองเบื้องต้น
        </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([0.6, 1, 0.5])
    with col2:
        if st.button("เริ่มการวิเคราะห์", key="start_analysis"):
            st.session_state.page = 'analysis'
            st.rerun()

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

    st.markdown("<br><br>", unsafe_allow_html=True)

def show_analysis_page():
    """Display the analysis page"""
    load_styles()
    initialize_analysis_session_state()
    
    # Back button
    if st.button("← กลับหน้าแรก", key="back_to_home"):
        st.session_state.page = 'home'
        st.rerun()
    
    # Load model
    model = load_model()
    
    display_logo()
    st.markdown("<h1 class='title'>SixtyScan</h1>", unsafe_allow_html=True)
    st.markdown("<p class='description' style='font-size: 32px; margin-bottom: 56px;'>ตรวจโรคพาร์กินสันจากเสียง</p>", unsafe_allow_html=True)

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
    <div class='card'>
        <h2>1. สระ</h2>
        <p class='instructions'>กรุณาออกเสียงแต่ละสระ 5-8 วินาทีอย่างชัดเจน โดยกดปุ่มบันทึกทีละสระ</p>
    </div>
    """, unsafe_allow_html=True)

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
                st.markdown(f"<div style='color: black; font-size: 16px; margin-bottom: 8px; text-align: center;'>Mel Spectrogram: <b>\"{sound}\"</b></div>", unsafe_allow_html=True)
                st.image(spec_image, use_container_width=True)

    uploaded_vowels = st.file_uploader("อัปโหลดไฟล์เสียงสระ (7 ไฟล์)", type=["wav", "mp3", "m4a"], accept_multiple_files=True)
    if uploaded_vowels and len([f for f in st.session_state.vowel_files if f is not None]) < 7:
        cleanup_temp_files()
        st.session_state.vowel_files = []
        for file in uploaded_vowels[:7]:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(file.read())
                st.session_state.vowel_files.append(tmp.name)

    # Pataka recording
    st.markdown("""
    <div class='card'>
        <h2>2. พยางค์</h2>
        <p class='instructions'>กรุณาออกเสียงคำว่า <b>"พา - ทา - คา"</b> ให้จบภายใน 6 วินาที</p>
    </div>
    """, unsafe_allow_html=True)

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
            st.markdown("<div style='color: black; font-size: 16px; margin-bottom: 8px; text-align: center;'>Mel Spectrogram: <b>\"พา-ทา-คา\"</b></div>", unsafe_allow_html=True)
            st.image(spec_image, use_container_width=True)

    uploaded_pataka = st.file_uploader("อัปโหลดไฟล์เสียงพยางค์", type=["wav", "mp3", "m4a"], accept_multiple_files=False)
    if uploaded_pataka and not st.session_state.pataka_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded_pataka.read())
            st.session_state.pataka_file = tmp.name

    # Sentence recording
    st.markdown("""
    <div class='card'>
        <h2>3. ประโยค</h2>
        <p class='sentence-instruction'>กรุณาอ่านประโยค <b>"วันนี้อากาศแจ่มใสนกร้องเสียงดังเป็นจังหวะ"</b></p>
    </div>
    """, unsafe_allow_html=True)

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
            st.markdown("<div style='color: black; font-size: 16px; margin-bottom: 8px; text-align: center;'>Mel Spectrogram: <b>\"วันนี้อากาศแจ่มใสนกร้องเสียงดังเป็นจังหวะ\"</b></div>", unsafe_allow_html=True)
            st.image(spec_image, use_container_width=True)

    uploaded_sentence = st.file_uploader("อัปโหลดไฟล์เสียงประโยค", type=["wav", "mp3", "m4a"], accept_multiple_files=False)
    if uploaded_sentence and not st.session_state.sentence_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded_sentence.read())
            st.session_state.sentence_file = tmp.name

    # Buttons
    col1, col2 = st.columns([1, 0.16])
    with col1:
        button_col1, button_col2 = st.columns([1, 1])
        with button_col1:
            predict_btn = st.button("วิเคราะห์", key="predict", type="primary")
        with button_col2:
            loading_placeholder = st.empty()
    with col2:
        if st.button("ลบข้อมูล", key="clear", type="secondary"):
            st.session_state.clear_button_clicked = True
            st.rerun()

    # Reset clear_clicked flag
    if st.session_state.clear_clicked:
        st.session_state.clear_clicked = False

    # Prediction logic
    if predict_btn:
        valid_vowel_files = [f for f in st.session_state.vowel_files if f is not None]
        
        if len(valid_vowel_files) == 7 and st.session_state.pataka_file and st.session_state.sentence_file:
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
            
            all_probs = predict_from_model(valid_vowel_files, st.session_state.pataka_file, st.session_state.sentence_file, model)
            final_prob = np.mean(all_probs)
            percent = int(final_prob * 100)
            
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
            
            # Display all spectrograms in the results section
            st.markdown("### 📊 การวิเคราะห์ Mel Spectrogram ทั้งหมด")
            
            # Create a grid layout for all spectrograms
            spec_cols = st.columns(3)
            
            # Display vowel spectrograms
            for i, (sound, file_path) in enumerate(zip(vowel_sounds, valid_vowel_files)):
                with spec_cols[i % 3]:
                    spec_image = create_mel_spectrogram_display(file_path, f"สระ \"{sound}\"")
                    if spec_image:
                        st.markdown(f"<div style='color: black; font-size: 14px; margin-bottom: 8px; text-align: center;'>Mel Spectrogram: <b>\"{sound}\"</b></div>", unsafe_allow_html=True)
                        st.image(spec_image, use_container_width=True)
            
            # Display pataka spectrogram
            col_idx = len(vowel_sounds) % 3
            with spec_cols[col_idx]:
                spec_image = create_mel_spectrogram_display(st.session_state.pataka_file, "พยางค์")
                if spec_image:
                    st.markdown("<div style='color: black; font-size: 14px; margin-bottom: 8px; text-align: center;'>Mel Spectrogram: <b>\"พา-ทา-คา\"</b></div>", unsafe_allow_html=True)
                    st.image(spec_image, use_container_width=True)
            
            # Display sentence spectrogram
            col_idx = (len(vowel_sounds) + 1) % 3
            with spec_cols[col_idx]:
                spec_image = create_mel_spectrogram_display(st.session_state.sentence_file, "ประโยค")
                if spec_image:
                    st.markdown("<div style='color: black; font-size: 14px; margin-bottom: 8px; text-align: center;'>Mel Spectrogram: <b>\"ประโยค\"</b></div>", unsafe_allow_html=True)
                    st.image(spec_image, use_container_width=True)
            
            st.markdown("""
            <div style='margin-top: 20px; padding: 20px; background-color: #f0f2f6; border-radius: 10px;'>
                <h4 style='color: #4A148C; margin-bottom: 10px;'>💡 เกี่ยวกับ Mel Spectrogram</h4>
                <p style='font-size: 16px; margin-bottom: 8px;'>• <b>สีเข้ม (น้ำเงิน/ม่วง):</b> ความถี่ที่มีพลังงานต่ำ</p>
                <p style='font-size: 16px; margin-bottom: 8px;'>• <b>สีอ่อน (เหลือง/แดง):</b> ความถี่ที่มีพลังงานสูง</p>
                <p style='font-size: 16px; margin-bottom: 8px;'>• <b>แกน X:</b> เวลา (วินาที)</p>
                <p style='font-size: 16px; margin-bottom: 8px;'>• <b>แกน Y:</b> ความถี่ Mel</p>
                <p style='font-size: 16px;'>• รูปแบบของ Spectrogram สามารถช่วยระบุความผิดปกติของการออกเสียงได้</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("กรุณาอัดเสียงหรืออัปโหลดให้ครบทั้ง 7 สระ พยางค์ และประโยค", icon="⚠")

# =============================
# Main App Logic
# =============================
if st.session_state.page == 'home':
    show_home_page()
elif st.session_state.page == 'analysis':
    show_analysis_page()
