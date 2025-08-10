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
    # Global Styles
    # =============================
    def load_styles():
        st.markdown("""
            <style>
                /* Import fonts */
                @import url('https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;500;600;700;800;900&display=swap');
                
                /* Global Reset */
                .stApp {
                    background-color: #f5f5f5 !important;
                    margin: 0 !important;
                    padding: 0 !important;
                    font-family: 'Prompt', sans-serif !important;
                }
                
                /* Hide Streamlit elements */
                .stDeployButton {display:none;}
                footer {visibility: hidden;}
                .stApp > header {visibility: hidden;}
                #MainMenu {visibility: hidden;}
                
                /* Remove the keyboard arrow icon */
                .stButton > button > div[data-testid="stMarkdownContainer"] p {
                    display: none;
                }
                
                /* Header Styles */
                .header {
                    background: linear-gradient(135deg, #4A148C 0%, #6A1B9A 50%, #8E24AA 100%);
                    padding: 16px 40px;
                    display: flex;
                    align-items: center;
                    justify-content: flex-start;
                    margin: 0;
                    width: 100%;
                    box-sizing: border-box;
                }
                
                .header-logo {
                    height: 48px;
                    width: auto;
                    margin-right: 20px;
                }
                
                .header-title {
                    color: white;
                    font-family: 'Prompt', sans-serif;
                    font-size: 24px;
                    font-weight: 500;
                    margin: 0;
                    text-align: left;
                }
                
                .header-datetime {
                    background: rgba(255, 255, 255, 0.2);
                    color: white;
                    padding: 8px 16px;
                    border-radius: 20px;
                    font-family: 'Prompt', sans-serif;
                    font-size: 14px;
                    font-weight: 400;
                    margin-left: auto;
                }
                
                /* Main Content Area - Reduced top padding */
                .main-content {
                    padding: 40px 60px 80px 60px;
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
                    font-family: 'Prompt', sans-serif;
                    font-size: 64px;
                    font-weight: 700;
                    color: #2d2d2d;
                    line-height: 1.2;
                    margin-bottom: 40px;
                    margin-top: 0;
                }
                
                .title-highlight {
                    color: #4A148C;
                }
                
                /* Home page buttons - Keep custom styling */
                .home-button-primary {
                    font-size: 32px !important;
                    padding: 24px 48px !important;
                    border-radius: 50px !important;
                    font-weight: 900 !important;
                    font-family: 'Prompt', sans-serif !important;
                    min-width: 280px !important;
                    height: 80px !important;
                    margin: 10px 0 !important;
                    border: none !important;
                    cursor: pointer !important;
                    transition: all 0.3s ease !important;
                    box-shadow: 0 4px 15px rgba(74, 20, 140, 0.3) !important;
                    text-align: center !important;
                    display: flex !important;
                    align-items: center !important;
                    justify-content: center !important;
                    background: linear-gradient(135deg, #1976D2 0%, #9C27B0 100%) !important;
                    color: white !important;
                }
                
                .home-button-secondary {
                    font-size: 32px !important;
                    padding: 24px 48px !important;
                    border-radius: 50px !important;
                    font-weight: 900 !important;
                    font-family: 'Prompt', sans-serif !important;
                    min-width: 280px !important;
                    height: 80px !important;
                    margin: 10px 0 !important;
                    border: none !important;
                    cursor: pointer !important;
                    transition: all 0.3s ease !important;
                    box-shadow: 0 4px 15px rgba(74, 20, 140, 0.3) !important;
                    text-align: center !important;
                    display: flex !important;
                    align-items: center !important;
                    justify-content: center !important;
                    background: linear-gradient(135deg, #4A148C 0%, #6A1B9A 100%) !important;
                    color: white !important;
                }
                
                /* Home page button hover effects */
                .home-button-primary:hover, .home-button-secondary:hover {
                    transform: translateY(-3px) !important;
                    box-shadow: 0 8px 25px rgba(74, 20, 140, 0.4) !important;
                }
                
                .home-button-primary:active, .home-button-secondary:active {
                    transform: translateY(0px) !important;
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
                
                /* Analysis page styles */
                .card {
                    background-color: #ffffff;
                    border-radius: 16px;
                    padding: 40px;
                    box-shadow: 0 4px 20px rgba(0,0,0,0.06);
                    margin-bottom: 40px;
                }
                
                .card h2 {
                    font-size: 48px;
                    margin-bottom: 20px;
                    color: #222;
                    font-weight: 600;
                    font-family: 'Prompt', sans-serif;
                }
                
                .instructions {
                    font-size: 28px !important;
                    color: #333;
                    margin-bottom: 24px;
                    font-weight: 400;
                    font-family: 'Prompt', sans-serif;
                }
                
                .pronounce {
                    font-size: 24px !important;
                    color: #000;
                    font-weight: 400;
                    margin-top: 0;
                    margin-bottom: 24px;
                    font-family: 'Prompt', sans-serif;
                }
                
                .sentence-instruction {
                    font-size: 24px !important;
                    font-weight: 400 !important;
                    color: #333 !important;
                    margin-bottom: 24px !important;
                    font-family: 'Prompt', sans-serif !important;
                    display: block !important;
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
                {logo_html}
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
                        ทันทีด้วย <span class="title-highlight">SixtyScan</span>
                    </h1>
                    <div class="button-container">
        """, unsafe_allow_html=True)
        
        # Create custom styled buttons for home page only
        if st.button("เริ่มใช้งาน", key="start_analysis"):
            st.session_state.page = 'analysis'
            st.rerun()
            
        # Apply custom styling to the home page buttons
        st.markdown("""
            <script>
                // Apply custom classes to home page buttons
                const buttons = document.querySelectorAll('[data-testid="stButton"] button');
                if (buttons.length >= 1) {
                    buttons[0].className += ' home-button-primary';
                }
            </script>
        """, unsafe_allow_html=True)
            
        if st.button("คู่มือ", key="guide_manual"):
            st.session_state.page = 'guide'
            st.rerun()
            
        # Apply styling to second button
        st.markdown("""
            <script>
                const buttons = document.querySelectorAll('[data-testid="stButton"] button');
                if (buttons.length >= 2) {
                    buttons[1].className += ' home-button-secondary';
                }
            </script>
        """, unsafe_allow_html=True)
        
        st.markdown("""
                    </div>
                </div>
                <div class="content-right">
        """, unsafe_allow_html=True)
        
        # Display woman image
        if woman_image_b64:
            st.markdown(f"""
                <img src="data:image/jpg;base64,{woman_image_b64}" class="woman-image" alt="Woman using phone">
            """, unsafe_allow_html=True)
        else:
            # Fallback placeholder
            st.markdown("""
                <div style="width: 100%; max-width: 500px; height: 400px; background: linear-gradient(135deg, #e3f2fd, #f3e5f5); 
                           border-radius: 20px; display: flex; align-items: center; justify-content: center; 
                           box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);">
                    <div style="text-align: center; color: #666;">
                        <div style="font-size: 48px; margin-bottom: 10px;">📱</div>
                        <div style="font-size: 18px;">insert.jpg<br>not found</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
                </div>
            </div>
        """, unsafe_allow_html=True)

    def show_guide_page():
        """Display the guide/manual page"""
        load_styles()
        show_header()
        
        # Back button - normal Streamlit button
        if st.button("← กลับหน้าแรก", key="back_to_home_from_guide"):
            st.session_state.page = 'home'
            st.rerun()
        
        st.markdown("""
            <div style="max-width: 1000px; margin: 40px auto; padding: 0 40px;">
                <h1 style="text-align: center; color: #4A148C; font-size: 48px; margin-bottom: 40px; font-family: 'Prompt', sans-serif;">คู่มือการใช้งาน SixtyScan</h1>
                
                <div style="background: white; padding: 40px; border-radius: 16px; box-shadow: 0 4px 20px rgba(0,0,0,0.06); margin-bottom: 30px;">
                    <h2 style="color: #4A148C; font-size: 32px; margin-bottom: 20px; font-family: 'Prompt', sans-serif;">การเตรียมตัวก่อนการตรวจ</h2>
                    <ul style="font-size: 20px; line-height: 1.6; font-family: 'Prompt', sans-serif;">
                        <li>หาสถานที่เงียบ ปราศจากเสียงรบกวน</li>
                        <li>ใช้ไมโครโฟนหรืออุปกรณ์บันทึกเสียงที่มีคุณภาพ</li>
                        <li>นั่งหรือยืนในท่าที่สบาย</li>
                        <li>พักผ่อนเพียงพอก่อนการตรวจ</li>
                    </ul>
                </div>
                
                <div style="background: white; padding: 40px; border-radius: 16px; box-shadow: 0 4px 20px rgba(0,0,0,0.06); margin-bottom: 30px;">
                    <h2 style="color: #4A148C; font-size: 32px; margin-bottom: 20px; font-family: 'Prompt', sans-serif;">ขั้นตอนการตรวจ</h2>
                    <div style="font-size: 20px; line-height: 1.6; font-family: 'Prompt', sans-serif;">
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
                    <h2 style="color: #4A148C; font-size: 32px; margin-bottom: 20px; font-family: 'Prompt', sans-serif;">ข้อควรระวัง</h2>
                    <ul style="font-size: 20px; line-height: 1.6; color: #d32f2f; font-family: 'Prompt', sans-serif;">
                        <li><strong>ระบบนี้เป็นเพียงการตรวจคัดกรองเบื้องต้น</strong></li>
                        <li><strong>ไม่สามารถทดแทนการวินิจฉัยโดยแพทย์ได้</strong></li>
                        <li><strong>หากมีข้อสงสัยควรปรึกษาแพทย์เฉพาะทาง</strong></li>
                    </ul>
                </div>
            </div>
        """, unsafe_allow_html=True)

    def show_analysis_page():
        """Display the analysis page - desktop version with normal Streamlit buttons for analysis functions"""
        load_styles()
        initialize_analysis_session_state()
        
        # Header
        show_header()
        
        # Back button - normal Streamlit button
        if st.button("← กลับหน้าแรก", key="back_to_home"):
            st.session_state.page = 'home'
            st.rerun()
        
        # Load model
        model = load_model()
        
        st.markdown("<h1 style='text-align: center; font-size: 48px; color: #4A148C; margin: 20px 0; font-family: \"Prompt\", sans-serif;'>การวิเคราะห์เสียง</h1>", unsafe_allow_html=True)

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
                    st.markdown(f"<div style='color: black; font-size: 16px; margin-bottom: 8px; text-align: center; font-family: \"Prompt\", sans-serif;'>Mel Spectrogram: <b>\"{sound}\"</b></div>", unsafe_allow_html=True)
                    st.image(spec_image, use_container_width=True)

        # File uploader for vowels - normal Streamlit component
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
                st.markdown("<div style='color: black; font-size: 16px; margin-bottom: 8px; text-align: center; font-family: \"Prompt\", sans-serif;'>Mel Spectrogram: <b>\"พา-ทา-คา\"</b></div>", unsafe_allow_html=True)
                st.image(spec_image, use_container_width=True)

        # File uploader for pataka - normal Streamlit component
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
                st.markdown("<div style='color: black; font-size: 16px; margin-bottom: 8px; text-align: center; font-family: \"Prompt\", sans-serif;'>Mel Spectrogram: <b>\"วันนี้อากาศแจ่มใสนกร้องเสียงดังเป็นจังหวะ\"</b></div>", unsafe_allow_html=True)
                st.image(spec_image, use_container_width=True)

        # File uploader for sentence - normal Streamlit component
        uploaded_sentence = st.file_uploader("อัปโหลดไฟล์เสียงประโยค", type=["wav", "mp3", "m4a"], accept_multiple_files=False)
        if uploaded_sentence and not st.session_state.sentence_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(uploaded_sentence.read())
                st.session_state.sentence_file = tmp.name

        # Action buttons - Normal Streamlit buttons
        col1, col2 = st.columns([1, 1])
        with col1:
            predict_btn = st.button("🔍 วิเคราะห์", key="predict", type="primary")
        with col2:
            if st.button("🗑️ ลบข้อมูล", key="clear", type="secondary"):
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
                    all_probs = predict_from_model(valid_vowel_files, st.session_state.pataka_file, st.session_state.sentence_file, model)
                    final_prob = np.mean(all_probs)
                    percent = int(final_prob * 100)

                if percent <= 50:
                    level = "ระดับต่ำ (Low)"
                    label = "Non Parkinson"
                    diagnosis = "ไม่เป็นพาร์กินสัน"
                    box_color = "#e6f9e6"
                    advice = """
                    <ul style='font-size:28px; font-family: "Prompt", sans-serif;'>
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
                    <ul style='font-size:28px; font-family: "Prompt", sans-serif;'>
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
                    <ul style='font-size:28px; font-family: "Prompt", sans-serif;'>
                        <li>พบแพทย์เฉพาะทางโดยเร็วที่สุด</li>
                        <li>บันทึกอาการทุกวัน</li>
                        <li>หากได้รับยา: ติดตามผลอย่างละเอียด</li>
                    </ul>
                    """

                st.markdown(f"""
                    <div style='background-color:{box_color}; padding: 32px; border-radius: 14px; font-size: 30px; color: #000000; font-family: "Prompt", sans-serif;'>
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
                            st.markdown(f"<div style='color: black; font-size: 14px; margin-bottom: 8px; text-align: center; font-family: \"Prompt\", sans-serif;'>Mel Spectrogram: <b>\"{sound}\"</b></div>", unsafe_allow_html=True)
                            st.image(spec_image, use_container_width=True)
                
                # Display pataka spectrogram
                col_idx = len(vowel_sounds) % 3
                with spec_cols[col_idx]:
                    spec_image = create_mel_spectrogram_display(st.session_state.pataka_file, "พยางค์")
                    if spec_image:
                        st.markdown("<div style='color: black; font-size: 14px; margin-bottom: 8px; text-align: center; font-family: \"Prompt\", sans-serif;'>Mel Spectrogram: <b>\"พา-ทา-คา\"</b></div>", unsafe_allow_html=True)
                        st.image(spec_image, use_container_width=True)
                
                # Display sentence spectrogram
                col_idx = (len(vowel_sounds) + 1) % 3
                with spec_cols[col_idx]:
                    spec_image = create_mel_spectrogram_display(st.session_state.sentence_file, "ประโยค")
                    if spec_image:
                        st.markdown("<div style='color: black; font-size: 14px; margin-bottom: 8px; text-align: center; font-family: \"Prompt\", sans-serif;'>Mel Spectrogram: <b>\"ประโยค\"</b></div>", unsafe_allow_html=True)
                        st.image(spec_image, use_container_width=True)
                
                st.markdown("""
                <div style='margin-top: 20px; padding: 20px; background-color: #f0f2f6; border-radius: 10px;'>
                    <h4 style='color: #4A148C; margin-bottom: 10px; font-family: "Prompt", sans-serif;'>💡 เกี่ยวกับ Mel Spectrogram</h4>
                    <p style='font-size: 16px; margin-bottom: 8px; font-family: "Prompt", sans-serif;'>• <b>สีเข้ม (น้ำเงิน/ม่วง):</b> ความถี่ที่มีพลังงานต่ำ</p>
                    <p style='font-size: 16px; margin-bottom: 8px; font-family: "Prompt", sans-serif;'>• <b>สีอ่อน (เหลือง/แดง):</b> ความถี่ที่มีพลังงานสูง</p>
                    <p style='font-size: 16px; margin-bottom: 8px; font-family: "Prompt", sans-serif;'>• <b>แกน X:</b> เวลา (วินาที)</p>
                    <p style='font-size: 16px; margin-bottom: 8px; font-family: "Prompt", sans-serif;'>• <b>แกน Y:</b> ความถี่ Mel</p>
                    <p style='font-size: 16px; font-family: "Prompt", sans-serif;'>• รูปแบบของ Spectrogram สามารถช่วยระบุความผิดปกติของการออกเสียงได้</p>
                </div>
                """, unsafe_allow_html=True)
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
