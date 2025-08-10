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

def run_mobile_app():
    """Main function to run the mobile version"""
    # =============================
    # Initialize Session State
    # =============================
    if 'page' not in st.session_state:
        st.session_state.page = 'home'

    # =============================
    # Page Config - Mobile optimized
    # =============================
    st.set_page_config(
        page_title="SixtyScan", 
        layout="centered",
        initial_sidebar_state="collapsed"
    )

    # =============================
    # Logo Loading Function - Mobile Specific
    # =============================
    @st.cache_data
    def load_logo():
        """Load mobile-specific logo with fallback options for reliability"""
        logo_paths = [
            "mobilelogo.png",           # Mobile-specific logo - same directory
            "./mobilelogo.png",         # Explicit relative path
            "assets/mobilelogo.png",    # If in assets folder
            "images/mobilelogo.png",    # If in images folder
            # Fallback to regular logo if mobile logo not found
            "logo.png",                 # Regular logo as fallback
            "./logo.png",
            "assets/logo.png",
            "images/logo.png"
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
        """Display mobile logo without title and description"""
        logo_b64 = load_logo()
        if logo_b64:
            st.markdown(f"""
            <img src="data:image/png;base64,{logo_b64}" class="logo" alt="SixtyScan Mobile Logo">
            """, unsafe_allow_html=True)

    # =============================
    # Mobile-Optimized Styles
    # =============================
    def load_mobile_styles():
        st.markdown("""
            <style>
                @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Thai:wght@400;600&family=Lexend+Deca:wght@700&display=swap');
                
                /* Global - Mobile optimized */
                html, body {
                    background-color: #f2f4f8;
                    font-family: 'Noto Sans Thai', sans-serif;
                    font-weight: 400;
                }
                
                /* Hide Streamlit elements */
                .stDeployButton {display:none;}
                footer {visibility: hidden;}
                .stApp > header {visibility: hidden;}
                
                /* Mobile-optimized logo */
                .logo {
                    display: block;
                    margin-left: auto;
                    margin-right: auto;
                    width: 140px;
                    margin-bottom: 20px;
                }
                
                /* Mobile About Us Section */
                .about-section {
                    background-color: #ffffff;
                    border-radius: 12px;
                    padding: 20px;
                    box-shadow: 0 4px 20px rgba(0,0,0,0.06);
                    margin: 20px auto;
                    max-width: 100%;
                }
                
                .about-title {
                    font-size: 24px;
                    color: #4A148C;
                    font-weight: 600;
                    font-family: 'Noto Sans Thai', sans-serif;
                    margin-bottom: 15px;
                    text-align: center;
                }
                
                .about-content {
                    font-size: 16px;
                    color: #333;
                    font-weight: 400;
                    font-family: 'Noto Sans Thai', sans-serif;
                    line-height: 1.5;
                    text-align: justify;
                }
                
                /* Mobile Card container */
                .card {
                    background-color: #ffffff;
                    border-radius: 12px;
                    padding: 20px;
                    box-shadow: 0 4px 20px rgba(0,0,0,0.06);
                    margin-bottom: 20px;
                }
                
                /* Mobile Section headers */
                .card h2 {
                    font-size: 28px;
                    margin-bottom: 15px;
                    color: #222;
                    font-weight: 600;
                    font-family: 'Noto Sans Thai', sans-serif;
                }
                
                /* Mobile Instructions text */
                .instructions {
                    font-size: 16px !important;
                    color: #333;
                    margin-bottom: 16px;
                    font-weight: 400;
                    font-family: 'Noto Sans Thai', sans-serif;
                }
                
                /* Mobile Pronunciation display */
                .pronounce {
                    font-size: 18px !important;
                    color: #000;
                    font-weight: 400;
                    margin-top: 0;
                    margin-bottom: 12px;
                    font-family: 'Noto Sans Thai', sans-serif;
                }
                
                .pronounce b, .instructions b, .sentence-instruction b {
                    font-weight: 700 !important;
                }
                
                .sentence-instruction {
                    font-size: 16px !important;
                    font-weight: 400 !important;
                    color: #333 !important;
                    margin-bottom: 16px !important;
                    font-family: 'Noto Sans Thai', sans-serif !important;
                    display: block !important;
                    line-height: 1.4 !important;
                }
                
                /* Mobile-optimized button styling */
                .stButton > button {
                    font-size: 20px !important;
                    padding: 16px 24px !important;
                    border-radius: 25px !important;
                    font-weight: 700 !important;
                    background: linear-gradient(135deg, #009688, #00bcd4) !important;
                    color: white !important;
                    border: none !important;
                    box-shadow: 0 4px 15px rgba(0, 150, 136, 0.3) !important;
                    transition: all 0.3s ease !important;
                    font-family: 'Noto Sans Thai', sans-serif !important;
                    width: 100% !important;
                    min-height: 50px !important;
                }
                
                .stButton > button:hover {
                    background: linear-gradient(135deg, #00796b, #0097a7) !important;
                    box-shadow: 0 6px 20px rgba(0, 150, 136, 0.4) !important;
                    transform: translateY(-2px) !important;
                }
                
                .stButton > button:active {
                    transform: translateY(0px) !important;
                }
                
                /* Mobile audio input styling */
                .stAudioInput {
                    margin-bottom: 16px;
                }
                
                /* Mobile file uploader */
                .stFileUploader {
                    margin-bottom: 16px;
                }
                
                /* Mobile specific utility classes */
                .mobile-spacing {
                    margin-bottom: 16px;
                }
                
                .mobile-text-small {
                    font-size: 14px !important;
                }
                
                .mobile-center {
                    text-align: center;
                }
            </style>
        """, unsafe_allow_html=True)

    # =============================
    # Analysis Functions (same as desktop)
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
            with st.spinner("กำลังดาวน์โหลดโมเดล..."):
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

    def create_mel_spectrogram_display_mobile(file_path, title="Mel Spectrogram"):
        """Create a mobile-optimized mel spectrogram for display"""
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

            # Mobile-optimized figure size
            fig, ax = plt.subplots(figsize=(6, 3), dpi=100, facecolor='white')
            
            img = librosa.display.specshow(mel_db, sr=sr, ax=ax, x_axis='time', y_axis='mel', 
                                          cmap='plasma', fmax=8000)
            
            ax.set_xlabel('เวลา (วินาที)', fontsize=10)
            ax.set_ylabel('ความถี่ Mel', fontsize=10)
            
            # Smaller colorbar for mobile
            cbar = plt.colorbar(img, ax=ax, format='%+2.0f dB', shrink=0.8)
            cbar.set_label('พลังงาน (dB)', fontsize=8)
            
            ax.grid(True, alpha=0.3)
            plt.tight_layout()

            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=120, facecolor='white')
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
    # Mobile Page Functions
    # =============================
    def show_mobile_home_page():
        """Display the mobile-optimized home page - logo only"""
        load_mobile_styles()
        
        # Only display the logo - no title or description
        display_logo()

        # Full width button for mobile
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

    def show_mobile_analysis_page():
        """Display the mobile-optimized analysis page - logo only at top"""
        load_mobile_styles()
        initialize_analysis_session_state()
        
        # Mobile back button
        if st.button("← กลับหน้าแรก", key="back_to_home"):
            st.session_state.page = 'home'
            st.rerun()
        
        # Load model
        model = load_model()
        
        # Only display the logo - no title or description
        display_logo()

        # Clear button logic
        if 'clear_button_clicked' in st.session_state and st.session_state.clear_button_clicked:
            cleanup_temp_files()
            st.session_state.vowel_files = []
            st.session_state.pataka_file = None
            st.session_state.sentence_file = None
            st.session_state.clear_clicked = True
            st.session_state.clear_button_clicked = False
            st.success("ลบข้อมูลเรียบร้อยแล้ว", icon="🗑️")
            st.rerun()

        # Mobile progress indicator
        progress_text = f"ความคืบหน้า: {len([f for f in st.session_state.vowel_files if f is not None])}/7 สระ"
        if st.session_state.pataka_file:
            progress_text += " ✓พยางค์"
        if st.session_state.sentence_file:
            progress_text += " ✓ประโยค"
        st.info(progress_text, icon="📊")

        # Mobile vowel recordings with collapsible sections
        with st.expander("1. บันทึกเสียงสระ (7 เสียง)", expanded=True):
            st.markdown("<p class='instructions'>กรุณาออกเสียงแต่ละสระ 5-8 วินาทีอย่างชัดเจน</p>", unsafe_allow_html=True)
            
            vowel_sounds = ["อา", "อี", "อือ", "อู", "ไอ", "อำ", "เอา"]
            
            for i, sound in enumerate(vowel_sounds):
                st.markdown(f"<p class='pronounce mobile-center'>ออกเสียง <b>\"{sound}\"</b></p>", unsafe_allow_html=True)
                
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
                        
                        # Show mobile spectrogram immediately
                        spec_image = create_mel_spectrogram_display_mobile(st.session_state.vowel_files[i], f"สระ \"{sound}\"")
                        if spec_image:
                            with st.expander(f"ดู Spectrogram: \"{sound}\""):
                                st.image(spec_image, use_container_width=True)
                else:
                    st.audio_input(f"🎤 บันทึกเสียง {sound}", key=f"vowel_{i}_new")
            
            # Mobile file upload option
            st.markdown("**หรืออัปโหลดไฟล์:**")
            uploaded_vowels = st.file_uploader("อัปโหลดไฟล์เสียงสระ (7 ไฟล์)", type=["wav", "mp3", "m4a"], accept_multiple_files=True)
            if uploaded_vowels and len([f for f in st.session_state.vowel_files if f is not None]) < 7:
                cleanup_temp_files()
                st.session_state.vowel_files = []
                for file in uploaded_vowels[:7]:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                        tmp.write(file.read())
                        st.session_state.vowel_files.append(tmp.name)

        # Mobile Pataka recording
        with st.expander("2. บันทึกพยางค์", expanded=True):
            st.markdown("<p class='instructions'>กรุณาออกเสียงคำว่า <b>\"พา - ทา - คา\"</b> ให้จบภายใน 6 วินาที</p>", unsafe_allow_html=True)

            if not st.session_state.clear_clicked:
                pataka_bytes = st.audio_input("🎤 บันทึกเสียงพยางค์", key="pataka")
                if pataka_bytes:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                        tmp.write(pataka_bytes.read())
                        if st.session_state.pataka_file and os.path.exists(st.session_state.pataka_file):
                            os.unlink(st.session_state.pataka_file)
                        st.session_state.pataka_file = tmp.name
                    st.success("บันทึกพยางค์สำเร็จ", icon="✅")
                    
                    # Show mobile spectrogram immediately
                    spec_image = create_mel_spectrogram_display_mobile(st.session_state.pataka_file, "พยางค์")
                    if spec_image:
                        with st.expander("ดู Spectrogram: \"พา-ทา-คา\""):
                            st.image(spec_image, use_container_width=True)
            else:
                pataka_bytes = st.audio_input("🎤 บันทึกเสียงพยางค์", key="pataka_new")

            st.markdown("**หรืออัปโหลดไฟล์:**")
            uploaded_pataka = st.file_uploader("อัปโหลดไฟล์เสียงพยางค์", type=["wav", "mp3", "m4a"], accept_multiple_files=False)
            if uploaded_pataka and not st.session_state.pataka_file:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    tmp.write(uploaded_pataka.read())
                    st.session_state.pataka_file = tmp.name

        # Mobile Sentence recording
        with st.expander("3. บันทึกการอ่านประโยค", expanded=True):
            st.markdown("<p class='sentence-instruction'>กรุณาอ่านประโยค<br><b>\"วันนี้อากาศแจ่มใสนกร้องเสียงดังเป็นจังหวะ\"</b></p>", unsafe_allow_html=True)

            if not st.session_state.clear_clicked:
                sentence_bytes = st.audio_input("🎤 บันทึกการอ่านประโยค", key="sentence")
                if sentence_bytes:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                        tmp.write(sentence_bytes.read())
                        if st.session_state.sentence_file and os.path.exists(st.session_state.sentence_file):
                            os.unlink(st.session_state.sentence_file)
                        st.session_state.sentence_file = tmp.name
                    st.success("บันทึกประโยคสำเร็จ", icon="✅")
                    
                    # Show mobile spectrogram immediately
                    spec_image = create_mel_spectrogram_display_mobile(st.session_state.sentence_file, "ประโยค")
                    if spec_image:
                        with st.expander("ดู Spectrogram: \"ประโยค\""):
                            st.image(spec_image, use_container_width=True)
            else:
                sentence_bytes = st.audio_input("🎤 บันทึกการอ่านประโยค", key="sentence_new")

            st.markdown("**หรืออัปโหลดไฟล์:**")
            uploaded_sentence = st.file_uploader("อัปโหลดไฟล์เสียงประโยค", type=["wav", "mp3", "m4a"], accept_multiple_files=False)
            if uploaded_sentence and not st.session_state.sentence_file:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    tmp.write(uploaded_sentence.read())
                    st.session_state.sentence_file = tmp.name

        # Mobile action buttons
        st.markdown("### การดำเนินการ")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            predict_btn = st.button("🔍 วิเคราะห์", key="predict", type="primary")
        with col2:
            if st.button("🗑️ ลบ", key="clear", type="secondary"):
                st.session_state.clear_button_clicked = True
                st.rerun()

        loading_placeholder = st.empty()

        # Reset clear_clicked flag
        if st.session_state.clear_clicked:
            st.session_state.clear_clicked = False

        # Mobile prediction logic
        if predict_btn:
            valid_vowel_files = [f for f in st.session_state.vowel_files if f is not None]
            
            if len(valid_vowel_files) == 7 and st.session_state.pataka_file and st.session_state.sentence_file:
                loading_placeholder.markdown("""
                    <div style="display: flex; align-items: center; justify-content: center; margin: 20px 0;">
                        <div style="width: 30px; height: 30px; border: 4px solid #f3f3f3; border-top: 4px solid #009688; border-radius: 50%; animation: spin 1s linear infinite;"></div>
                        <span style="margin-left: 15px; font-size: 18px; color: #009688; font-weight: 600;">กำลังวิเคราะห์...</span>
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
                    <ul style='font-size:16px; line-height:1.5;'>
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
                    <ul style='font-size:16px; line-height:1.5;'>
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
                    <ul style='font-size:16px; line-height:1.5;'>
                        <li>พบแพทย์เฉพาะทางโดยเร็วที่สุด</li>
                        <li>บันทึกอาการทุกวัน</li>
                        <li>หากได้รับยา: ติดตามผลอย่างละเอียด</li>
                    </ul>
                    """

                # Mobile-optimized results display
                st.markdown(f"""
                    <div style='background-color:{box_color}; padding: 20px; border-radius: 12px; font-size: 16px; color: #000000; margin: 20px 0;'>
                        <div style='text-align: center; font-size: 24px; font-weight: bold; margin-bottom: 15px;'>{label}</div>
                        <p style='margin-bottom: 8px;'><b>ระดับความน่าจะเป็น:</b> {level}</p>
                        <p style='margin-bottom: 12px;'><b>ความน่าจะเป็นของพาร์กินสัน:</b> {percent}%</p>
                        <div style='height: 24px; background: linear-gradient(to right, green, yellow, red); border-radius: 12px; margin-bottom: 12px; position: relative;'>
                            <div style='position: absolute; left: {percent}%; top: -2px; bottom: -2px; width: 4px; background-color: black; border-radius: 2px;'></div>
                        </div>
                        <p style='margin-bottom: 8px;'><b>ผลการวิเคราะห์:</b> {diagnosis}</p>
                        <p style='margin-bottom: 8px; font-weight: bold;'>คำแนะนำ:</p>
                        {advice}
                    </div>
                """, unsafe_allow_html=True)
                
                # Mobile spectrograms in expandable section
                with st.expander("📊 ดูการวิเคราะห์ Mel Spectrogram ทั้งหมด", expanded=False):
                    vowel_sounds = ["อา", "อี", "อือ", "อู", "ไอ", "อำ", "เอา"]
                    
                    # Display vowel spectrograms
                    st.markdown("**เสียงสระ:**")
                    for i, (sound, file_path) in enumerate(zip(vowel_sounds, valid_vowel_files)):
                        with st.expander(f"Spectrogram สระ \"{sound}\""):
                            spec_image = create_mel_spectrogram_display_mobile(file_path, f"สระ \"{sound}\"")
                            if spec_image:
                                st.image(spec_image, use_container_width=True)
                    
                    # Display pataka spectrogram
                    st.markdown("**เสียงพยางค์:**")
                    with st.expander("Spectrogram \"พา-ทา-คา\""):
                        spec_image = create_mel_spectrogram_display_mobile(st.session_state.pataka_file, "พยางค์")
                        if spec_image:
                            st.image(spec_image, use_container_width=True)
                    
                    # Display sentence spectrogram
                    st.markdown("**เสียงประโยค:**")
                    with st.expander("Spectrogram ประโยค"):
                        spec_image = create_mel_spectrogram_display_mobile(st.session_state.sentence_file, "ประโยค")
                        if spec_image:
                            st.image(spec_image, use_container_width=True)
                
                # Mobile info section
                st.markdown("""
                <div style='margin-top: 15px; padding: 15px; background-color: #f0f2f6; border-radius: 8px;'>
                    <h4 style='color: #4A148C; margin-bottom: 8px; font-size: 16px;'>💡 เกี่ยวกับ Mel Spectrogram</h4>
                    <p style='font-size: 14px; margin-bottom: 6px;'>• <b>สีเข้ม (น้ำเงิน/ม่วง):</b> ความถี่ที่มีพลังงานต่ำ</p>
                    <p style='font-size: 14px; margin-bottom: 6px;'>• <b>สีอ่อน (เหลือง/แดง):</b> ความถี่ที่มีพลังงานสูง</p>
                    <p style='font-size: 14px; margin-bottom: 6px;'>• <b>แกน X:</b> เวลา (วินาที)</p>
                    <p style='font-size: 14px; margin-bottom: 6px;'>• <b>แกน Y:</b> ความถี่ Mel</p>
                    <p style='font-size: 14px;'>• รูปแบบของ Spectrogram สามารถช่วยระบุความผิดปกติของการออกเสียงได้</p>
                </div>
                """, unsafe_allow_html=True)
                
            else:
                missing_items = []
                valid_vowel_count = len([f for f in st.session_state.vowel_files if f is not None])
                if valid_vowel_count < 7:
                    missing_items.append(f"สระ ({valid_vowel_count}/7)")
                if not st.session_state.pataka_file:
                    missing_items.append("พยางค์")
                if not st.session_state.sentence_file:
                    missing_items.append("ประโยค")
                
                st.warning(f"กรุณาบันทึกเสียงให้ครบ: {', '.join(missing_items)}", icon="⚠️")

    # =============================
    # Main Mobile App Logic
    # =============================
    if st.session_state.page == 'home':
        show_mobile_home_page()
    elif st.session_state.page == 'analysis':
        show_mobile_analysis_page()
