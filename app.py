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
from pydub import AudioSegment

# =============================
# Logo Loader
# =============================
@st.cache_data
def load_logo():
    for path in ["logo.png", "./logo.png", "assets/logo.png", "images/logo.png"]:
        if os.path.exists(path):
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode()
    return None

def display_logo():
    logo_b64 = load_logo()
    if logo_b64:
        st.markdown(f"""<img src="data:image/png;base64,{logo_b64}" class="logo">""", unsafe_allow_html=True)

# =============================
# Model Download
# =============================
MODEL_PATH = "best_resnet18.pth"
if not os.path.exists(MODEL_PATH):
    gdown.download("https://drive.google.com/uc?id=1_oHE9B-2PgSqpTQCC9HrG7yO0rsnZtqs", MODEL_PATH, quiet=False)

# =============================
# Page Config & Styles
# =============================
st.set_page_config(page_title="SixtyScan", layout="centered")
st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Thai:wght@400;600&family=Lexend+Deca:wght@700&display=swap');
html, body { background-color: #f2f4f8; font-family: 'Noto Sans Thai', sans-serif; }
.logo { display:block; margin:0 auto 24px auto; width: 180px; }
h1.title { text-align:center; font-family:'Lexend Deca',sans-serif; font-size:84px; color:#4A148C; font-weight:700; margin-bottom:10px; }
p.subtitle { text-align:center; font-size:32px; margin-bottom:56px; }
.card { background-color:#fff; border-radius:16px; padding:40px; box-shadow:0 4px 20px rgba(0,0,0,0.06); margin-bottom:40px; }
.card h2 { font-size:48px; margin-bottom:20px; color:#222; font-weight:600; }
.instructions { font-size:28px; margin-bottom:24px; }
.pronounce { font-size:24px; margin: 0 0 24px 0; }

/* Big mic button */
.big-mic-wrapper { text-align: center; margin: 20px 0; }
.big-mic-button {
    border-radius: 50%;
    width: 120px; height: 120px;
    display: flex; align-items: center; justify-content: center;
    margin: auto; cursor: pointer;
    box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    transition: background-color 0.3s, transform 0.2s;
}
.big-mic-button i { font-size: 50px; color: white; }
.big-mic-button:hover { transform: scale(1.05); }
.stAudio > div { transform: scale(1.6); transform-origin: center; margin-top: 12px; }

/* Action buttons */
.predict-btn, .clear-btn {
    font-size: 38px !important; padding: 1.4em 2.7em; border-radius: 14px;
    font-weight: bold; width: 100%; max-width: 300px; margin: 10px auto;
}
.predict-btn { background-color: #009688; color: white; border: none; }
.clear-btn { background-color: #cfd8dc; color: black; border: none; }
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
# Audio to Mel Tensor
# =============================
def audio_to_mel_tensor(file_path):
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
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    return transform(image).unsqueeze(0)

# =============================
# Predict Function
# =============================
def predict_from_model(vowel_paths, pataka_path, sentence_path):
    inputs = [audio_to_mel_tensor(p) for p in vowel_paths]
    inputs.append(audio_to_mel_tensor(pataka_path))
    inputs.append(audio_to_mel_tensor(sentence_path))
    with torch.no_grad():
        return [F.softmax(model(x), dim=1)[0][1].item() for x in inputs]

# =============================
# Header
# =============================
display_logo()
st.markdown("<h1 class='title'>SixtyScan</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>ตรวจโรคพาร์กินสันจากเสียง</p>", unsafe_allow_html=True)

# =============================
# Session Init
# =============================
if "vowel_paths" not in st.session_state: st.session_state.vowel_paths = []
if "pataka_path" not in st.session_state: st.session_state.pataka_path = None
if "sentence_path" not in st.session_state: st.session_state.sentence_path = None

# =============================
# Step 1: Vowels
# =============================
st.markdown("<div class='card'><h2>1. สระ</h2><p class='instructions'>บันทึกทีละสระ 5-8 วินาที</p></div>", unsafe_allow_html=True)
vowel_sounds = ["อา", "อี", "อือ", "อู", "ไอ", "อำ", "เอา"]
for sound in vowel_sounds:
    st.markdown(f"<p class='pronounce'>ออกเสียง <b>{sound}</b></p>", unsafe_allow_html=True)
    st.markdown(f"""
        <div class="big-mic-wrapper">
            <div class="big-mic-button" style="background-color:#009688;" onclick="document.getElementById('rec-{sound}').click()">
                <i class="fa fa-microphone"></i>
            </div>
        </div>
    """, unsafe_allow_html=True)
    audio_bytes = st.audio_input(f"บันทึกเสียง {sound}", key=f"rec-{sound}")
    if audio_bytes:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_bytes.read())
            st.session_state.vowel_paths.append(tmp.name)
        st.success(f"บันทึก {sound} สำเร็จ ✅")

st.markdown("--- หรือ อัปโหลดไฟล์เสียง ---")
uploaded_vowels = st.file_uploader("อัปโหลดสระ (7 ไฟล์)", type=["wav", "mp3", "m4a"], accept_multiple_files=True)
if uploaded_vowels:
    st.session_state.vowel_paths = []
    for file in uploaded_vowels[:7]:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(file.read())
            st.session_state.vowel_paths.append(tmp.name)

# =============================
# Step 2: Pataka
# =============================
st.markdown("<div class='card'><h2>2. พยางค์</h2></div>", unsafe_allow_html=True)
st.markdown(f"""
    <div class="big-mic-wrapper">
        <div class="big-mic-button" style="background-color:#3f51b5;" onclick="document.getElementById('rec-pataka').click()">
            <i class="fa fa-microphone"></i>
        </div>
    </div>
""", unsafe_allow_html=True)
pataka_bytes = st.audio_input("บันทึกเสียง พา-ทา-คา", key="rec-pataka")
if pataka_bytes:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(pataka_bytes.read())
        st.session_state.pataka_path = tmp.name
    st.success("บันทึกพยางค์สำเร็จ ✅")

uploaded_pataka = st.file_uploader("อัปโหลดไฟล์พยางค์", type=["wav", "mp3", "m4a"])
if uploaded_pataka:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_pataka.read())
        st.session_state.pataka_path = tmp.name

# =============================
# Step 3: Sentence
# =============================
st.markdown("<div class='card'><h2>3. ประโยค</h2></div>", unsafe_allow_html=True)
st.markdown(f"""
    <div class="big-mic-wrapper">
        <div class="big-mic-button" style="background-color:#9c27b0;" onclick="document.getElementById('rec-sentence').click()">
            <i class="fa fa-microphone"></i>
        </div>
    </div>
""", unsafe_allow_html=True)
sentence_bytes = st.audio_input("บันทึกการอ่านประโยค", key="rec-sentence")
if sentence_bytes:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(sentence_bytes.read())
        st.session_state.sentence_path = tmp.name
    st.success("บันทึกประโยคสำเร็จ ✅")

uploaded_sentence = st.file_uploader("อัปโหลดไฟล์ประโยค", type=["wav", "mp3", "m4a"])
if uploaded_sentence:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_sentence.read())
        st.session_state.sentence_path = tmp.name

# =============================
# Action Buttons
# =============================
col1, col2 = st.columns([1, 1])
with col1:
    predict_btn = st.button("วิเคราะห์", key="predict", type="primary")
with col2:
    if st.button("ลบข้อมูล", key="clear", type="secondary"):
        st.session_state.vowel_paths = []
        st.session_state.pataka_path = None
        st.session_state.sentence_path = None
        st.experimental_rerun()

# =============================
# Prediction Logic
# =============================
if predict_btn:
    if len(st.session_state.vowel_paths) == 7 and st.session_state.pataka_path and st.session_state.sentence_path:
        all_probs = predict_from_model(st.session_state.vowel_paths, st.session_state.pataka_path, st.session_state.sentence_path)
        percent = int(np.mean(all_probs) * 100)
        st.success(f"ความน่าจะเป็นพาร์กินสัน: {percent}%")
    else:
        st.warning("กรุณาอัดเสียงหรืออัปโหลดให้ครบ ⚠️")
