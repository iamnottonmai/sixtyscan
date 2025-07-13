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

# =============================
# Page Config & Font Styles
# =============================
st.set_page_config(page_title="SixtyScan", layout="centered")
st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Noto+Sans+Thai&display=swap" rel="stylesheet">
    <style>
        html, body {
            background-color: #f2f4f8;
            font-family: 'Noto Sans Thai', sans-serif;
        }
        h1.title {
            text-align: center;
            font-size: 84px;
            color: #4A148C;
            margin-bottom: 20px;
            font-weight: bold;
        }
        p.subtitle {
            text-align: center;
            font-size: 42px;
            color: #333;
            margin-bottom: 56px;
        }
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
            font-weight: bold;
        }
        .instructions {
            font-size: 34px !important;
            color: #333;
            margin-bottom: 24px;
            font-weight: bold;
        }
        .pronounce {
            font-size: 36px !important;
            color: #000;
            font-weight: bold;
            margin-top: 0;
            margin-bottom: 24px;
        }
        .predict-btn, .clear-btn {
            font-size: 38px !important;
            padding: 1.4em 2.7em;
            border-radius: 14px;
            font-weight: bold;
        }
        .predict-btn {
            background-color: #009688;
            color: white;
        }
        .clear-btn {
            background-color: #cfd8dc;
            color: black;
        }
    </style>
""", unsafe_allow_html=True)

# =============================
# Session State Initialization
# =============================
vowel_sounds = ["อา", "อี", "อือ", "อู", "ไอ", "อำ", "เอา"]

if "page" not in st.session_state:
    st.session_state.page = "main"
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "question_answers" not in st.session_state:
    st.session_state.question_answers = [None] * 12
if "final_result" not in st.session_state:
    st.session_state.final_result = None

# Initialize audio keys
for sound in vowel_sounds:
    st.session_state.setdefault(f"vowel_{sound}", None)
st.session_state.setdefault("pataka_audio", None)
st.session_state.setdefault("sentence_audio", None)
st.session_state.setdefault("uploaded_vowels", None)
st.session_state.setdefault("uploaded_pataka", None)
st.session_state.setdefault("uploaded_sentence", None)

# =============================
# Load Model
# =============================
@st.cache_resource
def load_model():
    model_path = "best_resnet18.pth"
    if not os.path.exists(model_path):
        gdown.download("https://drive.google.com/uc?id=1_oHE9B-2PgSqpTQCC9HrG7yO0rsnZtqs", model_path, quiet=False)
    model = ResNet18Classifier()
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model

model = load_model()

# =============================
# Audio Preprocessing
# =============================
def audio_to_mel_tensor(file_path):
    y, sr = librosa.load(file_path, sr=22050)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    fig = plt.figure(figsize=(2.24, 2.24), dpi=100)
    plt.axis('off')
    librosa.display.specshow(mel_db, sr=sr)
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
# Header
# =============================
st.markdown("<h1 class='title'>SixtyScan</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>ตรวจจับพาร์กินสันผ่านการวิเคราะห์เสียง</p>", unsafe_allow_html=True)

# =============================
# Vowel Recordings
# =============================
st.markdown("""
<div class='card'>
    <h2>1. พยัญชนะ</h2>
    <p class='instructions'>กรุณาออกเสียงแต่ละพยางค์ 5-8 วินาทีอย่างชัดเจน โดยกดปุ่มบันทึกทีละไฟล์ หรืออัปโหลดไฟล์เสียงด้านล่าง</p>
</div>
""", unsafe_allow_html=True)

vowel_paths = []

for sound in vowel_sounds:
    st.markdown(f"<p class='pronounce'>ออกเสียง \"{sound}\"</p>", unsafe_allow_html=True)
    audio_bytes = st.audio_input(f"🎤 บันทึกเสียง {sound}", key=f"vowel_{sound}")
    if audio_bytes:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_bytes.read())
            vowel_paths.append(tmp.name)
        st.success(f"บันทึกเสียง \"{sound}\" สำเร็จ", icon="✅")

if not vowel_paths:
    uploaded_vowels = st.file_uploader("หรืออัปโหลดไฟล์เสียงพยัญชนะ (7 ไฟล์)", type=["wav", "mp3", "m4a"], accept_multiple_files=True, key="uploaded_vowels")
    if uploaded_vowels:
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
    <p class='instructions'>กรุณาออกเสียงคำว่า "พา - ทา - คา" ให้จบภายใน 6 วินาที หรืออัปโหลดไฟล์เสียงด้านล่าง</p>
</div>
""", unsafe_allow_html=True)

st.markdown("<p class='pronounce'>ออกเสียง \"พา - ทา - คา\"</p>", unsafe_allow_html=True)

pataka_path = None
pataka_bytes = st.audio_input("🎤 บันทึกเสียงพยางค์", key="pataka_audio")
if pataka_bytes:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(pataka_bytes.read())
        pataka_path = tmp.name
    st.success("บันทึกพยางค์สำเร็จ", icon="✅")

if not pataka_path:
    uploaded_pataka = st.file_uploader("หรืออัปโหลดไฟล์เสียงพยางค์", type=["wav", "mp3", "m4a"], accept_multiple_files=False, key="uploaded_pataka")
    if uploaded_pataka:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded_pataka.read())
            pataka_path = tmp.name

# =============================
# Sentence Recording
# =============================
st.markdown("""
<div class='card'>
    <h2>3. ประโยค</h2>
    <p class='instructions'>กรุณาอ่านประโยคอย่างชัดเจน หรืออัปโหลดไฟล์เสียงด้านล่าง</p>
</div>
""", unsafe_allow_html=True)

st.markdown("<p class='pronounce'>อ่านประโยค \"วันนี้อากาศแจ่มใสนกร้องเสียงดังเป็นจังหวะ\"</p>", unsafe_allow_html=True)

sentence_path = None
sentence_bytes = st.audio_input("🎤 บันทึกการอ่านประโยค", key="sentence_audio")
if sentence_bytes:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(sentence_bytes.read())
        sentence_path = tmp.name
    st.success("บันทึกประโยคสำเร็จ", icon="✅")

if not sentence_path:
    uploaded_sentence = st.file_uploader("หรืออัปโหลดไฟล์เสียงประโยค", type=["wav", "mp3", "m4a"], accept_multiple_files=False, key="uploaded_sentence")
    if uploaded_sentence:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded_sentence.read())
            sentence_path = tmp.name

# =============================
# Buttons
# =============================
col1, col2 = st.columns([1, 1])
with col1:
    predict_btn = st.button("🔍 วิเคราะห์", type="primary")
with col2:
    clear_btn = st.button("ล้างข้อมูล", type="secondary")

# =============================
# Prediction Result Display
# =============================
if predict_btn:
    if len(vowel_paths) == 7 and pataka_path and sentence_path:
        all_probs = predict_from_model(vowel_paths, pataka_path, sentence_path)
        final_prob = np.mean(all_probs)
        percent = int(final_prob * 100)

        # Result rendering omitted for brevity (same as your original)

    else:
        st.warning("กรุณาอัดเสียงหรืออัปโหลดให้ครบทั้ง 7 พยัญชนะ พยางค์ และประโยค", icon="⚠️")

# =============================
# Clear Button Logic
# =============================
if clear_btn:
    for sound in vowel_sounds:
        st.session_state[f"vowel_{sound}"] = None
    for key in ["pataka_audio", "sentence_audio", "uploaded_vowels", "uploaded_pataka", "uploaded_sentence"]:
        st.session_state[key] = None
    st.session_state.uploaded_files = []
    st.session_state.question_answers = [None] * 12
    st.session_state.final_result = None
    st.rerun()
