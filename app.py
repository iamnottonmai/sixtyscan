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
# Page Config & Font Styles
# =============================
st.set_page_config(page_title="SixtyScan", layout="centered")
st.markdown("""
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
            width: 100%;
            max-width: 300px;
            display: block;
            margin: 10px auto;
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
def audio_to_mel_tensor(file_path):
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
# Header
# =============================
st.markdown("<h1 class='title'>SixtyScan</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>ตรวจจับพาร์กินสันผ่านการวิเคราะห์เสียง</p>", unsafe_allow_html=True)

# =============================
# Vowel Recordings (7)
# =============================
st.markdown("""
<div class='card'>
    <h2>1. พยัญชนะ</h2>
    <p class='instructions'>กรุณาออกเสียงแต่ละพยางค์ 5-8 วินาทีอย่างชัดเจน โดยกดปุ่มบันทึกทีละไฟล์ หรืออัปโหลดไฟล์เสียงด้านล่าง</p>
</div>
""", unsafe_allow_html=True)

vowel_sounds = ["อา", "อี", "อือ", "อู", "ไอ", "อำ", "เอา"]
vowel_paths = []

for sound in vowel_sounds:
    st.markdown(f"<p class='pronounce'>ออกเสียง \"{sound}\"</p>", unsafe_allow_html=True)
    audio_bytes = st.audio_input(f"🎤 บันทึกเสียง {sound}")
    if audio_bytes:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_bytes.read())
            vowel_paths.append(tmp.name)
        st.success(f"บันทึกเสียง \"{sound}\" สำเร็จ", icon="✅")

uploaded_vowels = st.file_uploader("หรืออัปโหลดไฟล์เสียงพยัญชนะ (7 ไฟล์)", type=["wav", "mp3", "m4a"], accept_multiple_files=True)
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
    <p class='instructions'>กรุณาออกเสียงคำว่า "พา - ทา - คา" ให้จบภายใน 6 วินาที หรืออัปโหลดไฟล์เสียงด้านล่าง</p>
</div>
""", unsafe_allow_html=True)

st.markdown("<p class='pronounce'>ออกเสียง \"พา - ทา - คา\"</p>", unsafe_allow_html=True)

pataka_path = None
pataka_bytes = st.audio_input("🎤 บันทึกเสียงพยางค์")
if pataka_bytes:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(pataka_bytes.read())
        pataka_path = tmp.name
    st.success("บันทึกพยางค์สำเร็จ", icon="✅")

uploaded_pataka = st.file_uploader("หรืออัปโหลดไฟล์เสียงพยางค์", type=["wav", "mp3", "m4a"], accept_multiple_files=False)
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
    <p class='instructions'>กรุณาอ่านประโยคอย่างชัดเจน หรืออัปโหลดไฟล์เสียงด้านล่าง</p>
</div>
""", unsafe_allow_html=True)

st.markdown("<p class='pronounce'>อ่านประโยค \"วันนี้อากาศแจ่มใสนกร้องเสียงดังเป็นจังหวะ\"</p>", unsafe_allow_html=True)

sentence_path = None
sentence_bytes = st.audio_input("🎤 บันทึกการอ่านประโยค")
if sentence_bytes:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(sentence_bytes.read())
        sentence_path = tmp.name
    st.success("บันทึกประโยคสำเร็จ", icon="✅")

uploaded_sentence = st.file_uploader("หรืออัปโหลดไฟล์เสียงประโยค", type=["wav", "mp3", "m4a"], accept_multiple_files=False)
if uploaded_sentence and not sentence_path:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_sentence.read())
        sentence_path = tmp.name

# =============================
# Buttons Layout
# =============================
col1, col2 = st.columns([1, 1])
with col1:
    predict_btn = st.button("วิเคราะห์", key="predict", type="primary")
with col2:
    clear_btn = st.button("ลบข้อมูล", key="clear", type="secondary")

# =============================
# Prediction Logic
# =============================
if predict_btn:
    if len(vowel_paths) == 7 and pataka_path and sentence_path:
        all_probs = predict_from_model(vowel_paths, pataka_path, sentence_path)
        final_prob = np.mean(all_probs)
        percent = int(final_prob * 100)

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
        st.warning("กรุณาอัดเสียงหรืออัปโหลดให้ครบทั้ง 7 พยัญชนะ พยางค์ และประโยค", icon="⚠️")

# =============================
# Clear Button Logic
# =============================
if clear_btn:
    st.rerun()
