import streamlit as st
import numpy as np
from pathlib import Path
from predict import predict_from_model  # your model logic
from utils import handle_uploaded_audio  # your helper function
from PIL import Image

st.set_page_config(page_title="SixtyScan", layout="wide")

# Load custom style
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# =============================
# Logo + Title + Subtitle
# =============================
st.markdown("<img src='logo.png' class='logo'>", unsafe_allow_html=True)
st.markdown("<div class='title'>SixtyScan</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>ตรวจโรคพาร์กินสันจากเสียง</div>", unsafe_allow_html=True)

# =============================
# Audio Upload Sections
# =============================
# Create placeholders for the 7 vowel files, 1 pataka, and 1 sentence
vowel_sounds = ["อะ", "อิ", "อึ", "อุ", "เอ", "แอ", "โอ"]
vowel_paths = []
pataka_path = None
sentence_path = None

# Vowel Section
with st.container():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h2>สระ</h2>", unsafe_allow_html=True)
    for vowel in vowel_sounds:
        uploaded = st.file_uploader(f"สระ: {vowel}", type=["wav", "mp3"], key=vowel)
        if uploaded:
            path = handle_uploaded_audio(uploaded, vowel)
            vowel_paths.append(path)
        else:
            vowel_paths.append(None)
    st.markdown("</div>", unsafe_allow_html=True)

# Pataka Section
with st.container():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h2>พยางค์</h2>", unsafe_allow_html=True)
    pataka = st.file_uploader("พูดคำว่า 'ปะ-ตะ-กะ'", type=["wav", "mp3"], key="pataka")
    if pataka:
        pataka_path = handle_uploaded_audio(pataka, "pataka")
    st.markdown("</div>", unsafe_allow_html=True)

# Sentence Section
with st.container():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h2>ประโยค</h2>", unsafe_allow_html=True)
    sentence = st.file_uploader("พูดประโยค “วันนี้วันพระ”", type=["wav", "mp3"], key="sentence")
    if sentence:
        sentence_path = handle_uploaded_audio(sentence, "sentence")
    st.markdown("</div>", unsafe_allow_html=True)

# =============================
# Buttons Layout
# =============================
col1, col2 = st.columns([1, 0.18])
with col1:
    button_col1, button_col2 = st.columns([1, 1])
    with button_col1:
        predict_btn = st.button("วิเคราะห์", key="predict", type="primary")
    with button_col2:
        loading_placeholder = st.empty()
with col2:
    st.markdown("""<div style="display: flex; justify-content: flex-end;">""", unsafe_allow_html=True)
    clear_btn = st.button("ลบข้อมูล", key="clear", type="secondary")
    st.markdown("</div>", unsafe_allow_html=True)

# =============================
# Prediction Logic
# =============================
if predict_btn:
    if len(vowel_paths) == 7 and pataka_path and sentence_path:
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

        loading_placeholder.empty()

        if percent <= 50:
            level = "ระดับต่ำ (Low)"
            label = "Non Parkinson"
            diagnosis = "ไม่เป็นพาร์กินสัน"
            box_color = "#e6f9e6"
            advice = """
            <ul style='font-size:28px;'>
                <li>ถ้าไม่มีอาการ: ควรตรวจปีละครั้ง (ไม่บังคับ)</li>
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
            <div style='background-color:{box_color}; padding: 32px; border-radius: 14px; font-size: 30px; color: #000000; font-family: "Noto Sans Thai", sans-serif;'>
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
# Clear Button Logic
# =============================
if clear_btn:
    st.session_state.clear()
    st.markdown("""
        <script>window.location.reload(true);</script>
        <meta http-equiv="refresh" content="0">
    """, unsafe_allow_html=True)
