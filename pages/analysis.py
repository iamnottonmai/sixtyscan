import streamlit as st

st.set_page_config(page_title="Analysis Test")

st.title("Analysis Page Works!")
st.write("This is your analysis page.")

if st.button("Back to Home"):
    st.switch_page("app.py")
