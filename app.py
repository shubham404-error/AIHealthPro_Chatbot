import streamlit as st
from pathlib import Path
import google.generativeai as genai
st.set_page_config(page_title="AIHealthPro Chatbot", page_icon=":robot:")
st.title("AIHealthPro Chatbot")
st.image("logo.png",width=200)
st.write("AIHealthPro is a chatbot that can help you with your health questions.")
