import streamlit as st
from pathlib import path
import google.generativeai as genai
st.set_page_config(page_title="AIHealthPro Chatbot", page_icon=":robot:")
st.title("AIHealthPro Chatbot")
st.image("https://www.cdc.gov/healthcommunication/ToolsTemplates/Photos/HealthPro_508.gif",width=200)
st.write("AIHealthPro is a chatbot that can help you with your health questions.")
