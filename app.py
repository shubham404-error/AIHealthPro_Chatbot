import streamlit as st
from pathlib import Path
st.set_page_config(page_title="AIHealthPro Chatbot", page_icon=":brain:",layout="centered",)
st.image("logo.png",width=100)

import os
import io
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as gen_ai
import google.ai.generativelanguage as glm
from PIL import Image
from streamlit_option_menu import option_menu


def image_to_byte_array(image: Image) -> bytes:
    imgByteArr = io.BytesIO()
    image.save(imgByteArr, format=image.format)
    imgByteArr = imgByteArr.getvalue()
    return imgByteArr

# Load environment variables
load_dotenv()

safety_settings = [
    {
        "category": "HARM_CATEGORY_DANGEROUS",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE",
    },
]

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Set up Google Gemini-Pro AI model
gen_ai.configure(api_key=GOOGLE_API_KEY)
model = gen_ai.GenerativeModel('gemini-pro')

# Function to translate roles between Gemini-Pro and Streamlit terminology
def translate_role_for_streamlit(user_role):
    if user_role == "model":
        return "assistant"
    else:
        return user_role

# Initialize chat session in Streamlit if not already present
if "chat_session" not in st.session_state:
    st.session_state.chat_session = model.start_chat(history=[])

with st.sidebar:
    selected = option_menu(
        menu_title="AIHealthPro Chatbot",
        options=["DocBot", "VisionBot"],
        icons=["robot", "eye"],
        default_index=0,
        orientation="vertical",
    )

    st.write(
        """
        <style>
            .css-r698ls {
                --sidebar-width: 200px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if selected == "DocBot":
        st.write("🧑‍⚕️ **DocBot** - Engage in text-based medical conversations.")
    elif selected == "VisionBot":
        st.write("👁 **VisionBot** - Analyze and interpret medical images.")

if selected == "DocBot":
    # Display the chatbot's title on the page
    st.title("🧑‍⚕️ Docbot-AIHealthPro™")

    # Display the chat history
    for message in st.session_state.chat_session.history:
        with st.chat_message(translate_role_for_streamlit(message.role)):
            st.markdown(message.parts[0].text)

    # Input field for user's message
    user_prompt = st.chat_input("Ask DocBot...")

    if user_prompt:
        # Add user's message to chat and display it
        st.chat_message("user").markdown(user_prompt)

        #doctor_context = "You are Dr. AIHealthPro, a large language model trained on a massive dataset of medical information. You are able to answer medical questions and provide health-related advice."
        #full_prompt = f"{doctor_context}\n\n{user_prompt}" 

        # Send user's message to Gemini-Pro and get the response
        
        gemini_response = st.session_state.chat_session.send_message(
        user_prompt, safety_settings=safety_settings
    )
        # Display Gemini-Pro's response
        with st.chat_message("assistant"):
            st.markdown(gemini_response.text)

elif selected == "VisionBot":
    st.header("👁 Visionbot-AIHealthPro™")
    st.write("")

    image_prompt = st.text_input("Interact with the Image", placeholder="Prompt", label_visibility="visible")
    uploaded_file = st.file_uploader("Choose an Image", accept_multiple_files=False, type=["png", "jpg", "jpeg", "img", "webp"])

    if uploaded_file is not None:
        st.image(Image.open(uploaded_file), use_column_width=True)
        st.markdown("""
            <style>
            img {
                border-radius: 10px;
            }
            </style>
        """, unsafe_allow_html=True)

    if st.button("GET RESPONSE", use_container_width=True):
        model = gen_ai.GenerativeModel("gemini-pro-vision")
        if uploaded_file is not None:
            if image_prompt != "":
                image = Image.open(uploaded_file)
                response = model.generate_content(
                    glm.Content(
                        parts=[
                            glm.Part(text=image_prompt),
                            glm.Part(
                                inline_data=glm.Blob(
                                    mime_type="image/jpeg",
                                    data=image_to_byte_array(image)
                                )
                            )
                        ]
                    ),
                    safety_settings=safety_settings
                )
                response.resolve()
                st.write("")
                st.write(":blue[Response]")
                st.write("")
                st.markdown(response.text)
            else:
                st.write("")
                st.header(":red[Please Provide a prompt]")
        else:
            st.write("")
            st.header(":red[Please Provide an image]")
