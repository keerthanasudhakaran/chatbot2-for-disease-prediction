import streamlit as st
import os
from audio_recorder_streamlit import audio_recorder
from streamlit_float import *
#from utils import text_to_speech, autoplay_audio, speech_to_text ,get_answer
from utils import  autoplay_audio ,get_answer

from CNN import *
import time

# Float feature initialization
float_init()


def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hi! How may I assist you today?"}
        ]



initialize_session_state()

st.title("Healthcare Chatbot 🤖")

# Create footer containers for the microphone and picture upload
footer_container_left = st.container()
footer_container_right = st.container()

with footer_container_left:
    audio_bytes = audio_recorder()

key='file_uploader'


with footer_container_right:
    with st.form("file_form",clear_on_submit=True):
        uploaded_picture = st.file_uploader("Upload Picture Of CT Scan", type=["png", "jpg"])
        submitted=st.form_submit_button('Submit')
    st.markdown('<style>div.Widget.row-widget.stFileUploader > div{float:right !important;}</style>', unsafe_allow_html=True)

# Add text input for user messages
user_input = st.text_input("Type your symptoms here:")

if st.button("Predict"):
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

user_input=''

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if audio_bytes:
    # Write the audio bytes to a file
    with st.spinner("Transcribing..."):
        webm_file_path = "temp_audio.mp3"
        with open(webm_file_path, "wb") as f:
            f.write(audio_bytes)
        audio_bytes=None
        transcript = speech_to_text(webm_file_path)
        if transcript:
            st.session_state.messages.append({"role": "user", "content": transcript})
            with st.chat_message("user"):
                st.write(transcript)
            os.remove(webm_file_path)
        transcript=None

if submitted and uploaded_picture is not None:
    # Process the uploaded picture (you need to replace this with your image processing code)
    processed_result = predict(uploaded_picture)
    #processed_result = 'tumor'
    st.session_state.messages.append({"role": "user", "content": uploaded_picture.name})
    with st.chat_message("user"):
        st.write(uploaded_picture.name)
    with st.chat_message("assistant"):
        #st.image(uploaded_picture)
        #st.write('Image Uploaded..')
        if processed_result=='normal':
            resp='No worries. You are absolutely normal.'
        else:
            resp='The uploaded image of CT scan suggests that you have '+processed_result
        st.write(resp)
    st.session_state.messages.append({"role": "assistant", "content": resp})
    #with st.chat_message("assistant"):    
    #    st.write(resp)

    uploaded_picture=None
    
    

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking🤔..."):
            final_response = get_answer(st.session_state.messages)
        
        with st.spinner("Generating audio response..."):    
            audio_file = text_to_speech(final_response)
            autoplay_audio(audio_file)
        
        st.write(final_response)
        st.session_state.messages.append({"role": "assistant", "content": final_response})
        os.remove(audio_file)

# Float the footer containers and provide CSS to target them
footer_container_left.float("bottom: 0rem; left: 1rem; position: fixed;")
footer_container_right.float("bottom: 0rem; right: 1rem; position: fixed;")
