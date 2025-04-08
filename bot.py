import streamlit as st
from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import CTransformers
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
import pyaudio
import wave
import speech_recognition as sr
from translate import Translator
import os

# Load the PDF files from the path
loader = DirectoryLoader('data/', glob="*.pdf", loader_cls=PyPDFLoader)
documents = loader.load()

# Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
text_chunks = text_splitter.split_documents(documents)

# Create embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': "cpu"})

# Vector store
vector_store = FAISS.from_documents(text_chunks, embeddings)

# Create LLM
llm = CTransformers(model="llama-2-7b-chat.ggmlv3.q4_0.bin", model_type="llama", config={'max_new_tokens': 128, 'temperature': 0.01})

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Instantiate the ConversationalRetrievalChain
chain = ConversationalRetrievalChain.from_llm(llm=llm, chain_type='stuff',
                                              retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
                                              memory=memory)

st.title("Disease Prediction ChatBot 🧑🏽‍⚕️")

def conversation_chat(query):
    print(f"Received query: {query}")  # Console logging for debugging
    result = chain({"question": query, "chat_history": st.session_state['history']})
    st.session_state['history'].append((query, result["answer"]))
    return result["answer"]


def initialize_session_state():
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello! Ask me anything about  🤗"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey! 👋"]

def display_chat_history():
    reply_container = st.container()
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Question:", placeholder="Ask about your Mental Health", key='input')
            if st.button("🎤"):  # Mic button
                audio_file = record_audio()
                st.audio(audio_file, format='audio/wav', start_time=0)
                translated_text = transcribe_and_translate_audio(audio_file)
                st.write("Translated text:", translated_text)
                os.remove(audio_file)  # Remove the audio file after processing
                st.stop()  # Stop the execution after processing the audio
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            output = conversation_chat(user_input)

            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with reply_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs")
                message(st.session_state["generated"][i], key=str(i), avatar_style="fun-emoji")

def record_audio(file_name='output.wav', duration=8, sample_rate=44100, channels=1, chunk=1024):
    audio = pyaudio.PyAudio()
    stream = audio.open(format=pyaudio.paInt16,
                        channels=channels,
                        rate=sample_rate,
                        input=True,
                        frames_per_buffer=chunk)
    
    st.write("Recording...")
    frames = []
    for i in range(0, int(sample_rate / chunk * duration)):
        data = stream.read(chunk)
        frames.append(data)
    st.write("Recording complete.")
    
    stream.stop_stream()
    stream.close()
    audio.terminate()
    
    wf = wave.open(file_name, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
    wf.setframerate(sample_rate)
    wf.writeframes(b''.join(frames))
    wf.close()
    return file_name

def transcribe_and_translate_audio(audio_file):
    recognizer = sr.Recognizer()
    
    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data, language="ta-IN")  # Assuming input is in Tamil
        translated_text = translate_to_english(text)
    return translated_text

def translate_to_english(tamil_text):
    translator = Translator(to_lang="en", from_lang="ta")
    try:
        translation = translator.translate(tamil_text)
        return translation
    except Exception as e:
        print("Translation Error:", e)
        return None

# Initialize session state
initialize_session_state()

# Display chat history
display_chat_history()

