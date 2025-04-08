import streamlit as st
import pyaudio
import wave
import speech_recognition as sr
from translate import Translator
import os

# Function to record audio
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

# Function to transcribe and translate audio
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

# Streamlit UI
st.title("Simple Audio Chatbot")

if st.button("🎤"):
    audio_file = record_audio()
    st.audio(audio_file, format='audio/wav', start_time=0)
    translated_text = transcribe_and_translate_audio(audio_file)
    st.write("Translated text:", translated_text)
    os.remove(audio_file)  # Remove the audio file after processing

st.write("Thank you")
