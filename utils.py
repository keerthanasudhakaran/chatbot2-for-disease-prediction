from openai import OpenAI
import os
#from dotenv import load_dotenv
import base64
import streamlit as st
#import pyaudio
#import wave
import speech_recognition as sr
from translate import Translator
from huggingface_hub import hf_api
from huggingface_hub import login
from huggingface_hub import InferenceClient

hf_token=st.secrets["api_token"]
#openai_api_key=st.secrets["openai_api_key"]
#login(api_token)
print( "The token is : ", hf_token);
client=InferenceClient( provider = "hf-inference", api_key = hf_token , )
completion = client.chat.completions.create( model = "meta-llama/Llama-2-7b-chat-hf",
                                            messages= [
                                               {
                                                    "role" : "user",
                                                   "content" : "What is the capital of France?"
                                                }
                                            ],
                                            max_tokens = 500, )
print( completion.choices[0].message)


#hf_api.list_models(token="your_hugging_face_api_token")

#login(token=hf_token)
#hf_token= os.getenv("api_token")
#print(f"Token loaded: {hf_token}")  # This will print None if the token is not loaded properly

#if hf_token:
    # Authenticate with Hugging Face API using the token
#login(token=hf_token)
#else:
   #raise ValueError("Hugging Face token is missing. Please set the 'HUGGINGFACE_TOKEN' environment variable.")

import nltk
nltk.download('wordnet')

from gtts import gTTS

from streamlit_chat import message
from langchain.chains import RetrievalQA
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import CTransformers
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory


# Load the PDF files from the path
loader = DirectoryLoader('data/', glob="*.pdf", loader_cls=PyPDFLoader)
documents = loader.load()

# Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50, separators= ["---"])
text_chunks = text_splitter.split_documents(documents)


# Create embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': "cpu"})

print("embedding : ",embeddings)
# Vector store
vector_store = FAISS.from_documents(text_chunks, embeddings)

## Default LLaMA-2 prompt style
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

def get_prompt(instruction, new_system_prompt=DEFAULT_SYSTEM_PROMPT ):
    SYSTEM_PROMPT = B_SYS + new_system_prompt + E_SYS
    prompt_template =  B_INST + SYSTEM_PROMPT + instruction + E_INST
    return prompt_template


sys_prompt = """
You are a health assistant. You are supposed to predict the most relevant condition or disease or state or disorder in the given context that suits the given symptoms. 
Give only the most accurately matching disease or condition or state or disorder given in the context. Do not predict anything out of the context return only what is given in the context.
"""
instruction = """CONTEXT:/n/n {context}/n

Question: {question}"""
get_prompt(instruction, sys_prompt)

from langchain.prompts import PromptTemplate
prompt_template = get_prompt(instruction, sys_prompt)

llama_prompt = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

chain_type_kwargs = {"prompt": llama_prompt}

# Create LLM
#llm = CTransformers(model="llama-2-7b-chat.ggmlv3.q4_0.bin", model_type="llama", config={'max_new_tokens': 128, 'temperature': 0.01})
#llm = CTransformers(model="meta-Ilama/Llama-2-7b-chat-hf", model_type="llama", config={'max_new_tokens': 128, 'temperature': 0.01})
#llm = CTransformers(model="Llama-2-7b-chat-hf", model_type="llama", config={'max_new_tokens': 128, 'temperature': 0.01})

# Instantiate the ConversationalRetrievalChain

chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
    chain_type="stuff",
     chain_type_kwargs=chain_type_kwargs
)

def get_answer(messages):
    #system_message = [{"role": "system", "content": "You are an helpful AI chatbot, that answers questions asked by User."}]
    #comb_messages = system_message + messages
    print(messages[-1]['content'])
    response = chain({"query": messages[-1]['content']})
    return response["result"]
    #return 'Based on the symptoms provided in the context, the most likely disease or condition is epilepsy. The patient is experiencing a combination of seizure, hypometabolism, aura, muscle twitch, drowsiness, tremor, unresponsiveness, hemiplegia, myoclonus, and gurgle, which are all common symptoms of epilepsy. Additionally, the patient is wheelchair-bound, which could be due to the severity of their condition or as a result of a seizure-related injury.'
    #return 'Based on the symptoms provided in the context, the most accurately matching disease is coronary heart disease.'

def speech_to_text(audio_file):
    # Initialize the recognizer
    recognizer = sr.Recognizer()

    # Load the audio file
    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)

    try:
        # Recognize speech using the Google Web Speech API
        tamil_text = recognizer.recognize_google(audio_data, language="ta-IN")
        #return tamil_text
    except sr.UnknownValueError:
        print("Google Web Speech API could not understand the audio")
    except sr.RequestError as e:
        print(f"Could not request results from Google Web Speech API; {e}")

    translator = Translator(to_lang="en", from_lang="ta")
    try:
        transcript = translator.translate(tamil_text)
        return transcript
    except Exception as e:
        print("Translation Error:", e)
        return None



def text_to_speech(input_text):
    response = gTTS(text=input_text, lang='en') 
    webm_file_path = "temp_audio_play.mp3"
    response.save(webm_file_path)
    return webm_file_path

def autoplay_audio(file_path: str):
    with open(file_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode("utf-8")
    md = f"""
    <audio autoplay>
    <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
    </audio>
    """
    st.markdown(md, unsafe_allow_html=True)
