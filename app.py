import streamlit as st
import requests
import tempfile
import os
import sounddevice as sd
import soundfile as sf
import numpy as np
from io import BytesIO

# ------------------------------
# Page Config
# ------------------------------
st.set_page_config(page_title="AI Voice Studio", layout="centered")
st.title("🎙️ AI Voice Studio")
st.markdown("Convert text to speech, speech to text, and even clone voices!")

# ------------------------------
# Sidebar Settings
# ------------------------------
st.sidebar.header("🔑 API Settings")
hf_api_key = st.sidebar.text_input("Hugging Face API Key", type="password")
elevenlabs_api_key = st.sidebar.text_input("ElevenLabs API Key (Optional)", type="password")

# ------------------------------
# Function - Hugging Face TTS
# ------------------------------
def text_to_speech_hf(text, model="suno/bark"):
    if not hf_api_key:
        st.error("Please enter your Hugging Face API key in the sidebar.")
        return None
    headers = {"Authorization": f"Bearer {hf_api_key}"}
    payload = {"inputs": text}
    response = requests.post(
        f"https://api-inference.huggingface.co/models/{model}",
        headers=headers,
        json=payload
    )
    if response.status_code == 200:
        return response.content
    else:
        st.error(f"Error: {response.status_code} - {response.text}")
        return None

# ------------------------------
# Function - Hugging Face STT
# ------------------------------
def speech_to_text_hf(audio_file, model="openai/whisper-small"):
    if not hf_api_key:
        st.error("Please enter your Hugging Face API key in the sidebar.")
        return None
    headers = {"Authorization": f"Bearer {hf_api_key}"}
    with open(audio_file, "rb") as f:
        data = f.read()
    response = requests.post(
        f"https://api-inference.huggingface.co/models/{model}",
        headers=headers,
        data=data
    )
    if response.status_code == 200:
        return response.json().get("text", "")
    else:
        st.error(f"Error: {response.status_code} - {response.text}")
        return None

# ------------------------------
# Tabs
# ------------------------------
tab1, tab2, tab3 = st.tabs(["📝 Text → Audio", "🎤 Audio → Text", "📼 Record Audio"])

# ------------------------------
# Tab 1 - Text to Audio
# ------------------------------
with tab1:
    st.subheader("Text to Speech")
    text_input = st.text_area("Enter text (Hindi or English):")
    if st.button("Convert to Speech"):
        if text_input.strip():
            audio_data = text_to_speech_hf(text_input)
            if audio_data:
                st.audio(audio_data, format="audio/wav")
                st.download_button("Download Audio", audio_data, file_name="tts.wav")
        else:
            st.warning("Please enter some text.")

# ------------------------------
# Tab 2 - Audio to Text
# ------------------------------
with tab2:
    st.subheader("Speech to Text")
    uploaded_audio = st.file_uploader("Upload audio file", type=["wav", "mp3", "m4a"])
    if st.button("Transcribe Audio"):
        if uploaded_audio:
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(uploaded_audio.read())
                tmp_file_path = tmp_file.name
            transcript = speech_to_text_hf(tmp_file_path)
            if transcript:
                st.success("Transcription:")
                st.write(transcript)
        else:
            st.warning("Please upload an audio file.")

# ------------------------------
# Tab 3 - Record Audio
# ------------------------------
with tab3:
    st.subheader("Record Audio")
    duration = st.slider("Recording duration (seconds)", 3, 15, 5)
    if st.button("Start Recording"):
        st.info("Recording... Speak now!")
        fs = 44100
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype="float32")
        sd.wait()
        st.success("Recording finished!")
        temp_audio_path = tempfile.mktemp(".wav")
        sf.write(temp_audio_path, recording, fs)
        st.audio(temp_audio_path)
        with open(temp_audio_path, "rb") as f:
            st.download_button("Download Recording", f, file_name="recording.wav")
