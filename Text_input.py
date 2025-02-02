import asyncio
from dotenv import load_dotenv
import requests
import time
import os
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain

from deepgram import Deepgram

load_dotenv()

app = FastAPI()

# Set up CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class LanguageModelProcessor:
    def __init__(self):
        self.llm = ChatGroq(temperature=0, model_name="llama-3.1-8b-instant", groq_api_key=os.getenv("GROQ_API_KEY"))
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        
        with open('system_prompt.txt', 'r') as file:
            system_prompt = file.read().strip()
        
        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{text}")
        ])
        
        self.conversation = LLMChain(llm=self.llm, prompt=self.prompt, memory=self.memory)
    
    def process(self, text: str) -> str:
        self.memory.chat_memory.add_user_message(text)
        response = self.conversation.invoke({"text": text})
        llm_response = response.get("text", "")
        self.memory.chat_memory.add_ai_message(llm_response)
        return llm_response

class AvatarGeneration:
    def __init__(self):
        self.tts_api_key = os.getenv("ELEVENLABS_API_KEY")
        self.simli_api_key = os.getenv("SIMLI_API_KEY")
        self.url = "https://api.simli.ai/textToVideoStream"
    
    def speak(self, text: str) -> str:
        if not self.tts_api_key or not self.simli_api_key:
            raise ValueError("Missing API keys for TTS or Simli.")
        
        payload = {
            "ttsAPIKey": self.tts_api_key,
            "simliAPIKey": self.simli_api_key,
            "faceId": "743a34ba-435e-4c38-ac2b-c8b91d58a07e",
            "requestBody": {
                "audioProvider": "ElevenLabs",
                "text": text,
                "voiceName": "pMsXgVXv3BLzUgSXRplE",
                "model_id": "eleven_turbo_v2",
                "voice_settings": {"stability": 0.1, "similarity_boost": 0.3, "style": 0.2}
            }
        }
        
        response = requests.post(self.url, json=payload, headers={"Content-Type": "application/json"})
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=f"AvatarGeneration API error: {response.text}")
        return response.json().get("mp4_url", "")

async def transcribe_audio(audio_file: UploadFile) -> str:
    try:
        dg_client = Deepgram(os.getenv("DEEPGRAM_API_KEY"))
        audio_data = await audio_file.read()
        response = await dg_client.transcription.prerecorded(
            {"buffer": audio_data, "mimetype": audio_file.content_type},
            {"punctuate": True, "language": "en-US"}
        )
        return response['results']['channels'][0]['alternatives'][0]['transcript']
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription error: {str(e)}")

class ConversationManager:
    def __init__(self):
        self.llm_processor = LanguageModelProcessor()
        self.avatar_generator = AvatarGeneration()
    
    async def process_voice(self, audio_file: UploadFile = None, text: str = None):
        if text:
            transcription = text
        elif audio_file:
            transcription = await transcribe_audio(audio_file)
        else:
            raise HTTPException(status_code=400, detail="Either text or audio_file must be provided.")
        
        llm_response = self.llm_processor.process(transcription)
        mp4_url = self.avatar_generator.speak(llm_response)
        return {"User": transcription, "llm_response": llm_response, "avatar_response": mp4_url}

conversation_manager = ConversationManager()

@app.post("/process_voice/")
async def process_voice(audio_file: UploadFile = File(None), text: str = Form(None)):
    return await conversation_manager.process_voice(audio_file, text)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
