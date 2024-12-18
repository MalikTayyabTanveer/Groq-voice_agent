# require deepgram-sdk==2.12.0
import asyncio
from dotenv import load_dotenv
import requests
import shutil
import subprocess
import requests
import time
import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain

from deepgram import Deepgram

load_dotenv()

app = FastAPI()

class LanguageModelProcessor:
    def __init__(self):
        self.llm = ChatGroq(temperature=0, model_name="llama3-8b-8192", groq_api_key=os.getenv("GROQ_API_KEY"))
        # self.llm = ChatOpenAI(temperature=0, model_name="gpt-4-0125-preview", openai_api_key=os.getenv("OPENAI_API_KEY"))
        # self.llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-0125", openai_api_key=os.getenv("OPENAI_API_KEY"))

        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        # Load the system prompt from a file
        with open('system_prompt.txt', 'r') as file:
            system_prompt = file.read().strip()
        
        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{text}")
        ])

        self.conversation = LLMChain(
            llm=self.llm,
            prompt=self.prompt,
            memory=self.memory
        )

    def process(self, text):
        self.memory.chat_memory.add_user_message(text)  # Add user message to memory

        start_time = time.time()

        # Go get the response from the LLM
        response = self.conversation.invoke({"text": text})
        end_time = time.time()

        self.memory.chat_memory.add_ai_message(response['text'])  # Add AI response to memory

        elapsed_time = int((end_time - start_time) * 1000)
        print(f"LLM ({elapsed_time}ms): {response['text']}")
        return response['text']


class AvatarGeneration:
    def __init__(self):
        self.tts_api_key = os.getenv("ELEVENLABS_API_KEY")  # Get from environment
        self.simli_api_key = os.getenv("SIMLI_API_KEY")  # Get from environment
        self.url = "https://api.simli.ai/textToVideoStream"

    def speak(self, text: str):
        # Validate API keys
        if not self.tts_api_key or not self.simli_api_key:
            raise ValueError("API keys for TTS or Simli are missing in environment variables.")

        # Prepare payload
        payload = {
            "ttsAPIKey": self.tts_api_key,
            "simliAPIKey": self.simli_api_key,
            "faceId": "743a34ba-435e-4c38-ac2b-c8b91d58a07e",  # Placeholder faceId, adjust as needed
            "requestBody": {
                "audioProvider": "ElevenLabs",
                "text": text,
                "voiceName": "pMsXgVXv3BLzUgSXRplE",
                "model_id": "eleven_turbo_v2",
                "voice_settings": {
                    "stability": 0.1,
                    "similarity_boost": 0.3,
                    "style": 0.2
                }
            }
        }

        # Prepare headers
        headers = {"Content-Type": "application/json"}

        try:
            # Make the POST request
            response = requests.post(self.url, json=payload, headers=headers)

            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"AvatarGeneration API error: {response.text}"
                )

            # Return the response content
            return response.json()  # Adjust based on how the caller handles the response

        except requests.RequestException as e:
            raise HTTPException(status_code=500, detail=f"HTTP request failed: {str(e)}")


class TranscriptCollector:
    def __init__(self):
        self.reset()

    def reset(self):
        self.transcript_parts = []

    def add_part(self, part):
        self.transcript_parts.append(part)

    def get_full_transcript(self):
        return ' '.join(self.transcript_parts)

transcript_collector = TranscriptCollector()

async def transcribe_audio(audio_file: UploadFile):
    try:
        dg_client = Deepgram(os.getenv("DEEPGRAM_API_KEY"))
        mime_type = audio_file.content_type
        audio_data = await audio_file.read()
        mime_type = audio_file.content_type

        response = await dg_client.transcription.prerecorded(
            {"buffer": audio_data, "mimetype": mime_type},
            {"punctuate": True, "language": "en-US"}
        )

        transcript = response['results']['channels'][0]['alternatives'][0]['transcript']
        return transcript

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription error: {str(e)}")

# Specify allowed origins
origins = [
    "http://localhost:3000",  # Frontend URL
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Adjust for specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ConversationManager:
    def __init__(self):
        self.transcription_response = ""
        self.llm = LanguageModelProcessor()

    async def process_voice(self, audio_file: UploadFile):
    # Step 1: Transcribe audio
        self.transcription_response = await transcribe_audio(audio_file)
    
    # Step 2: Generate LLM response
        llm_response = self.llm.process(self.transcription_response)
    
    # Step 3: Generate Avatar video/audio
        avatar_gen = AvatarGeneration()
        avatar_response = avatar_gen.speak(llm_response)
        mp4_url = avatar_response.get("mp4_url")
    
    # Step 4: Return results
        return self.transcription_response, llm_response, mp4_url

conversation_manager = ConversationManager()

@app.post("/process_voice/")
async def process_voice(audio_file: UploadFile = File(...)):
    """Process voice input and return transcription, LLM response, and Avatar output."""
    transcription, llm_response, mp4_url = await conversation_manager.process_voice(audio_file)
    
    # Directly return the JSON response from AvatarGeneration
    return {
        "User": transcription,
        "llm_response": llm_response,
        "avatar_response": mp4_url  # JSON response from AvatarGeneration API
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
