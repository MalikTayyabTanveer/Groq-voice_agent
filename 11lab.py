# require deepgram-sdk==2.12.0
import asyncio
from dotenv import load_dotenv
import shutil
import subprocess
import requests
import time
import os
import json
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import base64
import io
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
import websockets

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


import base64
import json
import asyncio
import websockets
from fastapi import HTTPException
import shutil
import subprocess

class TextToSpeech:
    ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
    voice_id = 'kmSVBPu7loj4ayNinwWM'
    model_id = 'eleven_turbo_v2'

    async def speak(self, text):
        """
        Converts text to speech using ElevenLabs WebSocket API and returns Base64-encoded audio data.
        """
        uri = f"wss://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}/stream-input?model_id={self.model_id}"

        try:
            async with websockets.connect(uri) as websocket:
                # Send the initial message with the API key and text input
                await websocket.send(json.dumps({
                    "xi_api_key": self.ELEVENLABS_API_KEY,
                    "text": text,
                    "voice_settings": {
                        "stability": 0.5,
                        "similarity_boost": 0.8,
                        "use_speaker_boost": False
                    },
                    "generation_config": {
                        "chunk_length_schedule": [120, 160, 250, 290]
                    },
                }))

                # Ensure immediate generation of audio
                await websocket.send(json.dumps({"text": "", "flush": True}))

                # Collect and concatenate Base64 audio chunks
                audio_base64 = ""
                while True:
                    message = await websocket.recv()
                    if not message:  # WebSocket closes automatically when done
                        break

                    # Accumulate Base64-encoded audio chunks
                    try:
                        audio_base64 += message
                    except Exception as error:
                        print(f"Received non-audio data or invalid chunk: {message}")
                        continue  # Skip invalid chunks

                return audio_base64  # Return the full Base64 audio string

        except websockets.exceptions.WebSocketException as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error during WebSocket communication: {str(e)}"
            )



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
        self.transcription_response = await transcribe_audio(audio_file)
        llm_response = self.llm.process(self.transcription_response)
        tts = TextToSpeech()
        audio_base64 = await tts.speak(llm_response)  # Use ElevenLabs TTS
        return self.transcription_response, llm_response, audio_base64

conversation_manager = ConversationManager()

@app.post("/process_voice/")
async def process_voice(audio_file: UploadFile = File(...)):
    """Process voice input and return transcription, LLM response, and Base64 audio output."""
    transcription, llm_response, audio_base64 = await conversation_manager.process_voice(audio_file)
    return {
        "User": transcription,
        "llm_response": llm_response,
        "audio_file": audio_base64  # Return the audio as Base64
    }



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
