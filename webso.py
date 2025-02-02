import base64
import io
from dotenv import load_dotenv
import requests
import os
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydub import AudioSegment
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

# Load environment variables
load_dotenv()

app = FastAPI()


class LanguageModelProcessor:
    def __init__(self):
        self.llm = ChatGroq(
            temperature=0, model_name="llama3-8b-8192", groq_api_key=os.getenv("GROQ_API_KEY")
        )
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        with open("system_prompt.txt", "r") as file:
            system_prompt = file.read().strip()

        self.prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessagePromptTemplate.from_template("{text}"),
            ]
        )

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
                "voice_settings": {"stability": 0.1, "similarity_boost": 0.3, "style": 0.2},
            },
        }

        response = requests.post(self.url, json=payload, headers={"Content-Type": "application/json"})
        if response.status_code != 200:
            raise Exception(f"AvatarGeneration API error: {response.text}")
        return response.json().get("mp4_url", "")


def decode_and_convert_audio(base64_audio: str) -> bytes:
    try:
        audio_data = base64.b64decode(base64_audio)
        audio = AudioSegment.from_file(io.BytesIO(audio_data), format="mp3")
        wav_buffer = io.BytesIO()
        audio.export(wav_buffer, format="wav")
        return wav_buffer.getvalue()
    except Exception as e:
        raise Exception(f"Invalid audio format: {str(e)}")


@app.websocket("/ws/voice")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        dg_client = Deepgram(os.getenv("DEEPGRAM_API_KEY"))
        language_model_processor = LanguageModelProcessor()
        avatar_generator = AvatarGeneration()

        async for message in websocket.iter_text():  # Expecting JSON messages
            try:
                # Parse the incoming message as JSON
                data = eval(message)  # Alternatively, use `json.loads` if sending proper JSON
                input_type = data.get("type")
                input_data = data.get("data")

                if input_type == "audio":
                    # Step 1: Decode and convert audio to WAV
                    audio_bytes = decode_and_convert_audio(input_data)

                    # Step 2: Transcribe the audio using Deepgram
                    response = await dg_client.transcription.prerecorded(
                        {"buffer": audio_bytes, "mimetype": "audio/wav"},
                        {"punctuate": True, "language": "en-US"},
                    )

                    if response and "results" in response:
                        transcript = response["results"]["channels"][0]["alternatives"][0].get("transcript", "")
                        if not transcript.strip():
                            await websocket.send_text("Transcription failed or empty response.")
                            continue

                        # Step 3: Process the transcript using the language model
                        llm_response = language_model_processor.process(transcript)

                        # Step 4: Generate a video from the LLM response
                        video_url = avatar_generator.speak(llm_response)

                        # Step 5: Send the video URL back to the client
                        await websocket.send_text(video_url)
                    else:
                        await websocket.send_text("Transcription failed or empty response.")

                elif input_type == "text":
                    # Process the text input directly
                    llm_response = language_model_processor.process(input_data)

                    # Generate a video from the LLM response
                    video_url = avatar_generator.speak(llm_response)

                    # Send the video URL back to the client
                    await websocket.send_text(video_url)

                else:
                    await websocket.send_text("Invalid input type. Use 'text' or 'audio'.")
            except Exception as e:
                await websocket.send_text(f"Error processing input: {str(e)}")
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"WebSocket error: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)