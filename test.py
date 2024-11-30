from deepgram import Deepgram
import asyncio
import os
from dotenv import load_dotenv
load_dotenv()
async def test_connection():
    # Read and print the API key to ensure it's correctly loaded
    dg_api_key = os.getenv("DEEPGRAM_API_KEY")
    if not dg_api_key:
        print("Deepgram API Key not found in environment. Make sure DEEPGRAM_API_KEY is set.")
        return
    else:
        print(f"Deepgram API Key loaded: {dg_api_key[:4]}...")  # Print part of the key for verification

    # Initialize Deepgram with the loaded API key
    dg_client = Deepgram(dg_api_key)

    options = {
        "punctuate": True,
        "model": "nova-2",
        "language": "en-US"
    }

    async def on_received(data):
        print("Received data:", data)

    try:
        print("Testing WebSocket connection...")
        # Start a live transcription connection
        connection = await dg_client.transcription.live(options, on_received)
        
        await asyncio.sleep(5)  # Keep connection open briefly to receive data
        await connection.close()  # Close the connection
    except Exception as e:
        print("Error:", e)

# Run the test connection
asyncio.run(test_connection())
