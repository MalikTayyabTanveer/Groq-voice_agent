# Eva(Digital Human) 

This is a alpha demo showing a bot that uses Text-To-Speech, Speech-To-Text, and a language model to have a conversation with a user.

This demo is set up to use [Deepgram](www.deepgram.com) for the audio service, [Groq](https://groq.com/) the LLM and [Simli](https://www.simli.com/) for Avatar.

The files in `building_blocks` are the isolated components if you'd like to inspect them#
## Files

QuickAgent.py: Includes the demo of the voice agent.

audio.py: Starts the voice agent at port 8000, which takes voice input and returns a voice response.

audiotext.py: Starts the voice agent at port 8000, which takes voice input and returns both the audio (in base64) and text reply.

Simli.py: Starts the voice agent at port 8000, which takes voice input and returns an avatar speaking your response.
```


python3 QuickAgent.py
```
