import os
import wave
import requests
from faster_whisper import WhisperModel
from wyoming.server import AsyncServer
from wyoming.speech import Transcribe, Transcript, AudioChunk
from wyoming.audio import AudioFormat
from wyoming.event import Event

WEBHOOK_URL = os.getenv("N8N_WEBHOOK_URL", "http://localhost:5678/webhook/voice-full")
model_size = os.getenv("MODEL_SIZE", "tiny")
language = os.getenv("LANGUAGE", "pl")

model = WhisperModel(model_size, compute_type="int8")

class WhisperSTT(AsyncServer):
    async def handle_event(self, event: Event) -> Event:
        if event.type != Transcribe.event_type:
            return Transcript(text="")

        audio_chunk = AudioChunk.from_event(event)
        samples = audio_chunk.samples
        rate = audio_chunk.rate

        filename = "/tmp/audio.wav"
        with wave.open(filename, "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(rate)
            f.writeframes(samples)

        segments, _ = model.transcribe(filename, language=language)
        text = " ".join(segment.text for segment in segments)

        with open(filename, "rb") as audio_file:
            requests.post(
                WEBHOOK_URL,
                files={"audio": ("audio.wav", audio_file, "audio/wav")},
                data={"text": text}
            )

        return Transcript(text=text)

if __name__ == "__main__":
    import asyncio
    server = WhisperSTT.from_args()
    asyncio.run(server.run())
