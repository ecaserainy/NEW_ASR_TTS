from fastapi import FastAPI,WebSocket,WebSocketDisconnect
import base64
import uvicorn
from modules.Asr_pcm import save_pcm_to_wav, recognize_audio
from modules.Speexdsp_pcm import  aec
from modules.TTS_Audio import cosy_tts
from modules.llm_inference import generate_reply_from_text
from utils.logger import logger

app = FastAPI()

FRAME_DURATION_MS = 30
SAMPLE_RATE = 16000
PCM_SAMPLE_WIDTH = 2
PCM_FRAME_BYTES = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000) * PCM_SAMPLE_WIDTH

@app.websocket('/ws/audio')
async def websocket_audio(websocket: WebSocket):
    await websocket.accept()
    logger.info(f"Connected")

    try:
        buffer_mic = bytearray()
        while True:
            data = await websocket.receive_json()
            if data.get("action") == "ping":
                await websocket.send_json({"action": "pong"})
                continue
            logger.info(f"Received: {data}")
            if "mic" not in data or "ref" not in data:
                continue

            mic_pcm = base64.b64decode(data['mic'])
            ref_pcm = base64.b64decode(data['ref'])
            clean_pcm = aec.process(mic_pcm, ref_pcm)
            buffer_mic.extend(clean_pcm)

            if len(buffer_mic) >= PCM_FRAME_BYTES:
                audio_pcm = bytes(buffer_mic[:PCM_FRAME_BYTES])
                del buffer_mic[:PCM_FRAME_BYTES]

                save_pcm_to_wav(audio_pcm, "audio.wav", SAMPLE_RATE)
                text_pcm = recognize_audio("audio.wav")
                await websocket.send_text(text_pcm)

                reply_text = generate_reply_from_text(text_pcm)
                await websocket.send_text(reply_text)

                wav_path = f"output/{websocket.client.host}_resp.wav"
                cosy_tts.text_to_wav(reply_text, wav_path)
                with open(wav_path, "rb") as f:
                    audio_bytes = f.read()
                b64_audio = base64.b64encode(audio_bytes).decode()
                data_url = f"data:audio/wav;base64,{b64_audio}"
                await websocket.send_json({
                    "text": reply_text,
                    "tts_audio": data_url
                })
    except WebSocketDisconnect:
        logger.info(f"Stop Connection")
    finally:
        aec.destroy()
        logger.info("AEC resources destroyed, WebSocket session ended")

if __name__ == '__main__':
    uvicorn.run("main:app", host="0.0.0.0", port=8000,ws_ping_interval=20,ws_ping_timeout=20)




