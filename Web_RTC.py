import asyncio
import base64
import fractions

import numpy as np
import webrtcvad
import av
from aiortc import (
    RTCPeerConnection,
    RTCSessionDescription,
    RTCConfiguration,
    RTCIceServer,
    MediaStreamError,
    AudioStreamTrack
)
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from modules.Asr_pcm import recognize_audio
from modules.TTS_Audio import cosy_tts
from modules.llm_inference import generate_reply_from_text
from utils.logger import logger

app = FastAPI()

# 音频参数
SAMPLE_RATE = 16000
CHANNELS = 1
FRAME_DURATION_MS = 20  # ms

# STUN 服务器配置
STUN_SERVER = RTCConfiguration(
    iceServers=[RTCIceServer(urls=["stuns:stun.l.google.com:19302"])]
)

class TTSAudioTrack(AudioStreamTrack):
    """TTS 推送轨道：从队列取 PCM，输出给对端播放"""
    def __init__(self):
        super().__init__()
        self.queue = asyncio.Queue()
        self.pts = 0
        self.time_base = fractions.Fraction(1, SAMPLE_RATE)

    async def recv(self):
        pcm_bytes = await self.queue.get()
        pcm_array = np.frombuffer(pcm_bytes, dtype=np.int16)
        frame = av.AudioFrame.from_ndarray(pcm_array, layout="mono")
        frame.sample_rate = SAMPLE_RATE
        frame.pts = self.pts
        frame.time_base = self.time_base
        self.pts += int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)
        return frame

    async def push_pcm(self, pcm_bytes: bytes):
        await self.queue.put(pcm_bytes)

@app.websocket("/wss/audio")
async def websocket_audio(websocket: WebSocket):
    await websocket.accept()
    logger.info(f"WebSocket 连接建立: {websocket.client.host}")

    pc = RTCPeerConnection(configuration=STUN_SERVER)
    tts_track = TTSAudioTrack()
    pc.addTrack(tts_track)

    @pc.on("track")
    async def on_track(track):
        if track.kind == "audio":
            logger.info("收到远端音轨，开始处理")
            vad = webrtcvad.Vad(2)
            ring = bytearray()
            audio_buffer = bytearray()
            FRAME_BYTES = int(SAMPLE_RATE * 0.02) * CHANNELS * 2

            try:
                while True:
                    frame = await track.recv()
                    pcm = frame.to_ndarray().astype(np.int16).tobytes()
                    ring.extend(pcm)
                    while len(ring) >= FRAME_BYTES:
                        chunk = bytes(ring[:FRAME_BYTES])
                        del ring[:FRAME_BYTES]

                        try:
                            is_speech = vad.is_speech(chunk, SAMPLE_RATE)
                        except Exception as exc:
                            logger.warning(f"VAD 格式不符: {exc}")
                            continue

                        if is_speech:
                            audio_buffer.extend(chunk)
                            # 如果累积到 0.2s，则触发 ASR
                            if len(audio_buffer) >= int(SAMPLE_RATE * 0.2) * CHANNELS * 2:
                                utterance = bytes(audio_buffer)
                                audio_buffer.clear()

                                # 执行 ASR → LLM → TTS 流程
                                text = await recognize_audio(utterance)
                                if text:
                                    logger.info(f"ASR 结果: {text}")
                                    await websocket.send_text(text)

                                    reply = await generate_reply_from_text(text)
                                    logger.info(f"LLM 回复: {reply}")
                                    await websocket.send_text(reply)

                                    loop = asyncio.get_event_loop()
                                    pcm_reply = await loop.run_in_executor(
                                        None, cosy_tts.text_to_pcm, reply, SAMPLE_RATE, CHANNELS
                                    )

                                    # 推送 TTS
                                    await tts_track.push_pcm(pcm_reply)
            except MediaStreamError:
                logger.info("远端音轨结束")
            except WebSocketDisconnect:
                logger.info("WebSocket 断开")
    try:
        data = await websocket.receive_json()
        if "offer" in data:
            await pc.setRemoteDescription(RTCSessionDescription(sdp=data["offer"], type="offer"))
            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)
            await websocket.send_json({"answer": pc.localDescription.sdp})
            logger.info("WebRTC SDP 协商完成")

        @pc.on("iceconnectionstatechange")
        async def on_iceconnectionstatechange():
            logger.info(f"ICE 连接状态: {pc.iceConnectionState}")
            if pc.iceConnectionState == "failed":
                await pc.close()

        while True:
            try:
                data = await websocket.receive_json()
                if data.get("action") == "ping":
                    await websocket.send_json({"action": "pong"})
                    continue
            except WebSocketDisconnect:
                logger.info("WebSocket 断开连接")
                break
    except Exception as e:
        logger.error(f"WebSocket 错误: {e}")
    finally:
        await pc.close()
        logger.info("WebRTC 连接关闭，WebSocket 会话结束")

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(
        "webrtc_:app",
        host="0.0.0.0",
        port=8443,
        ws_ping_interval=20,
        ws_ping_timeout=20
    )
