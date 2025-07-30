import asyncio
import base64

import numpy as np
import webrtcvad
from aiortc import RTCPeerConnection, RTCSessionDescription, AudioStreamTrack, RTCConfiguration, RTCIceServer, \
    MediaStreamError
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from modules.Asr_pcm import  recognize_audio
from modules.TTS_Audio import cosy_tts
from modules.llm_inference import generate_reply_from_text
from utils.logger import logger

app = FastAPI()

FRAME_DURATION_MS = 30
SAMPLE_RATE = 16000
PCM_SAMPLE_WIDTH = 2
PCM_FRAME_BYTES = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000) * PCM_SAMPLE_WIDTH
CHANNELS = 1

STUN_SERVER = RTCConfiguration(
    iceServers=[
        RTCIceServer(urls=["stuns:stun.l.google.com:19302"])
    ]
)

class CustomAudioTrack(AudioStreamTrack):
    def __init__(self, websocket: WebSocket):
        super().__init__()
        self.websocket = websocket
        self.vad = webrtcvad.Vad()
        self.vad.set_mode(2)
        self.audio_sender = None
        self.is_talking = False
        self.silence_duration = 0
        self.min_silence_duration = 1.0  # 最小沉默时间（秒）
        self.buffer_mic = bytearray()
        self.new_speech_duration = 0
        self.min_speech_duration = 0.2  # 最小语音持续时间（秒）
        self.energy_threshold = 0.05  # 初始音量阈值
        self.background_noise_mean = 0  # 环境噪声均值
        self.current_reply = ""  # 当前回复文本
        self.calibration_duration = 10.0  # 校准时间（秒）
        self.update_interval = 60.0  # 阈值更新间隔（秒）
        self.current_reply = ""
        self.process_lock = asyncio.Lock()
        asyncio.create_task(self.update_threshold_periodically())

    async def calibrate_background_noise(self, audio_data):
        """校准环境噪声，动态设置音量阈值"""
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        self.background_noise_mean = np.mean(np.abs(audio_array))
        self.energy_threshold = max(0.05, self.background_noise_mean * 1.5)
        logger.info(f"校准音量阈值: {self.energy_threshold}")

    async def update_threshold_periodically(self):
        """定期更新音量阈值"""
        while True:
            audio_data = bytearray()
            start_time = asyncio.get_event_loop().time()
            while asyncio.get_event_loop().time() - start_time < self.calibration_duration:
                frame = await super().recv()
                audio_data.extend(bytes(frame.samples))
            await self.calibrate_background_noise(audio_data)
            await asyncio.sleep(self.update_interval)

    async def recv(self):
        frame = await super().recv()
        audio_data = bytes(frame.samples)

        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        energy = np.mean(np.abs(audio_array))

        is_speech = energy > self.energy_threshold and self.vad.is_speech(audio_data, sample_rate=SAMPLE_RATE)

        if is_speech:
            self.new_speech_duration += frame.samples.size / frame.sample_rate
            if self.new_speech_duration >= self.min_speech_duration:
                if not self.is_talking:
                    logger.info("检测到语音（用户或环境），开始处理")
                    self.is_talking = True
                self.silence_duration = 0
                await self.handle_interruption()
        else:
            self.new_speech_duration = 0
            self.silence_duration += frame.samples / (PCM_SAMPLE_WIDTH * CHANNELS * SAMPLE_RATE)
            if self.is_talking and self.silence_duration >= self.min_silence_duration:
                logger.info("语音停止")
                self.is_talking = False

        self.buffer_mic.extend(audio_data)
        if len(self.buffer_mic) >= PCM_FRAME_BYTES:
            audio_pcm = bytes(self.buffer_mic[:PCM_FRAME_BYTES])
            del self.buffer_mic[:PCM_FRAME_BYTES]
            await self.process_audio(audio_pcm)
        return frame

    async def process_audio(self, audio_pcm):
        async with self.process_lock:
            if not self.is_talking:
                return
            text = await recognize_audio(audio_pcm)
            if not text:
                return
            logger.info(f"ASR 结果: {text}")
            await self.websocket.send_text(text)
            self.current_reply = await generate_reply_from_text(text)
            logger.info(f"回复文本: {self.current_reply}")
            await self.websocket.send_text(self.current_reply)

            loop = asyncio.get_event_loop()
            pcm_data = await loop.run_in_executor(
                None, cosy_tts.text_to_pcm, self.current_reply, SAMPLE_RATE, CHANNELS
            )
            b64_audio = base64.b64encode(pcm_data).decode()
            data_url = f"data:audio/pcm;base64,{b64_audio}"
            await self.websocket.send_json({
                "text": self.current_reply,
                "tts_audio": data_url
            })
            frame_size = int(SAMPLE_RATE * 0.02) * PCM_SAMPLE_WIDTH * CHANNELS
            for i in range(0, len(pcm_data), frame_size):
                frame = pcm_data[i:i + frame_size]
                self.audio_sender.send_audio_pcm_data(
                    frame,
                    SAMPLE_RATE,
                    CHANNELS,
                    len(frame) // (PCM_SAMPLE_WIDTH * CHANNELS)
                )
                await asyncio.sleep(0.02)

    async def handle_interruption(self):
        logger.info("处理语音中断（用户或非用户声音）")
        if self.current_reply:
            interrupted_text = self.current_reply + "..."
            logger.info(f"中断回复文本: {interrupted_text}")
            await self.websocket.send_text(interrupted_text)
            self.current_reply = ""
        await stop_audio_playback(self.websocket)

async def stop_audio_playback(websocket: WebSocket):
    logger.info("发送停止播放指令")
    await websocket.send_json({"action": "stop_audio"})

@app.websocket('/wss/audio')
async def websocket_audio(websocket: WebSocket):
    await websocket.accept()
    logger.info(f"WebSocket 连接建立: {websocket.client.host}")

    pc = RTCPeerConnection(configuration=STUN_SERVER)

    @pc.on("track")
    async def on_track(track):
        if track.kind == "audio":
            logger.info("收到远端音轨，开始处理")
            while True:
                try:
                    frame = await track.recv()
                except MediaStreamError:
                    break
                pcm = frame.to_ndarray().tobytes()
            logger.info("远端音轨结束")

    tts_track = CustomAudioTrack(websocket)
    pc.addTrack(tts_track)
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
        ssl_keyfile="./certs/privkey.pem",
        ssl_certfile="./certs/fullchain.pem",
        ws_ping_interval=20,
        ws_ping_timeout=20
    )
