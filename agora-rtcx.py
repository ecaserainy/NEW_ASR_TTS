from agora.rtc.agora_service import AgoraService
from agora.rtc.audio_frame_observer import IAudioFrameObserver
from agora.rtc.rtc_connection_observer import IRTCConnectionObserver
from agora.rtc.local_user import ILocalUserObserver
import asyncio
import base64
import os
import numpy as np
import webrtcvad
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from modules.Asr_pcm import save_pcm_to_wav, recognize_audio
from modules.TTS_Audio import cosy_tts
from modules.llm_inference import generate_reply_from_text
from utils.logger import logger

# FastAPI 实例
app = FastAPI()

# 音频参数
FRAME_DURATION_MS = 30
SAMPLE_RATE = 16000
PCM_SAMPLE_WIDTH = 2
PCM_FRAME_BYTES = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000) * PCM_SAMPLE_WIDTH

# 声网配置
APP_ID = "your_agora_app_id"
TOKEN = "your_token"  # 使用 RtcTokenBuilder 生成
CHANNEL = "voice_channel"
LOG_PATH = "/home/asr_tts/logs/agorasdk.log"

# 音频帧观测器
class AudioFrameObserver(IAudioFrameObserver):
    def __init__(self, processor):
        self.processor = processor

    def on_playback_audio_frame_before_mixing(self, channel_id, user_id, audio_frame):
        audio_data = audio_frame.buffer
        self.processor.handle_audio_frame(audio_data)
        return True

# 连接观测器
class RTCConnectionObserver(IRTCConnectionObserver):
    def __init__(self, processor):
        self.processor = processor

    def on_connected(self, agora_rtc_conn, conn_info, reason):
        logger.info(f"连接成功: 频道 {conn_info.channel_id}, UID: {conn_info.local_user_id}, 原因: {reason}")

    def on_disconnected(self, agora_rtc_conn, conn_info, reason):
        logger.info(f"连接断开: 频道 {conn_info.channel_id}, UID: {conn_info.local_user_id}, 原因: {reason}")

    def on_token_privilege_will_expire(self, agora_rtc_conn, token):
        logger.warning(f"Token 即将过期: {token}")
        # TODO: 从服务端获取新 Token 并调用 renew_token
        new_token = "new_token_from_server"  # 替换为实际逻辑
        agora_rtc_conn.renew_token(new_token)
        logger.info("Token 已更新")

    def on_token_privilege_did_expire(self, agora_rtc_conn):
        logger.error("Token 已过期，需重新连接")
        # TODO: 获取新 Token 并重新 connect
        new_token = "new_token_from_server"  # 替换为实际逻辑
        agora_rtc_conn.connect(new_token, CHANNEL, "0")
        logger.info("重新连接频道")

    def on_user_joined(self, agora_rtc_conn, user_id):
        logger.info(f"远端用户加入: {user_id}")
        self.processor.local_user.subscribe_audio(user_id)

    def on_user_left(self, agora_rtc_conn, user_id, reason):
        logger.info(f"远端用户离开: {user_id}, 原因: {reason}")

    def on_error(self, agora_rtc_conn, error_code, error_msg):
        logger.error(f"连接错误: 错误码 {error_code}, 描述: {error_msg}")

# 本地用户观测器
class LocalUserObserver(ILocalUserObserver):
    def on_local_audio_track_state_changed(self, agora_local_user, agora_local_audio_track, state, error):
        logger.info(f"本地音频轨道状态: 轨道 {agora_local_audio_track}, 状态 {state}, 错误 {error}")

    def on_user_audio_track_state_changed(self, agora_local_user, user_id, agora_remote_audio_track, state, reason, elapsed):
        logger.info(f"远端音频轨道状态: 用户 {user_id}, 状态 {state}, 原因 {reason}, 时间 {elapsed}ms")

    def on_stream_message(self, agora_local_user, user_id, stream_id, data, length):
        logger.info(f"收到数据流消息: 用户 {user_id}, Stream ID {stream_id}, 数据长度 {length}")

    def on_user_info_updated(self, agora_local_user, user_id, msg, val):
        logger.info(f"用户媒体信息更新: 用户 {user_id}, 消息 {msg}, 值 {val}")

    def on_audio_meta_data_received(self, agora_local_user, user_id, data):
        logger.info(f"收到音频 Metadata: 用户 {user_id}, 数据 {data}")

# 自定义音频处理类
class CustomAudioProcessor:
    def __init__(self, websocket: WebSocket):
        self.websocket = websocket
        self.service = AgoraService()
        try:
            self.service.initialize({"app_id": APP_ID})
            self.service.set_log_file(LOG_PATH, 1024 * 1024)  # 设置日志路径，1MB
        except Exception as e:
            logger.error(f"AgoraService 初始化失败: {e}")
            raise
        self.connection = self.service.create_rtc_connection({"channel_id": CHANNEL, "subscribed_audio": False})
        self.connection.register_observer(RTCConnectionObserver(self))
        self.connection.connect(TOKEN, CHANNEL, "0")
        self.local_user = self.connection.get_local_user()
        self.local_user.set_user_role("HOST")
        self.local_user.register_local_user_observer(LocalUserObserver())
        self.local_user.set_playback_audio_frame_before_mixing_parameters(1, SAMPLE_RATE)  # 单声道，16kHz
        self.media_factory = self.service.create_media_node_factory()
        self.audio_sender = self.media_factory.create_audio_pcm_data_sender()
        self.audio_track = self.service.create_custom_audio_track_pcm(self.audio_sender)
        self.audio_track.set_max_buffer_audio_frame_number(5000)  # 50s 缓冲
        self.audio_track.set_send_delay_ms(150)  # 150ms 最小发送时长
        self.local_user.publish_audio(self.audio_track)
        self.local_user.register_audio_frame_observer(AudioFrameObserver(self))
        self.local_user.subscribe_all_audio()  # 订阅所有远端音频
        self.vad = webrtcvad.Vad()
        self.vad.set_mode(2)
        self.is_talking = False
        self.silence_duration = 0
        self.min_silence_duration = 1.0
        self.buffer_mic = bytearray()
        self.new_speech_duration = 0
        self.min_speech_duration = 0.2
        self.energy_threshold = 0.05
        self.background_noise_mean = 0
        self.current_reply = ""
        self.calibration_duration = 10.0
        self.update_interval = 60.0
        self.process_lock = asyncio.Lock()
        asyncio.create_task(self.update_threshold_periodically())

    def handle_audio_frame(self, audio_data):
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        energy = np.mean(np.abs(audio_array))
        is_speech = energy > self.energy_threshold and self.vad.is_speech(audio_data, sample_rate=SAMPLE_RATE)
        if is_speech:
            self.new_speech_duration += FRAME_DURATION_MS / 1000
            if self.new_speech_duration >= self.min_speech_duration:
                if not self.is_talking:
                    logger.info("检测到语音，开始处理")
                    self.is_talking = True
                self.silence_duration = 0
                asyncio.create_task(self.handle_interruption())
        else:
            self.new_speech_duration = 0
            self.silence_duration += FRAME_DURATION_MS / 1000
            if self.is_talking and self.silence_duration >= self.min_silence_duration:
                logger.info("语音停止")
                self.is_talking = False
        self.buffer_mic.extend(audio_data)
        if len(self.buffer_mic) >= PCM_FRAME_BYTES:
            audio_pcm = bytes(self.buffer_mic[:PCM_FRAME_BYTES])
            del self.buffer_mic[:PCM_FRAME_BYTES]
            asyncio.create_task(self.process_audio(audio_pcm))

    async def calibrate_background_noise(self, audio_data):
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        self.background_noise_mean = np.mean(np.abs(audio_array))
        self.energy_threshold = max(0.05, self.background_noise_mean * 1.5)
        logger.info(f"校准音量阈值: {self.energy_threshold}")

    async def update_threshold_periodically(self):
        while True:
            audio_data = bytearray()
            start_time = asyncio.get_event_loop().time()
            while asyncio.get_event_loop().time() - start_time < self.calibration_duration:
                await asyncio.sleep(0.1)
            if audio_data:
                await self.calibrate_background_noise(audio_data)
            await asyncio.sleep(self.update_interval)

    async def process_audio(self, audio_pcm):
        async with self.process_lock:
            if not self.is_talking:
                return
            wav_path = "audio.wav"
            save_pcm_to_wav(audio_pcm, wav_path, SAMPLE_RATE)
            text_pcm = recognize_audio(wav_path)
            logger.info(f"ASR 结果: {text_pcm}")
            await self.websocket.send_text(text_pcm)
            self.current_reply = await generate_reply_from_text(text_pcm)
            logger.info(f"回复文本: {self.current_reply}")
            await self.websocket.send_text(self.current_reply)
            output_wav = f"output/{self.websocket.client.host}_resp.wav"
            cosy_tts.text_to_wav(self.current_reply, output_wav)
            with open(output_wav, "rb") as f:
                audio_bytes = f.read()
            b64_audio = base64.b64encode(audio_bytes).decode()
            data_url = f"data:audio/wav;base64,{b64_audio}"
            await self.websocket.send_json({
                "text": self.current_reply,
                "tts_audio": data_url
            })
            self.audio_sender.send_audio_pcm_data(audio_bytes, SAMPLE_RATE, 1, len(audio_bytes) // PCM_SAMPLE_WIDTH)

    async def handle_interruption(self):
        logger.info("处理语音中断")
        self.audio_track.clear_sender_buffer()
        self.audio_track.set_enabled(0)  # 暂停音频轨道
        if self.current_reply:
            interrupted_text = self.current_reply + "..."
            logger.info(f"中断回复文本: {interrupted_text}")
            await self.websocket.send_text(interrupted_text)
            self.current_reply = ""
        await stop_audio_playback(self.websocket)
        self.audio_track.set_enabled(1)  # 恢复音频轨道

async def stop_audio_playback(websocket: WebSocket):
    logger.info("发送停止播放指令")
    await websocket.send_json({"action": "stop_audio"})

@app.websocket('/ws/audio')
async def websocket_audio(websocket: WebSocket):
    await websocket.accept()
    logger.info(f"WebSocket 连接建立: {websocket.client.host}")
    audio_processor = CustomAudioProcessor(websocket)
    try:
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
        audio_processor.local_user.unpublish_audio(audio_processor.audio_track)
        audio_processor.local_user.unregister_audio_frame_observer()
        audio_processor.local_user.unregister_local_user_observer()
        audio_processor.connection.unregister_observer()
        audio_processor.connection.disconnect()
        audio_processor.service.release()
        audio_processor.media_factory.release()
        logger.info("Agora RTC 连接关闭，WebSocket 会话结束")

if __name__ == '__main__':
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, ws_ping_interval=20, ws_ping_timeout=20, ssl_keyfile="/etc/letsencrypt/live/your-domain.com/privkey.pem", ssl_certfile="/etc/letsencrypt/live/your-domain.com/fullchain.pem")
