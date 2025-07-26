import asyncio
import json
import websockets
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack, RTCIceServer, RTCConfiguration
from aiortc.contrib.media import MediaBlackhole
from speexdsp import EchoCanceller
import webrtcvad
from pydub import AudioSegment
import io
import logging
import yaml
from utils.audio_code import decode_pcm, encode_aac
from utils.signaling import handle_ice_candidate
from utils.logger import setup_logging

# 加载配置（本地测试）
try:
    with open("config/server_config.yaml", "r") as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    config = {
        "ice_servers": [{"urls": "stun:stun.l.google.com:19302"}],
        "models": {
            "sensevoice": "/mnt/d/code/NEW_ASR_TTS/models/SenseVoiceSmall",
            "qwen": "/mnt/d/code/NEW_ASR_TTS/models/Qwen2.5-1.5B-Instruct",
            "cosyvoice": "/mnt/d/code/NEW_ASR_TTS/models/CosyVoice-300M-Instruct"
        }
    }
    logging.warning("未找到 server_config.yaml，使用默认配置")

# 设置日志
setup_logging()
logger = logging.getLogger(__name__)

# 模拟模型（替换为实际模型）
class SenseVoice:
    @staticmethod
    def from_pretrained(model_name):
        return SenseVoice()
    def transcribe(self, audio):
        return "用户语音文本"

class CosyVoice:
    @staticmethod
    def from_pretrained(model_name):
        return CosyVoice()
    def synthesize(self, text):
        return b"\x00" * 640  # 模拟 PCM（16kHz，40ms）
    last_speech_audio = b""

class Qwen2ForCausalLM:
    @staticmethod
    def from_pretrained(model_name):
        return Qwen2ForCausalLM()
    def generate(self, **kwargs):
        return [0]

class AutoTokenizer:
    @staticmethod
    def from_pretrained(model_name):
        return AutoTokenizer()
    def __call__(self, text, return_tensors):
        return {"input_ids": []}
    def decode(self, tokens, skip_special_tokens):
        return "AI 回复"

# 初始化模型
asr_model = SenseVoice.from_pretrained(config["models"]["sensevoice"])
tts_model = CosyVoice.from_pretrained(config["models"]["cosyvoice"])
llm_model = Qwen2ForCausalLM.from_pretrained(config["models"]["qwen"])
tokenizer = AutoTokenizer.from_pretrained(config["models"]["qwen"])

# 初始化 AEC 和 VAD
echo_canceller = EchoCanceller.create(frame_size=128, filter_length=2048, sample_rate=16000)
vad = webrtcvad.Vad()
vad.set_mode(2)  # 降低敏感度，适合本地测试
interrupt_flag = False
pcs = set()

def generate_response(text):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = llm_model.generate(**inputs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

class AudioSinkTrack(MediaStreamTrack):
    kind = "audio"
    def __init__(self):
        super().__init__()
        self.audio_data = bytearray()

    async def recv(self):
        return None  # 占位：发送 AAC

async def handle_webrtc(websocket, path):
    global interrupt_flag
    # 转换配置
    ice_servers = [
        RTCIceServer(
            urls=server["urls"],
            username=server.get("username"),
            credential=server.get("credential")
        )
        for server in config.get("ice_servers", [])
    ]

    # 构造 RTCConfiguration 实例
    rtc_config = RTCConfiguration(iceServers=ice_servers)

    # 正确传入配置
    pc = RTCPeerConnection(configuration=rtc_config)
    pcs.add(pc)
    sink = MediaBlackhole()

    @pc.on("track")
    async def on_track(track):
        logger.info(f"接收到轨道: {track.kind}")
        if track.kind == "audio":
            sink.addTrack(track)
            await sink.start()
            while True:
                try:
                    frame = await track.recv()
                    pcm_data = decode_pcm(frame.data)  # G.711 PCMU 转 16kHz
                    if not pcm_data:
                        continue
                    reference_audio = tts_model.last_speech_audio or b""
                    clean_audio = echo_canceller.process(pcm_data, reference_audio)
                    if vad.is_speech(clean_audio, sample_rate=16000):
                        if not interrupt_flag:
                            interrupt_flag = True
                            await websocket.send(json.dumps({"type": "interrupt"}))
                            logger.info("VAD 检测到语音，通知前端暂停")
                        text = asr_model.transcribe(clean_audio)
                        response = generate_response(text)
                        audio = tts_model.synthesize(response)
                        tts_model.last_speech_audio = audio
                        aac_audio = encode_aac(audio)
                        if aac_audio and not interrupt_flag:
                            await send_audio(pc, aac_audio)
                    else:
                        if interrupt_flag:
                            interrupt_flag = False
                            await websocket.send(json.dumps({"type": "resume"}))
                            logger.info("VAD 检测到无语音，通知前端恢复")
                except Exception as e:
                    logger.error(f"轨道处理错误: {e}")
                    break

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info(f"连接状态变化: {pc.connectionState}")
        if pc.connectionState == "closed":
            pcs.discard(pc)
            await sink.stop()
            logger.info("连接关闭，清理资源")

    try:
        message = await websocket.recv()
        offer = json.loads(message)
        if offer.get("type") == "offer":
            await pc.setRemoteDescription(RTCSessionDescription(sdp=offer["sdp"], type="offer"))
            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)
            await websocket.send(json.dumps({
                "type": "answer",
                "sdp": pc.localDescription.sdp
            }))
            logger.info("发送 SDP Answer")

        async for message in websocket:
            data = json.loads(message)
            if data.get("type") == "ice":
                await handle_ice_candidate(pc, data)
    except Exception as e:
        logger.error(f"信令处理错误: {e}")
    finally:
        await pc.close()
        pcs.discard(pc)
        await sink.stop()

async def send_audio(pc, audio):
    pass  # 占位：实现 AAC 发送

async def signaling():
    server = await websockets.serve(handle_webrtc, " 192.168.112.66", 8000)
    # 生产环境取消注释，使用 SSL
    # import ssl
    # ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    # ssl_context.load_cert_chain(
    #     certfile=config["server"]["ssl_cert"],
    #     keyfile=config["server"]["ssl_key"]
    # )
    # server = await websockets.serve(
    #     handle_webrtc,
    #     config["server"]["host"],
    #     config["server"]["port"],
    #     ssl=ssl_context
    # )
    logger.info("WebSocket 服务器启动于 ws://localhost:8000")
    await server.wait_closed()

if __name__ == "__main__":
    asyncio.run(signaling())