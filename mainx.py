import asyncio
import websockets
import json
import yaml
import numpy as np
from aiortc import RTCPeerConnection, RTCSessionDescription
from pydub import AudioSegment
import io
from speexdsp import EchoCanceller
from webrtcvad import Vad
import torch
from funasr import AutoModel
from cosyvoice.cosyvoice.cli.cosyvoice import CosyVoice2
from transformers import AutoModelForCausalLM, AutoTokenizer


class VoiceProcessor:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.sensevoice = AutoModel(
            model="iic/SenseVoiceSmall",
            model_dir=self.config['models']['sensevoice'],
            device="cuda:0" if torch.cuda.is_available() else "cpu"
        )

        self.qwen = AutoModelForCausalLM.from_pretrained(
            self.config['models']['qwen'], torch_dtype=torch.float16
        ).to(self.device)
        self.qwen_tokenizer = AutoTokenizer.from_pretrained(self.config['models']['qwen'])

        self.cosyvoice = CosyVoice2(self.config['models']['cosyvoice'])
        self.sample_rate = self.cosyvoice.sample_rate  # 16000 Hz
        self.vad = Vad(3)
        self.echo_canceller = EchoCanceller(self.sample_rate, 1, frame_size=160)

    async def process_audio_to_text(self, pcm_data):
        audio = AudioSegment(
            data=pcm_data.tobytes(),
            sample_width=2,
            frame_rate=self.sample_rate,
            channels=1
        )
        wav_io = io.BytesIO()
        audio.export(wav_io, format="wav")
        wav_io.seek(0)
        result = self.sensevoice.generate(input=wav_io.read(), sample_rate=self.sample_rate)
        text = result[0]['text'] if result else ""
        return text

    async def process_text_to_response(self, text):
        inputs = self.qwen_tokenizer(text, return_tensors="pt").to(self.device)
        outputs = self.qwen.generate(**inputs, max_length=100)
        response = self.qwen_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

    async def process_text_to_speech(self, text, style="neutral"):
        for i, output in enumerate(self.cosyvoice.inference_instruct(
                text, style=style, stream=True
        )):
            audio_samples = output['tts_speech'].cpu().numpy()
            aac_audio = self.convert_to_aac(audio_samples)
            return aac_audio

    def convert_to_aac(self, pcm_samples):
        pcm_io = io.BytesIO(pcm_samples.tobytes())
        audio = AudioSegment.from_raw(pcm_io, sample_rate=16000, channels=1, sample_width=2)
        aac_io = io.BytesIO()
        audio.export(aac_io, format="aac", codec="aac")
        return aac_io.getvalue()


async def handle_webrtc(websocket, path):
    pc = RTCPeerConnection()
    processor = VoiceProcessor('/mnt/d/code/NEW_ASR_TTS/config/server_config.yaml')

    @pc.on("track")
    async def on_track(track):
        if track.kind == "audio":
            while True:
                frame = await track.recv()
                pcm_data = np.array(frame.samples, dtype=np.int16)
                is_speech = processor.vad.is_speech(pcm_data.tobytes(), processor.sample_rate)
                if is_speech:
                    clean_pcm = processor.echo_canceller.process(pcm_data)
                    text = await processor.process_audio_to_text(clean_pcm)
                    response = await processor.process_text_to_response(text)
                    aac_audio = await processor.process_text_to_speech(response, style="happy")
                    await websocket.send(aac_audio)

    async for message in websocket:
        data = json.loads(message)
        if data['type'] == 'offer':
            await pc.setRemoteDescription(RTCSessionDescription(data['sdp'], data['type']))
            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)
            await websocket.send(json.dumps({
                'type': 'answer',
                'sdp': pc.localDescription.sdp
            }))
        elif data['type'] == 'ice':
            await pc.addIceCandidate(data['candidate'])


async def main():
    server = await websockets.serve(
        handle_webrtc,
        "0.0.0.0",
        8000,
        ping_interval=None
    )
    print("WebSocket 服务器启动于 ws://0.0.0.0:8000")
    await server.wait_closed()


if __name__ == "__main__":
    asyncio.run(main())