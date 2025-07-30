import os

import numpy as np
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav

model_path = "/home/asr_tts/.virtualenvs/NEW_ASR_TTS/pretrained_models/CosyVoice2-0.5B"

class CosyTTS:
    def __init__(self, model_dir: str):
        os.environ["PYTHONPATH"] = os.getenv("PYTHONPATH", "") + ":third_party/Matcha-TTS"
        self.cosy = CosyVoice2(model_dir, load_jit=False, load_trt=False)
        self.sample_rate = self.cosy.sample_rate

    def text_to_pcm(self, text: str) -> bytes:
        prompt_speech_16k = load_wav('./asset/zero_shot_prompt.wav', 16000)
        pcm_data = bytearray()
        for i, j in enumerate(
                self.cosy.inference_zero_shot(text, "", stream=True, prompt_speech_16k=prompt_speech_16k)):
            wav = j["tts_speech"]
            pcm_array = wav.numpy().astype(np.int16).tobytes()
            pcm_data.extend(pcm_array)
        return bytes(pcm_data)

cosy_tts = CosyTTS(model_dir=model_path)