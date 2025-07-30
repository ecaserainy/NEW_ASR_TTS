import numpy as np
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

model_dir = "/home/asr_tts/.virtualenvs/NEW_ASR_TTS/models/SenseVoiceSmall"
model = AutoModel(
    model=model_dir,
    vad_model="/home/asr_tts/.virtualenvs/NEW_ASR_TTS/models/speech_fsmn_vad_zh-cn-16k-common-pytorch",
    vad_kwargs={"max_single_segment_time": 300},
    device="cuda:0",
    disable_update=True,
)

async def recognize_audio(pcm_data: bytes) -> str:
    pcm_array = np.frombuffer(pcm_data, dtype=np.int16) / 32768.0
    res = model.generate(
        input=pcm_array,
        cache={},
        language="auto",
        use_itn=True,
        batch_size_s=0.03,
        merge_vad=False,
        merge_length_s=0,
        is_final=False,
    )
    return rich_transcription_postprocess(res[0]["text"]) if res else ""

