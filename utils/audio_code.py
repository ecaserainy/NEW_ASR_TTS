from pydub import AudioSegment
import io
import logging

logger = logging.getLogger(__name__)

def decode_pcm(pcm_data):
    """处理 G.711 PCMU（8kHz）转为 16kHz PCM"""
    try:
        audio = AudioSegment(
            data=pcm_data,
            sample_width=1,  # G.711 8-bit
            frame_rate=8000,
            channels=1
        )
        audio = audio.set_frame_rate(16000).set_sample_width(2)
        return audio.raw_data
    except Exception as e:
        logger.error(f"PCM 处理错误: {e}")
        return None

def encode_aac(pcm_data):
    """编码 PCM 为 AAC（16kHz，32 kbps）"""
    try:
        audio = AudioSegment(
            data=pcm_data,
            sample_width=2,
            frame_rate=16000,
            channels=1
        )
        aac_io = io.BytesIO()
        audio.export(aac_io, format="aac", bitrate="32k", parameters=["-preset", "ultrafast"])
        return aac_io.getvalue()
    except Exception as e:
        logger.error(f"AAC 编码错误: {e}")
        return None