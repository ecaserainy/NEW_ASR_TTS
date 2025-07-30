import logging

logger = logging.getLogger(__name__)

async def handle_ice_candidate(pc, data):
    """处理 ICE 候选"""
    try:
        candidate = data.get("candidate")
        if candidate:
            await pc.addIceCandidate(candidate)
            logger.info("添加 ICE 候选")
    except Exception as e:
        logger.error(f"ICE 候选处理错误: {e}")