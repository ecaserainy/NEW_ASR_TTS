import ctypes

SAMPLE_RATE = 16000
FRAME_SIZE = 480
FILTER_LENGTH = 10 * FRAME_SIZE

speex = ctypes.cdll.LoadLibrary("libspeexdsp.so")

class SpeexEchoCanceller:
    def __init__(self):

        self.echo_state = speex.speex_echo_state_init(FRAME_SIZE, FILTER_LENGTH)
        if not self.echo_state:
            raise RuntimeError("Failed to init speex_echo_state")

        self.preprocess_state = speex.speex_preprocess_state_init(FRAME_SIZE, SAMPLE_RATE)
        if not self.preprocess_state:
            raise RuntimeError("Failed to init speex_preprocess_state")
        speex.speex_preprocess_ctl(self.preprocess_state, 24, ctypes.c_void_p(self.echo_state))
        vad = ctypes.c_int(1)
        speex.speex_preprocess_ctl(self.preprocess_state, 4, ctypes.byref(vad))
        noise_suppress = ctypes.c_int(-30)
        speex.speex_preprocess_ctl(self.preprocess_state, 18, ctypes.byref(noise_suppress))

    def process(self, mic_frame: bytes, ref_frame: bytes) -> bytes:
        mic_buf = (ctypes.c_short * FRAME_SIZE).from_buffer_copy(mic_frame)
        ref_buf = (ctypes.c_short * FRAME_SIZE).from_buffer_copy(ref_frame)
        out_buf = (ctypes.c_short * FRAME_SIZE)()

        speex.speex_echo_cancellation(self.echo_state, mic_buf, ref_buf, out_buf)
        speex.speex_preprocess_run(self.preprocess_state, out_buf)

        return ctypes.string_at(out_buf, FRAME_SIZE * 2)

    def destroy(self):
        speex.speex_echo_state_destroy(self.echo_state)
        speex.speex_preprocess_state_destroy(self.preprocess_state)

aec = SpeexEchoCanceller()