import time


class STTProcessor:
    def __init__(self, audio_stream, transcribe_func):
        self.audio_stream = audio_stream
        self.transcribe_func = transcribe_func
        self.is_running = False

    def start(self):
        self.audio_stream.start_stream()
        self.is_running = True

    def stop(self):
        self.audio_stream.stop_stream()
        self.is_running = False
