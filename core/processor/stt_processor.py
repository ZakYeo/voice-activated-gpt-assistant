import time


class STTProcessor:
    def __init__(self, audio_stream, transcribe_func, polling_frequency=10):
        self.audio_stream = audio_stream
        self.transcribe_func = transcribe_func
        self.polling_frequency = polling_frequency
        self.is_running = False

    def start(self):
        self.audio_stream.start_stream()
        self.is_running = True

    def stop(self):
        self.audio_stream.stop_stream()
        self.is_running = False

    def transcribe(self):
        result = []
        silence_chunk_count = 0
        timeout = 10
        start_time = time.time()
        silence_limit = self.polling_frequency * 3  # 3 seconds of silence

        while self.is_running and time.time() - start_time < timeout:
            chunk = self.audio_stream.read()
            if chunk == b"":  # Silence detected
                silence_chunk_count += 1
                if silence_chunk_count >= silence_limit:
                    break  # Stop after 3 seconds of silence
                continue
            else:
                silence_chunk_count = 0  # Reset if valid chunk received

            text = self.transcribe_func(chunk)
            if text:
                result.append(text)

        return "".join(result)
