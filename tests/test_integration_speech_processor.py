import whisper
import wave
import numpy as np


def wav_to_numpy(file_path):
    # Numpy bytes array is a required input for whisper to translate into text
    with wave.open(file_path, "rb") as wav_file:
        n_frames = wav_file.getnframes()
        audio_data = wav_file.readframes(n_frames)
        # Convert to NumPy array (16-bit PCM format)
        np_audio = np.frombuffer(audio_data, dtype=np.int16)

        # Normalize to floating-point values between -1 and 1
        np_audio = np_audio.astype(np.float32) / 32768.0
        return np_audio


def test_integration_with_whisper():
    audio_array = wav_to_numpy("./tests/test_speech_recording.wav")
    model = whisper.load_model("base")
    result = model.transcribe(audio_array, language="en")

    assert "testing 1, 2, 3, 1, 2, 3." in result["text"].lower()
