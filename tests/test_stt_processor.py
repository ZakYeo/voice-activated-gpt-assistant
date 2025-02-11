from unittest.mock import Mock
import pytest
from core.processor.stt_processor import STTProcessor


def test_start_stop():
    mock_audio_stream = Mock()
    processor = STTProcessor(audio_stream=mock_audio_stream)

    processor.start()
    mock_audio_stream.start_stream.assert_called_once()

    processor.stop()
    mock_audio_stream.stop_stream.assert_called_once()


def test_transcribe_audio_stream():
    mock_audio_stream = Mock()
    mock_audio_stream.read.side_effect = [
        b"audio_chunk_1", b"audio_chunk_2", None]  # None simulates end of stream

    transcribe_mock = Mock()
    transcribe_mock.side_effect = ["Hello", " world", ""]

    processor = STTProcessor(
        audio_stream=mock_audio_stream, transcribe_func=transcribe_mock)

    processor.start()
    result = processor.transcribe()
    processor.stop()

    assert transcribe_mock.call_count == 2
    assert result == "Hello world", f"Expected 'Hello world', got '{result}'"
