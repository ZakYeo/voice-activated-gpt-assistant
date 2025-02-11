from unittest.mock import Mock
import pytest
from core.processor.stt_processor import STTProcessor
import time


def test_start_stop():
    mock_audio_stream = Mock()
    processor = STTProcessor(
        audio_stream=mock_audio_stream, transcribe_func=lambda _: _)

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
    result = processor.transcribe(duration=1)
    processor.stop()

    assert transcribe_mock.call_count == 2
    assert result == "Hello world", f"Expected 'Hello world', got '{result}'"


def test_transcribe_for_duration():
    mock_audio_stream = Mock()
    mock_audio_stream.read.side_effect = [
        b"audio_chunk_1", b"audio_chunk_2", b"audio_chunk_3"] * 10

    transcribe_mock = Mock()
    transcribe_mock.side_effect = ["Hello", " world", "! "] * 10

    processor = STTProcessor(
        audio_stream=mock_audio_stream, transcribe_func=transcribe_mock)

    start_time = time.time()
    processor.transcribe(duration=2)
    elapsed_time = time.time() - start_time

    assert elapsed_time >= 2, f"Expected at least 2 seconds, got {elapsed_time:.2f} seconds"
