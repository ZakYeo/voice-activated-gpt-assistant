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


def test_transcribe_stops_after_3_seconds_of_silence():
    mock_audio_stream = Mock()
    frequency = 10  # Polling frequency: 10 times per second
    empty_chunks = [b""] * (3 * frequency)  # 3 seconds of silence

    mock_audio_stream.read.side_effect = [
        b"audio_chunk_1", b"audio_chunk_2"] + empty_chunks
    transcribe_mock = Mock()
    transcribe_mock.side_effect = ["Hello", " world"]

    processor = STTProcessor(audio_stream=mock_audio_stream,
                             transcribe_func=transcribe_mock, polling_frequency=frequency)

    processor.start()
    result = processor.transcribe()
    processor.stop()

    assert transcribe_mock.call_count == 2
    assert result == "Hello world", f"Expected 'Hello world', got '{result}'"


def test_transcribe_handles_valid_and_silence_periods():
    mock_audio_stream = Mock()
    frequency = 5  # Polling frequency: 5 times per second
    valid_chunks = [b"audio_chunk"] * 10
    empty_chunks = [b""] * (3 * frequency)  # 3 seconds of silence

    mock_audio_stream.read.side_effect = valid_chunks + empty_chunks
    transcribe_mock = Mock()
    transcribe_mock.side_effect = ["chunk"] * 10

    processor = STTProcessor(audio_stream=mock_audio_stream,
                             transcribe_func=transcribe_mock, polling_frequency=frequency)

    processor.start()
    result = processor.transcribe()
    processor.stop()

    assert transcribe_mock.call_count == 10
    assert "chunk" in result
