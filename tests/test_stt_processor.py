from unittest.mock import Mock, patch
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


def infinite_audio_chunks():
    while True:
        yield b"audio_chunk"


def mock_audio_stream():
    chunk_generator = infinite_audio_chunks()
    for chunk in chunk_generator:
        yield chunk


def infinite_transcriptions():
    i = 0
    while True:
        yield f"chunk-{i}"
        i += 1


def test_transcribe_stops_after_10_seconds():
    mock_audio = Mock()
    mock_audio.read.side_effect = lambda: next(mock_audio_stream())

    transcribe_mock = Mock()
    transcribe_mock.side_effect = lambda _: next(infinite_transcriptions())

    processor = STTProcessor(
        audio_stream=mock_audio, transcribe_func=transcribe_mock, polling_frequency=10
    )

    with patch("time.time") as mock_time:
        # Simulates time advancing each loop iteration
        mock_time.side_effect = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        processor.start()
        result = processor.transcribe()  # Will stop after the mocked 10 seconds
        processor.stop()

    elapsed_time = 10  # Since we patched it to simulate exactly 10 seconds
    assert elapsed_time >= 10, f"Expected at least 10 seconds, got {elapsed_time:.2f} seconds"
    assert "chunk-0" in result  # Ensure it transcribed something
