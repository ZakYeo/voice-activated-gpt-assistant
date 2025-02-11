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
