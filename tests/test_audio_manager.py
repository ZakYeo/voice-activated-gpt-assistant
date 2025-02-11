import pytest
from unittest.mock import Mock
from core.audio_manager import AudioManager


def test_start_stop_audio_capture():
    mock_microphone = Mock()
    audio_manager = AudioManager(microphone=mock_microphone)

    audio_manager.start_capture()
    mock_microphone.start_stream.assert_called_once()

    audio_manager.stop_capture()
    mock_microphone.stop_stream.assert_called_once()
