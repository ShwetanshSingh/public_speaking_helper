import os
import pytest
from unittest.mock import Mock, patch

from transcription_utils import WhisperModel

@pytest.fixture
def whisper_model():
    with patch('whisper.load_model') as mock_load:
        # Mock the whisper model
        mock_model = Mock()
        mock_load.return_value = mock_model
        model = WhisperModel()
        model.model = mock_model
        yield model

def test_transcribe_success(whisper_model):
    """Test successful transcription of audio file"""
    # Mock data
    expected_text = "This is a test transcription"
    mock_result = {"text": expected_text}
    whisper_model.model.transcribe.return_value = mock_result
    
    # Test
    result = whisper_model.transcribe("dummy_audio.mp3")
    
    # Assertions
    assert result == expected_text
    whisper_model.model.transcribe.assert_called_once_with("dummy_audio.mp3")

def test_transcribe_invalid_output(whisper_model):
    """Test handling of invalid model output"""
    # Mock data - returning a list instead of string
    mock_result = {"text": ["invalid", "output"]}
    whisper_model.model.transcribe.return_value = mock_result
    
    # Test and assert
    with pytest.raises(Exception) as exc_info:
        whisper_model.transcribe("dummy_audio.mp3")
    assert str(exc_info.value) == "Model output is not string, but list[Unknown]"

def test_transcribe_with_actual_env(whisper_model):
    """Test that environment variable for model type is respected"""
    test_model_type = "medium"
    with patch.dict(os.environ, {'WHISPER_MODEL': test_model_type}):
        with patch('whisper.load_model') as mock_load:
            WhisperModel()
            mock_load.assert_called_once_with(test_model_type)

def test_transcribe_with_default_model(whisper_model):
    """Test that default model type is used when env var is not set"""
    with patch.dict(os.environ, clear=True):
        with patch('whisper.load_model') as mock_load:
            WhisperModel()
            mock_load.assert_called_once_with("base")
