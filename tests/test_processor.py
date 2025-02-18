import pytest
from src.preprocessing.processor import PromptProcessor

def test_processor_initialization():
    processor = PromptProcessor()
    assert processor.model_name == "mistral"