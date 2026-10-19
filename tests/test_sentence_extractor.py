from src.sentence_extractor import extract_sentences


def test_extract_multiple_sentences():
    text = "Hello world. How are you? I am fine!"
    assert extract_sentences(text) == [
        "Hello world.",
        "How are you?",
        "I am fine!",
    ]


def test_empty_text():
    assert extract_sentences("") == []


def test_whitespace_only():
    assert extract_sentences("   ") == []