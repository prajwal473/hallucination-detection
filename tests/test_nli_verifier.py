import pytest

from src.nli_verifier import verify_claim


def test_empty_claim_raises_error():
    with pytest.raises(ValueError, match="Claim must not be empty"):
        verify_claim("", "The Eiffel Tower is in Paris.")


def test_empty_evidence_raises_error():
    with pytest.raises(ValueError, match="Evidence must not be empty"):
        verify_claim("The Eiffel Tower is in Paris.", "")


def test_nli_returns_expected_fields():
    result = verify_claim(
        "The Eiffel Tower is located in London.",
        "The Eiffel Tower is located in Paris, France.",
    )

    assert result["label"] in {
        "supported",
        "contradicted",
        "uncertain",
    }

    assert 0.0 <= result["entailment_score"] <= 1.0
    assert 0.0 <= result["contradiction_score"] <= 1.0
    assert 0.0 <= result["neutral_score"] <= 1.0
    