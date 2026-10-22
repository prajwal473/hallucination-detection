from src.detector import detect


def test_empty_response_returns_empty_list():
    assert detect("", ["Some evidence"]) == []


def test_whitespace_response_returns_empty_list():
    assert detect("   ", ["Some evidence"]) == []


def test_claim_without_evidence_is_uncertain():
    result = detect(
        "The moon is made of cheese.",
        [],
    )

    assert len(result) == 1
    assert result[0]["label"] == "uncertain"


def test_detector_returns_expected_fields():
    result = detect(
        "The Eiffel Tower is located in Paris.",
        ["The Eiffel Tower is located in Paris, France."],
    )

    assert len(result) == 1
    assert result[0]["claim"] == "The Eiffel Tower is located in Paris."
    assert result[0]["label"] in {
        "supported",
        "contradicted",
        "uncertain",
    }
    assert "evidence" in result[0]
    assert "entailment_score" in result[0]
    assert "contradiction_score" in result[0]
    assert "neutral_score" in result[0]


def test_unrelated_evidence_should_not_contradict_claim():
    result = detect(
        "The Earth orbits the Sun.",
        ["The Earth has one natural satellite, the Moon."],
    )

    assert len(result) == 1
    assert result[0]["label"] == "uncertain"


def test_different_water_facts_should_not_be_contradictory():
    result = detect(
        "Water boils at 100 degrees Celsius at standard atmospheric pressure.",
        [
            "At standard atmospheric pressure, water freezes at 0 degrees Celsius."
        ],
    )

    assert len(result) == 1
    assert result[0]["label"] == "uncertain"