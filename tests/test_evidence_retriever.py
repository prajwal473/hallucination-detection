from src.evidence_retriever import retrieve_evidence


def test_retrieves_relevant_evidence():
    evidence_store = [
        "The Eiffel Tower is in Paris.",
        "The Statue of Liberty is in New York.",
    ]

    result = retrieve_evidence(
        "Where is the Eiffel Tower?",
        evidence_store,
    )

    assert result
    assert result[0] == "The Eiffel Tower is in Paris."


def test_empty_claim_returns_empty_list():
    assert retrieve_evidence("", ["Some evidence"]) == []


def test_empty_evidence_store_returns_empty_list():
    assert retrieve_evidence("A claim", []) == []


def test_top_k_limits_results():
    evidence_store = [
        "The Eiffel Tower is in Paris.",
        "The Eiffel Tower was completed in 1889.",
    ]

    result = retrieve_evidence(
        "The Eiffel Tower",
        evidence_store,
        top_k=1,
    )

    assert len(result) == 1