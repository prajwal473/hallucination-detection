from src.sentence_extractor import extract_sentences
from src.evidence_retriever import retrieve_evidence
from src.nli_verifier import verify_claim


def detect(
    response_text: str,
    evidence_store: list[str],
    top_k: int = 3,
) -> list[dict]:
    """
    Analyze an LLM response sentence by sentence.

    Returns a list of dictionaries containing each claim,
    its retrieved evidence, NLI scores, and predicted label.
    """
    if not response_text or not response_text.strip():
        return []

    sentences = extract_sentences(response_text)
    results = []

    for sentence in sentences:
        evidence_list = retrieve_evidence(
            sentence,
            evidence_store,
            top_k=top_k,
        )

        if not evidence_list:
            results.append({
                "claim": sentence,
                "evidence": [],
                "label": "uncertain",
                "entailment_score": 0.0,
                "contradiction_score": 0.0,
                "neutral_score": 1.0,
            })
            continue

        # Verify against the highest-ranked evidence first.
        # Verify the claim against the highest-ranked evidence.
        best_evidence = evidence_list[0]

        best_result = verify_claim(
            sentence,
            best_evidence,
        )

        results.append({
            "claim": sentence,
            "evidence": evidence_list,
            "label": best_result["label"],
            "entailment_score": best_result["entailment_score"],
            "contradiction_score": best_result["contradiction_score"],
            "neutral_score": best_result["neutral_score"],
        })
    return results


if __name__ == "__main__":
    evidence_store = [
        "The Eiffel Tower is located in Paris, France.",
        "The Eiffel Tower was completed in 1889.",
        "London is the capital of the United Kingdom.",
        "The Statue of Liberty is located in New York Harbor.",
    ]

    response = (
        "The Eiffel Tower is located in Paris. "
        "The Eiffel Tower is located in London."
    )

    results = detect(response, evidence_store)

    for index, result in enumerate(results, start=1):
        print(f"\nClaim {index}")
        print("-" * 40)

        for key, value in result.items():
            print(f"{key}: {value}")