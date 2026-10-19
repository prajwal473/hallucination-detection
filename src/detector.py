import re

from src.sentence_extractor import extract_sentences
from src.evidence_retriever import retrieve_evidence
from src.nli_verifier import verify_claim


STOP_WORDS = {
    "the", "is", "a", "an", "and", "or", "of", "in",
    "on", "at", "to", "for", "it", "was", "are",
    "were", "be", "this", "that", "with", "by", "as",
    "located",
}


def get_content_words(text: str) -> set[str]:
    """Extract meaningful words for a simple relevance check."""
    words = re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())
    return {word for word in words if word not in STOP_WORDS}


def detect(
    response_text: str,
    evidence_store: list[str],
    top_k: int = 3,
) -> list[dict]:
    """Analyze an LLM response sentence by sentence."""

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

        best_evidence = evidence_list[0]

        claim_words = get_content_words(sentence)
        evidence_words = get_content_words(best_evidence)

        overlap = claim_words & evidence_words

        # Require sufficient lexical overlap before trusting NLI.
        if not claim_words or len(overlap) < 2:
            results.append({
                "claim": sentence,
                "evidence": evidence_list,
                "label": "uncertain",
                "entailment_score": 0.0,
                "contradiction_score": 0.0,
                "neutral_score": 1.0,
            })
            continue

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