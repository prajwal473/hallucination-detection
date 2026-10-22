import re
from src.sentence_extractor import extract_sentences
from src.evidence_retriever import retrieve_evidence_with_scores
from src.nli_verifier import verify_claim

def get_relation_group(text: str) -> str | None:
    """Identify the main relationship expressed in a sentence."""

    patterns = {
        "location": r"\b(located in|is in|stands in)\b",
        "capital": r"\b(capital of)\b",
        "temperature_freeze": r"\b(freezes|freeze)\b",
        "temperature_boil": r"\b(boils|boil)\b",
        "orbit": r"\b(orbits|orbit)\b",
        "identity": r"\b(known as|called)\b",
        "sides": r"\b(sides|side)\b",
        "size": r"\b(larger|largest|smaller|smallest)\b",
        "completion": r"\b(completed|completion)\b",
        "authorship": r"\b(wrote|written by|authored|attributed to)\b",
    }

    text = text.lower()

    for group, pattern in patterns.items():
        if re.search(pattern, text):
            return group

    return None


MIN_RELEVANCE_SCORE = 0.40
MIN_NLI_SCORE = 0.80


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
        retrieved = retrieve_evidence_with_scores(
            sentence,
            evidence_store,
            top_k=top_k,
        )

        relevant = [
            item
            for item in retrieved
            if item["similarity_score"] >= MIN_RELEVANCE_SCORE
        ]

        if not relevant:
            results.append({
                "claim": sentence,
                "evidence": [],
                "label": "uncertain",
                "entailment_score": 0.0,
                "contradiction_score": 0.0,
                "neutral_score": 1.0,
            })
            continue

        evidence_results = []

        for item in relevant:
            evidence = item["evidence"]

            forward = verify_claim(sentence, evidence)
            reverse = verify_claim(evidence, sentence)

            evidence_results.append({
                "evidence": evidence,
                "similarity_score": item["similarity_score"],
                "forward": forward,
                "reverse": reverse,
            })

        # Aggregate scores across retrieved evidence.
        entailment = max(
            item["forward"]["entailment_score"]
            for item in evidence_results
        )

        contradiction = max(
            item["forward"]["contradiction_score"]
            for item in evidence_results
        )

        neutral = max(
            item["forward"]["neutral_score"]
            for item in evidence_results
        )

        strong_support = any(
            item["forward"]["entailment_score"] >= MIN_NLI_SCORE
            for item in evidence_results
        )

        strong_contradiction = any(
            item["forward"]["contradiction_score"] >= MIN_NLI_SCORE
            and get_relation_group(sentence) is not None
            and get_relation_group(sentence)
            == get_relation_group(item["evidence"])
            for item in evidence_results
        )

        if strong_support and not strong_contradiction:
            label = "supported"
        elif strong_contradiction and not strong_support:
            label = "contradicted"
        else:
            label = "uncertain"

        results.append({
            "claim": sentence,
            "evidence": [
                item["evidence"]
                for item in evidence_results
            ],
            "label": label,
            "entailment_score": entailment,
            "contradiction_score": contradiction,
            "neutral_score": neutral,
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

    for index, result in enumerate(
        detect(response, evidence_store),
        start=1,
    ):
        print(f"\nClaim {index}")
        print("-" * 40)

        for key, value in result.items():
            print(f"{key}: {value}")