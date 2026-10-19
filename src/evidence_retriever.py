import re


def retrieve_evidence(claim: str, evidence_store: list[str], top_k: int = 3) -> list[str]:
    """
    Retrieve evidence sentences that share words with a claim.

    Args:
        claim: The sentence we want to verify.
        evidence_store: A list of known evidence sentences.
        top_k: Maximum number of evidence sentences to return.

    Returns:
        The most relevant evidence sentences.
    """
    if not claim or not claim.strip() or not evidence_store or top_k <= 0:
        return []

    stop_words = {
        "the", "is", "a", "an", "and", "or", "of", "in", "on",
        "at", "to", "for", "it", "was", "are", "were", "be",
        "this", "that", "with", "by", "as"
    }

    def tokenize(text: str) -> set[str]:
        words = re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())
        return {word for word in words if word not in stop_words}

    claim_words = tokenize(claim)

    if not claim_words:
        return []

    scored_evidence = []

    for evidence in evidence_store:
        evidence_words = tokenize(evidence)

        if not evidence_words:
            continue

        overlap = len(claim_words & evidence_words)
        score = overlap / len(claim_words)

        if overlap > 0:
            scored_evidence.append((score, evidence))

    scored_evidence.sort(key=lambda item: item[0], reverse=True)

    return [evidence for _, evidence in scored_evidence[:top_k]]


if __name__ == "__main__":
    evidence_store = [
        "The Eiffel Tower is located in Paris, France.",
        "The Statue of Liberty is located in New York Harbor.",
        "The Eiffel Tower was completed in 1889.",
        "London is the capital of the United Kingdom.",
    ]

    claim = "The Eiffel Tower is located in London."

    results = retrieve_evidence(claim, evidence_store, top_k=3)

    print(f"Claim: {claim}")
    print("\nRetrieved evidence:")

    for index, evidence in enumerate(results, start=1):
        print(f"{index}. {evidence}")