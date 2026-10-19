from transformers import pipeline

MODEL_NAME = "cross-encoder/nli-deberta-v3-small"

_nli_pipeline = None


def load_nli_model():
    """Load the pretrained NLI model once and reuse it."""
    global _nli_pipeline

    if _nli_pipeline is None:
        _nli_pipeline = pipeline(
            "text-classification",
            model=MODEL_NAME,
            tokenizer=MODEL_NAME,
            top_k=None,
        )

    return _nli_pipeline


def verify_claim(claim: str, evidence: str) -> dict:
    """
    Estimate whether evidence supports, contradicts, or is neutral
    toward a claim.
    """
    if not claim or not claim.strip():
        raise ValueError("Claim must not be empty.")

    if not evidence or not evidence.strip():
        raise ValueError("Evidence must not be empty.")

    model = load_nli_model()

    # Evidence is the premise; the claim is the hypothesis.
    result = model(
        {
            "text": evidence,
            "text_pair": claim,
        }
    )

    # Normalize the output into a list of label/score dictionaries.
    predictions = result[0] if result and isinstance(result[0], list) else result

    scores = {
        item["label"].lower(): float(item["score"])
        for item in predictions
    }

    entailment = scores.get("entailment", 0.0)
    contradiction = scores.get("contradiction", 0.0)
    neutral = scores.get("neutral", 0.0)

    label_scores = {
        "supported": entailment,
        "contradicted": contradiction,
        "uncertain": neutral,
    }

    predicted_label = max(label_scores, key=label_scores.get)

    return {
        "claim": claim,
        "evidence": evidence,
        "label": predicted_label,
        "entailment_score": entailment,
        "contradiction_score": contradiction,
        "neutral_score": neutral,
    }


if __name__ == "__main__":
    claim = "The Eiffel Tower is located in London."
    evidence = "The Eiffel Tower is located in Paris, France."

    result = verify_claim(claim, evidence)

    print("\nNLI Verification Result")
    print("----------------------")

    for key, value in result.items():
        print(f"{key}: {value}")