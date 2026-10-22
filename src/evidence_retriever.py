from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

_embedding_model = None


def load_embedding_model():
    """Load the embedding model once and reuse it."""
    global _embedding_model

    if _embedding_model is None:
        _embedding_model = SentenceTransformer(MODEL_NAME)

    return _embedding_model


def retrieve_evidence_with_scores(
    claim: str,
    evidence_store: list[str],
    top_k: int = 3,
) -> list[dict]:
    """
    Retrieve evidence ranked by semantic similarity.

    Returns a list of dictionaries containing evidence text
    and its cosine similarity score.
    """
    if not claim or not claim.strip():
        return []

    if not evidence_store or top_k <= 0:
        return []

    evidence_store = [
        evidence.strip()
        for evidence in evidence_store
        if evidence and evidence.strip()
    ]

    if not evidence_store:
        return []

    model = load_embedding_model()

    claim_embedding = model.encode([claim])
    evidence_embeddings = model.encode(evidence_store)

    similarities = cosine_similarity(
        claim_embedding,
        evidence_embeddings,
    )[0]

    ranked_indices = similarities.argsort()[::-1][:top_k]

    return [
        {
            "evidence": evidence_store[index],
            "similarity_score": float(similarities[index]),
        }
        for index in ranked_indices
    ]


def retrieve_evidence(
    claim: str,
    evidence_store: list[str],
    top_k: int = 3,
) -> list[str]:
    """
    Preserve the original interface by returning evidence text only.
    """
    results = retrieve_evidence_with_scores(
        claim,
        evidence_store,
        top_k,
    )

    return [item["evidence"] for item in results]


if __name__ == "__main__":
    evidence_store = [
        "Paris, France, is home to the famous Eiffel Tower.",
        "The Statue of Liberty stands in New York Harbor.",
        "The Eiffel Tower was completed in 1889.",
        "London is the capital of the United Kingdom.",
    ]

    claim = "The Eiffel Tower is located in Paris."

    results = retrieve_evidence_with_scores(
        claim,
        evidence_store,
        top_k=3,
    )

    print(f"Claim: {claim}")
    print("\nRetrieved evidence with similarity scores:")

    for index, result in enumerate(results, start=1):
        print(
            f"{index}. Similarity: "
            f"{result['similarity_score']:.4f} | "
            f"{result['evidence']}"
        )