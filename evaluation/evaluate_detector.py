import json
from pathlib import Path

from src.detector import detect


DATASET_PATH = Path("evaluation/smoke_dataset.json")


def evaluate():
    with DATASET_PATH.open("r", encoding="utf-8") as file:
        dataset = json.load(file)

    total = 0
    correct = 0

    print("\nHallucination Detection Evaluation")
    print("=" * 40)

    for index, sample in enumerate(dataset, start=1):
        response = sample["response"]
        evidence = sample["evidence"]
        expected_label = sample["expected_label"]

        results = detect(
            response_text=response,
            evidence_store=evidence,
        )

        if not results:
            print(f"\nExample {index}: No claims detected")
            continue

        prediction = results[0]["label"]

        print(f"\nExample {index}")
        print(f"Response: {response}")
        print(f"Expected: {expected_label}")
        print(f"Predicted: {prediction}")

        print("NLI scores:")
        print(
            f"  Entailment: "
            f"{results[0]['entailment_score']:.4f}"
        )
        print(
            f"  Contradiction: "
            f"{results[0]['contradiction_score']:.4f}"
        )
        print(
            f"  Neutral: "
            f"{results[0]['neutral_score']:.4f}"
        )

        is_correct = prediction == expected_label
        print(f"Correct: {'Yes' if is_correct else 'No'}")

        total += 1
        correct += int(is_correct)

    accuracy = correct / total if total else 0.0

    print("\n" + "=" * 40)
    print(f"Correct predictions: {correct}/{total}")
    print(f"Accuracy: {accuracy:.2%}")


if __name__ == "__main__":
    evaluate()