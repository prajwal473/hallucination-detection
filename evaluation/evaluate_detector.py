import json
import argparse
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from src.detector import detect


DEFAULT_DATASET_PATH = Path("evaluation/smoke_dataset.json")

LABELS = ["supported", "contradicted", "uncertain"]


def evaluate(dataset_path: Path):
    with dataset_path.open("r", encoding="utf-8") as file:
        dataset = json.load(file)

    expected_labels = []
    predicted_labels = []

    print("\nHallucination Detection Evaluation")
    print("=" * 45)

    for index, sample in enumerate(dataset, start=1):
        results = detect(
            response_text=sample["response"],
            evidence_store=sample["evidence"],
        )

        if not results:
            print(f"\nExample {index}: No claims detected")
            continue

        expected = sample["expected_label"]
        predicted = results[0]["label"]

        expected_labels.append(expected)
        predicted_labels.append(predicted)

        print(f"\nExample {index}")
        print(f"Response: {sample['response']}")
        print(f"Expected: {expected}")
        print(f"Predicted: {predicted}")

        if predicted != expected:
            print("Result: INCORRECT")
        else:
            print("Result: CORRECT")

    if not expected_labels:
        print("\nNo examples were evaluated.")
        return

    print("\n" + "=" * 45)
    print(f"Evaluated examples: {len(expected_labels)}")
    print(f"Accuracy: {accuracy_score(expected_labels, predicted_labels):.2%}")

    print("\nClassification Report")
    print(
        classification_report(
            expected_labels,
            predicted_labels,
            labels=LABELS,
            zero_division=0,
            digits=3,
        )
    )

    print("Confusion Matrix")
    print("Rows = actual labels; columns = predicted labels")
    print(f"Label order: {LABELS}")
    print(confusion_matrix(
        expected_labels,
        predicted_labels,
        labels=LABELS,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate the hallucination detector."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Path to the evaluation JSON dataset.",
    )
    args = parser.parse_args()
    evaluate(args.dataset)