import datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from helpers import prepare_dataset_nli
import torch
import os
import json

from sklearn.metrics import accuracy_score

def test_on_hans(model, tokenizer, output_dir):
    """Test the model on the HANS dataset and calculate accuracy."""
    print("Loading HANS dataset...")
    hans_dataset = datasets.load_dataset('hans', trust_remote_code=True)

    def preprocess_hans(examples):
        """Preprocess HANS examples for evaluation."""
        return prepare_dataset_nli(examples, tokenizer, max_seq_length=128)

    print("Preprocessing HANS dataset...")
    hans_eval_dataset = hans_dataset['validation'].map(
        preprocess_hans,
        batched=True,
        num_proc=2,
        remove_columns=hans_dataset['validation'].column_names
    )

    def compute_hans_metrics(eval_preds):
        """Compute accuracy for the HANS dataset."""
        logits, labels = eval_preds
        predictions = logits.argmax(axis=-1)
        return {'accuracy': accuracy_score(labels, predictions)}

    print("Evaluating on HANS...")
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        compute_metrics=compute_hans_metrics,  # Add custom metrics
    )
    hans_results = trainer.evaluate(hans_eval_dataset)

    print("HANS Evaluation Results:")
    print(hans_results)

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'hans_eval_metrics.json'), encoding='utf-8', mode='w') as f:
        json.dump(hans_results, f)

    return hans_results


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please ensure a compatible GPU and drivers are installed.")
    print("CUDA is available. Using device:", torch.cuda.get_device_name(0))

    # Ensure all operations use the GPU
    device = torch.device("cuda")

    # Load model and tokenizer (adjust as needed)
    model_name_or_path = "./anli_model"  # Replace with your model path if needed
    output_dir = "./output"  # Directory for evaluation outputs

    print("Loading model and tokenizer...")
    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, num_labels=3)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

    # Test on HANS
    hans_results = test_on_hans(model, tokenizer, output_dir)
    print("model is ", model_name_or_path)
    print("HANS evaluation completed. Results saved to:", output_dir)

if __name__ == "__main__":
    main()
