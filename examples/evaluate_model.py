# examples/evaluate_model.py
import torch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, TensorDataset
import argparse
from src.model import load_model_and_tokenizer
from eval.evaluate import evaluate_model
from src.utils import prepare_data

def main(args):
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_name, device)
    model.load_state_dict(torch.load(args.model_path))
    
    # Load and prepare dataset for evaluation
    dataset = load_dataset(args.dataset_name, split=args.dataset_split)
    input_ids, attention_mask, labels = prepare_data(dataset, tokenizer)
    eval_dataset = TensorDataset(input_ids, attention_mask, labels)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size)
    
    # Evaluate model
    accuracy, precision, recall, f1 = evaluate_model(model, eval_loader, device, tokenizer)
    print(f"Evaluation Results:\n  Accuracy: {accuracy:.4f}\n  Precision: {precision:.4f}\n  Recall: {recall:.4f}\n  F1 Score: {f1:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned LLaMA model on text detection.")
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B", help="Name of the model to use")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the saved fine-tuned model")
    parser.add_argument("--dataset_name", type=str, default="artem9k/ai-text-detection-pile", help="Hugging Face dataset name")
    parser.add_argument("--dataset_split", type=str, default="train[:10%]", help="Dataset split to use")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for evaluation")
    
    args = parser.parse_args()
    main(args)
