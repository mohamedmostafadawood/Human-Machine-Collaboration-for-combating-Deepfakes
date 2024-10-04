# examples/train_model.py
import torch
from transformers import LlamaForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
import argparse
from src.model import load_model_and_tokenizer
from src.train import train_model
from src.utils import prepare_data, save_model

def main(args):
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_name, device)

    # Load and prepare dataset
    dataset = load_dataset(args.dataset_name, split=args.dataset_split)
    input_ids, attention_mask, labels = prepare_data(dataset, tokenizer)
    
    # Split into training and validation sets
    full_dataset = TensorDataset(input_ids, attention_mask, labels)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    # Train the model
    trained_model = train_model(model, train_loader, args.num_epochs, args.learning_rate, device)

    # Save model
    save_model(trained_model, tokenizer, save_path=args.save_path)
    print(f"Model saved to {args.save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and save the LLaMA model for text detection.")
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B", help="Name of the model to use")
    parser.add_argument("--dataset_name", type=str, default="artem9k/ai-text-detection-pile", help="Hugging Face dataset name")
    parser.add_argument("--dataset_split", type=str, default="train[:10%]", help="Dataset split to use")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="Learning rate for training")
    parser.add_argument("--save_path", type=str, default="models/fine_tuned_model", help="Path to save the fine-tuned model")
    
    args = parser.parse_args()
    main(args)
