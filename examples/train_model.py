# examples/train_model.py
import yaml
import torch
from transformers import LlamaForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset, random_split
from src.model import load_model_and_tokenizer
from src.train import train_model
from src.utils import prepare_data, save_model

def main(config_path):
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model and tokenizer
    model_name = config['model']['name']
    save_path = config['model']['save_path']
    model, tokenizer = load_model_and_tokenizer(model_name, device)

    # Load and prepare dataset
    dataset_name = config['dataset']['name']
    dataset_split = config['dataset']['split']
    dataset = load_dataset(dataset_name, split=dataset_split)
    input_ids, attention_mask, labels = prepare_data(dataset, tokenizer, config['dataset']['max_length'])
    
    # Split into training and validation sets
    full_dataset = TensorDataset(input_ids, attention_mask, labels)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Create dataloaders
    batch_size = config['training']['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Train the model
    num_epochs = config['training']['num_epochs']
    learning_rate = config['training']['learning_rate']
    trained_model = train_model(model, train_loader, num_epochs, learning_rate, device)

    # Save model
    save_model(trained_model, tokenizer, save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train and save the LLaMA model for text detection.")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml", help="Path to the training configuration file")
    args = parser.parse_args()
    main(args.config)
