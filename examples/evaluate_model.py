# examples/evaluate_model.py
import yaml
import torch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, TensorDataset
from src.model import load_model_and_tokenizer
from src.evaluate import evaluate_model
from src.utils import prepare_data

def main(config_path):
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model and tokenizer
    model_name = config['model']['name']
    load_path = config['model']['load_path']
    model, tokenizer = load_model_and_tokenizer(model_name, device)
    model.load_state_dict(torch.load(load_path))
    
    # Load and prepare dataset for evaluation
    dataset_name = config['dataset']['name']
    dataset_split = config['dataset']['split']
    dataset = load_dataset(dataset_name, split=dataset_split)
    input_ids, attention_mask, labels = prepare_data(dataset, tokenizer, config['dataset']['max_length'])
    eval_dataset = TensorDataset(input_ids, attention_mask, labels)
    eval_loader = DataLoader(eval_dataset, batch_size=config['evaluation']['batch_size'])
    
    # Evaluate model
    accuracy, precision, recall, f1 = evaluate_model(model, eval_loader, device, tokenizer)
    print(f"Evaluation Results:\n  Accuracy: {accuracy:.4f}\n  Precision: {precision:.4f}\n  Recall: {recall:.4f}\n  F1 Score: {f1:.4f}")
    
    # Optional: Save results to file
    metrics_output = config['evaluation'].get('metrics_output')
    if metrics_output:
        with open(metrics_output, 'w') as f:
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall: {recall:.4f}\n")
            f.write(f"F1 Score: {f1:.4f}\n")
        print(f"Metrics saved to {metrics_output}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned LLaMA model on text detection.")
    parser.add_argument("--config", type=str, default="configs/eval_config.yaml", help="Path to the evaluation configuration file")
    args = parser.parse_args()
    main(args.config)
