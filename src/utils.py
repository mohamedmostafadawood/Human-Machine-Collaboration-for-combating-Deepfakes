# src/utils.py
import torch
from transformers import AutoTokenizer
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def prepare_data(dataset, tokenizer, max_length=128):
    """Prepare data by tokenizing and creating input tensors."""
    texts = dataset['text']
    labels = dataset['source']
    prompts = [f"Classify the following text as human-written or AI-generated: {text}" for text in texts]
    
    # Tokenize prompts with padding and truncation
    encoded_inputs = tokenizer(prompts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    label_tensor = torch.tensor([1 if label == "human" else 0 for label in labels], dtype=torch.long)
    
    return encoded_inputs['input_ids'], encoded_inputs['attention_mask'], label_tensor

def save_model(model, tokenizer, save_path="models/fine_tuned_model"):
    """Save the fine-tuned model and tokenizer for later evaluation."""
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    logger.info(f"Model and tokenizer saved to {save_path}")
