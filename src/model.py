# src/model.py
from transformers import LlamaForCausalLM, AutoTokenizer
import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model_and_tokenizer(model_name: str, device: str):
    """Load the LLaMA model and tokenizer."""
    logger.info(f"Loading model: {model_name}")
    model = LlamaForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Setting pad_token to eos_token.")
    
    model.config.pad_token_id = tokenizer.pad_token_id
    model = model.to(device)
    return model, tokenizer
