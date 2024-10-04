import torch
from transformers import LlamaForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import matplotlib.pyplot as plt
import logging
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_model_and_tokenizer(model_name: str, device: str):
    """Load the LLaMA model and tokenizer."""
    logger.info(f"Loading model: {model_name}")
    model = LlamaForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Setting pad_token to eos_token.")
    
    model.config.pad_token_id = tokenizer.pad_token_id
    model = model.to(device)
    return model, tokenizer

def prepare_data(dataset, tokenizer, max_length=128):
    """Prepare data for training, ensuring consistent tensor sizes."""
    texts = dataset['text']
    labels = dataset['source']
    prompts = [f"Classify the following text as human-written or AI-generated: {text}" for text in texts]
    
    # Tokenize prompts with padding and truncation
    encoded_inputs = tokenizer(prompts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    label_tensor = torch.tensor([1 if label == "human" else 0 for label in labels], dtype=torch.long)
    
    return encoded_inputs['input_ids'], encoded_inputs['attention_mask'], label_tensor

def train_model(model, train_loader, num_epochs, learning_rate, device):
    """Train the model with enhanced logging and higher epochs."""
    model.train()
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    loss_fn = CrossEntropyLoss(ignore_index=model.config.pad_token_id)
    train_losses = []

    for epoch in range(num_epochs):
        total_loss = 0
        logger.info(f"Starting Epoch {epoch+1}/{num_epochs}")

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            # Shift logits and labels for next token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()

            # Compute loss
            loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            train_losses.append(loss.item())
            logger.info(f"Batch loss: {loss.item()}")  # Log batch loss for tracking

        avg_train_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1} completed. Average Train Loss: {avg_train_loss:.4f}")

    plot_losses(train_losses)
    return model

def plot_losses(train_losses):
    """Plot training loss over time."""
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.legend()
    plt.savefig('loss_plot.png')
    logger.info("Loss plot saved as 'loss_plot.png'")

def evaluate_model(model, data_loader, device, tokenizer, dataset_name=""):
    """Evaluate model and compute metrics."""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits[:, -1, :]  # Get the last token's logits
            predicted_token_id = torch.argmax(logits, dim=-1)
            predicted_tokens = tokenizer.convert_ids_to_tokens(predicted_token_id)
            
            # Convert predictions to binary (1 for 'human', 0 for 'AI')
            predictions = [1 if 'human' in token.lower() else 0 for token in predicted_tokens]
            all_preds.extend(predictions)
            all_labels.extend(labels.cpu().numpy())

    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='binary', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='binary', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)
    
    logger.info(f"{dataset_name} Evaluation Results:")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    logger.info("\nConfusion Matrix:")
    logger.info(cm)

def save_model(model, tokenizer, save_path="fine_tuned_model"):
    """Save the fine-tuned model and tokenizer for later evaluation."""
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    logger.info(f"Model and tokenizer saved to {save_path}")

def main():
    # Parameters
    model_name = "meta-llama/Llama-3.2-1B"
    batch_size = 4
    num_epochs = 5  # Adjust as needed for resources
    learning_rate = 5e-6
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_name, device)

    # Load a larger dataset from Hugging Face
    logger.info("Loading dataset from Hugging Face...")
    dataset = load_dataset("artem9k/ai-text-detection-pile", split="train[:2%]")  # Load a subset (10% for example)
    logger.info(f"Loaded dataset with {len(dataset)} samples.")

    # Prepare data tensors
    input_ids, attention_mask, label_tensor = prepare_data(dataset, tokenizer)

    # Create dataset and split into train and validation sets
    full_dataset = TensorDataset(input_ids, attention_mask, label_tensor)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Train model
    trained_model = train_model(model, train_loader, num_epochs, learning_rate, device)

    # Evaluate on training data
    logger.info("Evaluating model on training data to check for learning...")
    evaluate_model(trained_model, train_loader, device, tokenizer, dataset_name="Training Set")

    # Evaluate on validation data
    logger.info("Evaluating model on validation data...")
    evaluate_model(trained_model, val_loader, device, tokenizer, dataset_name="Validation Set")

    # Save model for later evaluation
    save_model(trained_model, tokenizer)

if __name__ == "__main__":
    main()
