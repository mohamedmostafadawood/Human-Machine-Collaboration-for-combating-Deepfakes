# src/train.py
import torch
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_model(model, train_loader, num_epochs, learning_rate, device):
    """Train the model and log loss over epochs."""
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

            # Shift logits and labels for the next token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            
            # Calculate loss
            loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            train_losses.append(loss.item())

        avg_train_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1} completed. Average Train Loss: {avg_train_loss:.4f}")

    plot_losses(train_losses)
    return model

def plot_losses(train_losses):
    """Plot training loss over time and save it as a PNG."""
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.legend()
    plt.savefig('results/loss_plot.png')
    logger.info("Loss plot saved as 'results/loss_plot.png'")
