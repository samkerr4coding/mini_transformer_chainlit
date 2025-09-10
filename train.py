import torch.nn as nn
import torch.optim as optim
from data import get_batches, VOCAB_SIZE, PAD_IDX
from model.transformer import Transformer
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_masks(src, tgt):
    src_mask = (src != PAD_IDX).unsqueeze(-2).bool()
    tgt_mask = (tgt != PAD_IDX).unsqueeze(-2).bool()

    seq_len = tgt.size(1)
    nopeak_mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).bool()
    tgt_mask = tgt_mask & nopeak_mask.to(device)

    return src_mask, tgt_mask



def train():
    src, tgt = get_batches()
    src, tgt = src.to(device), tgt.to(device)

    model = Transformer(VOCAB_SIZE, VOCAB_SIZE, d_model=128, n_layers=4, n_heads=8, d_ff=512).to(device) # Increased model size
    optimizer = optim.Adam(model.parameters(), lr=0.0005, betas=(0.9, 0.98), eps=1e-9) # Adjusted learning rate
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    epochs = 1000  # Increased epochs
    model.train()

    best_loss = float('inf')  # Initialize best loss
    patience = 100  # Number of epochs to wait for improvement # Increased patience
    trigger = 0 #counter for patience

    for epoch in range(epochs):
        optimizer.zero_grad()

        # Prepare data and masks
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        src_mask, tgt_mask = create_masks(src, tgt_input)

        # Forward pass
        output = model(src, tgt_input, src_mask, tgt_mask)

        # Compute loss
        output = output.transpose(1, 2)  # Transpose for CrossEntropyLoss
        loss = criterion(output, tgt_output)

        # Backpropagation
        loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # Clip gradients, adjusted value

        optimizer.step()

        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

        # Early Stopping
        if loss.item() < best_loss:
            best_loss = loss.item()
            trigger = 0
        else:
            trigger += 1

            if trigger >= patience:
                print("Early stopping triggered!")
                break

    torch.save(model.state_dict(), "mini_transformer.pt")
    print("Training complete, model saved as mini_transformer.pt")

if __name__ == "__main__":
    train()