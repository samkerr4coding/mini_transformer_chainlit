import torch.nn as nn
import torch.optim as optim
from data import get_batches, VOCAB_SIZE, PAD_IDX
from model.transformer import Transformer
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_masks(src, tgt):
    # Create a mask for the source tensor, masking padding tokens
    src_mask = (src != PAD_IDX).unsqueeze(-2).bool()
    # Create a mask for the target tensor, masking padding tokens
    tgt_mask = (tgt != PAD_IDX).unsqueeze(-2).bool()

    # Get the sequence length of the target tensor
    seq_len = tgt.size(1)
    # Create a lower triangular matrix (nopeak mask) to prevent the model from attending to future tokens
    nopeak_mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).bool()
    # Combine the target mask with the nopeak mask
    tgt_mask = tgt_mask & nopeak_mask.to(device)

    # Return the source and target masks
    return src_mask, tgt_mask

def train():
    # Get batches of source and target data
    src, tgt = get_batches()
    # Move the source and target data to the device (GPU if available, otherwise CPU)
    src, tgt = src.to(device), tgt.to(device)

    # Initialize the Transformer model
    model = Transformer(VOCAB_SIZE, VOCAB_SIZE, d_model=128, n_layers=4, n_heads=8, d_ff=512).to(device) # Increased model size
    # Initialize the Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.0005, betas=(0.9, 0.98), eps=1e-9) # Adjusted learning rate
    # Initialize the CrossEntropyLoss criterion, ignoring padding tokens
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    # Set the number of training epochs
    epochs = 1000  # Increased epochs
    # Set the model to training mode
    model.train()

    # Initialize the best loss to infinity
    best_loss = float('inf')  # Initialize best loss
    # Set the patience for early stopping
    patience = 100  # Number of epochs to wait for improvement # Increased patience
    # Initialize the trigger for early stopping
    trigger = 0 #counter for patience

    # Training loop
    for epoch in range(epochs):
        # Zero the gradients
        optimizer.zero_grad()

        # Prepare data and masks
        # Create the target input by removing the last token from the target tensor
        tgt_input = tgt[:, :-1]
        # Create the target output by removing the first token from the target tensor
        tgt_output = tgt[:, 1:]

        # Create the source and target masks
        src_mask, tgt_mask = create_masks(src, tgt_input)

        # Forward pass
        # Pass the source, target input, and masks to the model
        output = model(src, tgt_input, src_mask, tgt_mask)

        # Compute loss
        # Transpose the output for CrossEntropyLoss
        output = output.transpose(1, 2)  # Transpose for CrossEntropyLoss
        # Calculate the loss between the model output and the target output
        loss = criterion(output, tgt_output)

        # Backpropagation
        # Compute the gradients
        loss.backward()

        # Gradient Clipping
        # Clip the gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # Clip gradients, adjusted value

        # Update the parameters
        optimizer.step()

        # Print the loss every 50 epochs
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

        # Early Stopping
        # If the loss is less than the best loss, update the best loss and reset the trigger
        if loss.item() < best_loss:
            best_loss = loss.item()
            trigger = 0
        # Otherwise, increment the trigger
        else:
            trigger += 1

            # If the trigger is greater than or equal to the patience, trigger early stopping
            if trigger >= patience:
                print("Early stopping triggered!")
                break

    # Save the model
    torch.save(model.state_dict(), "mini_transformer.pt")
    # Print a message indicating that training is complete and the model has been saved
    print("Training complete, model saved as mini_transformer.pt")

# Main function
if __name__ == "__main__":
    # Train the model
    train()
