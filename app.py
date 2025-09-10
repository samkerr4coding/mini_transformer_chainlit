# app.py
import chainlit as cl
import torch
from data import encode, decode, VOCAB_SIZE, MAX_LEN, idx2word
from model.transformer import Transformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Transformer(VOCAB_SIZE, VOCAB_SIZE, d_model=128, n_layers=4, n_heads=8, d_ff=512).to(device) # MATCH THE TRAINING ARCHITECTURE
model.load_state_dict(torch.load("mini_transformer.pt", map_location=device))
model.eval()
print("Model loaded successfully:", model)  # Debug: Check model loading

import torch
from data import PAD_IDX, SOS_IDX, EOS_IDX

def predict(model, src, src_mask, max_len=MAX_LEN, temperature=1.0): # Added temperature
    model.eval()
    src = src.to(device)
    memory = model.encoder(src, mask=src_mask)
    print("Encoder Memory:", memory) # Inspect the encoder output
    ys = torch.ones(1, 1).fill_(SOS_IDX).type(torch.long).to(device)

    for i in range(max_len - 1):
        tgt_mask = (ys != PAD_IDX).unsqueeze(-2).to(device)

        print(f"Iteration {i+1}:")
        print("src_mask shape:", src_mask.shape)
        print("src_mask:", src_mask)
        print("tgt_mask shape:", tgt_mask.shape)
        print("tgt_mask:", tgt_mask)
        print("ys shape:", ys.shape)
        print("ys:", ys)

        out = model.decoder(ys, memory, src_mask=src_mask, tgt_mask=tgt_mask)
        print("Decoder output shape:", out.shape)
        print("Decoder output:", out)

        out = model.out_linear(out[:, -1])
        print("Linear output shape:", out.shape)
        print("Linear output:", out)

        prob = torch.nn.functional.log_softmax(out / temperature, dim=-1) # Temperature scaling
        print("Probabilities:", prob)
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        print("Next word index:", next_word)  # Add this line
        print("Next word string:", idx2word[next_word]) # Add this line

        ys = torch.cat([ys, torch.ones(1, 1).fill_(next_word).type(torch.long).to(device)], dim=1)

        # TRACING ADDED HERE
        predicted_indices = [idx.item() for idx in ys[0]]
        predicted_words = [idx2word[idx] for idx in predicted_indices]
        print(f"Iteration {i+1} - Predicted indices: {predicted_indices}")
        print(f"Iteration {i+1} - Predicted words: {predicted_words}")

        if next_word == EOS_IDX:
            break
    return ys


@cl.on_message
async def main(message: cl.Message):
    input_phrase = message.content.strip()
    if not input_phrase:
        await cl.Message(content="Please enter a phrase.").send()
        return

    input_ids = encode(input_phrase)
    print("Encoded input:", input_ids)  # Debug: Check encoded input
    src = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)  # Add batch dimension
    print("Source tensor shape:", src.shape) # Debug: Check source tensor shape

    # Create the source mask
    src_mask = (src != PAD_IDX).unsqueeze(-2).to(device)

    with torch.no_grad():
        prediction = predict(model, src, src_mask, temperature=0.7) # Pass src_mask, adjust temperature
        print("Raw prediction:", prediction)  # Debug: Check raw prediction
        predicted_indices = [idx.item() for idx in prediction[0]]
        completion = decode(predicted_indices)

    if completion:
        # Split the completion into input and predicted parts
        input_length = len(input_phrase.split())
        completed_words = completion.split()
        input_words = completed_words[:input_length]
        predicted_words = completed_words[input_length:]

        # Color the input and predicted parts
        colored_input = f"<span style='color:red;'>{' '.join(input_words)}</span>"
        colored_completion = f"<span style='color:green;'>{' '.join(predicted_words)}</span>"

        # Combine the colored parts
        full_message = f"{colored_input} {colored_completion}"

        await cl.Message(
            content=full_message,
        ).send()
    else:
        await cl.Message(content="No completion found for that input.").send()
