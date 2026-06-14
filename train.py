import torch
import torch.nn as nn
from data import PAIRS
from preprocess import load_data, pairs_to_tensors, MAX_LENGTH, SOS_token, PAD_token
from model import EncoderRNN, AttnDecoderRNN

HIDDEN_SIZE = 128
NUM_LAYERS = 1
DROPOUT = 0.1
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
TEACHER_FORCING_RATIO = 0.9
BATCH_SIZE = 16

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    pairs, vocab = load_data(PAIRS, MAX_LENGTH)
    print(f"Pairs: {len(pairs)}, Vocab size: {vocab.num_words}")

    input_tensor, target_tensor = pairs_to_tensors(pairs, vocab, MAX_LENGTH, device)
    dataset = torch.utils.data.TensorDataset(input_tensor, target_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    encoder = EncoderRNN(vocab.num_words, HIDDEN_SIZE, NUM_LAYERS, DROPOUT).to(device)
    decoder = AttnDecoderRNN(vocab.num_words, HIDDEN_SIZE, NUM_LAYERS, DROPOUT).to(device)

    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=LEARNING_RATE)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_token)

    for epoch in range(1, NUM_EPOCHS + 1):
        total_loss = 0
        for input_batch, target_batch in loader:
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            encoder_outputs, encoder_hidden = encoder(input_batch)
            decoder_outputs = decoder(
                encoder_outputs, encoder_hidden,
                target_tensor=target_batch,
                max_length=target_batch.size(1),
                sos_token=SOS_token,
                teacher_forcing_ratio=TEACHER_FORCING_RATIO,
            )

            loss = criterion(
                decoder_outputs.reshape(-1, decoder_outputs.size(-1)),
                target_batch.reshape(-1),
            )
            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{NUM_EPOCHS}  Loss: {avg_loss:.4f}")

    torch.save({
        "encoder_state_dict": encoder.state_dict(),
        "decoder_state_dict": decoder.state_dict(),
        "vocab_word2index": vocab.word2index,
        "vocab_index2word": vocab.index2word,
        "vocab_num_words": vocab.num_words,
        "hidden_size": HIDDEN_SIZE,
        "num_layers": NUM_LAYERS,
        "max_length": MAX_LENGTH,
    }, "chatbot_model.pt")
    print("Done! Model saved as chatbot_model.pt")

if __name__ == "__main__":
    train()
