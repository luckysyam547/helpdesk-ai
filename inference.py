import torch
from preprocess import clean_text, MAX_LENGTH, SOS_token, EOS_token, UNK_token
from model import EncoderRNN, AttnDecoderRNN

def load_model(path="chatbot_model.pt", device="cpu"):
    checkpoint = torch.load(path, map_location=device)
    vocab_num_words = checkpoint["vocab_num_words"]
    hidden_size = checkpoint["hidden_size"]
    num_layers = checkpoint["num_layers"]

    encoder = EncoderRNN(vocab_num_words, hidden_size, num_layers).to(device)
    decoder = AttnDecoderRNN(vocab_num_words, hidden_size, num_layers).to(device)

    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    decoder.load_state_dict(checkpoint["decoder_state_dict"])
    encoder.eval()
    decoder.eval()

    word2index = checkpoint["vocab_word2index"]
    index2word = checkpoint["vocab_index2word"]
    max_length = checkpoint["max_length"]
    return encoder, decoder, word2index, index2word, max_length, device

@torch.no_grad()
def generate_response(text, encoder, decoder, word2index, index2word, max_length, device):
    cleaned = clean_text(text)
    indexes = [word2index.get(w, UNK_token) for w in cleaned.split(" ")]
    indexes.append(EOS_token)
    indexes += [0] * (max_length + 1 - len(indexes))
    indexes = indexes[:max_length + 1]

    input_tensor = torch.tensor([indexes], dtype=torch.long, device=device)
    encoder_outputs, encoder_hidden = encoder(input_tensor)
    decoder_outputs = decoder(
        encoder_outputs, encoder_hidden,
        target_tensor=None,
        max_length=max_length + 1,
        sos_token=SOS_token,
        teacher_forcing_ratio=0.0,
    )

    _, topi = decoder_outputs.topk(1, dim=-1)
    predicted_indices = topi.squeeze(-1).squeeze(0).tolist()

    words = []
    for idx in predicted_indices:
        if idx == EOS_token:
            break
        if idx == 0:
            continue
        words.append(index2word.get(idx, "<UNK>"))

    if not words:
        return "I am sorry I did not understand that. Could you rephrase?"
    return " ".join(words)

def chat_loop():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, decoder, word2index, index2word, max_length, device = load_model(
        "chatbot_model.pt", device)
    print("HelpDesk AI ready. Type quit to exit.\n")
    while True:
        text = input("You: ")
        if text.strip().lower() == "quit":
            break
        response = generate_response(
            text, encoder, decoder, word2index, index2word, max_length, device)
        print(f"Bot: {response}\n")

if __name__ == "__main__":
    chat_loop()
