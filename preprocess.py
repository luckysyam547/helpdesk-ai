import re
import torch

PAD_token = 0
SOS_token = 1
EOS_token = 2
UNK_token = 3
MAX_LENGTH = 12

def clean_text(text):
    text = text.lower().strip()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

class Vocabulary:
    def __init__(self, name):
        self.name = name
        self.word2index = {"<PAD>": PAD_token, "<SOS>": SOS_token,
                            "<EOS>": EOS_token, "<UNK>": UNK_token}
        self.word2count = {}
        self.index2word = {v: k for k, v in self.word2index.items()}
        self.num_words = 4

    def add_sentence(self, sentence):
        for word in sentence.split(" "):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

def filter_pair(pair, max_length=MAX_LENGTH):
    return len(pair[0].split(" ")) < max_length and len(pair[1].split(" ")) < max_length

def load_data(pairs_raw, max_length=MAX_LENGTH):
    pairs = []
    for q, a in pairs_raw:
        q_clean, a_clean = clean_text(q), clean_text(a)
        if q_clean and a_clean:
            pairs.append((q_clean, a_clean))
    pairs = [p for p in pairs if filter_pair(p, max_length)]
    vocab = Vocabulary("helpdesk")
    for q, a in pairs:
        vocab.add_sentence(q)
        vocab.add_sentence(a)
    return pairs, vocab

def sentence_to_indexes(vocab, sentence, max_length=MAX_LENGTH):
    indexes = [vocab.word2index.get(w, UNK_token) for w in sentence.split(" ")]
    indexes.append(EOS_token)
    indexes += [PAD_token] * (max_length + 1 - len(indexes))
    return indexes[:max_length + 1]

def pairs_to_tensors(pairs, vocab, max_length=MAX_LENGTH, device="cpu"):
    input_seqs, target_seqs = [], []
    for q, a in pairs:
        input_seqs.append(sentence_to_indexes(vocab, q, max_length))
        target_seqs.append(sentence_to_indexes(vocab, a, max_length))
    input_tensor = torch.tensor(input_seqs, dtype=torch.long, device=device)
    target_tensor = torch.tensor(target_seqs, dtype=torch.long, device=device)
    return input_tensor, target_tensor
