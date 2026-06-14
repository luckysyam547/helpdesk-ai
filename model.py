import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers=1, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers,
                           batch_first=True,
                           dropout=(0 if num_layers == 1 else dropout))

    def forward(self, input_seq):
        embedded = self.embedding(input_seq)
        outputs, hidden = self.gru(embedded)
        return outputs, hidden

class LuongAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, decoder_hidden, encoder_outputs):
        query = decoder_hidden[-1].unsqueeze(2)
        keys = self.attn(encoder_outputs)
        scores = torch.bmm(keys, query).squeeze(2)
        attn_weights = F.softmax(scores, dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
        return context, attn_weights

class AttnDecoderRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers=1, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.attention = LuongAttention(hidden_size)
        self.gru = nn.GRU(hidden_size * 2, hidden_size, num_layers,
                           batch_first=True,
                           dropout=(0 if num_layers == 1 else dropout))
        self.out = nn.Linear(hidden_size, vocab_size)

    def forward_step(self, input_token, hidden, encoder_outputs):
        embedded = self.dropout(self.embedding(input_token))
        context, attn_weights = self.attention(hidden, encoder_outputs)
        gru_input = torch.cat((embedded, context), dim=2)
        output, hidden = self.gru(gru_input, hidden)
        output = self.out(output.squeeze(1))
        return output, hidden, attn_weights

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None,
                max_length=13, sos_token=1, teacher_forcing_ratio=0.5):
        batch_size = encoder_outputs.size(0)
        device = encoder_outputs.device
        decoder_input = torch.full((batch_size, 1), sos_token, dtype=torch.long, device=device)
        decoder_hidden = encoder_hidden
        decoder_outputs = []
        seq_len = target_tensor.size(1) if target_tensor is not None else max_length

        for t in range(seq_len):
            decoder_output, decoder_hidden, _ = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_outputs.append(decoder_output)
            use_teacher_forcing = (
                target_tensor is not None and
                torch.rand(1).item() < teacher_forcing_ratio)
            if use_teacher_forcing:
                decoder_input = target_tensor[:, t].unsqueeze(1)
            else:
                _, topi = decoder_output.topk(1)
                decoder_input = topi.detach()

        decoder_outputs = torch.stack(decoder_outputs, dim=1)
        return decoder_outputs
