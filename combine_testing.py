import re
import torch
import torch.nn.functional as F
# =========================================================
# LOAD BOTH MODELS
# =========================================================
# eng -> tel
eng2tel_ckpt = torch.load(
    "eng2tel_lstm_full.pt",
    map_location="cpu",
    weights_only=False
)
# tel -> eng
tel2eng_ckpt = torch.load(
    "tel2eng_lstm_full.pt",
    map_location="cpu",
    weights_only=False
)
PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# =========================================================
# MODEL CLASSES
# =========================================================
class Attention(torch.nn.Module):
    def __init__(self, enc_hidden_dim, dec_hidden_dim):
        super().__init__()
        self.attn = torch.nn.Linear(
            enc_hidden_dim * 2 + dec_hidden_dim,
            dec_hidden_dim
        )
        self.v = torch.nn.Linear(dec_hidden_dim, 1, bias=False)
    def forward(self, hidden, encoder_outputs):
        src_len = encoder_outputs.shape[1]
        hidden = hidden.permute(1, 0, 2).repeat(1, src_len, 1)
        energy = torch.tanh(
            self.attn(torch.cat((hidden, encoder_outputs), dim=2))
        )
        return torch.softmax(self.v(energy).squeeze(2), dim=1)
class Encoder(torch.nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, dropout=0.5):
        super().__init__()
        self.embedding = torch.nn.Embedding(
            input_dim,
            emb_dim,
            padding_idx=PAD_IDX
        )
        self.lstm = torch.nn.LSTM(
            emb_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True
        )
        self.fc_hidden = torch.nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc_cell = torch.nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = torch.nn.Dropout(dropout)
    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        outputs, (hidden, cell) = self.lstm(embedded)
        hidden = torch.tanh(
            self.fc_hidden(
                torch.cat((hidden[-2], hidden[-1]), dim=1)
            )
        ).unsqueeze(0)
        cell = torch.tanh(
            self.fc_cell(
                torch.cat((cell[-2], cell[-1]), dim=1)
            )
        ).unsqueeze(0)
        return outputs, hidden, cell
class Decoder(torch.nn.Module):
    def __init__(
        self,
        output_dim,
        emb_dim,
        enc_hidden_dim,
        dec_hidden_dim,
        dropout=0.5
    ):
        super().__init__()
        self.embedding = torch.nn.Embedding(
            output_dim,
            emb_dim,
            padding_idx=PAD_IDX
        )
        self.attention = Attention(enc_hidden_dim, dec_hidden_dim)
        self.lstm = torch.nn.LSTM(
            emb_dim + enc_hidden_dim * 2,
            dec_hidden_dim,
            batch_first=True
        )
        self.fc = torch.nn.Linear(
            dec_hidden_dim + enc_hidden_dim * 2 + emb_dim,
            output_dim
        )
        self.dropout = torch.nn.Dropout(dropout)
    def forward(self, x, hidden, cell, encoder_outputs):
        embedded = self.dropout(self.embedding(x.unsqueeze(1)))
        attn_weights = self.attention(hidden, encoder_outputs)
        context = torch.bmm(
            attn_weights.unsqueeze(1),
            encoder_outputs
        )
        lstm_input = torch.cat((embedded, context), dim=2)
        output, (hidden, cell) = self.lstm(
            lstm_input,
            (hidden, cell)
        )
        prediction = self.fc(
            torch.cat((output, context, embedded), dim=2).squeeze(1)
        )
        return prediction, hidden, cell
class Seq2Seq(torch.nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
# =========================================================
# BUILD MODELS
# =========================================================
def build_model(ckpt, src_vocab, trg_vocab):
    cfg = ckpt["config"]
    encoder = Encoder(
        len(src_vocab),
        cfg["enc_emb_dim"],
        cfg["enc_hid_dim"],
        cfg["dropout"]
    )
    decoder = Decoder(
        len(trg_vocab),
        cfg["dec_emb_dim"],
        cfg["enc_hid_dim"],
        cfg["dec_hid_dim"],
        cfg["dropout"]
    )
    model = Seq2Seq(encoder, decoder, device).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model
eng2tel_model = build_model(
    eng2tel_ckpt,
    eng2tel_ckpt["eng_char2idx"],
    eng2tel_ckpt["tel_char2idx"]
)
tel2eng_model = build_model(
    tel2eng_ckpt,
    tel2eng_ckpt["tel_char2idx"],
    tel2eng_ckpt["eng_char2idx"]
)
# =========================================================
# HELPERS
# =========================================================
def encode(word, char2idx, max_len):
    seq = [SOS_IDX]
    for c in word:
        seq.append(char2idx.get(c, UNK_IDX))
    seq.append(EOS_IDX)
    seq = seq[:max_len] 
    seq += [PAD_IDX] * (max_len - len(seq))
    return seq
def beam_predict(
    word,
    model,
    src_vocab,
    trg_idx2char,
    max_src_len,
    beam_size=5,
    max_len=30
):
    src = torch.tensor([
        encode(word, src_vocab, max_src_len)
    ]).to(device)
    with torch.no_grad():
        enc_out, hidden, cell = model.encoder(src)
        beams = [(0.0, [SOS_IDX], hidden, cell)]
        completed = []
        for _ in range(max_len):
            new_beams = []
            for score, seq, h, c in beams:
                dec_input = torch.tensor([seq[-1]]).to(device)
                out, h2, c2 = model.decoder(
                    dec_input,
                    h,
                    c,
                    enc_out
                )
                log_probs = F.log_softmax(out, dim=-1).squeeze(0)
                topk_vals, topk_idxs = log_probs.topk(beam_size)
                for log_p, idx in zip(
                    topk_vals.tolist(),
                    topk_idxs.tolist()
                ):
                    new_seq = seq + [idx]
                    new_score = score + log_p
                    if idx == EOS_IDX:
                        completed.append(
                            (new_score / len(new_seq), new_seq)
                        )
                    else:
                        new_beams.append(
                            (new_score, new_seq, h2, c2)
                        )
            if not new_beams:
                break
            beams = sorted(
                new_beams,
                key=lambda x: x[0] / len(x[1]),
                reverse=True
            )[:beam_size]
        if not completed:
            completed = [
                (s / len(seq), seq)
                for s, seq, *_ in beams
            ]
        best_seq = max(completed, key=lambda x: x[0])[1]
        return ''.join(
            trg_idx2char.get(i, '')
            for i in best_seq
            if i not in (PAD_IDX, SOS_IDX, EOS_IDX)
        )
# =========================================================
# LANGUAGE DETECTION
# =========================================================
def is_telugu(text):
    for ch in text:
        if '\u0C00' <= ch <= '\u0C7F':
            return True
    return False
# =========================================================
# MAIN SMART TRANSLITERATION
# =========================================================
def transliterate_mixed(sentence):
    words = sentence.split()
    outputs = []
    for word in words:
        # Telugu -> English
        if is_telugu(word):
            result = beam_predict(
                word,
                tel2eng_model,
                tel2eng_ckpt["tel_char2idx"],
                {
                    int(k): v
                    for k, v in tel2eng_ckpt["eng_idx2char"].items()
                },
                tel2eng_ckpt["config"]["max_src_len"]
            )
        # English -> Telugu
        else:
            result = beam_predict(
                word.lower(),
                eng2tel_model,
                eng2tel_ckpt["eng_char2idx"],
                {
                    int(k): v
                    for k, v in eng2tel_ckpt["tel_idx2char"].items()
                },
                eng2tel_ckpt["config"]["max_src_len"]
            )
        outputs.append(result)
    return '  '.join(outputs)
# =========================================================
# DEMO
# =========================================================
while True:
    text = input("\nInput: ")
    if text.lower() == "exit":
        break
    output = transliterate_mixed(text)
    print("Output:", output)