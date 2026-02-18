"""Three-layer transformer model trained on a small text corpus.

Implements a simple transformer language model using PyTorch's
nn.TransformerEncoder with 3 encoder layers.
"""

import math

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class TextDataset(Dataset):
    """Simple word-level text dataset for language modeling.

    Parameters
    ----------
    texts : list of str
        List of text strings.
    seq_length : int
        Sequence length for input windows.
    """

    def __init__(self, texts, seq_length=10):
        self.seq_length = seq_length

        # Tokenize
        all_words = []
        for text in texts:
            all_words.extend(text.lower().split())

        # Build vocabulary
        unique_words = sorted(set(all_words))
        self.word2idx = {w: i + 1 for i, w in enumerate(unique_words)}
        self.word2idx["<pad>"] = 0
        self.idx2word = {i: w for w, i in self.word2idx.items()}
        self.vocab = self.word2idx

        # Encode all words
        self.encoded = [self.word2idx[w] for w in all_words]

    def __len__(self):
        return max(0, len(self.encoded) - self.seq_length)

    def __getitem__(self, idx):
        x = self.encoded[idx : idx + self.seq_length]
        y = self.encoded[idx + 1 : idx + self.seq_length + 1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class SimpleTransformer(nn.Module):
    """Three-layer transformer language model.

    Parameters
    ----------
    vocab_size : int
        Size of the vocabulary.
    d_model : int
        Embedding dimension.
    nhead : int
        Number of attention heads.
    num_layers : int
        Number of transformer encoder layers.
    dim_feedforward : int
        Feedforward dimension in the transformer.
    dropout : float
        Dropout rate.
    """

    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=3,
                 dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, vocab_size)

    def forward(self, src):
        """Forward pass.

        Parameters
        ----------
        src : Tensor of shape (batch, seq_len)
            Input token indices.

        Returns
        -------
        Tensor of shape (batch, seq_len, vocab_size)
            Logits for next token prediction.
        """
        # Causal mask
        seq_len = src.size(1)
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len)
        mask = mask.to(src.device)

        x = self.embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x, mask=mask)
        return self.output_proj(x)


def train_model(texts, epochs=50, d_model=128, nhead=4, num_layers=3,
                dim_feedforward=256, seq_length=10, lr=0.001, batch_size=16):
    """Train a transformer language model on a list of texts.

    Parameters
    ----------
    texts : list of str
        Training corpus.
    epochs : int
        Number of training epochs.
    d_model, nhead, num_layers, dim_feedforward : int
        Model hyperparameters.
    seq_length : int
        Input sequence length.
    lr : float
        Learning rate.
    batch_size : int
        Batch size.

    Returns
    -------
    model : SimpleTransformer
        Trained model.
    losses : list of float
        Training loss per epoch.
    """
    dataset = TextDataset(texts, seq_length=seq_length)
    vocab_size = len(dataset.word2idx)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    model = SimpleTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    losses = []
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        for x_batch, y_batch in loader:
            optimizer.zero_grad()
            output = model(x_batch)
            loss = criterion(output.view(-1, vocab_size), y_batch.view(-1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        losses.append(epoch_loss / max(n_batches, 1))

    return model, losses


if __name__ == "__main__":
    corpus = [
        "the cat sat on the mat",
        "the dog ran in the park",
        "a bird flew over the house",
        "the cat chased the dog around the yard",
        "the bird sat on the tree and sang a song",
        "the quick brown fox jumped over the lazy dog",
        "the sun was shining bright in the sky",
        "the children played in the garden all day",
    ]

    print("Training Three-Layer Transformer Model")
    print("=" * 50)

    model, losses = train_model(corpus, epochs=100, d_model=64, nhead=4,
                                 num_layers=3, dim_feedforward=128, seq_length=5)

    print(f"Final loss: {losses[-1]:.4f}")
    print(f"Loss reduction: {losses[0]:.4f} → {losses[-1]:.4f}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
