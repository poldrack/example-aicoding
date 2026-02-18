# transformer (#12)

Three-layer transformer model trained on a small text corpus.

## Approach

- Uses PyTorch's `nn.TransformerEncoder` with 3 encoder layers.
- Sinusoidal positional encoding and causal attention mask for autoregressive language modeling.
- Word-level tokenization with a simple vocabulary built from the training corpus.
- `TextDataset` produces sliding windows of `seq_length` tokens for next-token prediction.
- Trained with cross-entropy loss and Adam optimizer.
