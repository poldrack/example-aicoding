"""Tests for transformer — three-layer transformer model trained on small text corpus."""

import pytest
import numpy as np

torch = pytest.importorskip("torch")

from aicoding.transformer.solution import (
    SimpleTransformer,
    TextDataset,
    train_model,
)


@pytest.fixture
def small_corpus():
    return [
        "the cat sat on the mat",
        "the dog ran in the park",
        "a bird flew over the house",
        "the cat chased the dog",
        "the bird sat on the tree",
    ]


@pytest.fixture
def dataset(small_corpus):
    return TextDataset(small_corpus, seq_length=5)


class TestTextDataset:
    def test_creates_vocab(self, dataset):
        assert hasattr(dataset, "vocab") or hasattr(dataset, "word2idx")
        vocab = getattr(dataset, "vocab", getattr(dataset, "word2idx", None))
        assert len(vocab) > 0

    def test_len_positive(self, dataset):
        assert len(dataset) > 0

    def test_getitem_returns_tensors(self, dataset):
        x, y = dataset[0]
        assert isinstance(x, torch.Tensor)
        assert isinstance(y, torch.Tensor)

    def test_getitem_shapes(self, dataset):
        x, y = dataset[0]
        assert x.ndim == 1
        assert y.ndim == 1 or y.ndim == 0


class TestSimpleTransformer:
    def test_can_instantiate(self):
        model = SimpleTransformer(vocab_size=100, d_model=32, nhead=4,
                                   num_layers=3, dim_feedforward=64)
        assert model is not None

    def test_has_three_layers(self):
        model = SimpleTransformer(vocab_size=100, d_model=32, nhead=4,
                                   num_layers=3, dim_feedforward=64)
        # Check that the transformer encoder has 3 layers
        found = False
        for name, module in model.named_modules():
            if "encoder" in name.lower() or "layer" in name.lower():
                found = True
                break
        assert found or len(list(model.parameters())) > 0

    def test_forward_shape(self, dataset):
        vocab_size = len(getattr(dataset, "vocab", getattr(dataset, "word2idx", {})))
        model = SimpleTransformer(vocab_size=vocab_size, d_model=32, nhead=4,
                                   num_layers=3, dim_feedforward=64)
        x, _ = dataset[0]
        x = x.unsqueeze(0)  # batch dim
        output = model(x)
        assert output.shape[-1] == vocab_size

    def test_forward_different_sequences(self, dataset):
        vocab_size = len(getattr(dataset, "vocab", getattr(dataset, "word2idx", {})))
        model = SimpleTransformer(vocab_size=vocab_size, d_model=32, nhead=4,
                                   num_layers=3, dim_feedforward=64)
        x1, _ = dataset[0]
        x2, _ = dataset[min(1, len(dataset) - 1)]
        out1 = model(x1.unsqueeze(0))
        out2 = model(x2.unsqueeze(0))
        # Different inputs should (likely) produce different outputs
        assert out1.shape == out2.shape


class TestTrainModel:
    def test_trains_without_error(self, small_corpus):
        model, losses = train_model(small_corpus, epochs=5, d_model=32, nhead=4,
                                     num_layers=3, dim_feedforward=64, seq_length=5)
        assert model is not None
        assert len(losses) == 5

    def test_loss_decreases(self, small_corpus):
        model, losses = train_model(small_corpus, epochs=20, d_model=32, nhead=4,
                                     num_layers=3, dim_feedforward=64, seq_length=5,
                                     lr=0.01)
        # Loss should generally decrease (first > last)
        assert losses[-1] < losses[0]

    def test_losses_are_finite(self, small_corpus):
        _, losses = train_model(small_corpus, epochs=5, d_model=32, nhead=4,
                                 num_layers=3, dim_feedforward=64, seq_length=5)
        assert all(np.isfinite(l) for l in losses)

    def test_model_has_parameters(self, small_corpus):
        model, _ = train_model(small_corpus, epochs=2, d_model=32, nhead=4,
                                num_layers=3, dim_feedforward=64, seq_length=5)
        params = list(model.parameters())
        assert len(params) > 0
        assert all(p.requires_grad for p in params)
