"""Tests for the embedder. The slow group loads the real model once; the fast group checks
the test helpers (vector_at, FakeEmbedder) that later tests rely on instead of the model."""

import numpy as np
import pytest
from conftest import BASE, FakeEmbedder, HashEmbedder, vector_at

from gateway.config import Settings


@pytest.fixture(scope="module")
def emb():
    # Imported here so `-m "not slow"` never imports torch.
    from gateway.embedder import Embedder

    # Module scope: the model loads once for this whole file, not once per test.
    return Embedder(Settings(_env_file=None).embed_model)


@pytest.mark.slow
def test_vector_shape_dtype_and_unit_norm(emb):
    v = emb.embed("hi")
    assert v.shape == (384,)
    assert v.dtype == np.float32
    assert abs(np.linalg.norm(v) - 1) < 1e-5


@pytest.mark.slow
def test_paraphrase_closer_than_unrelated(emb):
    a = emb.embed("How do I reset my password?")
    b = emb.embed("Password reset steps?")
    c = emb.embed("What's the weather?")
    assert a @ b > a @ c


@pytest.mark.slow
def test_embedding_is_deterministic(emb):
    assert np.allclose(emb.embed("How do I reset my password?"), emb.embed("How do I reset my password?"))


@pytest.mark.parametrize("similarity", [0.0, 0.99, 1.0])
def test_vector_at_gives_unit_vector_with_exact_similarity(similarity):
    v = vector_at(similarity)
    assert v @ BASE == pytest.approx(similarity)
    assert np.linalg.norm(v) == pytest.approx(1.0)


def test_fake_embedder_returns_given_vector():
    fake = FakeEmbedder({"hello": BASE})
    assert np.array_equal(fake.embed("hello"), BASE)


def test_fake_embedder_unknown_text_raises():
    fake = FakeEmbedder({"hello": BASE})
    with pytest.raises(KeyError, match="goodbye"):
        fake.embed("goodbye")


def test_hash_embedder_same_text_same_vector():
    assert np.array_equal(HashEmbedder().embed("hello"), HashEmbedder().embed("hello"))


def test_hash_embedder_unit_norm():
    assert np.linalg.norm(HashEmbedder().embed("hello")) == pytest.approx(1.0)


def test_hash_embedder_different_texts_unrelated():
    emb = HashEmbedder()
    assert emb.embed("How do I reset my password?") @ emb.embed("password reset steps") < 0.5
