import numpy as np
import hashlib
import pytest

from backend.stego.baselines.keyed_random_matching import (
    embed_keyed_random_matching, 
    extract_keyed_random_matching, 
)

KEYED_RANDOM_MATCHING_POSITION_KEY = hashlib.sha256(b"GuardianPixel-baseline-test-keyed-random-matching").digest()


def test_embed_keyed_random_matching():
    generator = np.random.default_rng(42)
    
    rgb = generator.integers(
        0,
        256,
        size=(64, 64, 3),
        dtype=np.uint8,
    )
    
    bits = generator.integers(
        0,
        2,
        size=1000,
        dtype=np.uint8,
    )
    
    original = rgb.copy()

    stego = embed_keyed_random_matching(
        rgb, 
        bits, 
        key=KEYED_RANDOM_MATCHING_POSITION_KEY, 
        channel=1
    )

    extracted_bits = extract_keyed_random_matching(
        stego, 
        num_bits=len(bits), 
        key=KEYED_RANDOM_MATCHING_POSITION_KEY, 
        channel=1
    )

    assert np.array_equal(bits, extracted_bits)
    assert stego.shape == original.shape
    assert np.array_equal(rgb, original)



def test_only_selected_channel_changes():
    generator = np.random.default_rng(42)
    
    rgb = generator.integers(
        0,
        256,
        size=(64, 64, 3),
        dtype=np.uint8,
    )
    
    bits = generator.integers(
        0,
        2,
        size=1000,
        dtype=np.uint8,
    )
    
    # Chaning R channel
    stego = embed_keyed_random_matching(
        rgb, 
        bits, 
        key=KEYED_RANDOM_MATCHING_POSITION_KEY, 
        channel=0, 
    )

    assert np.array_equal(
        rgb[:, :, 1:], 
        stego[:, :, 1:]
    )

    # Changing G channel
    stego = embed_keyed_random_matching(
        rgb,
        bits, 
        key=KEYED_RANDOM_MATCHING_POSITION_KEY, 
        channel=1
    )

    assert np.array_equal(
        rgb[:, :, [0, 2]], 
        stego[:, :, [0, 2]]
    )

    # Changing B channel
    stego = embed_keyed_random_matching(
        rgb,
        bits, 
        key=KEYED_RANDOM_MATCHING_POSITION_KEY, 
        channel=2
    )

    assert np.array_equal(
        rgb[:, :, :2], 
        stego[:, :, :2]
    )



def test_payload_at_max_capacity():
    generator = np.random.default_rng(42)
    
    rgb = generator.integers(
        0,
        256,
        size=(64, 64, 3),
        dtype=np.uint8,
    )
    
    bits = generator.integers(
        0,
        2,
        size=64*64,
        dtype=np.uint8,
    )

    stego = embed_keyed_random_matching(
        rgb, 
        bits, 
        key=KEYED_RANDOM_MATCHING_POSITION_KEY, 
    )

    extracted = extract_keyed_random_matching(
        stego, 
        num_bits=len(bits), 
        key=KEYED_RANDOM_MATCHING_POSITION_KEY
    )

    assert np.array_equal(
        bits, 
        extracted
    )



def test_payload_too_large():
    generator = np.random.default_rng(42)
    
    rgb = generator.integers(
        0,
        256,
        size=(64, 64, 3),
        dtype=np.uint8,
    )
    
    bits = generator.integers(
        0,
        2,
        size=(64*64*4),
        dtype=np.uint8,
    )

    with pytest.raises(ValueError, match="Failed to generate positions:"):
        embed_keyed_random_matching(
            rgb, 
            bits, 
            key=KEYED_RANDOM_MATCHING_POSITION_KEY, 
    )



def test_invalid_channel_raises():
    generator = np.random.default_rng(42)
    
    rgb = generator.integers(
        0,
        256,
        size=(64, 64, 3),
        dtype=np.uint8,
    )
    
    bits = generator.integers(
        0,
        2,
        size=1000,
        dtype=np.uint8,
    )

    with pytest.raises(ValueError, match="Channel must be 0, 1 or 2."):
        embed_keyed_random_matching(
            rgb, 
            bits, 
            key=KEYED_RANDOM_MATCHING_POSITION_KEY, 
            channel= 4
    )



def test_invalid_key_type():
    generator = np.random.default_rng(42)
    
    rgb = generator.integers(
        0,
        256,
        size=(64, 64, 3),
        dtype=np.uint8,
    )
    
    bits = generator.integers(
        0,
        2,
        size=1000,
        dtype=np.uint8,
    )

    with pytest.raises(ValueError, match="Key must be bytes object."):
        embed_keyed_random_matching(
            rgb, 
            bits, 
            key="Now I am become death, The destroyer of the worlds."
    )



def test_payload_null():
    generator = np.random.default_rng(42)
    
    rgb = generator.integers(
        0,
        256,
        size=(64, 64, 3),
        dtype=np.uint8,
    )
    
    bits = np.array([], dtype=np.uint8)

    stego = embed_keyed_random_matching(
        rgb, 
        bits, 
        key=KEYED_RANDOM_MATCHING_POSITION_KEY, 
    )

    extracted = extract_keyed_random_matching(
        stego, 
        num_bits=len(bits), 
        key=KEYED_RANDOM_MATCHING_POSITION_KEY, 
    )

    assert np.array_equal(stego, rgb)
    assert extracted.size == 0



def test_wrong_key_fails_to_extract_correctly():
    generator = np.random.default_rng(42)
    
    rgb = generator.integers(
        0,
        256,
        size=(64, 64, 3),
        dtype=np.uint8,
    )
    
    bits = generator.integers(
        0,
        2,
        size=1000,
        dtype=np.uint8,
    )

    stego = embed_keyed_random_matching(
        rgb, 
        bits, 
        key=KEYED_RANDOM_MATCHING_POSITION_KEY
    )

    extracted = extract_keyed_random_matching(
        stego, 
        num_bits=len(bits), 
        key=b"Hello-this-is-me-rozz"
    )

    assert not np.array_equal(bits, extracted)



def test_different_keys_produces_different_stego_images():
    generator = np.random.default_rng(42)
    
    rgb = generator.integers(
        0,
        256,
        size=(64, 64, 3),
        dtype=np.uint8,
    )
    
    bits = generator.integers(
        0,
        2,
        size=1000,
        dtype=np.uint8,
    )

    stego1 = embed_keyed_random_matching(
        rgb, 
        bits, 
        key=b"different-key1-test"
    )

    stego2 = embed_keyed_random_matching(
        rgb, 
        bits, 
        key=b"different-key2-test"
    )

    assert not np.array_equal(stego1, stego2)



def test_reproducibility():
    generator = np.random.default_rng(42)
    
    rgb = generator.integers(
        0,
        256,
        size=(64, 64, 3),
        dtype=np.uint8,
    )
    
    bits = generator.integers(
        0,
        2,
        size=1000,
        dtype=np.uint8,
    )

    stego1 = embed_keyed_random_matching(
        rgb, 
        bits, 
        key=KEYED_RANDOM_MATCHING_POSITION_KEY
    )

    stego2 = embed_keyed_random_matching(
        rgb,
        bits,
        key=KEYED_RANDOM_MATCHING_POSITION_KEY
    )

    assert np.array_equal(stego1, stego2)



def test_short_key_raises():
    generator = np.random.default_rng(42)
    
    rgb = generator.integers(
        0,
        256,
        size=(64, 64, 3),
        dtype=np.uint8,
    )
    
    bits = generator.integers(
        0,
        2,
        size=1000,
        dtype=np.uint8,
    )

    with pytest.raises(ValueError, match="Failed to generate positions:"):
        embed_keyed_random_matching(
            rgb, 
            bits, 
            key=b"Trinity"
        )
