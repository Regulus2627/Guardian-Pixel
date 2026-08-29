import numpy as np
import pytest

from backend.stego.baselines.sequential_matching import embed_matching_lsb, extract_matching_lsb

def test_embed_matching_lsb():
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

    stego = embed_matching_lsb(
        rgb, 
        bits, 
        channel=2
    )

    extracted_bits = extract_matching_lsb(
        stego, 
        len(bits)
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

    # changing R channel
    stego = embed_matching_lsb(rgb, bits, channel=0)
    assert np.array_equal(
        rgb[:, :, 1:],
        stego[:, :, 1:]
    )

    # changing G channel
    stego = embed_matching_lsb(rgb, bits, channel=1)
    assert np.array_equal(
        rgb[:, :, [0, 2]], 
        stego[:, :, [0, 2]]
    )

    # changing B channel
    stego = embed_matching_lsb(rgb, bits, channel=2)
    assert np.array_equal(
        rgb[:, :, :2], 
        stego[:, :, :2]
    )



def test_pixels_after_payload_unchanged():
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

    # checking for R channel
    stego = embed_matching_lsb(rgb, bits, channel=0)

    original_pixels = rgb.reshape(-1, 3)
    stego_pixels = stego.reshape(-1, 3)

    assert np.array_equal(
        original_pixels[len(bits):, 0], 
        stego_pixels[len(bits):, 0]
    )

    # checking for G channel
    stego = embed_matching_lsb(rgb, bits, channel=1)

    stego_pixels = stego.reshape(-1, 3)

    assert np.array_equal(
        original_pixels[len(bits):, 1], 
        stego_pixels[len(bits):, 1]
    )

    # checking for B channel
    stego = embed_matching_lsb(rgb, bits, channel=2)

    stego_pixels = stego.reshape(-1, 3)

    assert np.array_equal(
        original_pixels[len(bits):, 2], 
        stego_pixels[len(bits):, 2]
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
        size=(64*64*2),
        dtype=np.uint8,
    )

    with pytest.raises(ValueError, match="Embedding failed:"):
        embed_matching_lsb(rgb, bits)



def test_payload_null():
    generator = np.random.default_rng(42)
                    
    rgb = generator.integers(
        0,
        256,
        size=(64, 64, 3),
        dtype=np.uint8,
    )

    bits = np.array([], dtype=np.uint8)

    stego = embed_matching_lsb(rgb, bits)
    extracted = extract_matching_lsb(stego, num_bits=0)

    assert np.array_equal(rgb, stego)
    assert extracted.size == 0



def test_payload_at_maximum_capacity():
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
        size=(64*64),
        dtype=np.uint8,
    )

    stego = embed_matching_lsb(rgb, bits)
    extracted = extract_matching_lsb(stego, len(bits))

    assert np.array_equal(bits, extracted)



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
        size= 1000,
        dtype=np.uint8,
    )

    with pytest.raises(ValueError, match="Channel must be 0, 1, or 2."):
        embed_matching_lsb(rgb, bits, channel=5)



def test_invalid_bit_values_raises():
    generator = np.random.default_rng(42)
                            
    rgb = generator.integers(
        0,
        256,
        size=(64, 64, 3),
        dtype=np.uint8,
    )
    
    bits = generator.integers(
        0,
        5,
        size= 1000,
        dtype=np.uint8,
    )

    with pytest.raises(ValueError, match="Embedding failed:"):
        embed_matching_lsb(rgb, bits)



def test_known_bit_pattern():
    rgb = np.array([[[137, 200, 50]]], dtype=np.uint8)  # 1x1x3 pixel image
    bits = np.array([0], dtype=np.uint8)  # want LSB = 0
    stego = embed_matching_lsb(rgb, bits, channel=0)
    assert stego[0, 0, 0] == 136 or stego[0, 0, 0] == 138



def test_unchanged_when_lsb_already_matches():
    rgb = np.array([[[137, 200, 50]]], dtype=np.uint8)   #137 has lsb 1
    bits = np.array([1], dtype=np.uint8)
    stego = embed_matching_lsb(rgb, bits, channel=0)
    assert np.array_equal(rgb, stego)



def test_edge_case_pixel_value_zero():
    rgb = np.array([[[0, 200, 50]]], dtype=np.uint8)
    bits = np.array([1], dtype=np.uint8)
    stego = embed_matching_lsb(rgb, bits, channel=0)
    assert (stego[0, 0, 0] == rgb[0, 0, 0] + 1)



def test_edge_case_pixel_value_255():
    rgb = np.array([[[255, 200, 50]]], dtype=np.uint8)
    bits = np.array([0], dtype=np.uint8)
    stego = embed_matching_lsb(rgb, bits, channel=0)
    assert (stego[0, 0, 0] == rgb[0, 0, 0] - 1)



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
        size= 1000,
        dtype=np.uint8,
    )

    stego1 = embed_matching_lsb(rgb, bits, channel=1)
    stego2 = embed_matching_lsb(rgb, bits, channel=1)

    assert np.array_equal(stego1, stego2)