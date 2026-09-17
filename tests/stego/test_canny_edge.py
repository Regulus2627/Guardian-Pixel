import numpy as np
import pytest

from backend.stego.baselines.canny_edge import embed_canny_lsb, extract_canny_lsb

# Helper to create an image with strong, robust edges
# A sharp contrast ensures cv2.Canny finds edges even after +-1 LSB modifications
def _create_strong_edge_image():
    img = np.zeros((20, 20, 3), dtype=np.uint8)
    img[5:15, 5:15, :] = 250  # Strong white box in the center
    return img

def test_round_trip_extraction():
    generator = np.random.default_rng(42)
    rgb = _create_strong_edge_image()
    bits = generator.integers(0, 2, size=10, dtype=np.uint8)
    
    stego = embed_canny_lsb(rgb, bits, channel=2)
    extracted = extract_canny_lsb(stego, num_bits=10, channel=2)
    
    assert np.array_equal(bits, extracted)

def test_maximum_pixel_change():
    """Verify that no pixel value changes by more than 1."""
    generator = np.random.default_rng(42)
    rgb = _create_strong_edge_image()
    bits = generator.integers(0, 2, size=15, dtype=np.uint8)
    
    stego = embed_canny_lsb(rgb, bits, channel=0)
    
    # Calculate absolute differences safely using integers
    diff = np.abs(stego.astype(int) - rgb.astype(int))
    assert np.max(diff) <= 1

def test_only_selected_channel_changes():
    rgb = _create_strong_edge_image()
    bits = np.array([1, 0, 1, 1], dtype=np.uint8)
    
    stego = embed_canny_lsb(rgb, bits, channel=1)
    
    # Channels 0 and 2 should be completely untouched
    assert np.array_equal(rgb[:, :, 0], stego[:, :, 0])
    assert np.array_equal(rgb[:, :, 2], stego[:, :, 2])

def test_over_capacity_raises():
    rgb = _create_strong_edge_image()
    # Requesting 10,000 bits will far exceed the edges found in a 20x20 image
    bits = np.zeros(10000, dtype=np.uint8)
    
    with pytest.raises(ValueError, match="Insufficient edge capacity"):
        embed_canny_lsb(rgb, bits, channel=2)

def test_zero_length_payload():
    rgb = _create_strong_edge_image()
    bits = np.array([], dtype=np.uint8)
    
    stego = embed_canny_lsb(rgb, bits, channel=0)
    assert np.array_equal(rgb, stego)
    
    extracted = extract_canny_lsb(stego, num_bits=0, channel=0)
    assert len(extracted) == 0

def test_invalid_channel_raises():
    rgb = _create_strong_edge_image()
    bits = np.array([1, 0], dtype=np.uint8)
    
    with pytest.raises(ValueError, match="Channel must be"):
        embed_canny_lsb(rgb, bits, channel=5)

def test_invalid_bit_values_raises():
    rgb = _create_strong_edge_image()
    # -1 would silently wrap to 255 if converted to uint8 before validation
    bad_bits = np.array([0, 1, 2, -1]) 
    
    with pytest.raises(ValueError, match="All payload bits must be strictly 0 or 1"):
        embed_canny_lsb(rgb, bad_bits, channel=0)

def test_image_shape_validation():
    """Test that non-3D or non-RGB arrays are rejected in both embed and extract."""
    bad_img_2d = np.zeros((10, 10), dtype=np.uint8)
    bad_img_4ch = np.zeros((10, 10, 4), dtype=np.uint8)
    bits = np.array([1, 0])
    
    with pytest.raises(ValueError, match="Image must have shape"):
        embed_canny_lsb(bad_img_2d, bits)
    
    with pytest.raises(ValueError, match="Image must have shape"):
        extract_canny_lsb(bad_img_4ch, num_bits=2)

def test_invalid_bits_array_raises():
    rgb = _create_strong_edge_image()

    bad_bits = np.array([[0, 1, 0, 0, 1], [0, 1, 1, 0, 1]])

    with pytest.raises(ValueError, match="Payload bits must be a 1D NumPy array."):
        embed_canny_lsb(rgb, bad_bits)


def test_invalid_rgb_dtype():
    generator = np.random.default_rng(42)

    rgb = generator.integers(
        0, 
        500, 
        size=(64, 64, 3), 
        dtype=np.int16
    )

    bits = np.array([0, 1, 1, 0, 0, 1, 0])

    with pytest.raises(ValueError, match="Image must have dtype uint8."):
        embed_canny_lsb(rgb, bits)