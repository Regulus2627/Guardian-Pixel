import cv2
import hashlib
import numpy as np

from backend.stego.lsb_matching import (
    embed_bits_at_positions,
    extract_bits_at_positions,
    LSBMatchingError
)
from backend.vision.preprocessing import rgb_to_luminance

# A constant direction key to ensure reproducibility for the baseline comparisons
CANNY_MATCHING_DIRECTION_KEY = hashlib.sha256(b"GuardianPixel-baseline-canny-matching").digest()

def _validate_image_shape(image: np.ndarray) -> None:
    """Validates that the input is a valid 3D RGB array."""
    if not isinstance(image, np.ndarray):
        raise ValueError("Image must be a NumPy array.")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Image must have shape (H, W, 3).")
    if image.dtype != np.uint8:
        raise ValueError("Image must have dtype uint8.")

def _get_canny_flat_positions(image: np.ndarray, channel: int, num_positions: int) -> np.ndarray:
    """Helper to calculate the flat 1D positions of Canny edges."""
    # 1. Convert to grayscale
    gray = rgb_to_luminance(image)

    # 2. Detect edges
    edge_map = cv2.Canny(gray, 100, 200)
    
    # 3. Extract 2D coordinates in row-major order
    edge_coords = np.argwhere(edge_map > 0)
    
    if len(edge_coords) < num_positions:
        raise ValueError(f"Insufficient edge capacity. Need {num_positions} bits, but only found {len(edge_coords)} edge pixels.")
        
    # 4. Flatten coordinates to 1D channel indices
    _, W, _ = image.shape
    rows = edge_coords[:, 0]
    cols = edge_coords[:, 1]
    
    flat_positions = (rows * W * 3) + (cols * 3) + channel
    
    return flat_positions[:num_positions]


def embed_canny_lsb(rgb: np.ndarray, bits: np.ndarray, channel: int = 2) -> np.ndarray:
    """Embed bits into Canny edge pixels using +-1 LSB matching."""
    # 1. Image Validation
    _validate_image_shape(rgb)
    
    # 2. Channel Validation
    if channel not in (0, 1, 2):
        raise ValueError("Channel must be 0 (Red), 1 (Green), or 2 (Blue).")
        
    # 3. Bit Validation (BEFORE dtype conversion)
    raw_bits = np.asarray(bits)
    if raw_bits.ndim != 1:
        raise ValueError("Payload bits must be a 1D NumPy array.")
    if not np.all((raw_bits == 0) | (raw_bits == 1)):
        raise ValueError("All payload bits must be strictly 0 or 1.")
    
    valid_bits = raw_bits.astype(np.uint8)
    
    if len(valid_bits) == 0:
        return rgb.copy()

    # 4. Get Edge Positions
    try:
        selected_positions = _get_canny_flat_positions(rgb, channel, len(valid_bits))
    except ValueError as e:
        raise ValueError(f"Capacity error: {str(e)}") from e

    # 5. Delegate to the core matching backend
    try:
        stego_result = embed_bits_at_positions(
            rgb, 
            valid_bits, 
            selected_positions, 
            CANNY_MATCHING_DIRECTION_KEY
        )
        return stego_result.stego_image
    except LSBMatchingError as e:
        raise ValueError(f"Failed to embed bits: {str(e)}") from e


def extract_canny_lsb(stego: np.ndarray, num_bits: int, channel: int = 2) -> np.ndarray:
    """Extract bits from Canny edge pixels."""
    # 1. Image and Extraction Validation
    _validate_image_shape(stego)
    
    if num_bits < 0:
        raise ValueError("num_bits must be non-negative.")
    if num_bits == 0:
        return np.array([], dtype=np.uint8)
        
    # 2. Channel Validation
    if channel not in (0, 1, 2):
        raise ValueError("Channel must be 0 (Red), 1 (Green), or 2 (Blue).")

    # 3. Reconstruct Edge Positions from the Stego Image
    try:
        selected_positions = _get_canny_flat_positions(stego, channel, num_bits)
    except ValueError as e:
        raise ValueError(f"Capacity error during extraction: {str(e)}") from e

    # 4. Delegate to the core matching backend
    try:
        extracted_bits = extract_bits_at_positions(stego, selected_positions)
        return extracted_bits
    except LSBMatchingError as e:
        raise ValueError(f"Failed to extract bits: {str(e)}") from e