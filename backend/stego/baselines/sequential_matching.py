import hashlib
import numpy as np
from backend.stego.lsb_matching import (
    embed_bits_at_positions, 
    extract_bits_at_positions, 
    LSBMatchingError
    )

# a constant direction key
SEQUENTIAL_MATCHING_DIRECTION_KEY = hashlib.sha256(b"GuardianPixel-baseline-sequential-matching").digest()


def embed_matching_lsb(
    rgb : np.ndarray, 
    bits : np.ndarray, 
    channel : int = 2
) -> np.ndarray:
    """Embed at sequential positions"""

    if channel not in (0, 1, 2):
        raise ValueError("Channel must be 0, 1, or 2.")

    # this calculates positions for flattened image
    positions = (np.arange(len(bits)) * 3) + channel

    try:
        result = embed_bits_at_positions(
            rgb, 
            bits, 
            positions=positions, 
            direction_key=SEQUENTIAL_MATCHING_DIRECTION_KEY
        )
    except LSBMatchingError as error:
        raise ValueError(f"Embedding failed: {error}") from error


    return result.stego_image

def extract_matching_lsb(
        stego : np.ndarray, 
        num_bits : int, 
        channel: int = 2,
) -> np.ndarray : 
    """Extract from sequential positions"""
   
    if channel not in (0, 1, 2):
        raise ValueError("Channel must be 0, 1, or 2.")

    positions = (np.arange(num_bits) * 3) + channel

    extracted = extract_bits_at_positions(
        stego, 
        positions=positions
    )

    return extracted