import numpy as np
import hashlib

from backend.core.positions import (
    generate_keyed_positions, 
    PositionError
)
from backend.stego.lsb_matching import (
    embed_bits_at_positions, 
    extract_bits_at_positions, 
    LSBMatchingError
)

KEYED_RANDOM_MATCHING_DIRECTION_KEY = hashlib.sha256(
    b"GuardianPixel-baseline-keyed-random-matching"
).digest()

def embed_keyed_random_matching(
    rgb: np.ndarray,
    bits: np.ndarray, 
    key: bytes,  
    channel: int = 2,
) -> np.ndarray:
    """Embed at keyed random positions"""

    if channel not in (0, 1, 2):
        raise ValueError("Channel must be 0, 1 or 2.")
    
    if not isinstance(key, bytes):
        raise ValueError("Key must be bytes object.")


    try: 
        positions = generate_keyed_positions(
            candidate_positions=np.arange(rgb.shape[0]*rgb.shape[1], dtype=np.int64) * 3 + channel, 
            count=len(bits), 
            key=key, 
            domain=b"GP_POSITIONS" 
    )
    except PositionError as error:
        raise ValueError(f"Failed to generate positions: {error}") from error

    try:
        result = embed_bits_at_positions(
            rgb, 
            bits, 
            positions=positions, 
            direction_key=KEYED_RANDOM_MATCHING_DIRECTION_KEY
        )
    except LSBMatchingError as error:
        raise ValueError(f"Embedding failed: {error}") from error

    return result.stego_image



def extract_keyed_random_matching(
    stego: np.ndarray, 
    num_bits: int, 
    key: bytes, 
    channel: int = 2, 
) -> np.ndarray:
    """Extract from keyed random positions"""

    if channel not in (0, 1, 2):
        raise ValueError("Channel must be 0, 1 or 2.")

    if not isinstance(key, bytes):
        raise ValueError("Key must be bytes object.")

    if num_bits < 0:
        raise ValueError("num_bits cannot be negative.")

    if num_bits > (stego.shape[0] * stego.shape[1]):
        raise ValueError("Requested bit count exceeds image capacity.")

    try: 
        positions = generate_keyed_positions(
            candidate_positions=np.arange(stego.shape[0]*stego.shape[1], dtype=np.int64) * 3 + channel, 
            count=num_bits, 
            key=key, 
            domain=b"GP_POSITIONS" 
    )
    except PositionError as error:
        raise ValueError(f"Failed to generate positions: {error}") from error

    try: 
        extracted = extract_bits_at_positions(
            stego, 
            positions=positions
    )
    except LSBMatchingError as error:
        raise ValueError(f"Failed to extract: {error}") from error

    return extracted