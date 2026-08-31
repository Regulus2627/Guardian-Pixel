import numpy as np


LSB_CLEAR_MASK = np.uint8(0xFE)


def embed_sequential_lsb(
    rgb: np.ndarray,
    bits: np.ndarray,
    channel: int = 2,
) -> np.ndarray:
    """Embed bits sequentially into the LSB of one RGB channel."""

    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.size == 0:
        raise ValueError("RGB must be a non-empty 3 channel image.")

    bits = np.asarray(bits)

    if bits.ndim != 1:
        raise ValueError("Bits array must be 1 dimensional.")

    if bits.size > 0 and not np.all((bits == 0) | (bits == 1)):
        raise ValueError("Bits must contain only 0 or 1.")

    bits = np.asarray(bits, dtype=np.uint8)

    if channel not in (0, 1, 2):
        raise ValueError("Channel must be 0, 1, or 2.")

    pixels = rgb.copy().reshape(-1, rgb.shape[-1])

    if len(bits) > pixels.shape[0]:
        raise ValueError("Payload is too large for this image.")

    # 0xFE = 11111110, so AND clears the existing LSB.
    pixels[:len(bits), channel] = (
        pixels[:len(bits), channel] & LSB_CLEAR_MASK
    ) | bits

    return pixels.reshape(rgb.shape)


def extract_sequential_lsb(
    stego: np.ndarray,
    num_bits: int,
    channel: int = 2,
) -> np.ndarray:
    """Extract bits sequentially from the LSB of one RGB channel."""

    if stego.ndim != 3 or stego.shape[2] != 3 or stego.size == 0:
        raise ValueError("Stego must be a non-empty 3 channel image.")

    if channel not in (0, 1, 2):
        raise ValueError("Channel must be 0, 1, or 2.")

    pixels = stego.reshape(-1, stego.shape[-1])

    if num_bits < 0:
        raise ValueError("num_bits cannot be negative.")

    if num_bits > pixels.shape[0]:
        raise ValueError(
            "Requested bit count exceeds image capacity."
        )

    return pixels[:num_bits, channel] & 1  