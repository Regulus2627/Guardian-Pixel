# GuardianPixel Vision Pipeline

## Input

A validated PNG or JPEG cover image and the exact number of encrypted
payload bits required by the embedding module.

## Processing

1. Validate and convert the image to RGB.
2. Convert RGB to luminance.
3. Generate a pretrained HED edge map.
4. Generate a local Shannon-entropy map.
5. Generate a local-variance map.
6. Normalize all feature maps to 0–1.
7. Fuse maps using configurable weights.
8. Divide the heatmap into 8×8 blocks.
9. Rank blocks by mean suitability.
10. Select sufficient blocks with a safety margin.
11. Encode the binary map using the smallest of RAW, RLE, zlib,
    ALL_SELECTED or NONE_SELECTED.

## Output

- Normalized HED map
- Normalized entropy map
- Normalized variance map
- Fused suitability heatmap
- Capacity-aware binary block map
- Encoded block-map bytes
- Timing and configuration information

## Default Fusion

S = 0.45 × HED + 0.35 × Entropy + 0.20 × Variance

These are initial experimental weights and must be validated on the
DIV2K validation set.

## Receiver

The receiver does not run HED, entropy, variance or fusion. It obtains
the encoded block map from Member 2's metadata protocol.

## Limitation

The suitability map does not prove universal undetectability. Its
benefit must be measured against sequential, random, Canny, entropy-only
and HED-only baselines at equal payload rates.