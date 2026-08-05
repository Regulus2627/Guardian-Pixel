# HED Model Files

GuardianPixel uses the pretrained Holistically-Nested Edge Detection
model for sender-side edge-map generation.

## Required files

- deploy.prototxt
- hed_pretrained_bsds.caffemodel

## Download weights

https://vcl.ucsd.edu/hed/hed_pretrained_bsds.caffemodel

## Expected SHA-256

4B6937684BCE9BE1EF5163C78EC812DFF9A23653BFBB451925210A64ECFAAAC7

The `.caffemodel` file is intentionally excluded from Git because it
is large. Every teammate must download it separately and verify its
checksum.