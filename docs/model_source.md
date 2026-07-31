# GuardianPixel HED Model Source

## Model

Holistically-Nested Edge Detection (HED)

## Architecture

VGG16-based fully convolutional edge-detection network with five
deeply supervised side outputs and one fused output.

## Original paper

Saining Xie and Zhuowen Tu,
“Holistically-Nested Edge Detection,” ICCV 2015.

## Source repository

https://github.com/s9xie/hed

## Network definition

https://raw.githubusercontent.com/s9xie/hed/master/examples/hed/deploy.prototxt

## Pretrained weights

https://vcl.ucsd.edu/hed/hed_pretrained_bsds.caffemodel

## Dataset

The pretrained model is associated with augmented BSDS500
edge-detection training data.

## Inference engine

OpenCV DNN with CPU inference.

## Local files

- backend/models/hed/deploy.prototxt
- backend/models/hed/hed_pretrained_bsds.caffemodel

## SHA-256

4B6937684BCE9BE1EF5163C78EC812DFF9A23653BFBB451925210A64ECFAAAC7

## GuardianPixel use

The model is used only on the sender side to produce an edge
probability map. It does not embed or extract the secret data.

## Limitation

HED detects learned image boundaries. An edge is not automatically
a secure steganographic position. The HED output will therefore be
combined with local entropy and local variance.