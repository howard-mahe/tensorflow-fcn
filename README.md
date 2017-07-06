# tensorflow-fcn
This is a one file Tensorflow implementation of [Fully Convolutional Networks](http://arxiv.org/abs/1411.4038) in Tensorflow. The code can easily be integrated in your semantic segmentation pipeline. The network can be applied directly or finetuned to perform semantic segmentation using tensorflow training code.

Deconvolution Layers are initialized as bilinear upsampling. Conv and FCN layer weights using VGG weights. Numpy load is used to read VGG weights. No Caffe or Caffe-Tensorflow is required to run this. **The .npy file for [VGG16] to be downloaded before using this needwork**. You can find the file here: ftp://mi.eng.cam.ac.uk/pub/mttt2/models/vgg16.npy

No Pascal VOC finetuning was applied to the weights. The model is meant to be finetuned on your own data.

### Tensorflow 1.0rc

This code requires `Tensorflow Version >= 1.0rc` to run.

Tensorflow 1.0 comes with a large number of breaking api changes. If you are currently running an older tensorflow version, I would suggest creating a new `virtualenv` and install 1.2.

## Usage

`python train.py` to start the training.

## Content

Currently the following Models are provided:

- FCN32
- FCN16
- FCN8

## Predecessors

Weights were generated using [Caffe to Tensorflow](https://github.com/ethereon/caffe-tensorflow). The VGG implementation is based on [tensorflow-vgg16](https://github.com/ry/tensorflow-vgg16) and numpy loading is based on [tensorflow-vgg](https://github.com/machrisaa/tensorflow-vgg). You do not need any of the above cited code to run the model, not do you need caffe.

## Install

`sudo apt-get install libpng-dev libjpeg8-dev libfreetype6-dev pkg-config` <br>

## Requirements

In addition to tensorflow the following packages are required:

numpy
scipy
pillow
matplotlib

Those packages can be installed by running `pip install -r requirements.txt` or `pip install numpy scipy pillow matplotlib`.

