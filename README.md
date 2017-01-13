functional-zoo
==============

PyTorch, unlike lua torch, has autograd in it's core, so using modular
structure of torch.nn` modules is not necessary, one can easily allocate
needed Variables and write a function that utilizes them, which is sometimes
more convenient. This repo contains model definitions in this functional way,
and pretrained weights for some models.

Weights are serialized as a dict of numpy arrays in `hdf5`, so should be easily
loadable in other frameworks.


## Models

All models were validated to produce reported accuracy using
[imagenet-validation.py](imagenet-validation.py) script (depends on
OpenCV python bindings).

### Folded

Models below have batch_norm parameters and statistics folded into convolutional
layers for speed. It is not recommended to use them for finetuning.

* [NIN](nin-export.ipynb)
* [ResNet-18](resnet-18-export.ipynb)
* [ResNet-34](resnet-34-export.ipynb)
