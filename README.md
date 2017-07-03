functional-zoo
==============

Model definitions and pretrained weights for PyTorch and Tensorflow

PyTorch, unlike lua torch, has autograd in it's core, so using modular
structure of `torch.nn` modules is not necessary, one can easily allocate
needed Variables and write a function that utilizes them, which is sometimes
more convenient. This repo contains model definitions in this functional way,
with pretrained weights for some models.

Weights are serialized as a dict of arrays in `hdf5`, so should be easily
loadable in other frameworks. Thanks to @edgarriba we have [cpp_parser](cpp_parser) for
loading weights in C++.

More models coming! We also plan to add definitions for other frameworks
in future, probably `tiny-dnn` first. Contributions are welcome.

See also imagenet classification with PyTorch [demo.ipynb](demo.ipynb)


## Models

All models were validated to produce reported accuracy using
[imagenet-validation.py](imagenet-validation.py) script (depends on
OpenCV python bindings).

To load weights in Python first do `pip install hickle`, then:

```python
import hickle as hkl
weights = hkl.load('resnet-18-export.hkl')
```

And the `weights` will be a dict of numpy arrays. See the notebooks for more
examples.


### Folded

Models below have batch_norm parameters and statistics folded into convolutional
layers for speed. It is not recommended to use them for finetuning.

#### ImageNet

| model | notebook | val error | download | size |
|:------|:--------:|:--------:|:--------:|:----:|
| VGG-16 | [vgg-16.ipynb](vgg-16.ipynb) | 30.09, 10.69 | url coming | 528 MB |
| NIN | [nin-export.ipynb](nin-export.ipynb) | 32.96, 12.29 | [url](https://s3.amazonaws.com/pytorch/h5models/nin-export.hkl) | 33 MB |
| ResNet-18 (fb) | [resnet-18-export.ipynb](resnet-18-export.ipynb) | 30.43, 10.76 | [url](https://s3.amazonaws.com/pytorch/h5models/resnet-18-export.hkl) | 42 MB |
| ResNet-18-AT | [resnet-18-at-export.ipynb](resnet-18-at-export.ipynb) | 29.44, 10.12 | [url](https://www.dropbox.com/s/z092wmrgyqn4ys5/resnet-18-at-export.hkl?dl=0) | 44.1 MB |
| ResNet-34 (fb) | [resnet-34-export.ipynb](resnet-34-export.ipynb) | 26.72, 8.74 | [url](https://s3.amazonaws.com/pytorch/h5models/resnet-34-export.hkl) | 78.3 MB |
| WRN-50-2 | [wide-resnet-50-2-export.ipynb](wide-resnet-50-2-export.ipynb) | 22.0, 6.05 | [url](https://s3.amazonaws.com/pytorch/h5models/wide-resnet-50-2-export.hkl) | 246 MB |


#### Fast Neural Style

Notebook: [fast-neural-style.ipynb](fast-neural-style.ipynb)

Models:

| model | download | size |
|:------|:--------:|:----:|
| candy.hkl | [url](https://s3.amazonaws.com/pytorch/h5models/fast-neural-style/candy.hkl) | 7.1 MB |
| feathers.hkl | [url](https://s3.amazonaws.com/pytorch/h5models/fast-neural-style/feathers.hkl) | 7.1 MB |
| wave.hkl | [url](https://s3.amazonaws.com/pytorch/h5models/fast-neural-style/wave.hkl) | 7.1 MB |


### Models with batch normalization

Coming
