# RFCN
tensorpack version

## Introduction
This is an implementation of Region-based Fully Convolutional Networks on Python 3, Tensorpack, and TensorFlow. The model generates bounding boxes for each instance of an object in the image. It uses Resnet-50 as backbone.

## Contribution
In this repository, I reuse the code from tensorpack/examples/FasterRCNN and implement the final layer VotePooling layer.

For convenience, I modify the structure of box coordinate regression layer where it has (C+1) x 4 channel but in the origin paper it has only 4 channels.

## Structure of ResnetC4RFCN
```
Resnet-50-C4Backbone-> RPN Module -> ROI Proposal Module 
                    |    
                    -> resnet-conv5 block -> conv2d(1x1 kernel) -> VotePooling(cls) 
                                          |
                                          -> conv2d(1x1 kernel) -> VotePooling(reg)
```
1. resnet-conv5 block can be replaced by aspp block
2. VotePooling layer needs the results of ROI Proposal Module, but I don't draw these edges in the graph above.

## Training details
The pretrain weights is COCO-R50C4-MaskRCNN-Standard.npz, you can download from here(http://models.tensorpack.com/FasterRCNN/).
You can use this instruction to start training: 
    python train.py --load './COCO-R50C4-MaskRCNN-Standard.npz'
More details about how to use this code to train on your own data, you can refer to tensorpack/examples/FasterRCNN.

## Results
Currently, I get mAP on VOC trainval is 72.5%.


