# Paper Notes
+ Format : Arxiv Index - Name - Time

## 1. 1608.06993 - [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993) - 20170819
+ One of CVPR2017 best paper, a new CNN archetecture
+ Advantages:
  + Alleviate gradient vanishing
  + strengthen feature propagation
  + encourage feature reuse
  + reduce number of parameters
  + Can be trained as similar steps in ResNet
+ Archetecture: http://cvmart.net/community/article/detail/93
  + 为了进行特征复用，在跨层连接时使用的是在特征维度上的 Concatenate 操作，而不是 Element-wise Addition 操作。
  + 不需要Elewise-wise操作，因此在每个单元结束时不需要1x1卷积重构维度
  + 采用 Pre-activation 的策略来设计单元，将 BN 操作从主支上移到分支之前, i.e BN->ReLU->1x1Conv->BN->ReLU->3x3Conv
  + 网络中每层都接受前面所有层的特征作为输入, 为避免特征维度增长过快, 在进行下采样之前先用一个卷积层将特征维度压缩一半
  + 增长率k(每个单元模块最后3x3的卷积核的数量)的设置. 
    + 每个单元模块最后是以 Concatenate 的方式来进行连接的，因此每经过一个单元模块，下一层的特征维度就会增长 k
    + k越大意味着在网络中流通的信息也越大，相应地网络的能力也越强，但是整个模型的尺寸和计算量也会变大
    + 本文中使用了 k=32 和 k=48 两种设置
+ Implementation: 
  + Original: https://github.com/liuzhuang13/DenseNet
  + PyTorch: https://github.com/gpleiss/efficient_densenet_pytorch
  + Tensorflow: https://github.com/YixuanLi/densenet-tensorflow

---

## 2. 1704.05548 - [Annotating Object Instances with a Polygon-RNN](https://arxiv.org/abs/1704.05548) - 20170819
+ One of CVPR2017 best paper, an approach for semi-automatic annotation of object instances
+ Cast annotation as polygon prediction task $\rightarrow$ Polygon RNN
+ Why semi-automic
  + Human interfere can make it more accurate
  + We need to assign ground-truth bounding box first before predicting the position of polygon using RNN
+ Previous attempts
  + Learning segmentation models from weak annotation such as image tags or bounding boxes $\rightarrow$ in-competitive performance
  + Producing(noisy) labels inside bounding boxes with a GrabCut type of approach $\rightarrow$ cannot be used as official ground-truth for a benchmark due to its inherent imprecisions.
+ Polygon-RNN: predicts a vertex at every time step
  + Inputs:
    + $x_t$ : a tensor at time step t, concatenate multiple features processed by CNN
    + $y_{t-1}, y_{t-2}$ : previous 2 predicted vertices (one-hot encoding)
    + $y_1$ : the one-hot encoding of the first predicted vertex
    + $x_t$ and $y_{t-1}, y_{t-2}$ can be used to predict the direction of next edge
    + $y_1$ can be used to predict the stop position of this edge
  + CNN : a modified vgg16
    + remove fully-connected layers as well as last max-pooling layer(pool5)
    + add additional convolutional layers with skip-connections(see ResNet), this allows CNN to extract following features  
      + low-level information about the edges and corners $rightarrow$ follow the object’s boundaries
      + semantic information about the object $\rightarrow$ see the object
      + concated output features (28x28x512) $\rightarrow$ another 3x3 conv + ReLU $\rightarrow$ 28x28x128
  + RNN : 
    + Convoluntional LSTM
      + preserve the spatial information received from the CNN
      + reduce the parameters
    + model the polygon with a two-layer ConvLSTM with kernel size of 3x3 and 16 channels, which outputs a vertex at each time step
    + 
    + formulate the vertex prediction as a classification task.

## 3. 1512.03385 - [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) - 20170820
+ These days I'm trying to perform hand detection via faster R-CNN, current base model is VGG16, the performance can be improved if I replace it with ResNet
+ ResNet152(ImageNet 2015 + CVPR 2016) is a very deep model and there has been a lot of related notes. The networks are easier to optimize, and can gain accuracy from considerably increased depth. 
+ Problems of deeper network
  + Vanishing / exploding gradients
  + Degradation: adding more layers will lead to higher training error
    + One solution of degradation: add "identity mapping" layers, and the other layers ar ecopied from the learned shallower model. However it's not easy to fit desired underlying mapping directly
    + This paper use deep residual learning framework to tackle degradation
+ Deep residual learning framework: 
  + H(x) denotes the desired underlying mapping, which is hard to get directly
  + To make this easier, we compute $F(x) := H(x) - x$ first, then $H(x) := F(x) + x$
  + $F(x) + x$ can be realized by feedforward NN with "short connections" $\rightarrow$ residual block
    + Short connections are those skipping one or more layers
    + We perform identity mapping on short connections
    + The outputs of short connections(x) are then added to the outputs (F(x)) of common stacked layers
+ Archetecture:
  + Build the plain network
    + Fewer filters and lower complexity than VGG16
    + Only one fully-connected layer
  + Insert short connections into plain network
    + If the dimensions are same, the identity can be inserted directly
    + If not(the dimensions increase), we use linear projection to match the dimensions
+ Implementation details
  + Augmentation: scale the images to 256x480
  + Cut 224x224 crop, with the per-pixel mean subtracted
  + Batch normalization between convolution and activation
  + SGD
  + No dropout
