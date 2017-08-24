# Paper Notes
+ Enjoy yourself :D

## 1. 1608.06993 - [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993)
&hearts; DL , CV , CNN, Recognition
+ One of CVPR2017 best paper, a new CNN archetecture
+ Advantages:
  + Alleviate gradient vanishing
  + strengthen feature propagation
  + encourage feature reuse
  + reduce number of parameters
  + Can be trained as similar steps in ResNet
+ Limitations: from [DPN](https://arxiv.org/abs/1707.01629)
  + Width of the densely connected path linearly increases as the depth rises
  + This may cause the number of parameters to grow quadratically compared with the residual networks if the implementation is not specifically optimized
+ Archetecture: 
  + Difference bwtween ResNet and DenseNet: 
    + ResNet adds the input features to the output features through the residual path: $x(l) = H_l(l-1) + x_{l-1}$
    + DenseNet uses a densely connected path to concatenate the input features with the output features : $x(l) = H_l([x_0,...,x_{l-1}])$
    + This enables each micro-block to receive raw information from all previous micro-blocks
  + Preactivation, i.e BN->ReLU->1x1Conv->BN->ReLU->3x3Conv
  + To make pooling easier(the dimensions may increase too fast), halve feature dimension using conv before pooling
  + Growth rate k(the number of 3x3 kernels after each part):
    + After each part, the dimension of next part will increase by k
    + Larger k means more information will be accessed, while the computational complexity will be increased
    + In this paper k = 32/48
+ Implementation: 
  + Original: https://github.com/liuzhuang13/DenseNet
  + PyTorch: https://github.com/gpleiss/efficient_densenet_pytorch
  + Tensorflow: https://github.com/YixuanLi/densenet-tensorflow
---

## 2. 1704.05548 - [Annotating Object Instances with a Polygon-RNN](https://arxiv.org/abs/1704.05548)
&hearts; DL , CV , RNN, Annotation
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
    + formulate the vertex prediction as a classification task.

---

## 3. 1512.03385 - [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
&hearts; DL , CV , CNN, Recognition
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
    + For the same output feature map size, the layers have the same number of filters
    + If the feature map size is halved(1/2), the number of filters is doubled $\rightarrow$ preserve the time complexity per layer
    + Only one fully-connected layer
    + Fewer filters and lower complexity than VGG16
  + Insert short connections into plain network
    + If the dimensions are same, the identity can be inserted directly
    + If not(the dimensions increase), we use linear projection to match the dimensions
  + 2 kinds of building blocks in this paper
    + ResNet-34
    + Bottleneck building block for ResNet-50/101/152
+ Implementation details
  + Scale the images to 256x480 as augmentation
  + Cut 224x224 crop, with the per-pixel mean subtracted
  + Standard color augmentation
  + Batch normalization between convolution and activation: alleviate gradient vanishing / exploding
  + SGD
  + No dropout

---

## 4. 1707.01629 - [Dual Path Networks](https://arxiv.org/abs/1707.01629)
&hearts; DL , CV , CNN, Recognition
+ Championship of ImageNet 2017(Object localization for 1000 categories) 
  + Enjoy the benefits of ResNet and DenseNet, bridge the densely connected networks with HORNN
  + Shares common features while maintaining the flexibility to explore new features through dual path architectures
  + Advantages
    + Effective feature reusage and reexploitation
    + Higher parameter efficiency, lower computational cost and lower memory consumption
    + Friendly for optimization
+ ResNet, DenseNet and HORNN
  + Terminologies:
    + t: t-th step
    + k: the index of current step
    + F: Feature extracting function, inputs hidden state, outputs extracted information
    + G: Transformation function, transforms gathered information to current hidden state
  + Observations:
    + If for all k, F and G are shared $\rightarrow$ ResNet and DenseNet can be seen as HORNN
    + If for all k,t, F is shared $\rightarrow$ ResNet can be seen as DenseNet
+ DPN
  + It will be easier to understand via codes, although it's still in progress: https://github.com/cypw/DPNs
---

## 5. 1506.01497 - [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497)
&hearts; DL , CV , RCNN, Object Detection
+ Related readings
  + 1311.2524 - [Rich feature hierarchies for accurate object detection and semantic segmentation (RCNN)](https://arxiv.org/abs/1311.2524)
  + 1504.08083 - [Fast RCNN](https://arxiv.org/abs/1504.08083)
  + 1702.02138 - [An Implementation of Faster RCNN with Study for Region Sampling](https://arxiv.org/abs/1702.02138)
+ RCNN: 
    + Input image
    + Extract ROI(about 2000) from a proposal method : selective search
    + For each proposal, get features : CNN
    + Classification : SVM + Bbox regression
+ Fast RCNN: ROI pooling
    + Extract features from input images : CNN 
    + Get ROI: projected region proposals
    + ROI pooling: 
        + Layer input: feature map
        + Layer output: rois(rectangular vectors)
        + Cut each proposal as MxN parts, perform max pooling for each part
        + In this way we can get fixed number of features from each rigion
    + Classification & Bbox Regression: combined as a multi-task model
+ Faster RCNN: RPN
    + Extract features from input images : CNN 
    + Get ROI proposals from RPN
        + Predict proposals from features
        + RPN has classification and bbox regression as well, but rougher
        + Takes image features as input and outputs a set of rectangular object proposals(anchor), each with an objectness score
    + Send proposal and features to ROI pooling
    + Make final classification and bbox regression

## 6. [Robust Hand Detection in Vehicles](http://ieeexplore.ieee.org/document/7899695/)
&hearts; DL , CV , RCNN, Object Detection
+ Modified Faster RCNN
    + Multiple scale Faster-RCNN
    + Weight normalization
    + Add new layer
+ Change 1 : Multiple scale Faster-RCNN
    + Combine both global and local features --> enhance hand detecting in an image
    + Collect features not only conv5, but also conv3 and conv4, then incorporate them
    + Implementation: 
        + For conv3, conv4, conv5, each conv is only followed with ReLU, remove Max-pooling layer.
        + Take their output as the input of 3 corresponding ROI pooling layers and normalization layers
        + Concat and shrink normalization layers as input of fc layers
        + roi pooling in fc layers: make prediction of class and position
+ Change 2: Weight normalization
    + Features in shallower layers: larger-scaled values
    + Features in deeper layers: smaller-scaled values
    + To combine the features of 3 conv layers, we need to normalize them
    + Implementation:
        + Put each feature into normalization layer(see the equations)
        + Each pixel xi is normalized, then multiply scaling factor ri
        + Use backpropagation to get ri in training step, we need to build loop here
        + After normalization, concate the layers
+ Change 3: Add new layer
    + Each RPN needs a normalization layer
    + Add two more ROI pooling layers in detector part
    + Each ROI pooling layer needs a normalization layer
    + After each concatenation(2 positions in total), we need a 1*1 conv layer
+ Some details
    + For RPN:
        + normalize each to_be_normalized layer
        + concat 3 normalized layers
        + change the dimension using 1 * 1 conv
    + For ROI pooling:
        + put each conv output into its ROI pooling (so there should be 3 ROI pooling layers)
        + normalize each layer
        + concat them
        + change the dimension using 1 * 1 conv

## 7. 1612.08242 - [YOLO9000 Better, Faster, Stronger](https://arxiv.org/abs/1612.08242)
&hearts; DL , CV , YOLO, Object Detection

## 8. 1704.03414 - [A-Fast-RCNN: Hard Positive Generation via Adversary for Object Detection](https://arxiv.org/abs/1704.03414)

## 9. 1703.06870 - [Mask R-CNN](https://arxiv.org/abs/1703.06870)
&hearts; DL , CV , RCNN, Object Detection
+ Extend Faster RCNN by adding a branch for predicting segmentation masks for each ROI, in parallel with existing branch for classification and bbox regression
+ Review of Faster RCNN
    + Two stages
        + Get candidate object bboxes from RPN
        + Extract features using RolPool from each candidate box
        + Finally perform classification and bbox regression.
    + The features used by both stages can be shared for faster inference
    + Faster-RCNN is not designed for pixel-to-pixel alignment
        + RoiPool performs coarse spatial quantization for feature extraction
        + How to fix: change it to RoiAlign which is quantization-free and can faithfully preserves exact spatial locations
+ Mask R-CNN
    + Two stages
        + Similar to faster RCNN in the first stage(RPN)
        + When predicting the class and box offset, Mask R-CNN also outputs a binary mask(third branch) for each RoI
    + $L = L_{cls} + L_{box} + L_{mask}$
        + $L_{mask}$ uses per-pixel sigmoid instead of softmax
        + For an ROI associated with ground-truch class k, $L_{mask}$ is only defined on k-th mask
        + Other mask outputs are irrelevent $\rightarrow$ we can generate masks for each class without competition among classes
        + Decouple mask and class prediction
    + Mask branch: 
        + Use a FCN(full collected network) to predics m*m mask from each ROI
        + Need ROI features to be well aligned $\rightarrow$ RoIAlign
    + RoIAlign: 
        + In RoIPool, quantization leads to misalignments
            + Does not affect classification
            + Affects pixel-accurate prediction
        + In RoIAlign, we avoid quantization: round(x / 16) $\rightarrow$ x / 16

    + Archetecture:
        + Extend two kinds of backbone heads:
            + ResNet
            + FPN(feature pyramid network)

+ Mask RCNN is easy to generalize
    + Instance segmentation
    + Bbox object detection
    + Person keypoint detection
        + Model a keypoint’s location as a one-hot m * m mask, where onlu a single pixel is labeled as foreground
        + Adopt Mask R-CNN to predict K masks, one for each of K keypoint types (e.g., left shoulder, right elbow).

## 10. 1612.03144 - [Feature Pyramid Networks for Object Detection](https://arxiv.org/abs/1612.03144)

