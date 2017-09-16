## [Robust Hand Detection in Vehicles](http://ieeexplore.ieee.org/document/7899695/)
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
