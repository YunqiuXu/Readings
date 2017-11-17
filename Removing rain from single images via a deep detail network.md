## Removing rain from single images via a deep detail network
+ Other refs:
    + https://www.leiphone.com/news/201707/ZXZ450ilP3PnyUUx.html
    + https://www.zhihu.com/question/57523080/answer/212756693
+ 类似ResNet的思路，回归带雨图像与原图的残差，而不是直接输出还原图像。这样一来可以使算法操作的图像目标值域缩小，稀疏性增强。实际上这一点在超分辨率等很多问题中已经被广泛应用。
+ 使用频域变换，分离图像中的低频部分和高频部分，只对高频部分做去雨操作。原因是雨滴基本只存在于高频部分，分离后可以使得操作目标进一步稀疏化，实验效果显著优于不做这一操作的结果。
