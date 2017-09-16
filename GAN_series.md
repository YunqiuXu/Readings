# Notes for GAN (updated on 20170916)
+ Ref:
    + [从头开始GAN](https://zhuanlan.zhihu.com/p/27012520)
    + [《Conditional Generative Adversarial Nets》阅读笔记](https://zhuanlan.zhihu.com/p/23648795)
    + [Image-to-Image Translation with Conditional Adversarial Networks》阅读笔记](https://zhuanlan.zhihu.com/p/24248684)
    + [异父异母的三胞胎：CycleGAN, DiscoGAN, DualGAN](https://zhuanlan.zhihu.com/p/26332365)
    + [CycleGAN（以及DiscoGAN和DualGAN）简介](https://zhuanlan.zhihu.com/p/27539515)

## 1. GAN
+ GAN = 生成网络 G + 判别网络 D
+ 两个网络的loss
    + G: $log(1-D(G(z)))$
        + 尽可能让 D(G(z)) 接近 1, 这样loss 才会小
        + 即尽可能让自己输出接近真实的数据
    + D: $min_G max_D V(D,G) = E_x\[log(D(x))\] + E_z\[log(1-D(G(z)))\]$
        + 分清真实数据和虚假数据
        + 对真实数据x, D尽可能输出1
        + 对生成的假数据G(z), D尽可能输出0
        + 这两部分组合起来得到总的loss function
+ 输入: 
    + G: 噪音z
    + D: 真实数据x以及G的输出G(z)(即生成的假数据)

## 2. DC-GAN
+ 开山之作之后的通用版, 使用神经网络，并且实现有效训练
+ 优化了网络结构，加入了 conv，batch_norm 等层，使得网络更容易训练
+ G网络使用4层反卷积，而D网络使用了4层卷积, 基本上G网络和D网络的结构正好是反过来的
+ [TF implementation](https://github.com/carpedm20/DCGAN-tensorflow)

## 3. CGAN(conditional)
+ 注意Hao Li那篇文章中用的就是这个
+ D和G的输入都增加了一个约束y用于引导数据的生成过程: 
    + 在G网络的输入在噪音z的基础上连接条件y, 再通过非线性函数映射到数据空间
    + 在D网络的输入在真实数据x的基础上也连接条件y, 再进一步判断x是真实数据的概率
    + 最后得到D(x|y), G(z|y)
+ 为什么这样做, 之前太自由了, 对于较大图片的情形不可控, 而CGAN加入了条件约束从无监督变为有监督
+ $min_G max_D V(D,G) = E_x\[log D(x|y)\] + E_z\[log (1 - D(G(z|y)))\]$

## 4. Pix2Pix
+ 注意Dat Tran的demo中用的就是这个
+ 根据CGAN提出用于image2image转译的通用框架
+ 本文结构: 以根据实景生成地图为例
    + G:
        + 输入噪音z(地图), 生成虚假实景数据
    + D:
        + 输入真实实景数据x或者虚假实景数据G(z) + 噪音z(地图), 判断输入的是真实数据还是虚假数据 
+ 实现细节:
    + 对目标函数进行修改
        + $min_G max_D V(D,G) = E_x\[log D(x, y)\] + E_z\[log (1 - D(x, G(x, z))\]$
        + 加入L1约束项 --> ||1 - D||_1 : 生成图像不仅要像真实数据(实景), 还要接近于输入图片(地图)
        
    + 在生成器中，用U-net结构代替encoder-decoder
        + 原因: 图像的底层特征也很重要
    + PatchGAN
        + 分块判断, 在图像的每个n*n区域判断是否为真
        + 平均给出结果 --> 有种ensemble learning的感觉
+ 本文适合生成色彩丰富的图片, 对低饱和度数据效果不好

##５．ＣycleGAN | DiscoGAN | DualGAN
+ 不需要成对数据, 只需要两类图片, 例如 folder_A(冬天) --> folder_B(夏天)
+ 和一般GAN一个生成器一个判别器不同, CycleGAN有两个生成器和判别器
    + 生成器G_A2B: 将A类图片转换成B类图片
    + 生成器G_B2A: 将B类图片转换成A类图片
    + 两个判别器D_a, D_b : 分辨两类中的真实图片和生成图片
+ 循环损失:
$$L_cyc (G_{A2B}, G_{B2A}, A,B) = E_{x \in A}[||G_{B2A}(G_{A2B}(x)) - x||_1] + E_{y \in B}[||G_{A2B}(G_{B2A}(y)) - y||_1]$$
+ 循环损失作为总损失的一部分:
$$L(G_{A2B}, G_{B2A}, A,B) = L_G(G_{A2B}, D_B, A,B) + L_G(G_{B2A}, D_A, B,A) + \lambdaL_cyc (G_{A2B}, G_{B2A}, A,B)$$
+ 训练细节:
    + 训练比配对的GAN要慢, batch size为1的情况下，每个epoch训1000张图，差不多要近100个epoch才能得到比较能够接受的结果
    + 自适应学习率的Adam并不适合GAN --> GAN中损失函数并不代表训练进度, 无法反映结果优劣
+ 一个insight: pix2pix产生的数据可以喂给CycleGAN进行训练
