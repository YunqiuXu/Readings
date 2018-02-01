# Reinforcement Learning From Imperfect Demonstration

![2018-02-01 20-56-03屏幕截图.png-54.4kB][1]

+ **Yunqiu Xu**
+ Normalized Actor-Critic (NAC):
    + Similar to DQfD, but can use imperfect (noisy) demonstration
    + **Normalize the Q function, reduce the Q values of actions unseen in the demonstration data**
    + Learn an initial policy network from demonstration, then refine it in real environment (same with DQfD here)
    + Good performance on realistic driving games

---

## 1. Introduction
+ Challenges:
    + DRL requires a large amount of interaction with an environment
    + The weakness of learning from demonstration: 
        + Does not leverage reward
        + Imitation: the expert demonstration should be noise-free and near optimal

+ Our work: NAC
    + Base on maximum entropy RL
    + Use both offline demonstration and online interaction experience
    + **Do not use supervised loss function**
    + Use a normalized formulation of soft Q-learning gradient (or a variant of policy gradient)
    + **Utilize rewards rather than simply imitating behavior by supervised learning**
    + Do not need optimal demonstration $\rightarrow$ **learn robustly even on corrupted demonstration**

## 2. Related Work
### 2.1 Learning from demonstration
+ DQfD / DDPGfD / DDPG + HER : Need expert demonstration (near optimal)
+ Our difference: 
    + Do not require explicit mechanism to determine expert data     + Learn to maximize reward using arbitrary data

### 2.2 Maximum Entropy RL and Soft Value Functions

+ Our work is based on
    + Harrnoja et al. 1702.08165 - Reinforcement Learning with deep energy-based policies
    + Schulman et al. 1704.06440 - Equivalence between policy gradients and soft q-learning

+ Maximum entropy RL

![2018-02-01 21-37-50屏幕截图.png-178.2kB][2]

+ Soft Value Functions

![2018-02-01 21-42-38屏幕截图.png-44.9kB][3]

+ Soft Q-learning and policy gradient

![2018-02-01 21-43-38屏幕截图.png-106.8kB][4]
        
+ Our difference:
    + Focus on learning from demonstration
    + Combine PG and Q-learning for RLfD without explicit imitation (supervised) loss

## 3. Robust Learning from Demonstration and Reward
+ Why can't us train only on good demonstration?
    + The agent can not understand why the action is good
    + **The agent can assign actions high Q-value, but can not assign other actions low Q-values**
+ NAC
    + Combine soft Q-learning and soft policy gradient
    + New Q-function gradient: **give actions not in demonstration low Q-values**

### 3.1 Details of NAC

![2018-02-01 22-46-59屏幕截图.png-139.8kB][5]

+ Different versions of NAC
    + Q-learning variant: omits $\nabla_{\theta}V(s)$ in Eq.9.
    + PG variant: requires importance sampling to use off-policy data

+ Algorithm: 

![2018-02-01 22-49-49屏幕截图.png-135.7kB][6]

### 3.2 Why NAF can reduce Q-value those not observed in demonstration
+ Traditional Q-learning:
    + Can not know whether the action itself is good
    + Can not know whether all actions in that states are good
    + So actions in demonstration may not necessary to have higher value than other actions in that state

![2018-02-01 22-56-52屏幕截图.png-5.6kB][7]
![2018-02-01 22-56-42屏幕截图.png-10.3kB][8]

+ In NAF, actor's update follows Eq.9.
    + Compared with soft Q-learning update in Eq.7, there is an extra term $-\nabla_{\theta}V_Q(s)$
    + When Q value $Q(s,a)$ increases, $V_Q(s)$ will decrease because of different sign
    + **$V_Q(s) = \alpha log \sum_a exp(Q(s,a) / \alpha))$ $\rightarrow$ If $V_Q(s)$ is decreasing, $Q(s,a)$ will not increase if actions are not observed in the demonstration**
    + **感觉这里还是略抽象, 需要进一步理解**

## 4. Experiment

+ Questions:
    + Can NAC benefit from both demonstrations and rewards
    + Can NAC handle imperfect demonstrations
    + Can NAC learn meaningful behaviors with a few demonstrations

+ Baselines:
    + DQfD: supervised loss + RL loss
    + Behavioral cloning: supervised imitation
    + DQN: similar to DQfD, DQN is pretrained with demonstration, then fine-tune to environment
    + Soft DQN: DQN + entropy regularized reward
    + NAC with importance sampling weighting:
        + **Why use importance weighting term: correct the action distribution mismatch between the demonstration and current policy**

+ Environments: Torcs and GTA

![2018-02-01 23-08-36屏幕截图.png-174.3kB][9]

+ Results

![2018-02-01 23-15-41屏幕截图.png-336.5kB][10]

![2018-02-01 23-16-42屏幕截图.png-223.3kB][11]

![2018-02-01 23-17-51屏幕截图.png-276.9kB][12]

## 5. Summary
+ 和之前DQfD等工作类似都是通过demonstration加速强化学习
+ 不同的是DQfD要求demonstration必须是很好的, 而NAC允许imperfect demonstration, 在demonstration不那么好时, NAC性能更好 (见Fig.4.)
+ NAC是如何达成这一目的的?
    + DQfD等方法的不足在于, 只能判定看到的动作还不错, 但对于没看到的动作没法判定它们不好
    + 结合了soft Q-learning以及soft policy gradient
    + 对于未在demonstration中出现的动作, 降低他们的Q-value
+ 还需再理解下原理, 有机会的话尝试实现下, 毕竟我们创建的demonstration不一定是完美的
    


        


  [1]: http://static.zybuluo.com/VenturerXu/fna9pidxqjkjfn00w2ixx2ia/2018-02-01%2020-56-03%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE.png
  [2]: http://static.zybuluo.com/VenturerXu/we06g223l5jvaj2q97juxgv4/2018-02-01%2021-37-50%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE.png
  [3]: http://static.zybuluo.com/VenturerXu/ae0xwef2sinmzrw6om15g4fo/2018-02-01%2021-42-38%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE.png
  [4]: http://static.zybuluo.com/VenturerXu/xi2r9qo323v5hx6sdy6y3oh8/2018-02-01%2021-43-38%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE.png
  [5]: http://static.zybuluo.com/VenturerXu/s3airu0fr0bsdyz5bfbzxtna/2018-02-01%2022-46-59%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE.png
  [6]: http://static.zybuluo.com/VenturerXu/dezqo6yo4mgycnvt1mc3zfeu/2018-02-01%2022-49-49%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE.png
  [7]: http://static.zybuluo.com/VenturerXu/kl1cinhaapc73h0r7tu88057/2018-02-01%2022-56-52%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE.png
  [8]: http://static.zybuluo.com/VenturerXu/oj1hkz42859ykrn1rwptx7ue/2018-02-01%2022-56-42%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE.png
  [9]: http://static.zybuluo.com/VenturerXu/d27iyju4e2cnffa56dtodmbf/2018-02-01%2023-08-36%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE.png
  [10]: http://static.zybuluo.com/VenturerXu/k1cchoulmvrbxuwn89xx3yui/2018-02-01%2023-15-41%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE.png
  [11]: http://static.zybuluo.com/VenturerXu/or922u99ztrwcrmxqa4ik6ts/2018-02-01%2023-16-42%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE.png
  [12]: http://static.zybuluo.com/VenturerXu/p3ij0brdi91hhs0jf7b3csws/2018-02-01%2023-17-51%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE.png