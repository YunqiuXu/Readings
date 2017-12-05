# Reading 171205
## 1. IRL
+ Recover reward function from demonstrations
+ Better than behavioral cloning
+ Hard to evaluate learned reward, especially for high-dim data (image)_
+ 1606.03476 - GAIL : GAIL is a kind of IRL, use GAN to generate reward function --> GAN is hard to train

## 2. H-DRL
+ Challenge : long term reward + sparse reward

+ 1703.01161 - FeUdal Net
    + Early version: Dayan & Hinton 1993
    + Sub-goal discovery
    + Manager + Worker
    + Can only handle single-task

+ 1704.03012 - Stocastic NN
    + Construct a pretraining environment with minimal domain knowledge
    + Learns a span of skills via SNN
    + Then it can be applied to more challenging scenario
    + Compared with FeUdal Net --> do not require domain-specific knowledge

+ 1709.04579 ARM-HSTRL
    + Use association rule mining to generate hierarchical structure automatically
    + Can handle multi-task


## 3. Meta-learning
+ 1703.03400 - MAML:
    + The initial parameters of the model are explicitly trained first
    + Then this model can be generated to new task with a small number of training samples / gradient steps on that task
    + Try to learn "as much as possible" with in a few gradient updates
    + The test error of training tasks will be treated as the training error of meta-learning process 

+ 1709.04905 - One-Shot Visual Imitation Learning via Meta-Learning
    + Try to achieve "one-shot" : less demonstrations for new task
    + MIL is an application of MAML
    + Share data across tasks:
        + Use these data (demonstrations of other tasks) for meta learning
        + Then we can learn a new task from its single demonstration
    + Note this work can be combined with [1707.03374]() --> learn from video or human

+ 1710.03641 - Continuous Adaptation via Meta-Learning in Nonstationary and Competitive Environments
    + Try to handle non-stationary environment
    + non-stationary environment can be seen as a sequence of stationary tasks $\rightarrow$ multitask problem
    + Develop a simple gradient-based meta-learning algorithm suitable for adaptation in dynamically changing and adversarial scenarios

+ 1710.09767 - Meta Learning Shared Hierarchies
    + Meta + H-rl --> speed up meta learning
    + Repeatedly reset the master policy to adapt the sub-policies for fast learning
    + Our work can achieve multi-task
    + Try to learn "as fast as possible" with a lot of updates
    + Warm-up period: optimize master policy
    + Joint update period: treat the master policy as an extension of the environment, optimize both master and subgoal

## 5. Others
+ 1707.05300 - Reverse Curriculum
    + Challenges: sparse reward + demonstrations
    + This work does not use reward engineering and demonstrations
        + Train the robot to reach the goal which the start position is nearby the goal
        + Then train it from further start position
    + How to choose start position:
        + Perform random walk from previous start states
        + You can get reward by starting from these states: can reach final state
        + But these are not best start states: require more training
    + Limitation: 
        + Choose goal: use goal generation
        + Sim-to-reality : use domain randomization
+ 1704.03732 - DQfD
    + Use demonstration in DQN --> accelerate the learning process
    + Be able to automatically assess the necessary ratio of demonstration data -->  prioritized replay
    + DQfD works by combining temporal difference updates with supervised classification of the demonstrator’s actions

## 4. Imitation learning:
+ Traditional and their limitation:
    + Off-policy : BC, can not 举一反三
    + On-policy : large work, inefficient
    + IRL : can not handle high dim

+ 1703.06907
    + Use domain randomization to handle sim-to-real transfer
    + Train on similated images and then transfer to real images by randomizing rendering in the simulator

+ 1703.07326 - One-Shot Imitation Learning
    + Train a policy on a large set of tasks
    + Input current state and one demonstration --> output curent control
    + Demonstration network : demonstration -> embedding
    + Context network : embedding + current state --> context embedding 
    + Manipulation network : context --> action

+ 1703.09327 - DART:
    + Add noise to broden supervisor's demonstration
    + Make off-policy (BC) more robust

+ 1707.02747 - Robust Imitation of Diverse Behaviors
    + Combine VAE with GAN
    + VAE: BC, learn semantic policy embeddings
    + GAN : Diverse generative adversarial imitation learning, produce diverse solutions
    + Note that the evaluation is based on the diversity of policies, not game score

+ 1707.03374 - Imitation from Observation: Learning to Imitate Behaviors from Raw Video via Context Translation
    + Handle imitation-from-observation problem: imitate from video
    + Context translation model: transform third-person viewpoint to first-person viewpoint
    + Simplify the problem as : 
        + the context can vary between the demonstrations and the learner
        + but the learner’s context still comes from the same distribution
    + Limitation: learn the translation model
        + Require a lot of demonstrations
        + Require observations of demonstrations from multiple contexts to learn to translate
