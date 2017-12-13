# An Efficient Approach to Model-Based Hierarchical Reinforcement Learning

+ **Yunqiu Xu**

---

## 1. Introduction

+ Current HRL can not handle real world problems: 
    + Multiple tasks
    + Changing subgoals
    + Uncertain subtask specifications
+ Two limitations of MAXQ-based (**pre-defined task hierarchy**) methods
    + All of the tasks / subtasks need to be clearly specified
    + Even similar subtasks have to be learned separately
+ Our work: context-sensitive reinforcement learning
    + Exploit common knowledge in subtasks $\rightarrow$ learn transition dynamics effiently
    + Actively evaluate different subtasks as execution choices
    + Based on simulation

## 2. Problem Setup
+ Task $T_i = \{I_i, G_i, F_i, A_i, T_i, R_i\}$
    + $I_i$ : input set
    + $G_i$ : goal, terminating states
    + $F_i$ : relevent features
    + $A_i$ : actions
    + $T_i$ : transition functions
    + $R_i$ : reward functions
    + 这个比我之前看得 Task 定义要复杂不少
    + Well-defined MDP : $\{F_i, A_i, T_i, R_i\}$

+ Fragment $\{F_j, A_j, T_j\}$
    + No goal states or local reward functions
    + Can describe **similar multiple tasks** $\rightarrow$ share same transition function, but differ in goals
    + How to generate: combine tasks with same $F_i$ and $A_i$
    + How it be used: facilitate efficient learning of task transition functions
    + 例如 Fig 4 中 Get 和 Put 两个任务都可以归于一个 Fragement

![2017-12-13 19-22-49屏幕截图.png-67.6kB][1]

+ General CSRL:
    + Node : task or fragment
    + Different from MAXQ:
        + A task can be decomposed as fragments / smaller tasks
        + Actions are defined inside each node $\rightarrow$ primitive actions won't appear as the leaf nodes
        + $F$ of a node is also the subset of $F$ of its parent
    + **Fragments allow us specify tasks without goal states**
        + 这里按照我的理解就是把两个动作/特征相似的任务归为一类
        + 例如抓取和放置都可以归类为定位
        + 因此学会抓取之后很快就能学会放置

## 3. Algorithm
### 3.1 Overview of CSRL

![2017-12-13 19-47-37屏幕截图.png-122.7kB][2]

+ $m$ : exploration threshold
+ $P_{k,a} = Parent(C_k, a)$ : the parent of $C_k$ with action $a$
+ Line 6 : init $P_{k,a}$
+ $n(P_{k,a} , a)$ : exploration count
    + If smaller than $m$ $\rightarrow$ transits to a fictitious component $C_k^f$
    + Else $\rightarrow$ update probability values
+ Line 18-21 : select a new task when a task is finished

### 3.2 Select a new task

![2017-12-13 20-20-23屏幕截图.png-66.6kB][3]

+ Model task selection as an S(semi)MDP
+ Recursively : given a task policy for any node, we can construct the task selection SMDP at the parent node

### 3.3 Task Simulation
+ Why simulate the task?
    + As the transition function has been computed for each node in the hierarchy
    + The parameters can be more efficiently estimated by simulating 
+ What do we simulate?
    + The effect of executing the task policy on the task’s parent node’s transition function

+ How to simulate?

![2017-12-13 20-26-09屏幕截图.png-84.8kB][4]

+ Given the policies of the child tasks, CSRL can simulate the results of executing the task on the root node

## 4. Experiments
+ Robot pickup and place : does not require task selection
+ Pickup and place with two objects
+ Household robot experiment
    + Requires multiple levels of reasoning
    + Cannot be solved using existing methods due to incomplete problem specification at the lower level

![2017-12-13 20-31-41屏幕截图.png-145.5kB][5]

## 5. Conclusion
+ Task learning mechanism: learn both task and global transition dynamics
+ Hierarchical execution mechanism: handle task selection by formulating and solving the underlying SMDP
+ Limitation:
    + Specify relevant features manually
    + Can not build sub-tasks automatically
    + No transfer learning : in the future we can try to perform transfer learning for those with similar hierarchy
        
            


  [1]: http://static.zybuluo.com/VenturerXu/95px5bwv1mgp9jsvp2x23krj/2017-12-13%2019-22-49%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE.png
  [2]: http://static.zybuluo.com/VenturerXu/259wjizgbwfeu307iu9y692b/2017-12-13%2019-47-37%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE.png
  [3]: http://static.zybuluo.com/VenturerXu/7h8n9igjc399fln92p450bcu/2017-12-13%2020-20-23%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE.png
  [4]: http://static.zybuluo.com/VenturerXu/qg4ugo9bxmidx44b4iidwolm/2017-12-13%2020-26-09%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE.png
  [5]: http://static.zybuluo.com/VenturerXu/ak23fyrfvw8rfh1id03v4lwn/2017-12-13%2020-31-41%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE.png