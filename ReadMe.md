# CartPole - Reinforcement Learning
Aims to solve CartPole with various RL-Techniques at an intermediate level.

## Techniques
- [X] [ Q-Tables ](#qtables)
- [X] [ Deep Q Learning(Fixed Targets, Memory, Double-Q) ](#qlearning)
- [X] [ Policy Gradient ](#policygradient)
- [ ] [ Actor Critic ](#actorcritic)

Deep Q-Learning techniques will be mixed and tested.

# Results 
X-Axis = Epochs/100
Q = Standard Q-Learning, T = Fixed Targets, M = Memory, DQ = Double-Q

<a name="qtables"></a>
## Q-Tables - Input Discretized
Since discretization leads to reduced accuracy this problem cant be completely solved.
Finding a sweetspot between accuracy-tradeoff and value-conservation might lead to a solution.
Another way of solving this problem could be by broadening the observation-space(10^4 -> 10^8 / 9,999 -> 99,999,999).

<p float="left">
  <img src="./Plots/CartPole-QTable/meanreward.png" width="425" />
  <img src="./Plots/CartPole-QTable/meanloss.png" width="425" /> 
  <img src="./Plots/CartPole-QTable/meanqvalue.png" width="425" />
</p>

<a name="qlearning"></a>
## Q-Learning
<p float="left">
  <img src="./Plots/CartPole-Q/meanreward.png" width="425" />
  <img src="./Plots/CartPole-Q/meanloss.png" width="425" /> 
  <img src="./Plots/CartPole-Q/meanqvalue.png" width="425" />
</p>

## Q-Learning(Fixed Targets)
Epochs = 30000, Replacement-Interval = 2, γ = 0.92, Exploration = 20%, Learning-Rate = 0.00025
Unstable and performs poorly after there is no exploration or a high rewardmean is reached.
Reducing the update interval(to 5-1) helps but is pointless.
Can be solved by using a Learning-Rate scheduler.

<p float="left">
  <img src="./Plots/CartPole-TQ/meanreward.png" width="425" />
  <img src="./Plots/CartPole-TQ/meanloss.png" width="425" /> 
  <img src="./Plots/CartPole-TQ/meanqvalue.png" width="425" />
</p>

## Q-Learning(Memory)
<p float="left">
  <img src="./Plots/CartPole-MQ/meanreward.png" width="425" />
  <img src="./Plots/CartPole-MQ/meanloss.png" width="425" /> 
  <img src="./Plots/CartPole-MQ/meanqvalue.png" width="425" />
</p>

## Q-Learning(Fixed Targets, Memory)
Epochs = 30000, Replacement-Interval = 150, γ = 0.99, Exploration = 6%, Learning-Rate = 0.00025, Memorysize = 500, Batchsize = 8, Warmup-Steps = 150
Unstable and performs poorly after there is no exploration or a high rewardmean is reached.

<p float="left">
  <img src="./Plots/CartPole-TMQ/meanreward.png" width="425" />
  <img src="./Plots/CartPole-TMQ/meanloss.png" width="425" /> 
  <img src="./Plots/CartPole-TMQ/meanqvalue.png" width="425" />
</p>

## Q-Learning(Double Q)
<p float="left">
  <img src="./Plots/CartPole-DQ/meanreward.png" width="425" />
  <img src="./Plots/CartPole-DQ/meanloss.png" width="425" /> 
  <img src="./Plots/CartPole-DQ/meanqvalue.png" width="425" />
</p>

## Q-Learning(Double Q, Memory)
<p float="left">
  <img src="./Plots/CartPole-MDQ/meanreward.png" width="425" />
  <img src="./Plots/CartPole-MDQ/meanloss.png" width="425" /> 
  <img src="./Plots/CartPole-MDQ/meanqvalue.png" width="425" />
</p>

<a name="policygradient"></a>
## PolicyGradient
<p float="left">
  <img src="./Plots/PolicyGradient/meanreward.png" width="425" />
  <img src="./Plots/PolicyGradient/meanloss.png" width="425" /> 
</p>

### Normalized Discounted Rewards(also smaller Network and higher Learning Rate)

<p float="left">  
  <img src="./Plots/PolicyGradient/normmeanreward.png" width="425" />
  <img src="./Plots/PolicyGradient/normmeanloss.png" width="425" /> 
</p>

<a name="actorcritic"></a>
## Actor Critic
<p float="left">
  <img src="./Plots/AC/meanreward.png" width="425" />
  <img src="./Plots/AC/meanloss.png" width="425" /> 
</p>
