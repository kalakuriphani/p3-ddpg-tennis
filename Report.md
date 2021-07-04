#Implementation
## Problem Statement:
In Tennis app unity environment the multi agent must get an average score of +0.5 over 100 consecutive episodes.
##Algorithm:
Deep Deterministic Policy Gradient (DDPG) is a reinforcement learning technique that combines both Q-learning and Policy gradients. DDPG being an actor-critic technique consists of two models: Actor and Critic. The actor is a policy network that takes the state as input and outputs the exact action (continuous), instead of a probability distribution over actions. The critic is a Q-value network that takes in state and action as input and outputs the Q-value. DDPG is an “off”-policy method. DDPG is used in the continuous action setting and the “deterministic” in DDPG refers to the fact that the actor computes the action directly instead of a probability distribution over actions.
DDPG is used in a continuous action setting and is an improvement over the vanilla actor-critic.<br>

![Alt text](images/ddpg.png?raw=true "DDPG Algorithm")
#Implementation
Following are the hyper-parameters used to solve the problem.

- BATCH_SIZE = 256
- BUFFER_SIZE = int(1e5)
- GAMMA = 0.9
- TAU = 1e-2
- LR_ACTOR = 2e-4
- LR_CRITIC = 2e-4
- POLICY_NOISE = 0.2
- NOISE_CLIP =0.5
-WEIGHT_DECAY = 0.0


## Neural Network Architecture
Two neural network models with one Actor , Critic for current and Actor, Critic for target models has been implemented with the following architecture

## Actor Model:

  - Total Layers: 3 with final Activation function with Tanh<br>
  - All input parameters normalized using BatchNorm1d  
  - First Layer with Input shape 24, output dimension 128 with Relu Activation
  - Second Layer with Input shape 128, output dimension 128 with Relu Activation
  - Third Layer with Input shape 128 and output dimension 2 with Tanh Activation

## Critic Model:

  - Total Layers: 3 
  - All input parameters normalized using BatchNorm1d  
  - First Layer with Input shape 24, output dimension 128 with Relu Activation
  - Second Layer with Input shape 128 + action_dim, output dimension 128 with Relu Activation
  - Third layer without any activation function.


### Training
The DDPG Agent took around 1412 episodes to achieve the average score of 0.5 <br>

![Alt text](images/scores.png?raw=true "Agent Score")
![Alt text](images/learning.png?raw=true "Agent Score")

### Inference
After successfully training the agent and tested the model with 30 episodes

![Alt text](images/scores_test.png?raw=true "Agent Test Score")



# Next Steps / Improvements:

- ## Implementing using TD3 Model (Twin Delay DDPG):
TD3 is the successor to the DDPG. Up unitl recently, DDPG was one of the most used algorithms for continuous control problems.
Although DDPG is capable of providing excellent results, it has its drawbacks. Like many RL algorithms training DDPG can be unstable and heavily reliant  on finding the hyper parameters.
This is caused by the algorithm continuously over estimating the Q values of the critic (value) network. These estimation errors build up over time and can lead to the agent falling into a 
local optima or experience catastrophic forgetting.
TD3 address this issues by focusing on reducing the overestimation bias. 
    - Using a pair of critic networks (The twin part of the title)
    - Delayed updates of the actor (The delayed part)
    - Action noise regularisation
<br>

Following are the Algorithm Steps:

1. Initialize Network
2. Initialize Replay Buffer
3. Select and carry out action with exploration noise
4. Store Transitions
5. Update Critic
6. Find the minimum of Critic Targets.
7. Update Actor
8. Update Target networks.
9. Repeat until sentient.




