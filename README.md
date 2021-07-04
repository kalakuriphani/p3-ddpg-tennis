# The Environment

![Alt text](images/env.png?raw=true "Unity ML-Agents Reacher Environment")
<p>
In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
This yields a single score for each episode.
The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.


## Environment Info 
Number of agents: 2 <br>
Number of Brains: 1 <br>
Number of external Brains: 1 <br>
Number of Visual Observations (per agent): 0 <br>
Vector Observation space type: continuous <br>
Vector Observation space size (per agent): 8 <br>
Number of stacked Vector Observation: 3 <br>
Vector Action space type: continuous <br>
Vector Action space size (per agent): 2 <br>



# Installation Steps:
The project is built on conda 3 and can be created by exporting the environment.yml file. By executing the following command

conda env create -f environment.yml

## Setup the environment:
Following installations are available and can be downloaded from here <br>
   -  <a href="https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip">Mac OSX</a>
   -  <a href="https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip">Linux</a>
   -  <a href="https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip">Windows 64</a>

## To Train the agent
Run the following command to train and run the agent. <br>
python train_agent.py

## Run the agent
To Run the use the following command <br>
python run_agent.py

