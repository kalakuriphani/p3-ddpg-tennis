from collections import deque
from tennis.maddpg_agent import MADDPG
import matplotlib.pyplot as plt
import numpy as np
import torch
import os

def make_plot(show=False):
    """Makes a pretty training plot call score.png.
    Args:
        show (bool):  If True, show the image.  If False, save the image.
    """



    # Load the previous scores and calculated running mean of 100 runs
    # ---------------------------------------------------------------------------------------
    with np.load('scores.npz') as data:
        scores = data['arr_0']
    cum_sum = np.cumsum(np.insert(scores, 0, 0))
    rolling_mean = (cum_sum[100:] - cum_sum[:-100]) / 100

    # Make a pretty plot
    # ---------------------------------------------------------------------------------------
    plt.figure()
    x_max = len(scores)
    y_min = scores.min() - 1
    x = np.arange(x_max)
    plt.scatter(x, scores, s=2, c='k', label='Raw Scores', zorder=4)
    plt.plot(x[99:], rolling_mean, lw=2, label='Rolling Mean', zorder=3)
    plt.scatter(x_max, rolling_mean[-1], c='g', s=40, marker='*', label='Episode {}'.format(x_max), zorder=5)
    plt.plot([0, x_max], [30, 30], lw=1, c='grey', ls='--', label='Target Score = 30', zorder=1)
    plt.plot([x_max, x_max], [y_min, rolling_mean[-1]], lw=1, c='grey', ls='--', label=None, zorder=2)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.legend()
    plt.xlim([0, x_max + 5])
    plt.ylim(bottom=y_min)
    if show:
        plt.show()
    else:
        plt.savefig('./images/scores.png', dpi=200)
    plt.close()

def make_plot_learning(show=False):
    with np.load('scores.npz') as data:
        scores = data['arr_0']
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Average Score')
    plt.xlabel('Episode #')
    if show:
        plt.show()
    else:
        plt.savefig('./images/learning.png', dpi=200)
    plt.close()

def train(agent,env,n_episodes=2_000,max_t=1000,save_models=True):
    """
    :param agent: The agent to train
    :param env: The trainining environment
    :param n_episodes: maximum number of training episodes
    :param max_t: maximum number of time steps per episode
    :return:
    """
    scores = list()
    scores_window = deque(maxlen=100)
    brain_name = env.brain_names[0]
    total_timesteps = 0
    brain = env.brains[brain_name]

    # Save pytorch_models weights and scores


    for i_episode in range(1,n_episodes+1):
        brain_info = env.reset(train_mode=True)[brain_name]
        states = brain_info.vector_observations
        # Reset the agents noise level
        agent.reset()
        # initialize score value calculations for 2 agents
        scores_episode = np.zeros(len(brain_info.agents))
        while True:
            actions = agent.act(states)  # get action from agent
            env_info = env.step(actions)[brain_name]  # send actions to the environment
            next_states = env_info.vector_observations  # get next states
            rewards = env_info.rewards  # get rewards
            dones = env_info.local_done  # see if episodes finished
            agent.step(states, actions, rewards, next_states, dones)  # perform optimization
            # Append stats
            scores_episode += rewards
            states = next_states

            # break if any agents are done
            if np.any(dones):
                break

        score = np.max(scores_episode)
        scores_window.append(score)
        scores.append(score)

        # display current stats
        print('\rEpisode {}\tAverage Score: {:.4f}\tCurrent Score: {:.4f}\t Max Score: {:.4f}'
              .format(i_episode, np.mean(scores_window), score, np.max(scores_window)), end="")


        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.4f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window) >= 0.5:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode,
                                                                                         np.mean(scores_window)))
            #agent.saveCheckPoints(True)
            #agent.save()
            if save_models and not os.path.exists("./pytorch_models"):
                os.makedirs("./pytorch_models")
            file_name = "%s_%s_%s" % ("DDPG", brain_name, str(0))
            agent.save("%s" % file_name, directory="./pytorch_models")
            break


    np.savez('scores.npz', scores)


def setup(env):
    """
    Setups up the environment for training
    """
    print("Setting up the environment")
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]
    action_size = brain.vector_action_space_size
    state_size = env_info.vector_observations.shape[1]
    num_agents = len(env_info.agents)

    #------- Setup MADDPG Agent ------#
    return MADDPG(num_agents,state_size,action_size)


