import os
from unityagents import UnityEnvironment
from tennis.maddpg_agent import MADDPG
import numpy as np
from tennis.environment import make_plot,make_plot_learning



if __name__ == '__main__':
    env = UnityEnvironment('./unity/Tennis.app')
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]
    state_dim = int(env_info.vector_observations.shape[1])
    action_dim = int(brain.vector_action_space_size)
    states = env_info.vector_observations
    num_agents = len(env_info.agents)
    agent = MADDPG(num_agents, state_dim, action_dim, 66)
    # Load agent
    agent.loadCheckPoints(isFinal=True)
    score = 0
    n_episodes = 30

    scores = []
    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations

        # initialize score value calculations for 2 agents
        scores_episode = np.zeros(num_agents)

        while True:
            actions = agent.act(states)  # get action from agent
            env_info = env.step(actions)[brain_name]  # send actions to the environment
            next_states = env_info.vector_observations  # get next states
            rewards = env_info.rewards  # get rewards
            dones = env_info.local_done  # see if episodes finished

            # Append stats
            scores_episode += rewards
            states = next_states

            # break if any agents are done
            if np.any(dones):
                break

        # calculate intermediate stats
        score = np.max(scores_episode)
        scores.append(score)

        # display current stats
        print('Episode {}\tAverage Score: {:.4f}\tCurrent Score: {:.4f}\t Max Score: {:.4f}'
              .format(i_episode, np.mean(scores), score, np.max(scores)))

    scores_file = './scores/scores_test.npz'
    np.savez(scores_file, scores)
    #make_plot_scores(scores_file,'./images/scores_test.png')
    #make_plot_learning(scores_file,'./images/scores_test.png')