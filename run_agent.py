import os
from unityagents import UnityEnvironment
from tennis.maddpg_agent import MADDPG


env = UnityEnvironment('unity/Tennis.app')

if __name__ == '__main__':

    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]
    action_size = brain.vector_action_space_size
    state_size = env_info.vector_observations.shape[1]
    states = env_info.vector_observations[0]
    score = 0
    agent = MADDPG(len(env_info.agents),state_size,action_size)
    agent.loadCheckPoints(True)

    while True:
        action = agent.act(states)
        env_info = env.step(action)[brain_name]
        next_state = env_info.vector_observations
        reward = env_info.rewards
        done = env_info.local_done
        score += reward
        state = next_state
        if done:
            break

    print("Score: {}".format(score))