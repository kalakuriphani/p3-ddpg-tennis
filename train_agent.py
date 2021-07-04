from tennis.environment import setup,train,make_plot_learning,make_plot
from unityagents import UnityEnvironment
import os

env = UnityEnvironment('./unity/Tennis.app')

if __name__ == '__main__':

    for name in ['./images/score.png','./images/learning.png']:
        if os.path.isfile(name):
            os.remove(name)

    agent = setup(env)
    print("Started Agent Training")
    train(agent,env)
    make_plot()
    make_plot_learning()
