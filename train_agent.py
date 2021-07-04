from tennis.environment import setup,train,make_plot_learning,make_plot
from unityagents import UnityEnvironment
import os

env = UnityEnvironment('./unity/Tennis.app')

if __name__ == '__main__':

    scores_file = './scores/scores_train.npz'
    scores_image = './images/scores.png'
    learning_image = './images/learning.png'

    for name in [scores_image,learning_image]:
        if os.path.isfile(name):
            os.remove(name)

    agent = setup(env)
    print("Started Agent Training")
    train(agent,env)

    make_plot(scores_file,scores_image)
    make_plot_learning(scores_file,learning_image)
