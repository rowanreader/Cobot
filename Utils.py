import numpy as np
import matplotlib.pyplot as plt

def plot_average_curve(x, scores, title, figure_file, y):
    running_avg = np.zeros(len(scores))
    plt.clf()
    num = len(running_avg)
    for i in range(num):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])

    x = [i + 1 for i in range(num)]
    plt.plot(x, running_avg)
    plt.title(title)
    plt.xlabel("Episodes")
    plt.ylabel(y)
    # plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)

def plot_learning_curve(x, scores, title, figure_file, y):
    plt.clf()
    plt.plot(x, scores)
    plt.title(title)
    plt.xlabel("Episodes")
    plt.ylabel(y)
    # plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)