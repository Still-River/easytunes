import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation
import matplotlib

def get_most_recent_log_file():
    files = os.listdir()
    files = [f for f in files if f.startswith('loss-') and f.endswith('.csv')]
    files.sort()
    return files[-1]

def update(frame):
    plt.clf()
    file = get_most_recent_log_file()
    df = pd.read_csv(file)
    sns.lineplot(x='batch_index', y='loss', hue='epoch', data=df)

def show_results_loop(interval=30000):  # interval in milliseconds
    fig = plt.figure()
    anim = FuncAnimation(fig, update, interval=interval)
    plt.show()

def show_res(file=None):
    if file is None:
        file = get_most_recent_log_file()
    df = pd.read_csv(file)
    sns.lineplot(x='batch_index', y='loss', hue='epoch', data=df)
    plt.show()

if __name__ == '__main__':
    show_results_loop()
