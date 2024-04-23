import os

def get_most_recent_model_checkpoint():
    files = os.listdir()
    files = [f for f in files if f.startswith('model_checkpoints_')]
    files.sort()
    folder = files[-1]

    files = os.listdir(folder)
    files = [f for f in files if f.startswith('model_epoch_') and f.endswith('.pth')]
    files.sort(key=lambda x: int(x.split('_')[2].split('.')[0]))

    return f'{folder}/{files[-1]}'