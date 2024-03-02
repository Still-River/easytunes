import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from simplify_midi import *


# Unique Token Counts
vocab_sizes = np.load('vocab_sizes.npy')
vocab_sizes = {'Pitch': vocab_sizes[0], 'Time': vocab_sizes[1], 'Duration': vocab_sizes[2]}
vocab_sizes = pd.DataFrame.from_dict(vocab_sizes, orient='index', columns=['Count'])
sns.set(font='Times New Roman')
sns.set_style('whitegrid')
plt.figure(figsize=(4.9, 3.8))
ax = sns.barplot(x=vocab_sizes.index, y='Count', data=vocab_sizes, color='#0891B2')
ax.set_xticklabels(ax.get_xticklabels(), fontsize=14, fontdict={'fontname': 'Times New Roman'})
ax.bar_label(ax.containers[0], fmt='%.0f', fontsize=14)
ax.set_xlabel('Parameter', fontsize=14, fontdict={'fontname': 'Times New Roman'})
ax.set_yticklabels(int(ax.get_yticks()), fontsize=14, fontdict={'fontname': 'Times New Roman'})
ax.set_ylim(0,150)
ax.set_ylabel('Count', fontsize=16, fontdict={'fontname': 'Times New Roman'})
ax.set_title('Unique Tokens per Note Parameter', fontsize=20, fontdict={'fontname': 'Times New Roman'})
plt.tight_layout()
plt.savefig('vocab_sizes.png', dpi=300)
plt.show()

# Song Length Distribution
songs = process_dataset('./ambrose_dataset')

songs = [dict_to_array(song[0]) for song in songs]

lengths = []
for song in songs:
    lengths.append(len(song))
lengths = np.array(lengths)
lengths = np.unique(lengths, return_counts=True)

# convert lengths to df
lengths = pd.DataFrame({'Length': lengths[0], 'Count': lengths[1]})
lengths = lengths.sort_values(by='Length')

# Create histogram of song lengths
sns.set(font='Times New Roman')
sns.set_style('whitegrid')
plt.figure(figsize=(4.9, 3.8))
# Plot histogram of song lengths
ax = sns.histplot(data=lengths, x='Length', weights='Count', bins=40, color='#0891B2')
ax.set_xlim(0, 3000)
ax.set_xticks([0, 1000, 2000, 3000])
ax.set_xticklabels([0, 1000, 2000, 3000], fontsize=14, fontdict={'fontname': 'Times New Roman'})
ax.set_xlabel('Number of Notes in Song', fontsize=14, fontdict={'fontname': 'Times New Roman'})
ax.set_yticklabels(int(ax.get_yticks()), fontsize=14, fontdict={'fontname': 'Times New Roman'})
ax.set_ylabel('Count', fontsize=16, fontdict={'fontname': 'Times New Roman'})
ax.set_title('Song Length Distribution', fontsize=20, fontdict={'fontname': 'Times New Roman'})
plt.tight_layout()
plt.savefig('song_length_distribution.png', dpi=300)
plt.show()

file = 'loss-2023-12-16-23-48-39.csv'
df = pd.read_csv(file)
df = df[df['epoch'] <= 3]
sns.lineplot(x='batch_index', y='loss', hue='epoch', data=df, color='#0891B2', linewidth=2.5, alpha=0.8, palette='deep')
plt.title('Loss over Epochs', fontsize=20, fontdict={'fontname': 'Times New Roman'})
plt.xlabel('Batch Index', fontsize=14, fontdict={'fontname': 'Times New Roman'})
plt.ylabel('Loss', fontsize=14, fontdict={'fontname': 'Times New Roman'})
plt.show()

# Time to Generate Song Bar Chart
times = {
    'Nvidia\n2070 RTX\nSuper': [6.702141046524048, 6.451699495315552, 6.393376350402832, 5.486905336380005, 5.75613260269165, 6.883164167404175, 6.365183115005493, 8.108211517333984, 7.5246593952178955, 7.258398771286011],
    'Ryzen 9\n3900X': [79.54835844039917, 79.62123203277588, 82.09951376914978, 81.28309082984924, 82.29498386383057, 84.25107312202454, 82.72647047042847, 83.26520705223083, 82.90304660797119, 83.69520163536072],
    'Intel\n1185G7': [78.15877747535706, 74.18605971336365, 74.14624500274658, 73.50156784057617, 74.20193076133728, 76.55239701271057, 78.10669279098511, 81.54228854179382, 80.27439665794373, 82.61540412902832]}

average_times = []
for key in times.keys():
    average_times.append(np.mean(times[key]))

stdev_times = []
for key in times.keys():
    stdev_times.append(np.std(times[key]))

times = pd.DataFrame({'Hardware': list(times.keys()), 'Time': average_times, 'Std Dev': stdev_times})

sns.set(font='OpenSans')
sns.set_style('whitegrid')
plt.figure(figsize=(8, 6))
ax = sns.barplot(x='Hardware', y='Time', data=times, color='#0891B2')
ax.set_xticklabels(ax.get_xticklabels(), fontsize=18, fontdict={'fontname': 'OpenSans'})
ax.bar_label(ax.containers[0], fmt='%.0f', fontsize=20, label_type='center', padding=25)
ax.set_ylim(0, 100)
ax.set_xlabel('')
for tick in ax.get_xticklabels():
    tick.set_rotation(45)
ax.set_yticklabels([int(ytick) for ytick in ax.get_yticks()], fontsize=20, fontdict={'fontname': 'OpenSans'})
ax.set_ylabel('Time (s)', fontsize=24, fontdict={'fontname': 'OpenSans'})
ax.set_title('Time to Generate Song', fontsize=28, fontdict={'fontname': 'OpenSans'})

for index, row in times.iterrows():
    ax.errorbar(index, row['Time'], yerr=row['Std Dev'], color='black', capsize=10, capthick=3, elinewidth=3)

plt.tight_layout()
plt.show()

plt.savefig('time_to_generate_song.png', dpi=300)