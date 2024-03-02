import os
import pickle
import numpy as np
import mido

### Code required to fix key signature errors (when not given, assume C)
class MetaSpec_key_signature(mido.midifiles.meta.MetaSpec_key_signature):
   def decode(self, message, data):
       try:
           super().decode(message, data)
       except mido.midifiles.meta.KeySignatureError:
           message.key = 'C'

mido.midifiles.meta.add_meta_spec(MetaSpec_key_signature)

def parse_midi(file_path):
    # Returns a list of notes as dicitonaries
    mid = mido.MidiFile(file_path)
    metadata = {
        'ticks_per_beat': mid.ticks_per_beat,
        'tempo': None,
        'time_signature': {
            'numerator': None,
            'denominator': None,
            'clocks_per_tick': None,
            'notated_32nd_notes_per_beat': None
        },
        'key_signature': None,

    }

    active_notes = []
    final_notes = []
    for msg in mid.tracks[0]:
        if msg.type == 'set_tempo':
            metadata['tempo'] = msg.tempo
        elif msg.type == 'time_signature':
            metadata['time_signature']['numerator'] = msg.numerator
            metadata['time_signature']['denominator'] = msg.denominator
            metadata['time_signature']['clocks_per_tick'] = msg.clocks_per_click
            metadata['time_signature']['notated_32nd_notes_per_beat'] = msg.notated_32nd_notes_per_beat
        elif msg.type == 'key_signature':
            metadata['key_signature'] = msg.key

    for track in mid.tracks:
        current_time = 0
        for msg in track:
            current_time += msg.time
            if msg.type == 'note_on' and msg.velocity > 0:
                active_notes.append({'channel': msg.channel, 'note': msg.note, 'start_time': round(current_time/metadata['ticks_per_beat'],4), 'end_time': None, 'duration': None})
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                for note in active_notes:
                    if note['note'] == msg.note and note['channel'] == msg.channel and note['end_time'] is None:
                        note['end_time'] = round(current_time/metadata['ticks_per_beat'],4)
                        note['duration'] = round(note['end_time'] - note['start_time'],4)
                        final_notes.append(note)
                        active_notes.remove(note)
                        del note['end_time']

    final_notes.sort(key=lambda x: x['start_time'])

    prev_start_time = 0
    for note in final_notes:
        note['delta'] = round(note['start_time'] - prev_start_time, 4)
        prev_start_time = note['start_time']

    return final_notes, metadata

def parse_note_data(note_data, metadata):
    # Returns a midi object from a list of notes
    midi = mido.MidiFile(ticks_per_beat=metadata['ticks_per_beat'])
    track = mido.MidiTrack()
    midi.tracks.append(track)

    track.append(mido.MetaMessage('text', text='Author', time=0))
    track.append(mido.MetaMessage('copyright', text='Title', time=0))
    track.append(mido.MetaMessage('set_tempo', tempo=metadata['tempo'], time=0))
    track.append(mido.MetaMessage('time_signature', numerator=metadata['time_signature']['numerator'], denominator=metadata['time_signature']['denominator'], clocks_per_click=metadata['time_signature']['clocks_per_tick'], notated_32nd_notes_per_beat=metadata['time_signature']['notated_32nd_notes_per_beat'], time=0))
    track.append(mido.MetaMessage('key_signature', key=metadata['key_signature'], time=0))
    track.append(mido.MetaMessage('end_of_track', time=0))

    tracks = []
    smallest_delta = float('inf')
    for note in note_data:
        note['channel'] = 0
        channel = note['channel']
        
        if note['delta'] < smallest_delta and note['delta'] > 0:
            smallest_delta = note['delta']

        if note['duration'] < smallest_delta and note['duration'] > 0:
            smallest_delta = note['duration']

        while len(tracks) <= channel:
            new_track = mido.MidiTrack()
            new_track.append(mido.MetaMessage('track_name', name='Piano', time=0))
            new_track.append(mido.MetaMessage('channel_prefix', channel=len(tracks), time=0))
            tracks.append(new_track)

    metadata['ticks_per_beat'] = int(smallest_delta ** -1)
    midi.ticks_per_beat = metadata['ticks_per_beat']

    end_time = 0
    time_tracker = 0
    for note in note_data:
        time_tracker += int(note['delta'] * metadata['ticks_per_beat'])
        note['start_time'] = int(time_tracker)
        note['duration'] = int(note['duration'] * metadata['ticks_per_beat'])
        note['end_time'] = int(note['start_time'] + note['duration'])

        if note['end_time'] > end_time:
            end_time = note['end_time']

    last_time = [0] * len(tracks)
    for time in range(0, end_time + 1):
        for note in note_data:
            if time == note['start_time']:
                tracks[note['channel']].append(mido.Message('note_on', channel=note['channel'], note=note['note'], velocity=100, time=note['start_time'] - last_time[note['channel']]))
                last_time[note['channel']] = time
            if time == note['end_time']:
                tracks[note['channel']].append(mido.Message('note_off', channel=note['channel'], note=note['note'], velocity=0, time=note['end_time'] - last_time[note['channel']]))
                last_time[note['channel']] = time
        
    for track in tracks:
        track.append(mido.MetaMessage('end_of_track', time=0))
        midi.tracks.append(track)

    return midi

def dict_to_array(song):
    # Converts a list of notes as dictionaries to a numpy array
    notes = []
    for note in song:
        notes.append([note['note'], note['delta'], note['duration']])
    return np.array(notes)

def array_to_dict(song):
    # Converts a numpy array to a list of notes as dictionaries
    notes = []
    for note in song:
        notes.append({'note': int(note[0]), 'delta': note[1], 'duration': note[2]})
    return notes

def process_dataset(folder):
   # Processes the entire folder of midi files and returns a list of songs
    fails = 0
    passes = 0

    songs = []
    for file in os.listdir(folder):
        if file.endswith('.mid'):
            try:                
                song = parse_midi(os.path.join(folder,file))
                songs.append(song)
                passes += 1
            except Exception as e:
                print(f"Failed to simplify {file}: {e}")
                fails += 1

    print(f"Failed to simplify {fails} of {passes+fails} files")

    return songs

def pad_songs(songs, max_length):
    # Pads or clips all songs to the same length
    token_size = len(songs[0][0])
    for i in range(len(songs)):
        if len(songs[i]) > max_length:
            songs[i] = songs[i][:max_length]
        elif len(songs[i]) < max_length:
            tokens_to_add = [[-1] * token_size] * (max_length - len(songs[i]))
            songs[i] = np.concatenate((songs[i], tokens_to_add), axis=0)
    return songs

def check_for_bad_songs(songs_numbers, lengths):
    # Ensures that all songs are within the bounds of the tokenized lookup table
    bad_songs = []
    for i in range(songs_numbers.shape[0]):
        for j in range(songs_numbers.shape[1]):
            for k in range(songs_numbers.shape[2]):
                if songs_numbers[i, j, k] >= lengths[k]:
                    bad_songs.append(i)
                    break

    return np.unique(bad_songs)


def create_unique_entries_lookup(songs):
    # Creates a lookup table for each parameter in the song
    # so that the note parameters can be tokenized
    unique_entries = []
    for i in range(songs.shape[2]):
        flattened = songs[:, :, i].flatten()
        unique_entries.append(np.unique(flattened))

    lengths = []
    for i in range(len(unique_entries)):
        lengths.append(len(unique_entries[i]))

    unique_entries_lookup = []
    for parameter in range(len(unique_entries)):
        temp = {}
        for value in range(len(unique_entries[parameter])):
            temp[unique_entries[parameter][value]] = value
        unique_entries_lookup.append(temp)

    reverse_unique_entries_dict = []
    for parameter in range(len(unique_entries_lookup)):
        temp = {}
        for key in unique_entries_lookup[parameter].keys():
            temp[unique_entries_lookup[parameter][key]] = key
        reverse_unique_entries_dict.append(temp)

    with open('data_processing_info.pkl', 'wb') as f:
        pickle.dump(unique_entries_lookup, f)
        pickle.dump(reverse_unique_entries_dict, f)
        pickle.dump(lengths, f)

    return unique_entries_lookup, reverse_unique_entries_dict, lengths

def convert_song_to_numbers(song):
    # Using the lookup table, converts a single song into a tokenized form

    with open('data_processing_info.pkl', 'rb') as f:
        unique_entries_lookup = pickle.load(f)
        reverse_unique_entries_dict = pickle.load(f)
        lengths = pickle.load(f)

    song_numbers = np.zeros(song.shape)
    for i in range(song.shape[0]):
        for j in range(song.shape[1]):
            song_numbers[i, j] = unique_entries_lookup[j][song[i, j]]
    return song_numbers

def convert_songs_to_numbers(songs):
    # Converts a list of songs into a tokenized form
    songs_numbers = np.zeros(songs.shape)
    for i in range(songs.shape[0]):
        songs_numbers[i] = convert_song_to_numbers(songs[i])
        if i % 100 == 0:
            print(f"Converted {i / songs.shape[0] * 100:.2f}% of songs to numbers")
    return songs_numbers

def convert_numbers_to_song(song):
    # Converts a tokenized song back into the untokenized version
    with open('data_processing_info.pkl', 'rb') as f:
        unique_entries_lookup = pickle.load(f)
        reverse_unique_entries_dict = pickle.load(f)
        lengths = pickle.load(f)

    song = song.astype(float)
    for i in range(song.shape[0]):
        for j in range(song.shape[1]):
            song[i, j] = reverse_unique_entries_dict[j][song[i, j]]
    return song

if __name__ == '__main__':
    # Process Dataset
    songs = process_dataset('./ambrose_dataset')

    songs = [dict_to_array(song[0]) for song in songs]

    lengths = []
    for song in songs:
        lengths.append(len(song))
    lengths = np.array(lengths)
    lengths = np.unique(lengths, return_counts=True)

    max_length = 600
    songs = pad_songs(songs, max_length)
    songs = np.array(songs)

    np.save('songs.npy', songs)

    unique_entries_lookup, reverse_unique_entries_dict, lengths = create_unique_entries_lookup(songs)
    songs_numbers = convert_songs_to_numbers(songs)

    np.save('songs_processed.npy', songs_numbers)
    np.save('vocab_sizes.npy', lengths)

    bad_songs = check_for_bad_songs(songs_numbers, lengths)
    if bad_songs.shape[0] > 0:
        songs_numbers = np.delete(songs_numbers, bad_songs, axis=0)

    np.save('songs_processed.npy', songs_numbers)

    # Example Parsing
    midi_file_path = './ambrose_dataset/Yellow.mid'
    note_data, metadata = parse_midi(midi_file_path)
    note_data = dict_to_array(note_data)
    note_data_tokenized = convert_song_to_numbers(note_data)
    # note_data above is used to train model,
    # model output would be note_data below
    note_data_reversed = convert_numbers_to_song(note_data_tokenized)
    note_data = array_to_dict(note_data_reversed)
    midi_original = mido.MidiFile(midi_file_path)
    midi_new = parse_note_data(note_data, metadata)
    midi_new.save('test.mid')