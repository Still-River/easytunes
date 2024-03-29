
# Easy Tunes: Simplified Music Generation Using Transformers

This repository contains a transformer model that can predict MIDI files.

- `simplify_midi.py` can be used to take a folder of MIDI songs, tokenize them, and save them as a numpy array of songs.
- `midi_transformer.py` contains the code for the trarnsformer itself, along with the dataset classes, training loops, and evaluation functions. It can be ran to train the model.
- `generate_midi.py` uses the trained model and creates a MIDI song, based on the metadata of a sample song.

For more information on the model architecture, limitations, etc., please check out the [poster](https://github.com/Still-River/easytunes/blob/main/Poster_Ingram.pdf) and [report](https://github.com/Still-River/easytunes/blob/main/Easy%20Tunes%20-%20Simplified%20Music%20Generation%20Using%20Transformers.pdf).
