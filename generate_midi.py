import torch
import numpy as np
from midi_transformer import model
from simplify_midi import *
import mido
import pickle
import time

def find_note_param_from_output(output):
    note_values = []
    for note_param in output:
        # Select based on probability distribution
        softmaxed_output = torch.nn.functional.softmax(note_param, dim=1)
        note_value = torch.multinomial(softmaxed_output, 1).item()
        note_values.append(note_value)

    return note_values

def generate_song(model, checkpoint, first_notes):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    with torch.no_grad():
        data = torch.tensor(first_notes).unsqueeze(0).long().to(device)
        for i in range(500):
            pitch_input = data[..., 0]
            time_input = data[..., 1]
            duration_input = data[..., 2]

            output = model(pitch_input, time_input, duration_input)
            output = find_note_param_from_output(output)
            if output == [-1, -1, -1]:
                    break

            output = torch.tensor(output).unsqueeze(0).unsqueeze(0).long().to(device)
            data = torch.cat((data, output), dim=1)

    song = data.squeeze(0).cpu().numpy()
    np.save('generated_song.npy', song)

    return song

def convert_generated_song_to_midi(song, metadata):
    note_data_reversed = convert_numbers_to_song(song)
    note_data = array_to_dict(note_data_reversed)
    midi = parse_note_data(note_data, metadata)
    midi.save('generated_song.mid')
    return midi

if __name__ == '__main__':
    # Process a midi file to use it's first few notes and metadata
    # as an input to start the model
    # Later this could be done by starting at random based on
    # sample distribution or the user selecting values

    # Test Single Song
    midi_file_path = './ambrose_dataset/Yellow.mid'
    note_data, metadata = parse_midi(midi_file_path)
    note_data = dict_to_array(note_data)
    note_data_tokenized = convert_song_to_numbers(note_data)

    first_few_notes = note_data_tokenized[:10]
    checkpoint = 'model_checkpoints_2023-12-16-23-48-39/model_epoch_4_266_early_stop.pth'
    generated_song = generate_song(model, checkpoint, first_few_notes)
    convert_song = convert_generated_song_to_midi(generated_song, metadata)

    # # Test 10 Songs for time performance
    # song_prompts = [
    #     'A_thousand_miles2C.mid',
    #     'AllIWantForChristmas.mid',
    #     'bachinvention05.mid',
    #     'Daydreamer.mid',
    #     'Yellow.mid',
    #     'FurElise.mid',
    #     'JoytotheWorld.mid',
    #     'SonataNo18.mid',
    #     'Moonlight_sonata_1st_movement1.mid',
    #     'Torn.mid'
    # ]

    # song_prompts = ['ambrose_dataset/' + song for song in song_prompts]

    # # Ensure all files exist before starting
    # for song in song_prompts:
    #     if not os.path.exists(song):
    #         print(f'{song} does not exist')
    #         exit()

    # # Start Timer
    # end_times = []
    # for song in song_prompts:
    #     start = time.time()
    #     note_data, metadata = parse_midi(song)
    #     note_data = dict_to_array(note_data)
    #     note_data_tokenized = convert_song_to_numbers(note_data)

    #     first_few_notes = note_data_tokenized[:40]
    #     generated_song = generate_song(model, 'model_checkpoints_2023-12-12-21-49-36/model_epoch_3.pth', first_few_notes)
    #     convert_song = convert_generated_song_to_midi(generated_song, metadata)
        
    #     end_times.append(time.time() - start)

    #     new_name = song.split('/')[-1]
    #     os.rename('generated_song.mid', new_name)

    # # Print end times, avergage, and std dev
    # print(end_times)
    # print(np.mean(end_times))
    # print(np.std(end_times))
    

