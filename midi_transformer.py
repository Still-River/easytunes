import os
import time
import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn

from third_party import PositionalEncoding
from helper import get_most_recent_model_checkpoint

class TransformerModel(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, dim_feedforward, max_len, vocab_sizes, dropout=0):
        super(TransformerModel, self).__init__()
        
        self.pitch_embedding = nn.Embedding(vocab_sizes['pitch'], d_model)
        self.delta_embedding = nn.Embedding(vocab_sizes['delta'], d_model)
        self.duration_embedding = nn.Embedding(vocab_sizes['duration'], d_model)
        self.embeddings = [self.pitch_embedding, self.delta_embedding, self.duration_embedding]

        self.combine_embeddings = nn.Linear(d_model * 3, d_model)

        self.pos_encoder = PositionalEncoding(d_model, max_len)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        
        self.pitch_decoder = nn.Linear(d_model, vocab_sizes['pitch'])
        self.delta_decoder = nn.Linear(d_model, vocab_sizes['delta'])
        self.duration_decoder = nn.Linear(d_model, vocab_sizes['duration'])
        self.decoders = [self.pitch_decoder, self.delta_decoder, self.duration_decoder]

        self.init_weights()

    def init_weights(self):
        for embedding in self.embeddings:
            nn.init.xavier_uniform_(embedding.weight)

        for decoder in self.decoders:
            nn.init.xavier_uniform_(decoder.weight)
            nn.init.constant_(decoder.bias, 0)

        nn.init.xavier_uniform_(self.combine_embeddings.weight)
        nn.init.constant_(self.combine_embeddings.bias, 0)

    def forward(self, pitch_data, delta_data, duration_data, padding_mask=None):
        pitch_embed = self.pitch_embedding(pitch_data)
        delta_embed = self.delta_embedding(delta_data)
        duration_embed = self.duration_embedding(duration_data)

        combined_embedding = torch.cat((pitch_embed, delta_embed, duration_embed), dim=-1)
        combined_embedding = self.combine_embeddings(combined_embedding)
        
        pos_encoded = self.pos_encoder(combined_embedding)

        if padding_mask is not None:
            output = self.transformer_encoder(pos_encoded, src_key_padding_mask=padding_mask)
        else:
            output = self.transformer_encoder(pos_encoded)

        pitch_output = self.pitch_decoder(output)
        delta_output = self.delta_decoder(output)
        duration_output = self.duration_decoder(output)

        return pitch_output[:, -1], delta_output[:, -1], duration_output[:, -1]
    
class MIDIDataset(Dataset):
    def __init__(self, data, bptt=None):
        self.data = data
        self.songs = data.shape[0]
        self.song_length = data.shape[1]
        self.note_params = data.shape[2]
        self.bptt = bptt if bptt is not None else self.song_length

    def __len__(self):
        if self.bptt == self.song_length:
            return self.songs 
        else:
            return self.songs * (self.song_length - self.bptt)

    def __getitem__(self, idx):
        if self.bptt == self.song_length:
            song = idx
            start, end = 0, self.song_length
        else:
            song = idx // (self.song_length - self.bptt)
            offset = idx % (self.song_length - self.bptt)
            start = offset
            end = start + self.bptt

        input_sequence = self.data[song, start:end]
        target_sequence = self.data[song, start+1:end+1]

        if input_sequence.shape[0] < self.bptt:
            padding = torch.zeros(self.bptt - input_sequence.shape[0], self.note_params)
            input_sequence = torch.cat((input_sequence, padding), dim=0)
            target_sequence = torch.cat((target_sequence, padding), dim=0)

        input_sequence = input_sequence.long()
        target_sequence = target_sequence.long()

        pitch_input = input_sequence[..., 0]
        delta_input = input_sequence[..., 1]
        duration_input = input_sequence[..., 2]

        pitch_target = target_sequence[..., 0]
        delta_target = target_sequence[..., 1]
        duration_target = target_sequence[..., 2]

        padding_mask = torch.zeros(self.bptt, dtype=torch.bool)
        padding_mask[pitch_input == 0] = True

        real_length = min(end - start, self.bptt)
        if real_length < self.bptt:
            padding_mask[real_length:] = True

        return pitch_input, delta_input, duration_input, pitch_target[-1], delta_target[-1], duration_target[-1], padding_mask
    
def collate_fn(batch):
    batch = [item for item in batch if not item[6].all()]

    if not batch:
        return None

    pitch_input = torch.stack([item[0] for item in batch])
    delta_input = torch.stack([item[1] for item in batch])
    duration_input = torch.stack([item[2] for item in batch])

    pitch_target = torch.stack([item[3] for item in batch])
    delta_target = torch.stack([item[4] for item in batch])
    duration_target = torch.stack([item[5] for item in batch])

    padding_mask = torch.stack([item[6] for item in batch])

    return pitch_input, delta_input, duration_input, pitch_target, delta_target, duration_target, padding_mask

def train(model, train_loader, criterion, optimizer, scheduler, epoch, device, loss_log_file='loss.csv', checkpoint_folder='model_checkpoints'):
    model.to(device)
    start_time = time.time()
    model.train()
    total_loss = 0.
    batch_index = 0
    for data in train_loader:
        try:
            if data is None:
                continue

            batch_index += 1

            pitch_input, delta_input, duration_input, pitch_target, delta_target, duration_target, padding_mask = [data_stream.to(device) for data_stream in data]

            pitch_output, delta_output, duration_output = model(pitch_input, delta_input, duration_input, padding_mask=padding_mask)

            pitch_loss = criterion(pitch_output, pitch_target)
            delta_loss = criterion(delta_output, delta_target)
            duration_loss = criterion(duration_output, duration_target)
            combined_loss = pitch_loss + delta_loss + duration_loss

            optimizer.zero_grad()
            combined_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Example with max_norm = 1.0
            optimizer.step()
            scheduler.step()

            total_loss += combined_loss.item()

            if batch_index % 10 == 0:
                elapsed = (time.time() - start_time) /60
                batches_per_minute = batch_index / elapsed
                expected_total_time = len(train_loader) / batches_per_minute
                expected_remaining_time = expected_total_time - elapsed
                print(f'Epoch: {epoch:02d}, Batch: {batch_index:05d}/{len(train_loader)}, time elapsed: {elapsed:.2f}/{expected_remaining_time:.2f} Loss: {combined_loss.item():.4f}')

            
            if batch_index % 100 == 0:
                with open(loss_log_file, 'a') as f:
                    f.write(f'{epoch},{batch_index},{combined_loss.item()}\n')

        except KeyboardInterrupt:
            print('Stopping early.')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'model_hyperparameters': model_hyperparameters,
                'train_load': train_loader,
                'val_load': val_loader
            }, f'{checkpoint_folder}/model_epoch_{epoch}_{batch_index}_early_stop.pth')
            print(f'Saved model checkpoint: {checkpoint_folder}/model_epoch_{epoch}_{batch_index}_early_stop.pth')
            break

    return total_loss / batch_index

def evaluate(model, val_loader, criterion, device, epoch):
    start_time = time.time()
    model.eval()
    total_loss = 0.
    batch_index = 0
    for data in val_loader:
        if data is None:
            continue

        batch_index += 1

        pitch_input, delta_input, duration_input, pitch_target, delta_target, duration_target, padding_mask = [data_stream.to(device) for data_stream in data]

        pitch_output, delta_output, duration_output = model(pitch_input, delta_input, duration_input, padding_mask=padding_mask)

        pitch_loss = criterion(pitch_output, pitch_target)
        delta_loss = criterion(delta_output, delta_target)
        duration_loss = criterion(duration_output, duration_target)
        combined_loss = pitch_loss + delta_loss + duration_loss

        total_loss += combined_loss.item()

        if batch_index % 10 == 0:
            elapsed = (time.time() - start_time) /60
            batches_per_minute = batch_index / elapsed
            expected_total_time = len(val_loader) / batches_per_minute
            expected_remaining_time = expected_total_time - elapsed
            print(f'Evaluation - Epoch {epoch:02d} Batch: {batch_index:04d}/{len(val_loader)}, time elapsed: {elapsed:.2f}/{expected_remaining_time:.2f} Loss: {combined_loss.item():.4f}')
        
    return total_loss / batch_index
    
vocab_sizes_array = np.load('vocab_sizes.npy')
vocab_sizes = {}
vocab_sizes['pitch'] = vocab_sizes_array[0]
vocab_sizes['delta'] = vocab_sizes_array[1]
vocab_sizes['duration'] = vocab_sizes_array[2]

data = np.load('songs_processed.npy')
max_len = 1024 # data.shape[1]

data = torch.from_numpy(data).long()

torch.manual_seed(0)
shuffle = torch.randperm(data.shape[0])
data = data[shuffle]

split = int(0.9 * data.shape[0])
train_data = data[:split]
val_data = data[split:]

#########################################################################################
################################### Model parameters ####################################
#########################################################################################

d_model = 2 ** 8
nhead = 8
num_encoder_layers = 10
dim_feedforward = 2 ** 12
bptt = 32
epochs = 100
batch_size = 2 ** 9

model_hyperparameters = {
    'd_model': d_model,
    'nhead': nhead,
    'num_encoder_layers': num_encoder_layers,
    'dim_feedforward': dim_feedforward,
    'bptt': bptt,
    'epochs': epochs,
    'batch_size': batch_size,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TransformerModel(d_model, nhead, num_encoder_layers, dim_feedforward, max_len, vocab_sizes).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0005,)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 100, gamma=0.99)

prev_checkpoint = None
# prev_checkpoint = get_most_recent_model_checkpoint()

#########################################################################################

train_dataset = MIDIDataset(train_data, bptt=bptt)
val_dataset = MIDIDataset(val_data, bptt=bptt)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

criterion = nn.CrossEntropyLoss(ignore_index=0)

if __name__ == '__main__':
    date = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    checkpoint_folder = 'model_checkpoints_' + date

    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder)

    if prev_checkpoint is not None:
        print('Loading model from checkpoint')
        checkpoint = torch.load(prev_checkpoint)

        d_model = checkpoint['model_hyperparameters']['d_model']
        nhead = checkpoint['model_hyperparameters']['nhead']
        num_encoder_layers = checkpoint['model_hyperparameters']['num_encoder_layers']
        dim_feedforward = checkpoint['model_hyperparameters']['dim_feedforward']
        max_len = checkpoint['model_hyperparameters']['bptt']

        model = TransformerModel(d_model, nhead, num_encoder_layers, dim_feedforward, max_len, vocab_sizes).to(device)

        model.load_state_dict(checkpoint['model_state_dict'])   
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        train_loader = checkpoint['train_load']
        val_loader = checkpoint['val_load']

    loss_log_file = f'loss-{date}.csv'
    with open(loss_log_file, 'w') as f:
        f.write('epoch,batch_index,loss\n')

    training_log_file = f'training-{date}.log'
    with open(training_log_file, 'w') as f:
        f.write(f'epoch,train_loss,val_loss\n')

    for epoch in range(1, epochs + 1):
        train_loss = train(model, train_loader, criterion, optimizer, scheduler, epoch, device, loss_log_file, checkpoint_folder)
        val_loss = evaluate(model, val_loader, criterion, device, epoch)

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'model_hyperparameters': model_hyperparameters,
            'train_load': train_loader,
            'val_load': val_loader
        }, f'{checkpoint_folder}/model_epoch_{epoch}.pth')

        with open(training_log_file, 'a') as f:
            f.write(f'{epoch},{train_loss},{val_loss}\n')

        print(f'Epoch: {epoch:02d}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')