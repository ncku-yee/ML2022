# Numerical Operations
import random
import numpy as np
import math

# Reading/Writing Data
import os
import json
from pathlib import Path

# For Progress Bar
from tqdm import tqdm

# Pytorch
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, random_split

def same_seed(seed): 
    """ Fixes random number generator seeds for reproducibility. """
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
def train_valid_split(dataset, train_ratio, seed):
    """ Split provided training data into training set and validation set. """
    train_set_size = int(train_ratio * len(dataset))
    valid_set_size = len(dataset) - train_set_size
    train_set, valid_set = random_split(dataset, [train_set_size, valid_set_size], generator=torch.Generator().manual_seed(seed))
    print(f"[Info]:\nTraining size: {train_set_size}\nValidation size: {valid_set_size}")
    return train_set, valid_set

def print_model(model, dataloader):
    """
    Print the model architecture.
    Arguments:
        model: Your Neural Network model.
        dataloader: Training data loader.
    Returns:
        void: No return value.
    """
    # Get the input size of training data.
    input_size = None
    for X, y in dataloader:
        input_size = X.shape[1:]        # input_size = (C, H, W)
        break

    # Pass model and the input size as parameters.
    print(model)

class SpeakerDataset(Dataset):
    def __init__(self, data_dir, segment_len=128):
        self.data_dir = data_dir
        self.segment_len = segment_len

        # Load the mapping from speaker neme to their corresponding "id". 
        mapping_path = Path(data_dir) / "mapping.json"   # Concatenate the path.
        mapping = json.load(mapping_path.open())
        self.speaker2id = mapping["speaker2id"]

        # Load metadata of training data, get each speaker's utterances' info.
        metadata_path = Path(data_dir) / "metadata.json" # Concatenate the path.
        metadata = json.load(open(metadata_path))["speakers"]
        
        # Get the total number of speaker.
        self.speaker_num = len(metadata.keys())
        self.data = []                                   # Each element records a speaker's utterances list and its id(label). 
        for speaker in metadata.keys():
            for utterances in metadata[speaker]:
                self.data.append([utterances["feature_path"], self.speaker2id[speaker]])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        feat_path, speaker = self.data[index]
        # Load preprocessed mel-spectrogram.
        mel = torch.load(os.path.join(self.data_dir, feat_path))

        # Segmemt mel-spectrogram into "segment_len" frames.
        if len(mel) > self.segment_len:
            # Randomly get the starting point of the segment.
            start = random.randint(0, len(mel) - self.segment_len)
            # Get a segment with "segment_len" frames.
            mel = torch.FloatTensor(mel[start:start+self.segment_len])
        else:
            mel = torch.FloatTensor(mel)
        # Turn the speaker id into long for computing loss later.
        speaker = torch.FloatTensor([speaker]).long()
        return mel, speaker

    def get_speaker_number(self):
        return self.speaker_num

def collate_batch(batch):
    """
    Arguments:
        batch: A list of tuples with (feature, label)
               feature -> mel-spectrogram.
               label   -> speaker id.
    """
    # Process features within a batch.
    """ Collate a batch of data. """
    mel, speaker = zip(*batch)
    # Because we train the model batch by batch, we need to pad the features in the same batch to make their lengths the same.
    # batch_first=True to see the first dimension as batch_size.
    mel = pad_sequence(mel, batch_first=True, padding_value=-20)    # pad log 10^(-20) which is very small value.
    # mel: (batch size, segment_len, n_mels) n_mels: 40(The dimention of mel-spectrogram)
    return mel, torch.FloatTensor(speaker).long()


def get_dataloader(data_dir, train_ratio, seed, segment_len, batch_size, n_workers):
    """Generate dataloader"""
    dataset = SpeakerDataset(data_dir, segment_len)
    speaker_num = dataset.get_speaker_number()
    # Split dataset into training dataset and validation dataset
    trainset, validset = train_valid_split(dataset, train_ratio, seed)

    train_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=n_workers,
        pin_memory=True,
        collate_fn=collate_batch,
    )
    valid_loader = DataLoader(
        validset,
        batch_size=batch_size,
        num_workers=n_workers,
        drop_last=True,
        pin_memory=True,
        collate_fn=collate_batch,
    )

    return train_loader, valid_loader, speaker_num

def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
        The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
        The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
        The total number of training steps.
        num_cycles (:obj:`float`, `optional`, defaults to 0.5):
        The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
        following a half-cosine).
        last_epoch (:obj:`int`, `optional`, defaults to -1):
        The index of the last epoch when resuming training.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    def lr_lambda(current_step):
        # Warmup
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        # decadence
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(
            0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


def model_fn(batch, model, criterion, device):
    """Forward a batch through the model."""

    mels, labels = batch
    mels = mels.to(device)
    labels = labels.to(device)

    outs = model(mels)
    
    loss = criterion(outs, labels)

    # Get the speaker id with highest probability.
    preds = outs.argmax(1)
    # Compute accuracy.
    accuracy = torch.mean((preds == labels).float())

    return loss, accuracy


def valid(dataloader, model, criterion, device): 
    """Validate on validation set."""

    model.eval()
    running_loss = 0.0
    running_accuracy = 0.0
    pbar = tqdm(total=len(dataloader.dataset), ncols=0, desc="Valid", unit=" uttr")

    for i, batch in enumerate(dataloader):
        with torch.no_grad():
            loss, accuracy = model_fn(batch, model, criterion, device)
            running_loss += loss.item()
            running_accuracy += accuracy.item()
        
        pbar.update(dataloader.batch_size)
        pbar.set_postfix(
            loss=f"{running_loss / (i+1):.2f}",
            accuracy=f"{running_accuracy / (i+1):.2f}",
        )

    pbar.close()
    model.train()

    return running_loss / len(dataloader),  running_accuracy / len(dataloader)