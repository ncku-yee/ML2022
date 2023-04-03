# Reading/Writing Data
import os
from pathlib import Path

# For Progress Bar
from tqdm import tqdm

# Pytorch
import torch
import torch.nn as nn

# Conformer
from conformer import Conformer

from utils import *


""" public acc: 0.78900 """
class Classifier(nn.Module):
    def __init__(self, d_model=256, segment_len=128, n_spks=600, dropout=0.3, device='cuda'):
        super().__init__()  
        # Project the dimension of features from that of input into d_model.
        self.prenet = nn.Linear(40, d_model)
        """
        Arguments:
            num_classes            : Number of classification classes.
            input_dim              : Dimension of input vector.
            encoder_dim            : Dimension of conformer encoder.
            num_encoder_layers     : Number of conformer blocks.
            num_attention_heads    : Number of attention heads(default=8).
            feed_forward_dropout_p : Probability of feed forward module dropout(default=0.1).
            attention_dropout_p    : Probability of attention module dropout(default=0.1).
            conv_dropout_p         : Probability of conformer convolution module dropout(default=0.1).
            half_step_residual     : Flag indication whether to use half step residual or not(fefault=True).
        """
        self.conformer = Conformer(num_classes=d_model, 
                                   input_dim=d_model, 
                                   encoder_dim=512,
                                   num_encoder_layers=4).to(device)
        self.segment_len = segment_len
        # TODO:
        #   Change Transformer to Conformer.
        #   https://arxiv.org/abs/2005.08100
        """
        Arguments:
            d_model        : The number of expected features in the input (required).
            nhead          : The number of heads in the multiheadattention models (required).
            dim_feedforward: The dimension of the feedforward network model (default=2048).
            dropout        : The dropout value (default=0.1).
            activation     : The activation function of intermediate layer, relu or gelu (default=relu).
        """
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, dim_feedforward=256, nhead=8, dropout=dropout, batch_first=True,
        )
        
        """
        TransformerEncoder is stacked of TransformerEncoderLayer.
        Arguments:
            encoder_layer : An instance of the TransformerEncoderLayer() class (required).
            num_layers    : The number of sub-encoder-layers in the encoder (required).
            norm          : The layer normalization component (optional).
        """
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)

        self.attention_pooling = nn.Sequential(
            nn.Linear(segment_len, 1),
            nn.Softmax(dim=1),
        )
        
        # Project the the dimension of features from d_model into speaker nums.
        self.pred_layer = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(d_model, n_spks),
        )

    def forward(self, mels):
        """
        Arguments:
            mels: (batch size, segment_len, n_mels) n_mels: 40(The dimention of mel-spectrogram)
        Returns:
            out: (batch size, n_spks)
        """
        # out: (batch size, segment_len, d_model)
        out = self.prenet(mels)
        """ Conformer """
        # out: (batch size, output len, d_model)
        out, out_len = self.conformer(out, self.segment_len)
        """ self-attention pooling """
        out = self.encoder(out)
        """ Mean pooling """
        stats = out.mean(dim=1)
        """ Prediction layer """
        out = self.pred_layer(stats)
        return out

def parse_args():
    """arguments"""
    config = {
        "seed": 5401314,
        "train_ratio": 0.95,
        "data_dir": "./Dataset",
        "segment_len": 256, 
        "save_path": "./models/model_classifier1.ckpt",
        "batch_size": 128,
        "n_workers": 8,
        "valid_steps": 2000,
        "warmup_steps": 1000,
        "save_steps": 2000,
        "total_steps": 20000,
    }

    return config

def main(
    seed,
    train_ratio,
    data_dir,
    segment_len,
    save_path,
    batch_size,
    n_workers,
    valid_steps,
    warmup_steps,
    total_steps,
    save_steps,
):
    """Main function."""
    same_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info]: Use {device} now!")

    train_loader, valid_loader, speaker_num = get_dataloader(data_dir, train_ratio, seed, segment_len, batch_size, n_workers)
    train_iterator = iter(train_loader)
    print(f"[Info]: Finish loading data!",flush = True)

    model = Classifier(n_spks=speaker_num, segment_len=segment_len).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10000)
    print(f"[Info]: Finish creating model!",flush = True)

    if not os.path.isdir('./models'):
        os.mkdir('./models')                    # Create directory of saving models.

    best_accuracy = -1.0
    best_state_dict = None

    train_batch_acc_record, train_batch_loss_record = [], []
    valid_acc_record, valid_loss_record = [], []
    
    pbar = tqdm(total=valid_steps, ncols=0, desc="Train", unit=" step")

    for step in range(total_steps):
        # Get data
        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            batch = next(train_iterator)

        loss, accuracy = model_fn(batch, model, criterion, device)
        batch_loss = loss.item()
        batch_accuracy = accuracy.item()

        # Record accuracy and loss of each epoch.
        train_batch_acc_record.append(batch_accuracy)
        train_batch_loss_record.append(batch_loss)
        
        # Updata model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Log
        pbar.update()
        pbar.set_postfix(
            loss=f"{batch_loss:.2f}",
            accuracy=f"{batch_accuracy:.2f}",
            step=step + 1,
        )

        # Do validation
        if (step + 1) % valid_steps == 0:
            pbar.close()

            valid_loss, valid_accuracy = valid(valid_loader, model, criterion, device)
            
            # Record accuracy and loss of each epoch.
            valid_acc_record.append(valid_accuracy)
            valid_loss_record.append(valid_loss)

            # keep the best model
            if valid_accuracy > best_accuracy:
                best_accuracy = valid_accuracy
                best_state_dict = model.state_dict()

            pbar = tqdm(total=valid_steps, ncols=0, desc="Train", unit=" step")

        # Save the best model so far.
        if (step + 1) % save_steps == 0 and best_state_dict is not None:
            torch.save(best_state_dict, save_path)
            pbar.write(f"Step {step + 1}, best model saved. (accuracy={best_accuracy:.4f})")

    pbar.close()

    return train_batch_acc_record, train_batch_loss_record, valid_acc_record, valid_loss_record

# Taining
config = parse_args()
train_acc, train_loss, valid_acc, valid_loss = main(**config)