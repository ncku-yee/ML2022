# Numerical Operations
import numpy as np

# Reading/Writing Data
import os
import csv
import json
from pathlib import Path

# For Progress Bar
from tqdm import tqdm

# Pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Conformer
from conformer import Conformer

# Garbage Collection
import gc

""" public acc: 0.78900 """
class Classifier1(nn.Module):
    def __init__(self, d_model=256, segment_len=128, n_spks=600, dropout=0.3, device='cuda'):
        super().__init__()
        # Project the dimension of features from that of input into d_model.
        self.prenet = nn.Linear(40, d_model)
        self.conformer = Conformer(num_classes=d_model, 
                                   input_dim=d_model, 
                                   encoder_dim=512,
                                   num_encoder_layers=4).to(device)
        self.segment_len = segment_len

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, dim_feedforward=256, nhead=8, dropout=dropout, batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)

        self.attention_pooling = nn.Sequential(
            nn.Linear(segment_len, 1),
            nn.Softmax(dim=1),
        )

        self.pred_layer = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(d_model, n_spks),
        )

    def forward(self, mels):
        out = self.prenet(mels)
        """ Conformer """
        out, out_len = self.conformer(out, self.segment_len)
        """ self-attention """
        out = self.encoder_layer(out)
        """ Mean pooling """
        stats = out.mean(dim=1)
        """ Prediction layer """
        out = self.pred_layer(stats)
        return out

""" public acc: 0.82150 """
class Classifier2(nn.Module):
    def __init__(self, d_model=256, segment_len=128, n_spks=600, dropout=0.3, device='cuda'):
        super().__init__()
        # Project the dimension of features from that of input into d_model.
        self.prenet = nn.Linear(40, d_model)
        self.conformer = Conformer(num_classes=d_model, 
                                   input_dim=d_model, 
                                   encoder_dim=512,
                                   num_encoder_layers=4).to(device)
        self.segment_len = segment_len

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, dim_feedforward=256, nhead=8, dropout=dropout, batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)

        self.attention_pooling = nn.Sequential(
            nn.Linear(segment_len, 1),
            nn.Softmax(dim=1),
        )

        self.pred_layer = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(d_model, n_spks),
        )

    def forward(self, mels):
        out = self.prenet(mels)
        """ Conformer """
        out, out_len = self.conformer(out, self.segment_len)
        """ Self-attention """
        out = self.encoder(out)
        """ Mean pooling """
        stats = out.mean(dim=1)
        """ Prediction layer """
        out = self.pred_layer(stats)
        return out

""" public acc: 0.82075 """
class Classifier3(nn.Module):
    def __init__(self, d_model=256, segment_len=128, n_spks=600, dropout=0.3, device='cuda'):
        super().__init__()
        self.prenet = nn.Linear(40, d_model)
        self.conformer = Conformer(num_classes=d_model, 
                                   input_dim=d_model, 
                                   encoder_dim=512,
                                   num_encoder_layers=6).to(device)
        self.segment_len = segment_len

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, dim_feedforward=256, nhead=8, dropout=dropout, batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)

        self.pred_layer = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(d_model, n_spks),
        )
        

    def forward(self, mels):
        out = self.prenet(mels)
        """ Conformer """
        out, out_len = self.conformer(out, self.segment_len)
        """ Self-attention """
        out = self.encoder(out)
        """ Mean pooling """
        stats = out.mean(dim=1)
        """ Prediction layer """
        out = self.pred_layer(stats)
        return out

""" public acc: 0.80050 """
class Classifier4(nn.Module):
    def __init__(self, d_model=256, segment_len=128, n_spks=600, dropout=0.3, device='cuda'):
        super().__init__()
        self.prenet = nn.Linear(40, d_model)
        self.conformer = Conformer(num_classes=d_model, 
                                   input_dim=d_model, 
                                   encoder_dim=512,
                                   num_encoder_layers=4).to(device)
        self.segment_len = segment_len

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, dim_feedforward=256, nhead=8, dropout=dropout, batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=8)

        self.attention_pooling = nn.Sequential(
            nn.Linear(segment_len, 1),
            nn.Softmax(dim=1),
        )

        self.pred_layer = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(d_model, n_spks),
        )
        

    def forward(self, mels):
        out = self.prenet(mels)
        """ Conformer """
        out, out_len = self.conformer(out, self.segment_len)
        """ Self-attention """
        out = self.encoder_layer(out)
        """ Mean pooling """
        stats = out.mean(dim=1)
        """ Prediction layer """
        out = self.pred_layer(stats)
        return out

""" public acc: 0.78600 """
class Classifier5(nn.Module):
    def __init__(self, d_model=256, segment_len=128, n_spks=600, dropout=0.3, device='cuda'):
        super().__init__()
        self.prenet = nn.Linear(40, d_model)
        self.conformer = Conformer(num_classes=d_model, 
                                   input_dim=d_model, 
                                   encoder_dim=512,
                                   num_encoder_layers=4).to(device)
        self.segment_len = segment_len

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, dim_feedforward=512, nhead=8, dropout=dropout, batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)

        self.pred_layer = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(d_model, n_spks),
        )

    def forward(self, mels):
        out = self.prenet(mels)
        """ Conformer """
        out, out_len = self.conformer(out, self.segment_len)
        """ Self-attention """
        out = self.encoder(out)
        """ Mean pooling """
        stats = out.mean(dim=1)
        """ Prediction layer """
        out = self.pred_layer(stats)
        return out

""" public acc: 0.79725 """
class Classifier6(nn.Module):
    def __init__(self, d_model=256, segment_len=128, n_spks=600, dropout=0.3, device='cuda'):
        super().__init__()
        self.prenet = nn.Linear(40, d_model)

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, dim_feedforward=512, nhead=8, dropout=dropout, batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6)

        self.pred_layer = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(d_model, n_spks),
        )

    def forward(self, mels):
        out = self.prenet(mels)
        """ Transformer Encoder """
        out = self.encoder(out)
        """ Mean pooling """
        stats = out.mean(dim=1)
        """ Prediction layer """
        out = self.pred_layer(stats)
        return out

class InferenceDataset(Dataset):
    def __init__(self, data_dir):
        testdata_path = Path(data_dir) / "testdata.json"
        metadata = json.load(testdata_path.open())
        self.data_dir = data_dir
        self.data = metadata["utterances"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        utterance = self.data[index]
        feat_path = utterance["feature_path"]
        mel = torch.load(os.path.join(self.data_dir, feat_path))

        return feat_path, mel


def inference_collate_batch(batch):
    """Collate a batch of data."""
    feat_paths, mels = zip(*batch)

    return feat_paths, torch.stack(mels)



def ensemble_soft_voting(model_mapping, data_dir, segment_len=128, speaker_num=600, **kwargs):
    """
    Soft-voting ensemble.
    Arguments:
        model_mapping : Your model's path and class name mapping.
        classes       : Output classes.
        **kwargs      : Arguments for Test Time Augmentation.
    Returns:
        Numpy array   : Return the prediction.
    """
    # First to gain the test_loader without augmentation
    test_set = InferenceDataset(data_dir)
    test_loader = DataLoader(
        test_set, 
        batch_size=1, 
        shuffle=False, 
        num_workers=8, 
        pin_memory=True, 
        collate_fn=inference_collate_batch
    )
    print(f"[Info]: Finish loading data!",flush = True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info]: Use {device} now!")

    with torch.no_grad():
        ensemble_preds = np.empty((len(test_loader.sampler), speaker_num, 0))  # Prediction of the test dataset.
        paths = []
        for model_path, class_name in model_mapping.items():
            model = class_name(n_spks=speaker_num, segment_len=segment_len).to(device)
            model.load_state_dict(torch.load(model_path))
            """ Evaluation mode """
            model.eval()
            predictions = np.empty((0, speaker_num))

            for feat_paths, mels in tqdm(test_loader):
                mels = mels.to(device)
                pred = model(mels)
                paths += feat_paths
                predictions = np.vstack((predictions, pred.detach().cpu().numpy()))
            ensemble_preds = np.dstack((ensemble_preds, predictions))

            """ Delete the model """
            del model
            gc.collect()
        mean_ensemble_preds = np.mean(ensemble_preds, axis=2)
        prediction = mean_ensemble_preds.argmax(axis=-1)
    return np.array(paths), prediction

model_list = [f"./models/model_classifier{i+1}.ckpt" for i in range(6)]
class_list = []
for model in model_list:
    name = model.split("/")[-1].split('.')[0].split('_')[-1]
    class_list.append(globals()[name.capitalize()])
model_mapping = dict(zip(model_list, class_list))

paths, prediction = ensemble_soft_voting(model_mapping, "./Dataset", segment_len=256, speaker_num=600)


mapping_path = Path("./Dataset") / "mapping.json"
mapping = json.load(mapping_path.open())


# Writing CSV
results = [["Id", "Category"]]
ids = [mapping["id2speaker"][str(pred)] for pred in prediction]
output_path = "./prediction.csv"
with open(output_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(results)
    writer.writerows(zip(paths, ids))