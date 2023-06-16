"""
Adapted from https://github.com/chavinlo/musicgen_trainer/blob/main/train.y
"""

import os

MODEL_PATH = '/src/models/'
os.environ['TRANSFORMERS_CACHE'] = MODEL_PATH
os.environ['TORCH_HOME'] = MODEL_PATH

import typing as tp
import subprocess
import datetime 
from cog import BaseModel, Input, Path
import torchaudio
from audiocraft.models import MusicGen
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch.nn.functional as F
from zipfile import ZipFile
import shutil

from torch.utils.data import Dataset
    
from audiocraft.modules.conditioners import (
    ClassifierFreeGuidanceDropout
)
from audiocraft.models.loaders import load_compression_model, load_lm_model, HF_MODEL_CHECKPOINTS_MAP


class TrainingOutput(BaseModel):
    weights: Path

MODEL_OUT = "/src/tuned_weights.tensors"
CHECKPOINT_DIR = "checkpoints"
SAVE_STRATEGY = "epoch"
DIST_OUT_DIR = "tmp/model"


def _load_model(
    model_path: str,  model_name: tp.Optional[str] = None,
) -> MusicGen:

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compression_model = load_compression_model(model_name, device=device, cache_dir=model_path)
    lm = load_lm_model(model_name, device=device, cache_dir=model_path)

    return MusicGen(model_name, compression_model, lm)

class AudioDataset(Dataset):
    def __init__(self, 
                data_dir
                ):
        self.data_dir = data_dir
        self.data_map = []

        dir_map = os.listdir(data_dir)
        for d in dir_map:
            name, ext = os.path.splitext(d)
            if ext == '.wav':
                if os.path.exists(os.path.join(data_dir, name + '.txt')):
                    self.data_map.append({
                        "audio": os.path.join(data_dir, d),
                        "label": os.path.join(data_dir, name + '.txt')
                    })
                else:
                    raise ValueError(f'No label file for {name}')
                
    def __len__(self):
        return len(self.data_map)
    
    def __getitem__(self, idx):
        data = self.data_map[idx]
        audio = data['audio']
        label = data['label']

        return audio, label

def count_nans(tensor):
    nan_mask = torch.isnan(tensor)
    num_nans = torch.sum(nan_mask).item()
    return num_nans

def preprocess_audio(audio_path, model: MusicGen, duration: int = 30):
    wav, sr = torchaudio.load(audio_path)
    wav = torchaudio.functional.resample(wav, sr, model.sample_rate)
    wav = wav.mean(dim=0, keepdim=True)
    end_sample = int(model.sample_rate * duration)

        # Add padding if necessary
    if wav.shape[1] < end_sample:
        pad_size = end_sample - wav.shape[1]
        wav = F.pad(wav, pad=(0, pad_size))

    wav = wav[:, :end_sample]

    assert wav.shape[0] == 1
    assert wav.shape[1] == model.sample_rate * duration

    wav = wav.cuda()
    wav = wav.unsqueeze(1)

    with torch.no_grad():
        gen_audio = model.compression_model.encode(wav)

    codes, scale = gen_audio

    assert scale is None

    return codes

def fixnan(tensor: torch.Tensor):
    nan_mask = torch.isnan(tensor)
    result = torch.where(nan_mask, torch.zeros_like(tensor), tensor)
    
    return result

def one_hot_encode(tensor, num_classes=2048):
    shape = tensor.shape
    one_hot = torch.zeros((shape[0], shape[1], num_classes))

    for i in range(shape[0]):
        for j in range(shape[1]):
            index = tensor[i, j].item()
            one_hot[i, j, index] = 1

    return one_hot

def train(
        dataset_path: Path = Input("Path to dataset directory",),
        model_name: str = Input(description="Model version to train.", default="melody", choices=["melody", "large"]),
        lr: float = Input(description="Learning rate", default=1e-4),
        epochs: int = Input(description="Number of epochs to train for", default=5),
        save_step: int = Input(description="Save model every n steps", default=None),
        batch_size: int = Input(description="Batch size", default=1),
) -> TrainingOutput:
    
    # For local runs, we'll support overriding `dataset_path` if `train_data` exists. 
    # That way we can avoid having to tar/untar the dataset for every training run.
    if os.path.exists('/src/train_data'):
        dataset_path = '/src/train_data'
    else:
        # decompress file at dataset_path
        subprocess.run(['tar', '-xvzf', dataset_path, '-C', '/src/train_data'])
        dataset_path = '/src/train_data'

    if batch_size > 1:
        raise ValueError("Batch size > 1 not supported yet")
    
    output_dir = DIST_OUT_DIR
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    directory = Path(output_dir)

    model = _load_model(MODEL_PATH, model_name=model_name)
    model.lm = model.lm.to(torch.float32) #important
    
    dataset = AudioDataset(dataset_path)
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    learning_rate = lr
    model.lm.train()

    scaler = torch.cuda.amp.GradScaler()

    #from paper
    optimizer = AdamW(model.lm.parameters(), lr=learning_rate, betas=(0.9, 0.95), weight_decay=0.1)

    criterion = nn.CrossEntropyLoss()

    num_epochs = epochs

    save_step = save_step
    save_models = False if save_step is None else True

    current_step = 0
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    for epoch in range(num_epochs):
        for batch_idx, (audio, label) in enumerate(train_dataloader):
            optimizer.zero_grad()

            #where audio and label are just paths
            audio = audio[0]
            label = label[0]

            audio = preprocess_audio(audio, model) #returns tensor
            text = open(label, 'r').read().strip()

            attributes, _ = model._prepare_tokens_and_attributes([text], None)

            conditions = attributes
            null_conditions = ClassifierFreeGuidanceDropout(p=1.0)(conditions)
            conditions = conditions + null_conditions
            tokenized = model.lm.condition_provider.tokenize(conditions)
            cfg_conditions = model.lm.condition_provider(tokenized)
            condition_tensors = cfg_conditions

            codes = torch.cat([audio, audio], dim=0)

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                lm_output = model.lm.compute_predictions(
                    codes=codes,
                    conditions=[],
                    condition_tensors=condition_tensors
                )

                codes = codes[0]
                logits = lm_output.logits[0]
                mask = lm_output.mask[0]

                codes = one_hot_encode(codes, num_classes=2048)

                codes = codes.cuda()
                logits = logits.cuda()
                mask = mask.cuda()

                mask = mask.view(-1)
                masked_logits = logits.view(-1, 2048)[mask]
                masked_codes = codes.view(-1, 2048)[mask]

                loss = criterion(masked_logits,masked_codes)

            assert count_nans(masked_logits) == 0
            
            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.lm.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()

            print(f"Epoch: {epoch}/{num_epochs}, Batch: {batch_idx}/{len(train_dataloader)}, Loss: {loss.item()}")

            current_step += 1

            if save_models:
                if current_step % save_step == 0:
                    torch.save(model.lm.state_dict(), f"{output_dir}/{timestamp}_lm_{current_step}.pt")

    torch.save(model.lm.state_dict(), f"{output_dir}/{timestamp}_lm_final.pt")

    out_path = "training_output.zip"
    with ZipFile(out_path, "w") as zip:
        for file_path in directory.rglob("*"):
            print(file_path)
            zip.write(file_path, arcname=file_path.relative_to(directory))

    return TrainingOutput(weights=Path(out_path))



if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--model_id', type=str, required=False, default='melody')
    parser.add_argument('--lr', type=float, required=False, default=0.0001)
    parser.add_argument('--epochs', type=int, required=False, default=5)
    # parser.add_argument('--use_wandb', type=int, required=False, default=0)
    parser.add_argument('--save_step', type=int, required=False, default=None)
    args = parser.parse_args()

    train(
        dataset_path=args.dataset_path,
        model_id=args.model_id,
        lr=args.lr,
        epochs=args.epochs,
        save_step=args.save_step,
    )