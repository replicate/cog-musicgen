# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import os
# We need to set `TRANSFORMERS_CACHE` before any imports, which is why this is up here.
MODEL_PATH = '/src/models/'
os.environ['TRANSFORMERS_CACHE'] = MODEL_PATH
os.environ['TORCH_HOME'] = MODEL_PATH


import shutil

from tempfile import TemporaryDirectory
from pathlib import Path
from distutils.dir_util import copy_tree
from typing import Optional
from cog import BasePredictor, Input, Path
import torch
from huggingface_hub import snapshot_download, login
import datetime
import gradio as gr

# Model specific imports
import torchaudio
import subprocess
import typing as tp

from audiocraft.models import MusicGen
from audiocraft.models.loaders import load_compression_model, load_lm_model, HF_MODEL_CHECKPOINTS_MAP
from audiocraft.data.audio import audio_write



class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        # self.model = torch.load("./weights.pth")

        # Specify model information here ---------------
        self.model_id = "facebook/musicgen-melody"
        self.model_cls = MusicGen
        self.remote_model_path = None
        self.model_load_args = dict()

        # Configure these variables if you want, but you don't need to for most models -------------
        self.model_path = self.tokenizer_path = MODEL_PATH
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load the model
        if not os.path.exists(self.model_path):
            # If path doesn't exist, we need to download the model
            # If a remote path is specified, we'll try to download from there
            downloaded = False
            if self.remote_model_path:
                # try to download from remote
                downloaded = self._maybe_download(self.model_id, self.model_path, self.remote_model_path)

            if not downloaded:
                # download from HuggingFace Hub
                self.model = self._load_model(model_path=self.model_path, cls=self.model_cls, model_id=self.model_id, **self.model_load_args)
        else:
            self.model = self._load_model(model_path=self.model_path, cls=self.model_cls, model_id=self.model_id, **self.model_load_args)

    def _load_model(
            self, model_path: str, cls: Optional[any] = None, load_args: Optional[dict] = None, model_id: Optional[str] = None, device: Optional[str] = None,
        ) -> MusicGen:

        if device is None:
            device = self.device


        name = next((key for key, val in HF_MODEL_CHECKPOINTS_MAP.items() if val == model_id), None)
        compression_model = load_compression_model(name, device=device, cache_dir=model_path)
        lm = load_lm_model(name, device=device, cache_dir=model_path)

        return MusicGen(name, compression_model, lm)

    def _load_tokenizer(
            self,
            tokenizer_path: str,
    ):
        return None

    def predict(
        self,
        description: str = Input(description="Music description for generation"),
        melody: Path = Input(description="mp3 format file to use for melody", default=None),
        duration: int = Input(description="Duration of the generated audio in seconds", default=8),
        strategy: str = Input(description="Strategy for generating audio", default="loudness"),
        seed: int = Input(description="Seed for random number generator. Default is -1 for random seed", default=-1),
        save_as_video: bool = Input(description="Save the generated audio as a video", default=False),
    ) -> Path:

        # Set seed or get random seed
        if seed == -1:
            seed = torch.seed()
        else:
            torch.manual_seed(seed)

        self.model.set_generation_params(duration=duration)
        if melody:
            melody, sr = torchaudio.load(melody)
            wav = self.model.generate_with_chroma([description], melody[None], sr)
        else:
            wav = self.model.generate([description])

        # Get the current timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        for idx, one_wav in enumerate(wav):
            # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
            path = audio_write(f'{idx}-{timestamp}', one_wav.cpu(), self.model.sample_rate, strategy=strategy)

        if save_as_video:
            waveform_video = gr.make_waveform(path)
            video_path = f"/tmp/{idx}-{timestamp}-waveform.mp4"
            waveform_video.export(video_path)
            return Path(video_path)

        return Path(path)

    def _maybe_download(self, model_id: str, model_path: str, remote_path: str = None) -> bool:
        """
        Sometimes we want to try to download from a remote location other than the Hugging Face Hub. We implement that here.
        If the download is possible, return True. Otherwise, return False.
        """

        if remote_path.startswith("gs://"):
            try:
                subprocess.check_call(["gcloud", "storage", "cp", remote_path, model_path])
                return True
            except subprocess.CalledProcessError as e:
                print(f"Failed to download '{remote_path}': {e}")
                return False

        else:
            raise ValueError(f"Only implemented for GCS. If you need to download from a different location, you can implement your own download method")
