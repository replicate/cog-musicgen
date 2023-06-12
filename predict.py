# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import os
import shutil

from tempfile import TemporaryDirectory
from pathlib import Path
from distutils.dir_util import copy_tree
from typing import Optional
from cog import BasePredictor, Input, Path
import torch
from huggingface_hub import snapshot_download, login

# Model specific imports 
import torchaudio
import subprocess
from audiocraft.models import MusicGen
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
        self.model_path = self.tokenizer_path = os.path.join('./models/', self.model_id)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print("model path is set to", self.model_path)
        print(f"Does model path exist? {os.path.exists(self.model_path)}")
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
                downloaded = self._download_model_from_hf_hub(self.model_id, self.model_path)
            if not downloaded:
                raise ValueError(f"Failed to download model from remote location {self.remote_model_path} or HuggingFace Hub {self.model_id}")
            
        self.model = self._load_model(model_path=self.model_path, cls=self.model_cls, model_id=self.model_id, **self.model_load_args)
        self.tokenizer = self._load_tokenizer(self.tokenizer_path)

    def _load_model(
            self, model_path: str, cls: Optional[any] = None, load_args: Optional[dict] = None, model_id: Optional[str] = None, device: Optional[str] = None,
        ) -> MusicGen:
    
        if device is None:
            device = self.device
        
        # Load the model
        # Note: audiocraft 
        from audiocraft.models.loaders import load_compression_model, load_lm_model, HF_MODEL_CHECKPOINTS_MAP
        name = next((key for key, val in HF_MODEL_CHECKPOINTS_MAP.items() if val == model_id), None)
        # name = model_path

        compression_model_path = os.path.join(model_path, 'compression_state_dict.bin')
        compression_model = load_compression_model(compression_model_path, device=device)

        lm_model_path = os.path.join(model_path, 'state_dict.bin')

        print('model_path:', lm_model_path)
        print('is file? ', os.path.isfile(lm_model_path))

        lm = load_lm_model(lm_model_path, device=device)
        print('loaded!')

        if name == 'melody':
            lm.condition_provider.conditioners['self_wav'].match_len_on_eval = True

        return MusicGen(name, compression_model, lm)

    def _load_tokenizer(
            self,
            tokenizer_path: str,
    ):
        return None

    def predict(
        self,
        description: str = Input(description="Music description for generation"),
        duration: int = Input(description="Duration of the generated audio in seconds", default=2
        ),
    ) -> Path:
        self.model.set_generation_params(duration=2)  # generate 8 seconds.
        wav = self.model.generate([description])
        # output_path = os.path.join(f"output.wav")
        # audio_write(output_path, wav[0].cpu(), self.model.sample_rate, strategy="loudness")

        for idx, one_wav in enumerate(wav):
            # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
            path = audio_write(f'{idx}', one_wav.cpu(), self.model.sample_rate, strategy="loudness")

        return Path(path)



    def _download_model_from_hf_hub(
            self, 
            model_id: str, 
            model_path: None, 
            allow_patterns: list = ["*.bin", "*.json", "*.md", "tokenizer.model", "checkpoint.pt", "*.py", "*.safetensors"],
            ignore_patterns: Optional[list] = None,
            revision: Optional[str] = None,
            rm_existing_model: Optional[bool] = False, 
    ):
        """Download model from HuggingFace Hub""" 

        if rm_existing_model:
        # logger.info(f"Removing existing model at {model_path}")
            if os.path.exists(model_path):
                shutil.rmtree(model_path)

        # setup temporary directory
        with TemporaryDirectory() as tmpdir:
            # logger.info(f"Downloading {model_id} weights to temp...")

            snapshot_dir = snapshot_download(
                repo_id=model_id, 
                cache_dir=tmpdir,
                allow_patterns=allow_patterns,
                ignore_patterns = ignore_patterns,
                revision=revision,
            )

            # copy snapshot to model dir
            # logger.info(f"Copying weights to {model_path}...")
            copy_tree(snapshot_dir, str(model_path))
        
        return True
    

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