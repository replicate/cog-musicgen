# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import os
import random

# We need to set `TRANSFORMERS_CACHE` before any imports, which is why this is up here.
MODEL_PATH = "/src/models/"
os.environ["TRANSFORMERS_CACHE"] = MODEL_PATH
os.environ["TORCH_HOME"] = MODEL_PATH


import shutil

from tempfile import TemporaryDirectory
from pathlib import Path
from distutils.dir_util import copy_tree
from typing import Optional
from cog import BasePredictor, Input, Path
import torch
import datetime

# Model specific imports
import torchaudio
import subprocess
import typing as tp
import numpy as np

from audiocraft.models import MusicGen
from audiocraft.models.loaders import (
    load_compression_model,
    load_lm_model,
    HF_MODEL_CHECKPOINTS_MAP,
)
from audiocraft.data.audio import audio_write


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.melody_model = self._load_model(
            model_path=MODEL_PATH,
            cls=MusicGen,
            model_id="facebook/musicgen-melody",
        )

        self.large_model = self._load_model(
            model_path=MODEL_PATH,
            cls=MusicGen,
            model_id="facebook/musicgen-large",
        )

    def _load_model(
        self,
        model_path: str,
        cls: Optional[any] = None,
        load_args: Optional[dict] = {},
        model_id: Optional[str] = None,
        device: Optional[str] = None,
    ) -> MusicGen:

        if device is None:
            device = self.device

        name = next(
            (key for key, val in HF_MODEL_CHECKPOINTS_MAP.items() if val == model_id),
            None,
        )
        compression_model = load_compression_model(
            name, device=device, cache_dir=model_path
        )
        lm = load_lm_model(name, device=device, cache_dir=model_path)

        return MusicGen(name, compression_model, lm)

    def predict(
        self,
        model_version: str = Input(
            description="Model to use for generation. If set to 'encode-decode', the audio specified via 'input_audio' will simply be encoded and then decoded.",
            default="melody",
            choices=["melody", "large", "encode-decode"],
        ),
        prompt: str = Input(
            description="A description of the music you want to generate.", default=None
        ),
        input_audio: Path = Input(
            description="An audio file that will influence the generated music. If `continuation` is `True`, the generated music will be a continuation of the audio file. Otherwise, the generated music will mimic the audio file's melody.",
            default=None,
        ),
        duration: int = Input(
            description="Duration of the generated audio in seconds.", default=8, le=30
        ),
        continuation: bool = Input(
            description="If `True`, generated music will continue `melody`. Otherwise, generated music will mimic `audio_input`'s melody.",
            default=False,
        ),
        continuation_start: int = Input(
            description="Start time of the audio file to use for continuation.",
            default=0,
            ge=0,
        ),
        continuation_end: int = Input(
            description="End time of the audio file to use for continuation. If -1 or None, will default to the end of the audio clip.",
            default=None,
            ge=0,
        ),
        normalization_strategy: str = Input(
            description="Strategy for normalizing audio.",
            default="loudness",
            choices=["loudness", "clip", "peak", "rms"],
        ),
        top_k: int = Input(
            description="Reduces sampling to the k most likely tokens.", default=250
        ),
        top_p: float = Input(
            description="Reduces sampling to tokens with cumulative probability of p. When set to  `0` (default), top_k sampling is used.",
            default=0.0,
        ),
        temperature: float = Input(
            description="Controls the 'conservativeness' of the sampling process. Higher temperature means more diversity.",
            default=1.0,
        ),
        classifier_free_guidance: int = Input(
            description="Increases the influence of inputs on the output. Higher values produce lower-varience outputs that adhere more closely to inputs.",
            default=3,
        ),
        output_format: str = Input(
            description="Output format for generated audio.",
            default="wav",
            choices=["wav", "mp3"],
        ),
        seed: int = Input(
            description="Seed for random number generator. If None or -1, a random seed will be used.",
            default=None,
        ),
    ) -> Path:

        if prompt is None and input_audio is None:
            raise ValueError("Must provide either prompt or input_audio")
        if continuation and not input_audio:
            raise ValueError("Must provide `input_audio` if continuation is `True`.")
        if model_version == "large" and input_audio and not continuation:
            raise ValueError(
                "Large model does not support melody input. Set `model_version='melody'` to condition on audio input."
            )

        model = self.melody_model if model_version == "melody" else self.large_model

        set_generation_params = lambda duration: model.set_generation_params(
            duration=duration,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            cfg_coef=classifier_free_guidance,
        )

        if not seed or seed == -1:
            seed = torch.seed() % 2 ** 32 - 1
            set_all_seeds(seed)
        set_all_seeds(seed)
        print(f"Using seed {seed}")

        if not input_audio:
            set_generation_params(duration)
            wav = model.generate([prompt], progress=True)

        elif model_version == "encode-decode":
            encoded_audio = self._preprocess_audio(input_audio, model)
            set_generation_params(duration)
            wav = model.compression_model.decode(encoded_audio).squeeze(0)

        else:
            input_audio, sr = torchaudio.load(input_audio)
            input_audio = input_audio[None] if input_audio.dim() == 2 else input_audio

            continuation_start = 0 if not continuation_start else continuation_start
            if continuation_end is None or continuation_end == -1:
                continuation_end = input_audio.shape[2] / sr

            if continuation_start > continuation_end:
                raise ValueError(
                    "`continuation_start` must be less than or equal to `continuation_end`"
                )

            input_audio_wavform = input_audio[
                ..., int(sr * continuation_start) : int(sr * continuation_end)
            ]
            input_audio_duration = input_audio_wavform.shape[-1] / sr

            if continuation:
                if (
                    duration + input_audio_duration
                    > model.lm.cfg.dataset.segment_duration
                ):
                    raise ValueError(
                        "duration + continuation duration must be <= 30 seconds"
                    )

                set_generation_params(duration + input_audio_duration)
                wav = model.generate_continuation(
                    prompt=input_audio_wavform,
                    prompt_sample_rate=sr,
                    descriptions=[prompt],
                    progress=True,
                )

            else:
                set_generation_params(duration)
                wav = model.generate_with_chroma(
                    [prompt], input_audio_wavform, sr, progress=True
                )

        audio_write(
            "out",
            wav[0].cpu(),
            model.sample_rate,
            strategy=normalization_strategy,
        )
        wav_path = "out.wav"

        if output_format == "mp3":
            mp3_path = "out.mp3"
            subprocess.call(["ffmpeg", "-i", wav_path, "-y", mp3_path])
            os.remove(wav_path)
            path = mp3_path
        else:
            path = wav_path

        return Path(path)

    def _preprocess_audio(
        audio_path, model: MusicGen, duration: tp.Optional[int] = None
    ):

        wav, sr = torchaudio.load(audio_path)
        wav = torchaudio.functional.resample(wav, sr, model.sample_rate)
        wav = wav.mean(dim=0, keepdim=True)

        # Calculate duration in seconds if not provided
        if duration is None:
            duration = wav.shape[1] / model.sample_rate

        # Check if duration is more than 30 seconds
        if duration > 30:
            raise ValueError("Duration cannot be more than 30 seconds")

        end_sample = int(model.sample_rate * duration)
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


# From https://gist.github.com/gatheluck/c57e2a40e3122028ceaecc3cb0d152ac
def set_all_seeds(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
