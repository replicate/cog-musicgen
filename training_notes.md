# Steps for training MusicGen on YouTube Audio

This guide provides step-by-step instructions for training MusicGen on your own audio using `Cog` and Replicate.

We decided to use the NES Mega Man 2 [soundtrack](https://www.youtube.com/watch?v=lDC4X8Dgxr4) as training data because, well, it's awesome. And it allows us to demonstrate some useful preprocessing steps. Specifically, you'll see how to: 

1. Setup your development environment
2. Extract audio from YouTube
3. Segment an audio sequence into sliding windows of a specific duration
4. Preprocess audio files for MusicGen
5. Use machine learning to generate metadata for audio files
6. Format data so that you can use it to fine-tune MusicGen
7. Use your fine-tuned MusicGen checkpoint to generate your own music

# 1. Setup your development environment

We'll use cog to prepare our development environment. 

To install cog: 

```
# instructions to install cog
```

Then, clone this [repository](https://github.com/replicate/cog-musicgen/) and run `cog build`. 

```sh
git clone [repository](https://github.com/replicate/cog-musicgen
cd cog-musicgen
cog build
```

Finally, we'll cache MusicGen's weights by running `cog predict`. 

```
cog predict -i prompt=test
```

This command will trigger the `Predictor.setup()` method defined in `predict.py` and, assuming the  model has not already been cached locally, the weights will be downloaded and cached.

# 1. Extract audio from YouTube

We'll use `youtube-dl` to download [this](https://www.youtube.com/watch?v=lDC4X8Dgxr4) YouTube version of the Mega Man 2 soundtrack. 

```bash
cog run youtube-dl -x --audio-format wav -o data/nes_mega_man_2_soundtrack.mp4 https://www.youtube.com/watch?v=lDC4X8Dgxr4
```