# Cog implementation of MusicGen - Melody - 1.5B
[![Replicate](https://replicate.com/joehoover/musicgen-melody/badge)](https://replicate.com/joehoover/musicgen-melody) 

MusicGen - Melody - 1.5B is Audiocraft provides the code and models for MusicGen, [a simple and controllable model for music generation][arxiv].  MusicGen is a single stage auto-regressive Transformer model trained over a 32kHz <a href="https://github.com/facebookresearch/encodec">EnCodec tokenizer</a> with 4 codebooks sampled at 50 Hz. Unlike existing methods like [MusicLM](https://arxiv.org/abs/2301.11325), MusicGen doesn't require a self-supervised semantic representation, and it generates all 4 codebooks in one pass. By introducing a small delay between the codebooks, the authors show they can predict them in parallel, thus having only 50 auto-regressive steps per second of audio. They used 20K hours of licensed music to train MusicGen. Specifically, they relied on an internal dataset of 10K high-quality music tracks, and on the ShutterStock and Pond5 music data.


For more information about this model, see [here](https://github.com/facebookresearch/audiocraft).

You can demo this model or learn how to use it with Replicate's API [here](https://replicate.com/joehoover/musicgen-melody). 

# Run with Cog

[Cog](https://github.com/replicate/cog) is an open-source tool that packages machine learning models in a standard, production-ready container. 
You can deploy your packaged model to your own infrastructure, or to [Replicate](https://replicate.com/), where users can interact with it via web interface or API.

## Prerequisites 

**Cog.** Follow these [instructions](https://github.com/replicate/cog#install) to install Cog, or just run: 

```
sudo curl -o /usr/local/bin/cog -L "https://github.com/replicate/cog/releases/latest/download/cog_$(uname -s)_$(uname -m)"
sudo chmod +x /usr/local/bin/cog
```

Note, to use Cog, you'll also need an installation of [Docker](https://docs.docker.com/get-docker/).

* **GPU machine.** You'll need a Linux machine with an NVIDIA GPU attached and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) installed. If you don't already have access to a machine with a GPU, check out our [guide to getting a 
GPU machine](https://replicate.com/docs/guides/get-a-gpu-machine).

## Step 1. Clone this repository

```sh
git clone https://github.com/replicate/cog-musicgen-melody
```

## Step 2. Run the model

To run the model, you need a local copy of the model's Docker image. You can satisfy this requirement by specifying the image ID in your call to `predict` like:

```
cog predict r8.im/joehoover/musicgen-melody@sha256:1a53415e6c4549e3022a0af82f4bd22b9ae2e747a8193af91b0bdffe63f93dfd -i description=tense staccato strings. plucked strings. dissonant. scary movie. -i duration=8
```

For more information, see the Cog section [here](https://replicate.com/joehoover/musicgen-melody/api#run)

Alternatively, you can build the image yourself, either by running `cog build` or by letting `cog predict` trigger the build process implicitly. For example, the following will trigger the build process and then execute prediction: 

```
cog predict -i description=tense staccato strings. plucked strings. dissonant. scary movie. -i duration=8
```

Note, the first time you run `cog predict`, model weights and other requisite assets will be downloaded if they're not available locally. This download only needs to be executed once.

# Run on replicate

## Step 1. Ensure that all assets are available locally

If you haven't already, you should ensure that your model runs locally with `cog predict`. This will guarantee that all assets are accessible. E.g., run: 

```
cog predict -i description=tense staccato strings. plucked strings. dissonant. scary movie. -i duration=8
```

## Step 2. Create a model on Replicate.

Go to [replicate.com/create](https://replicate.com/create) to create a Replicate model. If you want to keep the model private, make sure to specify "private".

## Step 3. Configure the model's hardware

Replicate supports running models on variety of CPU and GPU configurations. For the best performance, you'll want to run this model on an A100 instance.

Click on the "Settings" tab on your model page, scroll down to "GPU hardware", and select "A100". Then click "Save".

## Step 4: Push the model to Replicate


Log in to Replicate:

```
cog login
```

Push the contents of your current directory to Replicate, using the model name you specified in step 1:

```
cog push r8.im/username/modelname
```

[Learn more about pushing models to Replicate.](https://replicate.com/docs/guides/push-a-model)

# Licenses

* All code in this repository is licensed under the Apache License 2.0 license.
* The code in the [Audiocraft](https://github.com/facebookresearch/audiocraft) repository is released under the MIT license as found in the [LICENSE file](LICENSE).
* The weights in the [Audiocraft](https://github.com/facebookresearch/audiocraft) repository are released under the CC-BY-NC 4.0 license as found in the [LICENSE_weights file](LICENSE_weights).

