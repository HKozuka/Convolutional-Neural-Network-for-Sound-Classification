# Convolutional-Neural-Network-for-Sound-Classification

**Achieved 94% test accuracy on UrbanSound8K using a fine-tuned ConvNeXt Convolutional Neural Network**

Supervised fine-tuning of a **ConvNeXt** convolutional neural network for urban sound classification — demonstrating how audio can be reframed as an image classification problem.

## Overview

Raw audio clips are transformed into **spectrograms** (2D frequency-vs-time representations), which are then fed into a pretrained ConvNeXt model fine-tuned to classify 10 urban sound categories. The key insight: by converting sound into an image-like matrix, powerful computer vision models can be applied directly to audio tasks.

## Dataset

**[UrbanSound8K](https://www.kaggle.com/datasets/chrisfilo/urbansound8k)** — 8,732 labeled audio clips (up to 4 seconds each) from real-world urban environments, organized into 10 folds. This project uses **Fold 1** for training and validation.

| Class ID | Label |
|----------|-------|
| 0 | air_conditioner |
| 1 | car_horn |
| 2 | children_playing |
| 3 | dog_bark |
| 4 | drilling |
| 5 | engine_idling |
| 6 | gun_shot |
| 7 | jackhammer |
| 8 | siren |
| 9 | street_music |

## Pipeline

1. **Audio → Spectrogram**: Load `.wav` files with `torchaudio`, convert stereo to mono, apply Short-Time Fourier Transform (`n_fft=1024`)
2. **Preprocessing**: Resize spectrograms to `224×224`, expand to 3 channels to match ImageNet input format
3. **Model**: `ConvNeXt-Base` pretrained on ImageNet, final classification head replaced for 10-class output
4. **Training**: AdamW optimizer, cross-entropy loss, 80/20 train-validation split, batch size 32

## Tech Stack

- `PyTorch` + `torchaudio` — model training and audio processing
- `torchvision` — ConvNeXt model and image transforms
- `scikit-learn` — train/validation splitting
- `plotly` / `matplotlib` — visualization

## Usage

Open `ConvNeXt_sound_classifier.ipynb` in **Google Colab** or Jupyter. Mount your Google Drive, place the UrbanSound8K Fold 1 data at the expected path, and run cells top to bottom.

## References

- Liu et al., [*A ConvNet for the 2020s*](https://arxiv.org/abs/2201.03545), CVPR 2022
- Salamon et al., [*A Dataset and Taxonomy for Urban Sound Research*](https://dl.acm.org/doi/10.1145/2647868.2655045), ACM MM 2014
