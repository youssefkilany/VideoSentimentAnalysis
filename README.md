# Video Sentiment Analysis - [WIP]

This is a Multimodal model that detects Emotion and Sentiment Classification in videos.

## Getting Started

### Dependencies

* Torch
* Torchvision
* Torchaudio
* HuggingFace Transformers
* PySoundFile as a torchaudio backend

### Installing

* Loading, video and splitting into video/audio is all handled by FFmpeg, you can find it here [for downloading](https://www.ffmpeg.org/download.html).

* clone locally via HTTPs/SSH

```cli
git clone https://github.com/youssefkilany/VideoSentimentAnalysis.git
```

```cli
git@github.com:youssefkilany/VideoSentimentAnalysis.git
```

Then, you can install it as a local package

```cli
uv pip install .
```

or, if you want to use it while editing

```cli
uv install -e . 
```

### Executing program

You can simply run `run-training.bat` or `./run-training.sh` file.

## Help

Run this command for training cli documentation

```cli
python video_analysis/training/train.py -h
```
