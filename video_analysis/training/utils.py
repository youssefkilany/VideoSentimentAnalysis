from enum import Enum
from dataclasses import dataclass
import subprocess as sp

from torch.cuda import is_available as cuda_is_available

from video_analysis.install_ffmpeg import install_via_pip_alpine
from video_analysis.utils.logging import log_error, log_info


class Emotion(Enum):
    ANGER = 0
    DISGUST = 1
    FEAR = 2
    JOY = 3
    NEUTRAL = 4
    SADNESS = 5
    SURPRISE = 6


emotion_rev_map = {idx: val.name for idx, val in enumerate(list(Emotion))}


class Sentiment(Enum):
    NEUTRAL = 0
    POSITIVE = 1
    NEGATIVE = 2


sentiment_rev_map = {idx: val.name for idx, val in enumerate(list(Sentiment))}


class ComputeDevice(Enum):
    CPU = "cpu"
    CUDA = "cuda"


@dataclass
class Timestamp:
    hour: int
    minute: int
    second: int
    millis: int


@dataclass
class MeldDatasetConfig:
    csv_path: str
    video_dir: str
    device: ComputeDevice = ComputeDevice.CPU
    # Do more research on these settings
    tokenizer_id: str = "bert-base-uncased"
    seq_max_len: int = 128
    max_frames: int = 30
    model_input_w: int = 224
    model_input_h: int = 224
    n_fft: int = 1024
    hop_length: int = 512
    n_mels: int = 64
    max_mel_len: int = 300


@dataclass
class MeldVideoSample:
    id: int
    utterance: str
    speaker: str
    emotion: Emotion
    sentiment: Sentiment
    dialogue_id: int
    utterance_id: int
    season: int
    episode: int
    startTime: Timestamp
    endTime: Timestamp


@dataclass
class VideoMetadata:
    height: int
    width: int
    fps: int


def get_cuda(required=False):
    device = ComputeDevice.CUDA if cuda_is_available else ComputeDevice.CPU
    assert not required or (required and device == ComputeDevice.CUDA)
    return device


def check_requirements_if_valid():
    try:
        version = sp.run(["ffmpeg", "-version"], capture_output=True, check=True)
        version = version.stdout.decode("utf-8").split("-")[0]
        log_info(f"Found an installation of FFmpeg. {version}")
        return True

    except (sp.CalledProcessError, FileNotFoundError) as requirements_exc:
        installation_exc = None
        try:
            install_via_pip_alpine()
            return True
        except Exception as exc:
            installation_exc = exc

        log_error(
            "Script can't find FFmpeg installed on the system, and couldn't install it automatically. Please try to install it manually.",
            installation_exc,
            "Couldn't Find an installation of FFmpeg on the system.",
            requirements_exc,
        )
