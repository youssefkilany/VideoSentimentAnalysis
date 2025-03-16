from video_analysis.training.utils import (
    ComputeDevice,
    MeldDatasetConfig,
    MeldVideoSample,
    Emotion,
    Sentiment,
    Timestamp,
    VideoMetadata,
)
from video_analysis.utils.logging import log_error, log_info

import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

import os
import csv
import subprocess as sp

import uuid


def compute_weights(samples):
    emotions_weights = {emotion.value: 0 for emotion in list(Emotion)}
    sentiments_weights = {sentiment.value: 0 for sentiment in list(Sentiment)}

    skipped = 0
    for sample in samples:
        if sample is None:
            skipped += 1
            continue

        emotions_weights[sample.emotion.value] += 1
        sentiments_weights[sample.sentiment.value] += 1

    total = len(samples) - skipped
    emotions_weights = [
        emotion_cnt / total for emotion_cnt in emotions_weights.values()
    ]
    sentiments_weights = [
        sentiment_cnt / total for sentiment_cnt in sentiments_weights.values()
    ]
    return emotions_weights, sentiments_weights


class MeldDataset(Dataset):
    def __init__(self, config: MeldDatasetConfig, verbose=True):
        super().__init__()
        self.verbose = verbose
        self.csv_path = config.csv_path
        self.video_dir = config.video_dir

        # I had some problems with memory leak in GPU, this reasoning is according to this comment which have the same error msg as the I'm having.
        # TODO: solution is simply remove all logic of creating data on gpu from the start in the dataset, to the trainer and training loop at the beginning, which is the recommend way as per this answer.
        # make sure to check that this msg actually doesn't show up when num_workers>0
        # https://discuss.pytorch.org/t/124445/14
        self.device = config.device

        self.max_frames = config.max_frames
        self.seq_max_len = config.seq_max_len
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_id)

        self.model_input_w = config.model_input_w
        self.model_input_h = config.model_input_h

        self.n_fft = config.n_fft
        self.hop_length = config.hop_length
        self.n_mels = config.n_mels
        self.max_mel_len = config.max_mel_len

        self.samples = self.load_data()
        self.emotions_weights, self.sentiments_weights = compute_weights(self.samples)

    def load_csv(self) -> list[MeldVideoSample]:
        samples: list[MeldVideoSample] = []
        with open(self.csv_path) as csv_file:
            csv_lines = csv.reader(csv_file)
            next(csv_lines)
            samples = [MeldVideoSample(*row) for row in csv_lines]
        return samples

    @staticmethod
    def parse_timestamp(timestamp_str: str) -> Timestamp:
        h, m, sms = timestamp_str.split(":")
        s, ms = sms.split(",")
        splits = [int(v) for v in [h, m, s, ms]]
        return Timestamp(*splits)

    def load_data(self) -> list[MeldVideoSample]:
        samples = self.load_csv()
        transformed_samples = []
        attribute_transform_fns = {
            # the benefit of having explicit transform function for every field, is
            # that we can later easily add/remove more transformations if necessitate
            "id": int,
            "dialogue_id": int,
            "utterance_id": int,
            "season": int,
            "episode": int,
            "emotion": lambda emotion: Emotion[emotion.upper()],
            "sentiment": lambda sent: Sentiment[sent.upper()],
            "startTime": self.parse_timestamp,
            "endTime": self.parse_timestamp,
        }

        for sample in samples:
            for attribute, update_fn in attribute_transform_fns.items():
                val = sample.__getattribute__(attribute)
                try:
                    sample.__setattr__(attribute, update_fn(val))
                except Exception as e:
                    log_error(f"Invalid value of {attribute}, value = {val}", e)
            transformed_samples.append(sample)

        if self.verbose:
            log_info("CSV file is read & parse")

        return transformed_samples

    def tokenize_utterance(self, sample_utterance: str):
        return self.tokenizer(
            sample_utterance,
            padding="max_length",
            truncation=True,
            max_length=self.seq_max_len,
            return_tensors="pt",
        )

    def get_video_path(self, sample: MeldVideoSample) -> str:
        video_fname = f"dia{sample.dialogue_id}_utt{sample.utterance_id}.mp4"
        video_path = os.path.join(self.video_dir, video_fname)
        if not os.path.exists(self.video_dir):
            raise FileNotFoundError(
                f"Videos directory path is invalid, please check and try again. path = {self.video_dir}"
            )

        if not os.path.exists(video_path):
            raise FileNotFoundError(
                f"Video file name is invalid, please check and try again. video path = {video_path}"
            )
        return video_path

    def get_video_metadata(self, video_path: str) -> VideoMetadata:
        # ffprobe command
        "ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=s=x:p=0 input.mp4"

        ffprobe_command = [
            "ffprobe",
            "-v",
            "error",
            "-i",
            video_path,
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate",
            "-of",
            "csv=p=0",
        ]

        try:
            process = sp.run(ffprobe_command, stdout=sp.PIPE, bufsize=20)
            w, h, r_fps = process.stdout.decode("utf-8").strip("\r\n").split(",")
            r_fps = r_fps.split("/")
            fps = int(round(int(r_fps[0]) / int(r_fps[1])))
            return VideoMetadata(int(h), int(w), fps)
        except Exception as e:
            log_error(
                f"This video is corrupted, or not readable, path: {video_path}", e
            )

    def _preprocess_frames(self, raw_frames):
        frames = (
            torch.frombuffer(raw_frames, dtype=torch.uint8)
            .reshape(-1, self.model_input_h, self.model_input_w, 3)
            .to(device=self.device.value)
        )

        # max limit is 30 frames, if less than fps
        if frames.shape[0] < 30:
            zeros_padding = torch.zeros(
                (30 - frames.shape[0], *frames.shape[1:]), device=self.device.value
            )
            frames = torch.cat([frames, zeros_padding])

        # normalize and permute from N, H, W, C -> N, C, H, W
        return frames.permute(0, 3, 1, 2) / 255

    def _preprocess_audio(self, waveform, sample_rate):
        # TODO: Do more research on Mel Spectrogram parameters
        mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
        )

        # normalize spectrogram
        mel_spec = mel_spectrogram(waveform).to(device=self.device.value)
        mel_spec = (mel_spec - mel_spec.mean()) / mel_spec.std()

        if mel_spec.shape[-1] < self.max_mel_len:
            padding = self.max_mel_len - mel_spec.shape[-1]
            mel_spec = torch.nn.functional.pad(mel_spec, (0, padding))
        else:
            mel_spec = mel_spec[..., : self.max_mel_len]

        return mel_spec

    def extract_audio_video(self, video_path: str, video_metadata: VideoMetadata):
        # ffmpeg command
        limit_args = [
            # limit video to seconds equivalent of 30(max) frames adjusted to video fps
            "-t",  # or set to a constant of 1 second
            str(self.max_frames / video_metadata.fps),
            # "-frames:v",  # limit video to 30 frames, accurate but doesn't work with audio stream
            # str(self.max_frames),
        ]

        extract_video_args = [
            "-an",  # remove audio from video stream
            "-f",  # convert to raw bytes to be read as colored 3-channel
            "rawvideo",
            "-filter:v",  # transform raw data rgb channels
            "format=rgb24",
            "-s",  # resize to model input wxh
            f"{self.model_input_w}x{self.model_input_h}",
            *limit_args,
            "pipe:1",  # pipe video output to stdout instead of the default, a file
        ]

        temp_audio_file = f"temp-load-audio-{uuid.uuid4()}.mp3"
        extract_audio_args = [
            "-vn",  # remove video from audio stream
            "-f",
            "wav",
            # "mp3",
            "-c:a",  # change audio stream codec to mp3
            # "copy",  # we can just copy audio stream as is
            "libmp3lame",  # also, we can  re-encode using libmp3lame for wav/mp3 formats
            "-ar",  # change sampling rate
            "16k",
            "-ac",  # set to mono audio, i.e. 1-channel
            "1",
            "-b:a",  # change audio bitrate from 128k to 64b for faster processing
            "64k",
            *limit_args,
            # "pipe:2",  # pipe audio output to stderr instead of the default, a file
            # torch doesn't load audio from bytes stream, so it's written to a temp file
            "-y",  # overwrite output file without prompting
            temp_audio_file,
        ]

        ffmpeg_command = [
            "ffmpeg",
            "-v",  # suppress info logging to errors only
            "error",
            "-i",  # input video path
            video_path,
            *extract_audio_args,
            *extract_video_args,
        ]

        # run ffmpeg and load all frames into numpy array (num_frames, H, W, 3)
        # 1e8 is about 100mb which is the max size of any video in MELD dataset
        process = sp.run(ffmpeg_command, stdout=sp.PIPE, stderr=sp.PIPE, bufsize=10**8)

        # audio = process.stderr
        # TODO: find a way to load audio bytes stream into torchaudio
        waveform, sample_rate = torchaudio.load(temp_audio_file)

        if os.path.exists(temp_audio_file):
            os.remove(temp_audio_file)

        return (waveform, sample_rate), process.stdout

    def _dataset_getitem(self, index: int):
        if type(index) is torch.Tensor:
            index = index.item()
        sample = self.samples[index]

        text_inputs = self.tokenize_utterance(sample.utterance)

        video_path = self.get_video_path(sample)
        video_metadata = self.get_video_metadata(video_path)
        (waveform, sample_rate), raw_frames = self.extract_audio_video(
            video_path, video_metadata
        )

        video_frames = self._preprocess_frames(raw_frames)
        audio_features = self._preprocess_audio(waveform, sample_rate)

        return dict(
            text_inputs=[
                text_inputs["input_ids"].squeeze().to(self.device.value),
                text_inputs["attention_mask"].squeeze().to(self.device.value),
            ],
            video_frames=video_frames,
            audio_features=audio_features,
            emotion=torch.tensor(sample.emotion.value, device=self.device.value),
            sentiment=torch.tensor(sample.sentiment.value, device=self.device.value),
        )

    def __getitem__(self, index):
        if index >= len(self.samples):
            log_error(f"Invalid index at {index}")
            return

        try:
            item = self._dataset_getitem(index)
            return item
        except Exception as e:
            log_error(
                f"Dataset failed to fetch item at index {index}",
                f"Item details: {self.samples[index]}",
                e,
            )

    def __len__(self):
        return len(self.samples)


def prepare_meld_datasets(datasets_dir, device=ComputeDevice.CPU):
    default_config = {
        "tokenizer_id": MeldDatasetConfig.tokenizer_id,
        "seq_max_len": MeldDatasetConfig.seq_max_len,
        "device": device,
    }

    train_dataset = MeldDataset(
        MeldDatasetConfig(**datasets_dir["train"], **default_config), verbose=False
    )

    val_dataset = MeldDataset(
        MeldDatasetConfig(**datasets_dir["val"], **default_config), verbose=False
    )

    test_dataset = MeldDataset(
        MeldDatasetConfig(**datasets_dir["test"], **default_config), verbose=False
    )

    return train_dataset, val_dataset, test_dataset


def collate_fn(batch):
    return torch.utils.data.default_collate(list(filter(None, batch)))


def prepare_meld_dataloaders(datasets_dir, batch_size=8, device=ComputeDevice.CPU):
    def update_dl(dataloader, dataset: MeldDataset):
        dataloader.emotions_weights = dataset.emotions_weights
        dataloader.sentiments_weights = dataset.sentiments_weights
        return dataloader

    train_dataset, val_dataset, test_dataset = prepare_meld_datasets(
        datasets_dir, device=device
    )

    default_config = {
        "batch_size": batch_size,
        "collate_fn": collate_fn,
        "num_workers": 0,
    }

    train_dl = DataLoader(train_dataset, **default_config, shuffle=True)
    update_dl(train_dl, train_dataset)
    val_dl = DataLoader(val_dataset, **default_config)
    test_dl = DataLoader(test_dataset, **default_config)

    return train_dl, val_dl, test_dl
