from contextlib import nullcontext
from datetime import datetime
import json
import os
from pathlib import Path

from sklearn.metrics import accuracy_score, precision_score
import torchaudio
from tqdm import tqdm
from video_analysis.training.models import MultimodalSentimentClassifier

import torch
from torch import nn, optim, inference_mode
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from video_analysis.utils.logging import log_info


class MultimodalTrainer:
    def __init__(
        self,
        model: MultimodalSentimentClassifier,
        train_dl: DataLoader,
        val_dl: DataLoader,
        test_dl: DataLoader,
        model_dir: Path,
        *args,
        base_dir: Path = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.device = train_dl.dataset[0]["text_inputs"][0].device.type
        self.model = model.to(self.device)

        self.train_dl, self.val_dl, self.test_dl = train_dl, val_dl, test_dl

        # TODO: add dataset stats logging on creating a Trainer instance
        log_info(f"train_dl Dataset size = {len(self.train_dl.dataset)}")
        log_info(f"val_dl Dataset size = {len(self.val_dl.dataset)}")
        log_info(f"test_dl Dataset size = {len(self.test_dl.dataset)}")

        if model_dir is None:
            model_dir = "./model"
        self.model_dir = model_dir

        if base_dir is None:
            base_dir = (
                "/opt/ml/output/tensorboard" if "SM_MODEL_DIR" in os.environ else "runs"
            )

        # TODO: this should be serialized/deserialized along other parameters for reference
        training_timestamp = datetime.now().strftime("%b%d-%H_%M_%S")
        self.run_id = f"run_{training_timestamp}"
        log_dir = f"{base_dir}/{self.run_id}"
        self.tb_writer = SummaryWriter(log_dir)
        self.total_training_epochs = 0
        self.total_training_steps = 0
        self.current_train_losses = None

        emotions_weights = torch.tensor(
            self.train_dl.emotions_weights, device=self.device
        )
        self.emotions_loss_fn = torch.nn.CrossEntropyLoss(
            label_smoothing=0.05, weight=emotions_weights
        )

        sentiments_weights = torch.tensor(
            self.train_dl.sentiments_weights, device=self.device
        )
        self.sentiments_loss_fn = torch.nn.CrossEntropyLoss(
            label_smoothing=0.05, weight=sentiments_weights
        )

        # TODO: find a systematic way of exploring this grid of lr parameters
        # TODO: do some mini experiments and log metrics
        self.optimizer = optim.Adam(
            [
                {"params": self.model.text_encoder.parameters(), "lr": 1e-5},
                {"params": self.model.audio_encoder.parameters(), "lr": 1e-5},
                {"params": self.model.video_encoder.parameters(), "lr": 1e-5},
                {"params": self.model.fusion_layer.parameters(), "lr": 1e-5},
                {"params": self.model.emotion_classifier.parameters(), "lr": 1e-5},
                {"params": self.model.sentiment_classifier.parameters(), "lr": 1e-5},
            ],
            weight_decay=1e-5,
        )

        self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=2
        )

    def tb_log_metrics(self, losses, metrics=None, is_training=False):
        if is_training:
            self.current_train_losses = losses
            for k in self.current_train_losses.keys():
                self.tb_writer.add_scalar(
                    f"loss/{k}/train",
                    self.current_train_losses[k],
                    self.total_training_steps,
                )

            return

        for k in self.current_train_losses.keys():
            self.tb_writer.add_scalar(
                f"loss/{k}/val",
                losses[k],
                self.total_training_steps,
            )

        if metrics:
            phase = "train" if is_training else "val"
            for output, output_metrics in metrics.items():
                for metric in output_metrics:
                    self.tb_writer.add_scalar(
                        f"{phase}/{output}/{metric}",
                        metrics[output][metric],
                        self.total_training_steps,
                    )

    def run_single_iteration(self, data_dl, is_training=False):
        losses = {"emotions": 0, "sentiments": 0, "total": 0}
        emotions_preds = []
        emotions_labels = []
        sentiments_preds = []
        sentiments_labels = []

        if is_training:
            self.model.train()
        else:
            self.model.eval()

        # enter a context manager conditionally
        # https://stackoverflow.com/a/34798330/12896502

        with nullcontext() if is_training else inference_mode() as _ctx:
            title = "Training" if is_training else "Validation"
            for batch in tqdm(data_dl, desc=f"{title} steps", total=len(data_dl)):
                text_inputs = batch["text_inputs"]
                audio_features = batch["audio_features"]
                video_frames = batch["video_frames"]
                emotions_gt, sentiments_gt = batch["emotion"], batch["sentiment"]

                emotions_pred, sentiments_pred = self.model(
                    text_inputs, audio_features, video_frames
                ).values()

                emotions_loss = self.emotions_loss_fn(emotions_pred, emotions_gt)
                sentiments_loss = self.sentiments_loss_fn(
                    sentiments_pred, sentiments_gt
                )
                total_loss = emotions_loss + sentiments_loss

                losses["emotions"] += emotions_loss.item()
                losses["sentiments"] += sentiments_loss.item()
                losses["total"] += total_loss.item()

                if is_training:
                    train_loss = {
                        "emotions": emotions_loss.item(),
                        "sentiments": sentiments_loss.item(),
                    }
                    self.total_training_steps += 1
                    self.tb_log_metrics(train_loss, is_training=is_training)

                    self.optimizer.zero_grad()
                    total_loss.backward()
                    nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()

                else:
                    emotions_labels.extend(emotions_gt.cpu().numpy())
                    emotions_pred = torch.argmax(emotions_pred, dim=1)
                    emotions_preds.extend(emotions_pred.cpu().numpy())
                    sentiments_labels.extend(sentiments_gt.cpu().numpy())
                    sentiments_pred = torch.argmax(sentiments_pred, dim=1)
                    sentiments_preds.extend(sentiments_pred.cpu().numpy())

        avg_losses = {k: v / len(data_dl) for k, v in losses.items()}

        if is_training:
            self.total_training_epochs += 1
            return avg_losses

        # update lr scheduler _only_ based on validation loss
        self.lr_scheduler.step(avg_losses["total"])

        metrics = {
            "emotions": {
                "precision": precision_score(
                    emotions_labels, emotions_preds, average="weighted", zero_division=1
                ),
                "accuracy": accuracy_score(emotions_labels, emotions_preds),
            },
            "sentiments": {
                "precision": precision_score(
                    sentiments_labels,
                    sentiments_preds,
                    average="weighted",
                    zero_division=1,
                ),
                "accuracy": accuracy_score(sentiments_labels, emotions_preds),
            },
        }
        self.tb_log_metrics(avg_losses, metrics, is_training)

        return avg_losses, metrics

    def log_gpu_memory_used(self):
        if self.device == "cuda":
            gpu_used = torch.cuda.max_memory_allocated() / 1024**3
            log_info(f"GPU Memory Used: {gpu_used:.2f} GB")

    def log_training_intro(self):
        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            log_info(f"GPU Memory Summary\n{torch.cuda.memory_summary()}")
            self.log_gpu_memory_used()
        else:
            log_info("Starting training with CPU")

        log_info(
            f"Available Audio backends: {' '.join(torchaudio.list_audio_backends())}"
        )

    def save_model(self):
        # TODO: update save model to only save trainable modules, since now it stores the whole BERT and r18_3d models which are unnecessary and is inefficient
        # start from here: https://stackoverflow.com/a/78035275
        model_path = f"{self.model_dir}/{self.run_id}-{self.total_training_epochs}-{self.total_training_steps}.pth"
        os.makedirs(self.model_dir, exist_ok=True)
        torch.save(self.model.state_dict(), model_path)

    def train_model(self, train_epochs: int, eval_every_n: int):
        def metric_json_fmt(name, val):
            return {"Name": name, "Value": val}

        best_loss = float("inf")

        self.log_training_intro()

        metrics_data = {
            "train_loss": [],
            "train_epoch": [],
            "val_loss": [],
            "val_epoch": [],
            "test_loss": [],
            "test_epoch": [],
        }

        start_epoch, end_epoch = (
            self.total_training_epochs,
            self.total_training_epochs + train_epochs,
        )
        for epoch in tqdm(
            range(start_epoch, end_epoch),
            desc="Training epoch",
            total=train_epochs,
            leave=False,
        ):
            train_loss = self.run_single_iteration(self.train_dl, is_training=True)

            metrics_data["train_loss"].append(train_loss["total"])
            metrics_data["train_epoch"].append(epoch)
            metrics_json = json.dumps(
                {"metrics": [metric_json_fmt("train:loss", train_loss["total"])]}
            )

            if (epoch - start_epoch + 1) % eval_every_n == 0:
                val_loss, val_metrics = self.run_single_iteration(self.val_dl)

                if val_loss["total"] < best_loss:
                    best_loss = val_loss["total"]
                    self.save_model()

                metrics_data["val_loss"].append(val_loss["total"])
                metrics_data["val_epoch"].append(epoch)
                metrics_json = json.dumps(
                    {
                        "metrics": [
                            metric_json_fmt("val:loss", val_loss["total"]),
                            *[
                                metric_json_fmt(f"val:{output}_{metric}", output)
                                for output, output_metrics in val_metrics.items()
                                for metric in output_metrics
                            ],
                        ]
                    }
                )

            self.log_gpu_memory_used()

        test_loss, test_metrics = self.run_single_iteration(self.test_dl)
        metrics_data["test_loss"].append(test_loss["total"])
        metrics_data["test_epoch"].append(self.total_training_epochs)
        metrics_json = json.dumps(
            {
                "metrics": [
                    metric_json_fmt("test:loss", test_loss["total"]),
                    *[
                        metric_json_fmt(f"val:{output}_{metric}", output)
                        for output, output_metrics in test_metrics.items()
                        for metric in output_metrics
                    ],
                ]
            }
        )
        metrics_json
