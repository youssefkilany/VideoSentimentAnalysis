from collections import namedtuple

import torch

from video_analysis.training.models import MultimodalSentimentClassifier
from video_analysis.training.trainer import MultimodalTrainer


def test_logging():
    Batch = namedtuple("Batch", ["text_inputs", "video_frames", "audio_features"])

    mock_batch = Batch(
        text_inputs=[torch.ones(1), torch.ones(1)],
        video_frames=torch.ones(1),
        audio_features=torch.ones(1),
    )
    mock_train_loader = torch.utils.data.DataLoader([mock_batch])
    mock_val_loader = torch.utils.data.DataLoader([mock_batch])

    classifier = MultimodalSentimentClassifier()
    trainer = MultimodalTrainer(
        classifier, mock_train_loader, mock_val_loader, base_dir="mock-runs"
    )

    mock_losses = {
        "total": 1.3,
        "emotion": 1.6,
        "sentiment": 1.9,
    }
    trainer.log_metrics(mock_losses, is_training=True)
    trainer.total_training_epochs += 1

    mock_losses = {k: v + 1 for k, v in mock_losses.items()}
    mock_metrics = {
        "emotion": {"accuracy": 1.2, "precision": 1.4},
        "sentiment": {"accuracy": 1.6, "precision": 1.8},
    }
    trainer.log_metrics(mock_losses, mock_metrics, is_training=False)
    trainer.total_training_epochs += 1

    mock_losses = {k: v + 1 for k, v in mock_losses.items()}
    trainer.log_metrics(mock_losses, is_training=True)
    trainer.total_training_epochs += 1

    mock_losses = {k: v + 1 for k, v in mock_losses.items()}
    mock_metrics = {
        "emotion": {"accuracy": 2.2, "precision": 2.4},
        "sentiment": {"accuracy": 2.6, "precision": 2.8},
    }
    trainer.log_metrics(mock_losses, mock_metrics, is_training=False)


if __name__ == "__main__":
    test_logging()
