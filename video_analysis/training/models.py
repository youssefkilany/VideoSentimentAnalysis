import torch
from torch import Tensor, nn
from torchvision.models.video.resnet import r3d_18

from transformers import BertModel


def freeze_module_grads(module: nn.Module):
    for param in module.parameters(recurse=True):
        param.requires_grad = False


class TextEncoder(nn.Module):
    def __init__(
        self,
        tokenizer_id: str,
        out_dim: int,
        drop_out_prob: float,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.encoder = BertModel.from_pretrained(tokenizer_id)
        freeze_module_grads(self.encoder)
        in_dim = self.encoder.pooler.dense.out_features

        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Dropout1d(drop_out_prob),
        )

    def forward(self, inputs_ids: Tensor, attention_mask: Tensor):
        outputs = self.encoder(inputs_ids, attention_mask).pooler_output
        outputs = self.proj(outputs)
        return outputs


class AudioEncoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, drop_out_prob: float, **kwargs):
        super().__init__(**kwargs)

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(in_dim, in_dim, kernel_size=3),
            nn.BatchNorm1d(in_dim),
            nn.ReLU(),
        )
        self.pool = nn.AvgPool1d(kernel_size=2)
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(in_dim, out_dim, kernel_size=3),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
        )
        self.ada_pool = nn.AdaptiveAvgPool1d(1)

        self.proj = nn.Sequential(
            nn.Linear(out_dim, out_dim), nn.ReLU(), nn.Dropout1d(drop_out_prob)
        )

        self.pipeline = nn.Sequential(
            self.conv_block1,
            self.pool,
            self.conv_block2,
            self.ada_pool,
            nn.Flatten(),
            self.proj,
        )

    def forward(self, inputs: Tensor):
        inputs = inputs.squeeze(1)
        return self.pipeline(inputs)


class VideoEncoder(nn.Module):
    def __init__(self, out_dim: int, drop_out_prob: float, **kwargs):
        super().__init__(**kwargs)

        self.backbone = r3d_18()
        freeze_module_grads(self.backbone)

        in_dim = self.backbone.fc.out_features
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Dropout1d(drop_out_prob),
        )

    def forward(self, inputs: Tensor):
        # [b, t, c, h, w] -> [b, c, t, h, w]
        inputs.transpose_(1, 2)
        return self.proj(self.backbone(inputs))


class FusionLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, drop_out_prob: float, **kwargs):
        super().__init__(**kwargs)

        # TODO: fix batchnorm error when batchsize is 1
        self.pipeline = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Dropout1d(drop_out_prob),
        )

    def forward(self, inputs: Tensor):
        return self.pipeline(inputs)


class ClassifierHead(nn.Module):
    def __init__(
        self, in_dim: int, hid_dim: int, out_dim: int, drop_out_prob: float, **kwargs
    ):
        super().__init__(**kwargs)

        self.pipeline = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.ReLU(),
            nn.Dropout1d(drop_out_prob),
            nn.Linear(hid_dim, out_dim),
        )

    def forward(self, inputs: Tensor):
        return self.pipeline(inputs)


class MultimodalSentimentClassifier(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        bert_tokenizer = "bert-base-uncased"
        in_dim: int = 64
        hid_dim: int = 264
        out_dim: int = 128
        drop_out_prob: float = 0.2

        self.text_encoder = TextEncoder(bert_tokenizer, out_dim, drop_out_prob)
        self.audio_encoder = AudioEncoder(in_dim, out_dim, drop_out_prob)
        self.video_encoder = VideoEncoder(out_dim, drop_out_prob)

        self.fusion_layer = FusionLayer(out_dim * 3, out_dim, drop_out_prob)

        emotion_cls = 7
        sentiment_cls = 3
        self.emotion_classifier = ClassifierHead(
            out_dim, hid_dim, emotion_cls, drop_out_prob
        )
        self.sentiment_classifier = ClassifierHead(
            out_dim, hid_dim, sentiment_cls, drop_out_prob
        )

    def forward(
        self, text_input: list[Tensor], audio_input: Tensor, video_input: Tensor
    ):
        inputs = [
            self.text_encoder(*text_input),
            self.audio_encoder(audio_input),
            self.video_encoder(video_input),
        ]
        self.embeddings = torch.cat(inputs, dim=1)
        backbone_output = self.fusion_layer(self.embeddings)

        return {
            "emotions": self.emotion_classifier(backbone_output),
            "sentiments": self.sentiment_classifier(backbone_output),
        }
