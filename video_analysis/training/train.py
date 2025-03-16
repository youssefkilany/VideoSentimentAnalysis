import argparse
import os


SM_MODEL_DIR = os.environ.get("", ".")
SM_CHANNEL_TRAINING = os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/train")
SM_CHANNEL_VALIDATION = os.environ.get(
    "SM_CHANNEL_VALIDATION", "/opt/ml/input/data/val"
)
SM_CHANNEL_TESTING = os.environ.get("SM_CHANNEL_TESTING", "/opt/ml/input/data/test")

# check if this is a Sagemaker environment
if os.environ.get("SM_CHANNEL_TRAINING", None) is not None:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def get_cli_args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--num-epochs", help="Number of training epochs", type=int, default=20
    )
    parser.add_argument(
        "--eval-every", help="run validation every n epochs", type=int, default=20
    )
    parser.add_argument("--batch-size", help="batch size", type=int, default=8)

    # input/output data and artifacts directories
    parser.add_argument(
        "--train-csv-path",
        help="training csv file path",
        type=str,
        default=f"{SM_CHANNEL_TRAINING}/train.csv",
    )
    parser.add_argument(
        "--val-csv-path",
        help="validation csv file path",
        type=str,
        default=f"{SM_CHANNEL_VALIDATION}/val.csv",
    )
    parser.add_argument(
        "--test-csv-path",
        help="testing csv file path",
        type=str,
        default=f"{SM_CHANNEL_TESTING}/test.csv",
    )
    parser.add_argument(
        "--train-dir",
        help="training data directory path",
        type=str,
        default=SM_CHANNEL_TRAINING,
    )
    parser.add_argument(
        "--val-dir",
        help="training data directory path",
        type=str,
        default=SM_CHANNEL_VALIDATION,
    )
    parser.add_argument(
        "--test-dir",
        help="training data directory path",
        type=str,
        default=SM_CHANNEL_TESTING,
    )
    parser.add_argument(
        "--model-dir",
        help="training outputs directory including the model",
        type=str,
        default=SM_MODEL_DIR,
    )

    return parser


def main():
    # these imports take time, so move it after cli parsing, so we can fail-fast cli usage

    from video_analysis.training.dataset import prepare_meld_dataloaders
    from video_analysis.training.models import MultimodalSentimentClassifier
    from video_analysis.training.trainer import MultimodalTrainer
    from video_analysis.training.utils import check_requirements_if_valid, get_cuda

    if not check_requirements_if_valid():
        return

    datasets_dir = {
        "train": {"csv_path": args.train_csv_path, "video_dir": args.train_dir},
        "val": {"csv_path": args.val_csv_path, "video_dir": args.val_dir},
        "test": {"csv_path": args.test_csv_path, "video_dir": args.test_dir},
    }

    device = get_cuda()
    meld_dataloaders = prepare_meld_dataloaders(datasets_dir, args.batch_size, device)

    model = MultimodalSentimentClassifier().to(device.value)
    trainer = MultimodalTrainer(model, *meld_dataloaders, args.model_dir, "cli-runs")
    trainer.train_model(args.num_epochs, args.eval_every)


if __name__ == "__main__":
    args = get_cli_args_parser().parse_args()

    main()
