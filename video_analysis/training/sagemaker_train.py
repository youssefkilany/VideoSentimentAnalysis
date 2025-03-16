from sagemaker.pytorch import PyTorch
from sagemaker.debugger import TensorBoardOutputConfig

import os
from dotenv import load_dotenv


def start_training_job():
    tensorboard_config = TensorBoardOutputConfig(
        s3_output_path=f"s3://{os.getenv('S3_BUCKET')}/tensorboard-logs/",
        container_local_output_path="/opt/ml/output/tensorboard-logs",
    )

    classifier = PyTorch(
        entry_point="training/train.py",
        source_dir=".",
        role=f"{os.getenv('SAGEMAKER_AUTHORIZED_ROLE')}",
        framework_version="2.5.1",
        py_version="py311",
        instance_count=1,
        instance_type="ml.g5.xlarge",
        hyperparameters={"batch-size": 16, "epochs": 10},
        tensorboard_config=tensorboard_config,
    )

    classifier.fit(
        {
            "training": f"s3://{os.getenv('S3_BUCKET')}/meld-dataset/train/",
            "validation": f"s3://{os.getenv('S3_BUCKET')}/meld-dataset/val/",
            "testing": f"s3://{os.getenv('S3_BUCKET')}/meld-dataset/test/",
        }
    )


if __name__ == "__main__":
    load_dotenv()
    start_training_job()
