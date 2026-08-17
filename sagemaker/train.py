"""SageMaker training entry point.

Reuses the exact model/training code from `src/` (no architecture duplication) and wraps it
with the argument parsing and directory conventions the SageMaker PyTorch container expects:
  - hyperparameters come in as CLI args (SageMaker injects them from the `hyperparameters`
    dict passed to the Estimator)
  - SM_MODEL_DIR / SM_OUTPUT_DATA_DIR / SM_CHANNEL_TRAIN are environment variables the
    container sets automatically

Run locally as a sanity check with:
    python sagemaker/train.py --epochs 1 --data-dir ./data --model-dir /tmp/model
"""

import argparse
import json
import os
import sys

import torch
from torchvision import datasets, transforms

# Make `src/` importable regardless of the process's working directory (SageMaker runs this
# script from /opt/ml/code, with the rest of the repo copied alongside it).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import FASHION_MNIST_MEAN, FASHION_MNIST_STD  # noqa: E402
from src.models import BaselineMLP, SimpleCNN, count_parameters  # noqa: E402
from src.train import train_model, evaluate  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    # Model / training hyperparameters (mirror the choices explored in the notebook).
    parser.add_argument("--model", type=str, default="cnn", choices=["cnn", "baseline"])
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)

    # SageMaker directory conventions (env vars set inside the training container; the
    # defaults let this script also run standalone outside SageMaker).
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "./model"))
    parser.add_argument("--data-dir", type=str, default=os.environ.get("SM_CHANNEL_TRAIN", "./data"))
    parser.add_argument("--output-data-dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR", "./output"))

    return parser.parse_args()


def build_dataloaders(data_dir, batch_size):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((FASHION_MNIST_MEAN,), (FASHION_MNIST_STD,))]
    )
    # `download=True` requires the training instance to have internet access, which is the
    # default for a standard (non VPC-isolated) SageMaker training job. If your training job
    # runs inside a VPC with no NAT/internet, pre-download Fashion-MNIST and upload the
    # resulting `data/FashionMNIST` folder to S3 as the `train` channel instead, then leave
    # `download=False` here.
    train_full = datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=transform)
    test_set = datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=transform)

    n_val = int(len(train_full) * 0.1)
    n_train = len(train_full) - n_val
    generator = torch.Generator().manual_seed(42)
    train_set, val_set = torch.utils.data.random_split(train_full, [n_train, n_val], generator=generator)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader = build_dataloaders(args.data_dir, args.batch_size)

    if args.model == "cnn":
        model = SimpleCNN(kernel_size=args.kernel_size)
    else:
        model = BaselineMLP()

    print(model)
    print(f"Trainable parameters: {count_parameters(model):,}")

    history, elapsed = train_model(model, train_loader, val_loader, epochs=args.epochs, lr=args.lr, device=device)
    test_loss, test_acc, _, _ = evaluate(model, test_loader, device=device)
    print(f"Training time: {elapsed:.1f}s  |  test_loss={test_loss:.4f}  test_acc={test_acc:.4f}")

    os.makedirs(args.model_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.model_dir, "model.pth"))

    # Persist the architecture choice alongside the weights so inference.py can rebuild the
    # exact same model before loading the state dict.
    config = {"model": args.model, "kernel_size": args.kernel_size}
    with open(os.path.join(args.model_dir, "config.json"), "w") as f:
        json.dump(config, f)

    os.makedirs(args.output_data_dir, exist_ok=True)
    with open(os.path.join(args.output_data_dir, "metrics.json"), "w") as f:
        json.dump({"test_loss": test_loss, "test_accuracy": test_acc, "train_time_s": elapsed}, f)


if __name__ == "__main__":
    main()
