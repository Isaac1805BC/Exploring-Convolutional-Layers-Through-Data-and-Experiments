"""SageMaker inference handler for the PyTorch serving container.

Implements the four functions the container looks for by name:
  model_fn    -> load the trained model once when the endpoint starts
  input_fn    -> deserialize the request body into a tensor
  predict_fn  -> run the forward pass
  output_fn   -> serialize the prediction back to JSON

Expected request body (application/json):
    {"image": [[...28 rows of 28 floats in [0, 1]...]]}
"""

import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import CLASS_NAMES, FASHION_MNIST_MEAN, FASHION_MNIST_STD  # noqa: E402
from src.models import BaselineMLP, SimpleCNN  # noqa: E402

CONTENT_TYPE = "application/json"


def model_fn(model_dir):
    with open(os.path.join(model_dir, "config.json")) as f:
        config = json.load(f)

    if config["model"] == "cnn":
        model = SimpleCNN(kernel_size=config["kernel_size"])
    else:
        model = BaselineMLP()

    state_dict = torch.load(os.path.join(model_dir, "model.pth"), map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def input_fn(request_body, content_type=CONTENT_TYPE):
    if content_type != CONTENT_TYPE:
        raise ValueError(f"Unsupported content type: {content_type}")

    payload = json.loads(request_body)
    image = torch.tensor(payload["image"], dtype=torch.float32)  # shape (28, 28), values in [0, 1]
    image = (image - FASHION_MNIST_MEAN) / FASHION_MNIST_STD
    return image.unsqueeze(0).unsqueeze(0)  # -> (1, 1, 28, 28)


def predict_fn(input_tensor, model):
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0)
    return probs


def output_fn(probs, accept=CONTENT_TYPE):
    predicted_idx = int(torch.argmax(probs).item())
    response = {
        "predicted_class": CLASS_NAMES[predicted_idx],
        "predicted_index": predicted_idx,
        "probabilities": {CLASS_NAMES[i]: float(p) for i, p in enumerate(probs)},
    }
    return json.dumps(response), CONTENT_TYPE
