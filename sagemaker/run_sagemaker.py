"""Reference driver code for training + deploying this model on SageMaker.

This is NOT meant to be run as a script top-to-bottom. Copy each `# --- Cell N ---` block into
its own cell of a Jupyter notebook running on a SageMaker Notebook Instance / SageMaker
Studio (kernel: "conda_pytorch_p310" or similar), and run them one at a time, in order.

IMPORTANT: create that notebook at the repo root (the same folder that directly contains
`sagemaker/` and `src/` as siblings) — the relative paths below (`source_dir="sagemaker"`,
`dependencies=["src"]`) assume the notebook's working directory is the repo root.

Prerequisites (see README.md "Task 6" section for the console click-path):
  - This whole repo uploaded/cloned onto the notebook instance, so `sagemaker/train.py`,
    `sagemaker/inference.py`, and `src/` are all present alongside this file.
  - An IAM role with SageMaker + S3 permissions (get_execution_role() below finds it
    automatically when running *inside* a SageMaker notebook).
"""

# --- Cell 1: setup ---------------------------------------------------------
import sagemaker
from sagemaker.pytorch import PyTorch

session = sagemaker.Session()
role = sagemaker.get_execution_role()
bucket = session.default_bucket()  # auto-created S3 bucket for this account/region
print("Using bucket:", bucket)
print("Using role:", role)

# --- Cell 2: launch the training job ---------------------------------------
# `source_dir` points at the repo root so both `sagemaker/train.py` and `src/` get uploaded
# together; `entry_point` is the script path relative to `source_dir`.
estimator = PyTorch(
    entry_point="train.py",
    source_dir="sagemaker",
    dependencies=["src"],             # bundles src/ alongside the entry point
    role=role,
    framework_version="2.1",
    py_version="py310",
    instance_type="ml.m5.large",      # CPU is enough for this model; no need for a GPU instance
    instance_count=1,
    hyperparameters={
        "model": "cnn",
        "kernel-size": 3,
        "epochs": 8,
        "batch-size": 256,
        "lr": 0.001,
    },
    output_path=f"s3://{bucket}/fashion-mnist-cnn/output",
)

# No local `train` channel is passed: train.py downloads Fashion-MNIST itself via
# torchvision (the training instance has internet access by default). This call blocks and
# streams training logs until the job finishes (a few minutes on ml.m5.large).
estimator.fit()

# --- Cell 3: deploy to a real-time endpoint ---------------------------------
predictor = estimator.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.large",
    entry_point="inference.py",
    source_dir="sagemaker",
)
print("Endpoint name:", predictor.endpoint_name)

# --- Cell 4: test the endpoint ----------------------------------------------
import json
from torchvision import datasets, transforms

test_set = datasets.FashionMNIST(root="/tmp/data", train=False, download=True, transform=transforms.ToTensor())
image, true_label = test_set[0]

predictor.serializer = sagemaker.serializers.JSONSerializer()
predictor.deserializer = sagemaker.deserializers.JSONDeserializer()

response = predictor.predict({"image": image.squeeze(0).tolist()})
print("True label index:", true_label)
print("Prediction:", response)

# --- Cell 5: clean up (IMPORTANT — the endpoint bills hourly while it exists) ---
predictor.delete_endpoint()
