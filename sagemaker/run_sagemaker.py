"""Reference driver code for training + deploying this model on SageMaker.

This mirrors `run_sagemaker.ipynb` cell-for-cell (open that notebook to actually run this —
nothing here needs copy-pasting). Kept as a plain-script version for readability/diffing.

Written against SageMaker Python SDK v3 (`sagemaker>=3`), which replaced the v2
`sagemaker.pytorch.PyTorch` estimator with `ModelTrainer`, and `estimator.deploy()` with
`ModelBuilder`. Every import/class/method below was verified against the installed SDK's
source (not just documentation), since v2-style code (`sagemaker.pytorch.PyTorch`,
`sagemaker.Session()`, `sagemaker.get_execution_role()`) raises
`ModuleNotFoundError: ... was removed in SDK v3` on this version.

IMPORTANT: run from the repo root (the same folder that directly contains `sagemaker/` and
`src/` as siblings) — the relative paths below (`source_dir="sagemaker"`) assume that.
"""

# --- Cell 1: setup ---------------------------------------------------------
from sagemaker.core.helper.session_helper import Session, get_execution_role
from sagemaker.core.image_uris import retrieve as retrieve_image_uri

session = Session()
role = get_execution_role()
region = session.boto_region_name
bucket = session.default_bucket()  # auto-created S3 bucket for this account/region
print("Region:", region)
print("Using bucket:", bucket)
print("Using role:", role)

# --- Cell 2: configure the training job -------------------------------------
from sagemaker.train import ModelTrainer
from sagemaker.core.training.configs import SourceCode, Compute

FRAMEWORK_VERSION = "2.1.0"
PY_VERSION = "py310"
INSTANCE_TYPE = "ml.m5.large"

training_image = retrieve_image_uri(
    framework="pytorch",
    region=region,
    version=FRAMEWORK_VERSION,
    py_version=PY_VERSION,
    instance_type=INSTANCE_TYPE,
    image_scope="training",
)
print("Training image:", training_image)

source_code = SourceCode(source_dir="sagemaker", entry_script="train.py")

model_trainer = ModelTrainer(
    training_image=training_image,
    role=role,
    sagemaker_session=session,
    base_job_name="fashion-mnist-cnn",
    source_code=source_code,
    compute=Compute(instance_type=INSTANCE_TYPE, instance_count=1),
    hyperparameters={
        "model": "cnn",
        "kernel-size": 3,
        "epochs": 8,
        "batch-size": 256,
        "lr": 0.001,
    },
)

# --- Cell 3: launch training -------------------------------------------------
# Blocks and streams training logs until the job finishes (a few minutes on ml.m5.large).
# train.py downloads Fashion-MNIST itself inside the training container -- no local `train`
# data channel needed.
model_trainer.train()

training_job = model_trainer._latest_training_job
training_job.refresh()
model_data = training_job.model_artifacts.s3_model_artifacts
print("Trained model artifacts:", model_data)

# --- Cell 4: deploy to a real-time endpoint ----------------------------------
# ModelBuilder (SDK v3's replacement for estimator.deploy()) accepts the ModelTrainer object
# directly and pulls the trained artifacts from it.
from sagemaker.serve.model_builder import ModelBuilder

inference_image = retrieve_image_uri(
    framework="pytorch",
    region=region,
    version=FRAMEWORK_VERSION,
    py_version=PY_VERSION,
    instance_type=INSTANCE_TYPE,
    image_scope="inference",
)
print("Inference image:", inference_image)

model_builder = ModelBuilder(
    model=model_trainer,
    role_arn=role,
    sagemaker_session=session,
    image_uri=inference_image,
    source_code=SourceCode(source_dir="sagemaker", entry_script="inference.py"),
    instance_type=INSTANCE_TYPE,
)

built_model = model_builder.build()
endpoint = model_builder.deploy(
    endpoint_name="fashion-mnist-cnn-endpoint",
    initial_instance_count=1,
    instance_type=INSTANCE_TYPE,
)
print("Endpoint name:", endpoint.endpoint_name)

# --- Cell 5: test the endpoint with one real test image ----------------------
import json
from torchvision import datasets, transforms

test_set = datasets.FashionMNIST(root="/tmp/data", train=False, download=True, transform=transforms.ToTensor())
image, true_label = test_set[0]

response = endpoint.invoke(
    body=json.dumps({"image": image.squeeze(0).tolist()}),
    content_type="application/json",
)
body = response.body.read() if hasattr(response.body, "read") else response.body
result = json.loads(body)

print("True label index:", true_label)
print("Prediction:", result)

# --- Cell 6: clean up (IMPORTANT -- the endpoint bills hourly while it exists) ---
endpoint.delete()
print("Endpoint deleted.")
