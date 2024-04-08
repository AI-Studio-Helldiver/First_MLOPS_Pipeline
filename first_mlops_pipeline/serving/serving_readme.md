## Installation

### Prerequisites

* Docker: You need docker engine to be installed in your device. Install from: https://docs.docker.com/get-docker/
* ClearML-Server : Model repository, Service Health, Control plane
* CLI : Configuration & model deployment interface

### Initial Setup

1. Setup your [**ClearML Server**](https://github.com/allegroai/clearml-server) or use the [Free tier Hosting](https://app.clear.ml)
2. Setup local access (if you haven't already), see instructions [here](https://clear.ml/docs/latest/docs/getting_started/ds/ds_first_steps#install-clearml)
3. Install clearml-serving CLI in the venv: 
```bash
pip3 install clearml-serving
```
OR if using poetry
```bash
poetry add clearml-serving
```
4. Create the Serving Service Controller
  - `clearml-serving create --name "serving example"`
  - The new serving service UID should be printed `New Serving Service created: id=aa11bb22aa11bb22`
5. Write down the Serving Service UID
6. Clone this repository (if you haven't already)
7. Edit the environment variables file (`first_mlops_pipeline/serving/docker/example.env`) with your clearml-server credentials and Serving Service UID. For example, you should have something like
```bash
cat docker/example.env
```
```bash
  CLEARML_WEB_HOST="https://app.clear.ml"
  CLEARML_API_HOST="https://api.clear.ml"
  CLEARML_FILES_HOST="https://files.clear.ml"
  CLEARML_API_ACCESS_KEY="<access_key_here>"
  CLEARML_API_SECRET_KEY="<secret_key_here>"
  CLEARML_SERVING_TASK_ID="<serving_service_id_here>"
```
8. Spin the clearml-serving containers with docker-compose (or if running on Kubernetes use the helm chart)
```bash
cd first_mlops_pipeline/serving/docker && docker-compose --env-file example.env -f docker-compose.yml up 
```
If you need Triton support (keras/pytorch/onnx etc.), use the triton docker-compose file
```bash
cd first_mlops_pipeline/serving/docker && docker-compose --env-file example.env -f docker-compose-triton.yml up 
```
:muscle: If running on a GPU instance w/ Triton support (keras/pytorch/onnx etc.), use the triton gpu docker-compose file
```bash
cd first_mlops_pipeline/serving/docker && docker-compose --env-file example.env -f docker-compose-triton-gpu.yml up 
```

> **Notice**: Any model that registers with "Triton" engine, will run the pre/post processing code on the Inference service container, and the model inference itself will be executed on the Triton Engine container.
