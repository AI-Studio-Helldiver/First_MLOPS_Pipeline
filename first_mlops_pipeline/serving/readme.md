# Train and Deploy Keras model with Nvidia Triton Engine

## First setup clearml-serving in your local device or server:
- Follow the steps in [serving_readme.md](https://github.com/GitarthVaishnav/First_MLOPS_Pipeline/blob/development/first_mlops_pipeline/serving/serving_readme.md)  (Docker takes fair amount of time)

## training cifar-10 classifier model


Run the python training code (model saving  has been updated - to remove the optimiser so serving can work well)

```bash
python first_mlops_pipeline/train_model.py --queue_name gitarth
```

The output will be a model created on the project "CIFAR-10 Project", by the name "Model Training" or 'serving_model'.

## setting up the serving service (can be done on a machine - local or cloud - needs docker)

Prerequisites, Keras/Tensorflow models require Triton engine support, please use `docker-compose-triton.yml` / `docker-compose-triton-gpu.yml` or if running on Kubernetes, the matching helm chart.

1. Create serving Service: `clearml-serving create --name "serving example"` (write down the service ID)
2. Create model endpoint: 

 `clearml-serving --id <service_id> model add --engine triton --endpoint "test_model_keras_try" --preprocess "./first_mlops_pipeline/serving/preprocess.py" --name "Model Training" --project "CIFAR-10 Project" --input-size 32 32 --input-name "input" --input-type float32 --output-size -1 10 --output-name "output" --output-type float32   
`

3. Make sure you have the `clearml-serving` `docker-compose-triton.yml` (or `docker-compose-triton-gpu.yml`) running, it might take it a minute or two to sync with the new endpoint.

4. Test new endpoint (do notice the first call will trigger the model pulling, so it might take longer, from here on, it's all in memory): \
  `curl -X POST "http://127.0.0.1:8080/serve/test_model_keras_try" -H "accept: application/json" -H "Content-Type: application/json" -d '{"url": "https://raw.githubusercontent.com/allegroai/clearml-serving/main/examples/pytorch/5.jpg"}'`
 \
  or send a local file to be classified with \
  `curl -X POST "http://127.0.0.1:8080/serve/test_model_keras_try" -H "Content-Type: image/jpeg" --data-binary "@<filepath.jpg>"`

> **_Notice:_**  You can also change the serving service while it is already running!
This includes adding/removing endpoints, adding canary model routing etc.
by default new endpoints/models will be automatically updated after 1 minute

> **_IMPORTANT:_** The Triton works only on x86_64 architecture, meaning intel/AMD processors. This won't work on a macbook with silicon chips.