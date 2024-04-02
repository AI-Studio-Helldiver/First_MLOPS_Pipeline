# CIFAR-10 Training and Evaluation with ClearML

This document outlines a machine learning workflow for training and evaluating a Convolutional Neural Network (CNN) on the CIFAR-10 dataset using TensorFlow and ClearML. The workflow is divided into three main scripts: `train_model.py`, `evaluate_model.py`, and `training_pipeline.py`. Each script serves a distinct purpose in the machine learning pipeline, from data preprocessing and model training to evaluation and pipeline orchestration.

## Individual Tasks

### Training the Model (`train_model.py`)

The `train_model.py` script is responsible for training a CNN model on the CIFAR-10 dataset. It performs the following tasks:

- Initializes a ClearML `Task` for tracking and logging the training process.
- Accesses a preprocessed version of the CIFAR-10 dataset stored in ClearML's dataset management system.
- Defines a CNN model architecture using TensorFlow's Keras API.
- Compiles and trains the model on the preprocessed dataset, using categorical cross-entropy as the loss function and accuracy as the performance metric.
- Logs training metrics (loss and accuracy) for both training and validation sets to ClearML for monitoring.
- Saves and uploads the trained model to ClearML for versioning and reproducibility.

To train the model, execute the following command, replacing the placeholders with appropriate values:

```bash
python train_model.py --processed_dataset_name <Dataset Name> --epochs <Number of Epochs> --project_name <ClearML Project Name> --queue_name <ClearML Queue Name>
```

### Evaluating the Model (`evaluate_model.py`)

The `evaluate_model.py` script is for evaluating the performance of the trained model on a test set. It includes:

- Initializing a ClearML `Task` for logging the evaluation process.
- Fetching the trained model and the preprocessed test dataset from ClearML.
- Evaluating the model's performance on the test dataset and printing the loss and accuracy.
- Generating a confusion matrix to understand the model's prediction capabilities across different classes.
- Logging a set of test images with their true and predicted labels for visual inspection.

To evaluate a trained model, run:

```bash
python evaluate_model.py --model_id <Trained Model ID> --processed_dataset_name <Dataset Name> --project_name <ClearML Project Name> --queue_name <ClearML Queue Name>
```

## Training Pipeline (`training_pipeline.py`)

The `training_pipeline.py` script automates the training and evaluation process using ClearML's PipelineController. This allows for easy orchestration of the workflow, from training the model to evaluating its performance, in a reproducible manner. The pipeline:

- Initializes a new pipeline task in ClearML.
- Adds parameters for the project name, dataset name, number of epochs, and queue name.
- Configures and adds steps for training and evaluating the model.
- Automatically manages dependencies between steps, ensuring the evaluation only starts after training is completed.
- Starts the pipeline execution locally or remotely, based on the configuration.

To run the entire pipeline, execute:

```bash
python training_pipeline.py --pipeline_name <Pipeline Name> --project_name <ClearML Project Name> --processed_dataset_name <Dataset Name> --epochs <Number of Epochs> --queue_name <ClearML Queue Name>
```