# CIFAR-10 Training Pipeline with ClearML

This documentation provides a detailed guide on how to configure and run the CIFAR-10 training pipeline using ClearML. The pipeline automates the process from data upload to model evaluation, including a step for hyperparameter optimization (HPO).

## Overview

The pipeline is defined in the `training_pipeline.py` script, which orchestrates several key stages: data uploading, preprocessing, model training, evaluation, and optimization. This script uses functions defined across different modules in the `first_mlops_pipeline` package.

## Running the Pipeline

To run the entire pipeline, you will use the `__main__.py` script, which allows you to set pipeline parameters through command-line arguments.

### Configuration Parameters

The script accepts the following parameters, which you can specify when running the pipeline:

- `--epochs`: Specifies the number of training epochs (default: 10).
- `--pipeline_name`: Sets the name of the pipeline (default: "CIFAR-10 Training Pipeline").
- `--project_name`: Defines the ClearML project name (default: "CIFAR-10 Project").
- `--raw_dataset_name`: Identifies the name for the raw dataset (default: "CIFAR-10 Raw").
- `--processed_dataset_name`: Identifies the name for the processed dataset (default: "CIFAR-10 Preprocessed").

### Execution Command

To execute the pipeline, use the following command:

```bash
python -m first_mlops_pipeline --epochs 20 --pipeline_name "CIFAR-10 Full Pipeline" --project_name "CIFAR-10 ML Project" --raw_dataset_name "CIFAR-10 Raw" --processed_dataset_name "CIFAR-10 Preprocessed"
```
OR
```bash
python __main__.py --epochs 20 --pipeline_name "CIFAR-10 Full Pipeline" --project_name "CIFAR-10 ML Project" --raw_dataset_name "CIFAR-10 Raw" --processed_dataset_name "CIFAR-10 Preprocessed"
```

This command initializes the CIFAR-10 pipeline with specified settings, such as the number of epochs and the project name.

## Pipeline Steps

### Step 1: Upload CIFAR-10 Raw Data

This step uploads the raw CIFAR-10 dataset to the ClearML server, making it available for preprocessing.

### Step 2: Preprocess CIFAR-10 Data

The raw data is then preprocessed to prepare it for training. This includes normalization and data augmentation tasks.

### Step 3: Train Model

A model is trained on the preprocessed data using specified settings like the number of epochs.

### Step 4: Evaluate Model

After training, the model is evaluated on a test set to assess its performance and accuracy.

### Step 5: Hyperparameter Optimization (HPO)

Optionally, an HPO step can be included to optimize model parameters, improving model performance based on a predefined search space.

## Monitoring and Results

Throughout the pipeline execution, all tasks are logged and monitored through the ClearML web interface. This provides a comprehensive view of the model’s metrics, outputs, and overall performance, facilitating easy tracking and analysis.