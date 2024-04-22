from clearml import PipelineController, Task

from first_mlops_pipeline.evaluate_model import evaluate_model, log_debug_images
from first_mlops_pipeline.preprocess_upload_cifar10 import (
    preprocess_and_upload_cifar10,
    save_preprocessed_data,
)
from first_mlops_pipeline.train_model import train_model
from first_mlops_pipeline.upload_cifar_raw import (
    save_numpy_arrays,
    upload_cifar10_as_numpy,
)


def create_cifar10_pipeline(
    epochs: int = 10,
    pipeline_name: str = "CIFAR-10 Training Pipeline",
    project_name: str = "CIFAR-10 Project",
    raw_dataset_name: str = "CIFAR-10 Raw",
    processed_dataset_name: str = "CIFAR-10 Preprocessed",
):
    from clearml import PipelineController, Task

    from first_mlops_pipeline.evaluate_model import evaluate_model, log_debug_images
    from first_mlops_pipeline.preprocess_upload_cifar10 import (
        preprocess_and_upload_cifar10,
        save_preprocessed_data,
    )
    from first_mlops_pipeline.train_model import train_model
    from first_mlops_pipeline.upload_cifar_raw import (
        save_numpy_arrays,
        upload_cifar10_as_numpy,
    )

    # Initialize a new pipeline controller task
    pipeline = PipelineController(
        name=pipeline_name,
        project=project_name,
        version="1.0",
        add_pipeline_tags=True,
        auto_version_bump=True,
        target_project=project_name,
    )

    # Add pipeline-level parameters with defaults from function arguments
    pipeline.add_parameter(name="project_name", default=project_name)
    pipeline.add_parameter(name="raw_dataset_name", default=raw_dataset_name)
    pipeline.add_parameter(
        name="processed_dataset_name", default=processed_dataset_name
    )
    pipeline.add_parameter(name="epochs", default=epochs)

    # Step 1: Upload CIFAR-10 Raw Data
    pipeline.add_function_step(
        name="upload_cifar10_raw_data",
        function=upload_cifar10_as_numpy,
        function_kwargs={
            "project_name": "${pipeline.project_name}",
            "dataset_name": "${pipeline.raw_dataset_name}",
            "queue_name": "${pipeline.queue_name}",
        },
        task_type=Task.TaskTypes.data_processing,
        task_name="Upload CIFAR-10 Raw Data",
        function_return=["raw_dataset_id"],
        helper_functions=[save_numpy_arrays],
        cache_executed_step=False,
    )

    # Step 2: Preprocess CIFAR-10 Data
    pipeline.add_function_step(
        name="preprocess_cifar10_data",
        function=preprocess_and_upload_cifar10,
        function_kwargs={
            "raw_dataset_id": "${upload_cifar10_raw_data.raw_dataset_id}",
            "processed_project_name": "${pipeline.project_name}",
            "processed_dataset_name": "${pipeline.processed_dataset_name}",
            "queue_name": "${pipeline.queue_name}",
        },
        task_type=Task.TaskTypes.data_processing,
        task_name="Preprocess and Upload CIFAR-10",
        function_return=["processed_dataset_id"],
        helper_functions=[save_preprocessed_data],
        cache_executed_step=False,
    )

    # Step 3: Train Model and save to reigstry
    pipeline.add_function_step(
        name="train_model",
        function=train_model,
        function_kwargs={
            "processed_dataset_name": "${pipeline.processed_dataset_name}",
            "epochs": "${pipeline.epochs}",
            "project_name": "${pipeline.project_name}",
            "queue_name": "${pipeline.queue_name}",
        },
        task_type=Task.TaskTypes.training,
        task_name="Train Model",
        function_return=["model_id"],
        cache_executed_step=False,
    )

    # Step 4: Evaluate Model
    pipeline.add_function_step(
        name="evaluate_model",
        function=evaluate_model,
        function_kwargs={
            "model_id": "${train_model.model_id}",
            "project_name": "${pipeline.project_name}",
            "processed_dataset_name": "${pipeline.processed_dataset_name}",
            "queue_name": "${pipeline.queue_name}",
        },
        task_type=Task.TaskTypes.testing,
        task_name="Evaluate Model",
        helper_functions=[log_debug_images],
        cache_executed_step=False,
    )

    # Start the pipeline
    pipeline.start_locally()
    print("CIFAR-10 pipeline initiated. Check ClearML for progress.")
