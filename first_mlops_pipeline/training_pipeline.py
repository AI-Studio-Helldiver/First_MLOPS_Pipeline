from clearml import PipelineController, Task

from first_mlops_pipeline.train_model import train_model
from first_mlops_pipeline.evaluate_model import evaluate_model, log_debug_images


def create_cifar10_training_pipeline(
    pipeline_name,
    dataset_project: str,
    processed_dataset_name: str,
    epochs: int,
    queue_name: str
):
    from clearml import PipelineController, Task
    from first_mlops_pipeline.train_model import train_model
    from first_mlops_pipeline.evaluate_model import evaluate_model, log_debug_images

    # Initialize a new pipeline controller task
    pipeline = PipelineController(
        name=pipeline_name,
        project=dataset_project,
        version="1.0",
        add_pipeline_tags=True,
        auto_version_bump=True,
        target_project=dataset_project,
    )

    # Add pipeline-level parameters with defaults from function arguments
    pipeline.add_parameter(name="project_name", default=dataset_project)
    pipeline.add_parameter(
        name="epochs", default=epochs
    )
    pipeline.add_parameter(
        name="processed_dataset_name", default=processed_dataset_name
    )
    pipeline.add_parameter(
        name="queue_name", default=queue_name
    )

    # Step 1: Train Model and save to reigstry
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
        helper_functions=[],
        cache_executed_step=False,
        execution_queue=queue_name,
    )

    # Step 2: Evaluate Model
    pipeline.add_function_step(
        name="evaluate_model",
        function=evaluate_model,
        function_kwargs={
            "raw_dataset_id": "${train_model.model_id}",
            "project_name": "${pipeline.project_name}",
            "processed_dataset_name": "${pipeline.processed_dataset_name}",
            "queue_name": "${pipeline.queue_name}",
        },
        task_type=Task.TaskTypes.testing,
        task_name="Evaluate Model",
        helper_functions=[log_debug_images],
        cache_executed_step=False,
        execution_queue=queue_name,
    )

    # Start the pipeline
    pipeline.start(queue=queue_name)
    print("CIFAR-10 training pipeline initiated. Check ClearML for progress.")


if __name__ == "__main__":
    import argparse

    # Create the parser
    parser = argparse.ArgumentParser(
        description="Run CIFAR-10 Training Pipeline"
    )
    parser.add_argument(
        "--pipeline_name",
        type=str,
        default="CIFAR-10 Training Pipeline",
        help="Name of the pipeline",
    )
    parser.add_argument(
        "--project_name",
        type=str,
        default="CIFAR-10 Project",
        help="Project name for datasets",
    )
    parser.add_argument(
        "--processed_dataset_name",
        type=str,
        default="CIFAR-10 Preprocessed",
        help="Name for the processed dataset",
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument(
        "--queue_name",
        type=str,
        required=True,
        help="ClearML queue name",
    )
    # Parse the arguments
    args = parser.parse_args()

    # Call the function with the parsed arguments
    create_cifar10_training_pipeline(
        pipeline_name=args.pipeline_name,
        dataset_project=args.project_name,
        processed_dataset_name=args.processed_dataset_name,
        epochs=args.epochs,
        queue_name=args.queue_name,
    )
