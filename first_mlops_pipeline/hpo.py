from clearml.automation import HyperParameterOptimizer, UniformParameterRange, GridSearch
from clearml import Task
import argparse


def hpo(base_task_id, queue_name):
    # Initialize ClearML Task for HPO
    task = Task.init(
        project_name="CIFAR-10 Project",
        task_name="HPO CIFAR-10 Training",
        task_type=Task.TaskTypes.optimizer,
    )

    # Define Hyperparameter Space
    param_ranges = [
        UniformParameterRange("Args/epochs", min_value=5, max_value=50, step_size=5),
        ### you could make anything like batch_size, number of nodes, loss function, a command line argument in base task and use it as a parameter to be optimised. ###
    ]

    # Setup HyperParameter Optimizer
    optimizer = HyperParameterOptimizer(
        base_task_id=base_task_id,
        hyper_parameters=param_ranges,
        objective_metric_title="epoch_accuracy",
        objective_metric_series="epoch_accuracy",
        objective_metric_sign="max",
        optimizer_class=GridSearch,
        execution_queue=queue_name,
        max_number_of_concurrent_tasks=1,
    )

    # Start the Optimization
    optimizer.start()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Hyperparameter Optimization for CIFAR-10 Pipeline"
    )
    parser.add_argument(
        "--base_task_id",
        type=str,
        required=True,
        help="Base Task ID for the CIFAR-10 Pipeline",
    )
    parser.add_argument(
        "--queue_name",
        type=str,
        default="gitarth",
        help="Execution queue name in ClearML",
    )

    args = parser.parse_args()
    hpo(args.base_task_id, args.queue_name)
