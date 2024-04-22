from clearml.automation import HyperParameterOptimizer, UniformIntegerParameterRange, GridSearch
from clearml import Task
import argparse

def job_complete_callback(
    job_id,  # type: str
    objective_value,  # type: float
    objective_iteration,  # type: int
    job_parameters,  # type: dict
    top_performance_job_id,  # type: str
):
    print(
        "Job completed!", job_id, objective_value, objective_iteration, job_parameters
    )
    if job_id == top_performance_job_id:
        print(
            "Objective reached {}".format(
                objective_value
            )
        )


def hpo(base_task_id, queue_name):
    # Initialize ClearML Task for HPO
    task = Task.init(
        project_name="CIFAR-10 Project",
        task_name="HPO CIFAR-10 Training",
        task_type=Task.TaskTypes.optimizer,
        reuse_last_task_id=False,
    )

    # Define Hyperparameter Space
    param_ranges = [
        UniformIntegerParameterRange(
            "Args/epochs", min_value=5, max_value=10, step_size=5
        ),
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
        max_number_of_concurrent_tasks=2,
        optimization_time_limit=60.0,
        # Check the experiments every 6 seconds is way too often, we should probably set it to 5 min,
        # assuming a single experiment is usually hours...
        pool_period_min=0.1,
        compute_time_limit=120,
        total_max_jobs=20,
        min_iteration_per_job=15000,
        max_iteration_per_job=150000,
    )
    # report every 12 seconds, this is way too often, but we are testing here J
    optimizer.set_report_period(0.2)
    # start the optimization process, callback function to be called every time an experiment is completed
    # this function returns immediately
    optimizer.start(job_complete_callback=job_complete_callback)
    # set the time limit for the optimization process (2 hours)
    optimizer.set_time_limit(in_minutes=90.0)
    # wait until process is done (notice we are controlling the optimization process in the background)
    optimizer.wait()
    # optimization is completed, print the top performing experiments id
    top_exp = optimizer.get_top_experiments(top_k=3)
    print([t.id for t in top_exp])
    # make sure background optimization stopped
    optimizer.stop()

    print("Done")
    return top_exp[0].id

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
