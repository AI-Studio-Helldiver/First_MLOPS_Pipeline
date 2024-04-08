# CIFAR-10 Hyperparameter Optimization with ClearML

This script automates the hyperparameter optimization (HPO) process for a CIFAR-10 machine learning model using ClearML and Optuna. The objective is to maximize the validation accuracy by tuning the number of epochs.

## Dependencies
- **clearml** for hyperparameter optimization and task management.
- This script requires that you have these packages installed in your environment. If you are using `poetry`, ensure you add them via the `poetry add` command. For `pip`, use `pip install`.

## Setup
Before running the script, you must have a ClearML account and the ClearML SDK configured on your machine. Follow the [official ClearML documentation](https://clear.ml/docs/latest/docs/) to set up your account and SDK.

## How to Run
1. Ensure you have Python 3.10 or newer installed on your system.
2. Install the required dependencies: 
   - If using **poetry**, run: `poetry add clearml`
   - If using **pip**, run: `pip install clearml`
3. Run the training first (code is updated by a bit to work with serving and hpo):
```shell
python first_mlops_pipeline/train_model.py --queue_name "your_queue_name_here" --epochs 5
```
4. Run the script with the necessary arguments. Here's an example command:

```shell
python first_mlops_pipeline/hpo.py --base_task_id "your_base_task_id_here" --queue_name "your_queue_name_here"
```

### Arguments
- `--base_task_id`: The base task ID for the CIFAR-10 Training Task. This is a required argument. This needs a training task to be run prior.
- `--queue_name`: The execution queue name in ClearML. Defaults to "default" if not specified.

## How It Works
The script initializes a ClearML task for hyperparameter optimization, defines the hyperparameter space (in this case, the range of epochs to test), and then starts the optimization process using GridSearch as the optimizer. The optimization aims to find the number of epochs that yields the highest validation accuracy for the CIFAR-10 model.

Upon completion, the results will be available in your ClearML dashboard, where you can analyze the performance of different configurations.

For more detailed documentation on using ClearML and Optuna for HPO, refer to the [ClearML Documentation](https://clear.ml/docs/latest/docs/).
