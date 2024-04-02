import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from clearml import Dataset, OutputModel, Task
import tensorflow
from tensorflow.keras.callbacks import Callback, LambdaCallback
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical


def train_model(processed_dataset_name, epochs, project_name, queue_name):
    import argparse
    import tensorflow
    import matplotlib.pyplot as plt
    import numpy as np
    from clearml import Dataset, OutputModel, Task
    from tensorflow.keras.callbacks import Callback, LambdaCallback
    from tensorflow.keras.datasets import cifar10
    from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.utils import to_categorical

    task: Task = Task.init(
        project_name=project_name,
        task_name="Model Training",
        task_type=Task.TaskTypes.training,
        auto_connect_frameworks="keras",
    )
    # task.execute_remotely(queue_name=queue_name, exit_process=True)  ## Uncomment when running as a single task, not as a pipeline

    # Access dataset
    dataset = Dataset.get(dataset_name=processed_dataset_name, dataset_project=project_name)
    dataset_path = dataset.get_local_copy()

    # Load the numpy arrays from the dataset
    train_images = np.load(f"{dataset_path}/train_images_preprocessed.npy")
    train_labels = np.load(f"{dataset_path}/train_labels_preprocessed.npy")
    test_images = np.load(f"{dataset_path}/test_images_preprocessed.npy")
    test_labels = np.load(f"{dataset_path}/test_labels_preprocessed.npy")

    train_labels, test_labels = to_categorical(train_labels), to_categorical(
        test_labels
    )

    model = Sequential(
        [
            Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3)),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation="relu"),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation="relu"),
            Flatten(),
            Dense(64, activation="relu"),
            Dense(10, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    # Inside your training function, after initializing your task:
    logger = task.get_logger()

    # Manual logging within model.fit() callback
    callbacks = [
        LambdaCallback(
            on_epoch_end=lambda epoch, logs: [
                logger.report_scalar(
                    "loss", "train", iteration=epoch, value=logs["loss"]
                ),
                logger.report_scalar(
                    "accuracy", "train", iteration=epoch, value=logs["accuracy"]
                ),
                logger.report_scalar(
                    "val_loss", "validation", iteration=epoch, value=logs["val_loss"]
                ),
                logger.report_scalar(
                    "val_accuracy",
                    "validation",
                    iteration=epoch,
                    value=logs["val_accuracy"],
                ),
            ]
        )
    ]

    H = model.fit(
        train_images,
        train_labels,
        epochs=int(epochs),
        validation_data=(test_images, test_labels),
        callbacks=callbacks,
    )

    # Save and upload the model to ClearML
    model_file_name = "model.h5"
    model.save(model_file_name)
    output_model = OutputModel(task=task)
    output_model.update_weights(
        model_file_name, upload_uri="https://files.clear.ml"
    )  # Upload the model weights to ClearML
    output_model.publish()  # Make sure the model is accessible
    task.upload_artifact("trained_model", artifact_object=model_file_name)
    if os.path.exists("model.h5"):
        os.remove("model.h5")
    return output_model.id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a CNN on CIFAR-10 and log with ClearML."
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
        "--project_name",
        # required=True,
        help="ClearML Project name",
        default="CIFAR-10 Project",
    )
    parser.add_argument(
        "--queue_name",
        # required=True,
        help="ClearML Queue name",
        default="gitarth"
    )
    args = parser.parse_args()

    train_model(
        args.processed_dataset_name,
        args.epochs,
        args.project_name,
        args.queue_name
    )
