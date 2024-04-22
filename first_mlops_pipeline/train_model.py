import argparse
import os
import tempfile

import matplotlib.pyplot as plt
import numpy as np
from clearml import Dataset, OutputModel, Task
import tensorflow
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Input, Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

def train_model(processed_dataset_name, epochs, project_name, queue_name, args):
    import argparse
    import tensorflow
    import matplotlib.pyplot as plt
    import numpy as np
    from clearml import Dataset, OutputModel, Task
    from tensorflow.keras.callbacks import Callback
    from tensorflow.keras.datasets import cifar10
    from tensorflow.keras.layers import (
        Conv2D,
        Dense,
        Flatten,
        MaxPooling2D,
        Input,
        Activation,
    )
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

    task: Task = Task.init(
        project_name=project_name,
        task_name="Model Training",
        task_type=Task.TaskTypes.training,
        auto_connect_frameworks="keras",
    )
    task.connect(args)
    task.execute_remotely(queue_name=queue_name, exit_process=True) 

    # Access dataset
    dataset = Dataset.get(dataset_name=processed_dataset_name, dataset_project=project_name)
    dataset_path = dataset.get_local_copy()
    print("loading images")
    # Load the numpy arrays from the dataset
    train_images = np.load(f"{dataset_path}/train_images_preprocessed.npy")
    train_labels = np.load(f"{dataset_path}/train_labels_preprocessed.npy")
    test_images = np.load(f"{dataset_path}/test_images_preprocessed.npy")
    test_labels = np.load(f"{dataset_path}/test_labels_preprocessed.npy")

    train_labels, test_labels = to_categorical(train_labels), to_categorical(
        test_labels
    )
    print("model")
    model = Sequential(
        [
            Input(shape=(32, 32, 3), name="input"),
            Conv2D(32, (3, 3), activation="relu"),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation="relu"),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation="relu"),
            Flatten(),
            Dense(64, activation="relu"),
            Dense(10),  # Remove the activation="softmax" from here
            Activation("softmax", name="output"),  # Add the softmax activation as a separate layer
        ]
    )
    print("compile model")
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    # Inside your training function, after initializing your task:
    logger = task.get_logger()
    output_folder = os.path.join(tempfile.gettempdir(), 'cifar10_example')
    # Manual logging within model.fit() callback
    callbacks = []
    board = TensorBoard(log_dir=output_folder, write_images=False)
    callbacks.append(board)
    model_store = ModelCheckpoint(filepath=os.path.join(output_folder, "weight.keras"))
    logger.report_text(model.summary())
    callbacks.append(model_store)
    print(model.summary())
    print("train model")
    H = model.fit(
        train_images,
        train_labels,
        epochs=int(epochs),
        validation_data=(test_images, test_labels),
        callbacks=callbacks,
    )
    score = model.evaluate(test_images, test_labels, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
    logger.report_scalar(title='evaluate', series='score', value=score[0], iteration=epochs)
    logger.report_scalar(
        title="evaluate", series="accuracy", value=score[1], iteration=epochs
    )
    # Save and upload the model to ClearML
    model_file_name = "serving_model.h5"
    model_file_path = os.path.join(output_folder, model_file_name)
    model.save(model_file_path, include_optimizer=False)
    output_model = OutputModel(task=task)
    output_model.update_weights(
        model_file_path, upload_uri="https://files.clear.ml"
    )  # Upload the model weights to ClearML
    output_model.publish()  # Make sure the model is accessible
    if os.path.exists(f"{model_file_path}"):
        os.remove(f"{model_file_path}")
    return output_model.id, task.id


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
        "--epochs", type=int, default=5, help="Number of training epochs"
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
        args.queue_name,
        args=args
    )
