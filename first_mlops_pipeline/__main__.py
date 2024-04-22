import argparse

if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description="Run CIFAR-10 Processing and Training Pipeline")

    # Add arguments
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--pipeline_name', type=str, default="CIFAR-10 Processing and Training Pipeline", help='Name of the pipeline')
    parser.add_argument('--project_name', type=str, default="CIFAR-10 Project", help='Project name for datasets')
    parser.add_argument('--raw_dataset_name', type=str, default="CIFAR-10 Raw", help='Name for the raw dataset')
    parser.add_argument('--processed_dataset_name', type=str, default="CIFAR-10 Preprocessed", help='Name for the processed dataset')
    # Parse the arguments
    args = parser.parse_args()

    # Import the create_cifar10_pipeline function inside the main block to avoid unnecessary imports when this script is imported as a module elsewhere
    from first_mlops_pipeline.pipeline import create_cifar10_pipeline

    # Call the function with the parsed arguments
    create_cifar10_pipeline(
        epochs=args.epochs,
        pipeline_name=args.pipeline_name,
        project_name=args.project_name,
        raw_dataset_name=args.raw_dataset_name,
        processed_dataset_name=args.processed_dataset_name,
    )
