"""Script to run the Inference Engine to perform deep learning-assisted log file analysis."""

__author__ = "tyronevb"
__date__ = "2021"


import argparse
import pandas as pd
import sys

sys.path.append("..")
from src.inference_engine import InferenceEngine  # noqa

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the Inference Engine to perform deep learning log file analysis. Inference Engine can be "
                    "run in either training mode or inference mode."
    )
    parser.add_argument(
        "-l",
        "--parsed_log_file",
        action="store",
        type=str,
        help="Absolute path to parsed log file, in .csv format. (Output from Data Miner).",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        action="store",
        type=str,
        help="Absolute path to the output directory where outputs of the Inference Engine are to be saved. Includes "
             "feature mapping, anomaly detection reports and trained model parameters. Include trailing /",
        required=True,
    )
    parser.add_argument(
        "-c",
        "--config_file",
        action="store",
        type=str,
        help="Path to configuration file (yaml) that contains "
             "the various parameters for the Inference Engine, "
             "and Feature Extractor",
        required=True,
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Display inference engine operation details",
                        dest="verbose")
    parser.add_argument("-m", "--mode", action="store", choices=["training", "inference"], default="training",
                        help="Specify which mode to operate the Inference Engine in. Either training mode or inference mode. "
                             "Configuration files must also be for the required mode.")
    parser.add_argument("-n", "--name", action="store", type=str,
                        help="A unique name for this instance of the Inference Engine. Used when naming output"
                             "files from the engine. Recommend using name representative of system under"
                             "test",
                        required=True)
    parser.add_argument("-d", "--compute_device", action="store", choices=["cpu", "cuda:0", None], default=None,
                        help="Specify which compute device to use for deep learning model training and inference")

    args = parser.parse_args()

    print("\n====================================")
    print("Running Inference Engine . . .\n")

    # create inference engine instance
    inference_engine = InferenceEngine(config_file=args.config_file,
                                       name=args.name,
                                       device=args.compute_device,
                                       output_dir=args.output_dir,
                                       verbose=args.verbose)

    # load parsed log file
    df_parsed_log = pd.read_csv(args.parsed_log_file)

    # extract features from given parsed log file
    features_dataset = inference_engine.get_features(df_parsed_log=df_parsed_log)

    if args.mode == "training":
        print(". . . running Inference Engine in training mode . . .\n")
        # for training mode, split into batches and encode as tensors
        data_loader = inference_engine.batch_and_load_data_to_tensors(features_dataset=features_dataset)
        # train the anomaly detection model
        inference_engine.train_model(train_data=data_loader)
    else:
        print(". . . running Inference Engine in inference mode . . .\n")
        # for inference mode, use the features dataset
        # use a previously trained model to perform inference and detect anomalies
        anomaly_detection_report = inference_engine.infer_and_detect_anomalies(input_dataset=features_dataset)

    print("====================================\n")

# end
