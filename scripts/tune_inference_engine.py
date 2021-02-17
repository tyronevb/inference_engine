"""Script to run the Inference Engine to perform deep learning-assisted log file analysis."""

__author__ = "tyronevb"
__date__ = "2021"


import argparse
import pandas as pd
import numpy as np
import sys
from datetime import datetime
from tqdm import tqdm
from sklearn.model_selection import ParameterGrid

sys.path.append("..")

from src.inference_engine import InferenceEngine  # noqa
from src.feature_extractor import FeatureExtractor  # noqa

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
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Display inference engine operation details", dest="verbose"
    )
    parser.add_argument(
        "-n",
        "--name",
        action="store",
        type=str,
        help="A unique name for this instance of the Inference Engine. Used when naming output"
        "files from the engine. Recommend using name representative of system under"
        "test",
        required=True,
    )
    parser.add_argument(
        "-d",
        "--compute_device",
        action="store",
        choices=["cpu", "cuda:0", None],
        default=None,
        help="Specify which compute device to use for deep learning model training and inference",
    )

    args = parser.parse_args()

    print("\n====================================")
    print("Tuning Inference Engine . . .\n")

    # create inference engine instance
    inference_engine = InferenceEngine(
        config_file=args.config_file,
        name=args.name,
        device=args.compute_device,
        output_dir=args.output_dir,
        verbose=args.verbose,
    )

    # load parsed log file
    df_parsed_log = pd.read_csv(args.parsed_log_file)

    # get number of unique keys
    num_unique_keys = df_parsed_log["EventId"].nunique()

    start_t = datetime.now()
    print("==========================")
    print("Starting tuning of Inference Engine . . .")
    print("Inference Engine Tuning - {timestamp}".format(timestamp=start_t.strftime("%d %b %Y , %H:%M:%S")))

    # parameters --> manually change this to tune different parameters and set others to optimal values
    parameter_dict = {
        "hidden_size": [2 ** x for x in range(4, 8)],
        "num_layers": range(1, 5),
        "bidirectional": [True, False],
        "window_size": range(1, 20),
        # "num_candidates": range(1, 13), }  # might not tune on this first
    }

    # create parameter grid
    parameter_grid = ParameterGrid(parameter_dict)

    results = []  # store evaluation results of the various models
    tuning_record = []  # store tuning record of the process

    # get new parameters from ParameterGrid
    for idx, params in tqdm(enumerate(parameter_grid), desc="\nTuning . . . .", mininterval=0.01):
        print("\nConsidering parameter set {}: {}".format(idx, params))
        # update InferenceEngine class attributes
        inference_engine.hidden_size = params["hidden_size"]
        inference_engine.num_lstm_layers = params["num_layers"]
        inference_engine.bidirectional_lstm = params["bidirectional"]
        inference_engine.window_size = params["window_size"]

        # set output size
        inference_engine.output_size = num_unique_keys + 2  # take into account PAD and OOV

        # update the Feature Extractor and LSTM AD Model
        inference_engine.update_component_parameters()

        # extract features - depends on window size
        # extract features from given parsed log file
        features_dataset = inference_engine.get_features(df_parsed_log=df_parsed_log)

        # re-run the process
        # load data into data_loader
        data_loader = inference_engine.batch_and_load_data_to_tensors(features_dataset=features_dataset)

        # print the model --> verify that parameters are changing
        print(inference_engine.model)

        # train the anomaly detection model
        inference_engine.train_model(train_data=data_loader)

        # generate anomaly detection report
        anomaly_detection_report = inference_engine.infer_and_detect_anomalies(
            input_dataset=features_dataset, load_model=False
        )

        # evaluate
        infer_time, accuracy = inference_engine.evaluate_model(anomaly_detection_report)
        print("Model Accuracy achieved: {model_acc}".format(model_acc=accuracy))

        results.append(accuracy)  # append accuracy to results --> criteria for tuning
        tuning_record.append([idx, params, accuracy])  # append tuning record to list

    print("Inference Engine tuning complete!\n")
    end_t = datetime.now()

    optimal_combo_idx = np.argmax(results)  # find index of combination with the highest accuracy
    optimal_parameters = parameter_grid[optimal_combo_idx]  # find combination yielding the highest accuracy

    # print out the optimal parameters
    print("Optimal combination of parameters for Inferene Engine: " "{optimal}\n".format(optimal=optimal_parameters))

    df_tuning_record = pd.DataFrame(tuning_record, columns=["Run", "Parameter Set", "Accuracy"])
    df_tuning_record.set_index("Run", inplace=True)

    tuning_file_name = "inference_engine_tuning_record_{date}.csv".format(date=start_t.strftime("%m-%d-%Y_%Hh%Mm%Ss"))

    df_tuning_record.to_csv("{output_dir}{filename}".format(output_dir=args.output_dir, filename=tuning_file_name))

    print("Number of combinations for tunable parameters: {combos}".format(combos=len(parameter_grid)))
    print(
        "Time taken to search entire parameter space: {tune_time} seconds".format(
            tune_time=(end_t - start_t).total_seconds()
        )
    )
    print("Tuning record available at {filename}".format(filename=tuning_file_name))
    print("==========================")

# end
