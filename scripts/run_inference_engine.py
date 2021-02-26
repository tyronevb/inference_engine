"""Script to run the Inference Engine to perform deep learning-assisted log file analysis."""

__author__ = "tyronevb"
__date__ = "2021"


import argparse
import pandas as pd
import sys
from ast import literal_eval
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

sys.path.append("..")

from src.inference_engine import InferenceEngine  # noqa
from src.feature_extractor import FeatureExtractor  # noqa
from utils.hdfs_data_loader import load_HDFS  # noqa

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
        "-m",
        "--mode",
        action="store",
        choices=["training", "inference"],
        default="training",
        help="Specify which mode to operate the Inference Engine in. Either training mode or inference mode. "
        "Configuration files must also be for the required mode.",
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
    parser.add_argument(
        "-e",
        "--evaluate",
        action="store",
        help="Evaluate the performance of the Inference Engine. Requires"
        "ground truth dataset with labelled anomalies corresponding to given log file to be provided. "
        "Provide as absolute path to ground truth csv. Only supported for HDFS dataset. The preprocessed, "
        "feature-extracted dataset must be provided with the -l flag",
        default=None,
    )
    parser.add_argument(
        "-t",
        "--transformation",
        action="store",
        help="Specify path to a stored feature transformation. Only required when running in evaluate mode"
        " and in inference mode. Transformation to be stored in yaml file.",
        default=None,
    )
    parser.add_argument(
        "-p",
        "--processed",
        action="store_true",
        help="Specify whether data is already processed into features.",
    )
    parser.add_argument(
        "-k",
        "--keys",
        action="store",
        help="Path to csv file containing all expected log event keys for this system",
        default=None,
    )

    args = parser.parse_args()

    print("\n====================================")
    print("Running Inference Engine . . .\n")

    # create inference engine instance
    inference_engine = InferenceEngine(
        config_file=args.config_file,
        name=args.name,
        device=args.compute_device,
        output_dir=args.output_dir,
        verbose=args.verbose,
    )

    if args.processed:
        # if the given log file is already processed into features of events
        # and input windows and next keys
        df_parsed_log = pd.read_csv(
            args.parsed_log_file, converters={"EventSequence": literal_eval}
        )  # required to read as a literal list dtype

        if args.mode == "training":
            # if in training mode . . .
            # instantiate a FeatureExtractor
            feature_extractor = FeatureExtractor(
                training_mode=True,
                data_transformation=None,
                output_dir=args.output_dir,
                name=args.name,
                event_keys=args.keys,
                verbose=args.verbose,
            )

            # feature_extractor.unique_keys = df_parsed_log["Label"].unique()  # get the unique keys
            # num_unique_keys = df_parsed_log["Label"].nunique() + 2  # get the number of unique keys, +2 for OOV and PAD
            features_dataset = feature_extractor.fit_transform(df_parsed_log)  # transform features
        else:
            # in inference mode . . .
            feature_extractor = FeatureExtractor(
                training_mode=False,
                data_transformation=args.transformation,
                output_dir=args.output_dir,
                name=args.name,
                event_keys=None,
                verbose=args.verbose,
            )
            features_dataset = feature_extractor.transform(df_parsed_log)
            # num_unique_keys = len(feature_extractor.data_transformation)
    else:
        # load parsed log file
        df_parsed_log = pd.read_csv(args.parsed_log_file)
        # extract features from given parsed log file
        features_dataset = inference_engine.get_features(df_parsed_log=df_parsed_log)
        # num_unique_keys = df_parsed_log["EventId"].nunique() + 2  # get the number of unique keys, +2 for OOV and PAD

    # output size must be specified in config file and must = number of unique events + 2
    # inference_engine.output_size = num_unique_keys
    # inference_engine.update_component_parameters()

    if args.mode == "training":
        print(". . . running Inference Engine in training mode . . .\n")
        # for training mode, split into batches and encode as tensors
        data_loader = inference_engine.batch_and_load_data_to_tensors(features_dataset=features_dataset)
        # train the anomaly detection model
        inference_engine.train_model(train_data=data_loader)

        # infer on the trained model --> for evaluating training performance
        anomaly_detection_report = inference_engine.infer_and_detect_anomalies(
            input_dataset=features_dataset, load_model=False
        )
    else:
        print(". . . running Inference Engine in inference mode . . .\n")
        # for inference mode, use the features dataset
        # use a previously trained model to perform inference and detect anomalies
        anomaly_detection_report = inference_engine.infer_and_detect_anomalies(input_dataset=features_dataset)

    # evaluate LSTM Model Performance
    infer_time, accuracy = inference_engine.evaluate_model(anomaly_detection_report)
    print("\nLSTM Model Performance:")
    print("Inference Time: {i_time}\nModel Accuracy: {acc}\n".format(i_time=infer_time, acc=accuracy))

    # evaluate anomaly detection performance --> precision, recall, f1_measure, false positives, false negatives
    # required ground truth of actual anomalies corresponding to the dataset
    if args.evaluate:
        print("Inference Engine Anomaly Detection Performance")

        if args.processed:
            # load the ground truth for anomalies
            df_anomaly_ground_truth = pd.read_csv(args.evaluate)

            # append true anomaly labels to the anomaly detection report
            anomaly_detection_report["anomaly_ground_truth"] = df_anomaly_ground_truth["SessionLabel"]

            # per session evaluation - for hdfs (anomalies are recorded per session)
            anomaly_detection_report = anomaly_detection_report.groupby("session_id", as_index=False).sum()
            anomaly_detection_report["anomaly_ground_truth"] = (
                anomaly_detection_report["anomaly_ground_truth"] > 0
            ).astype(int)
            anomaly_detection_report["anomaly"] = (anomaly_detection_report["anomaly"] > 0).astype(int)

            # per line evaluation - future support for other log files if ground truth is available
            # nothing special here - just use the columns
            # todo: add a flag to specify per session or per line anomaly evaluation

            y_true = anomaly_detection_report["anomaly_ground_truth"]
            y_pred = anomaly_detection_report["anomaly"]

        else:
            # load the ground truth for anomalies
            df_anomaly_ground_truth = pd.read_csv(args.evaluate)

            # merge ground truth labels with anomaly detection report
            anomaly_detection_report = pd.merge(anomaly_detection_report, df_anomaly_ground_truth, on=["session_id"])

            # per session evaluation - for hdfs (anomalies are recorded per session)
            anomaly_detection_report = anomaly_detection_report.groupby("session_id", as_index=False).sum()
            anomaly_detection_report["Anomaly Label (GT)"] = (
                anomaly_detection_report["Anomaly Label (GT)"] > 0
            ).astype(int)
            anomaly_detection_report["anomaly"] = (anomaly_detection_report["anomaly"] > 0).astype(int)

            # per line evaluation - future support for other log files if ground truth is available
            # nothing special here - just use the columns
            # todo: add a flag to specify per session or per line anomaly evaluation

            y_true = anomaly_detection_report["Anomaly Label (GT)"]
            y_pred = anomaly_detection_report["anomaly"]

        tn, fp, fn, tp = confusion_matrix(y_true=y_true, y_pred=y_pred).ravel()

        precision = precision_score(y_true=y_true, y_pred=y_pred)
        recall = recall_score(y_true=y_true, y_pred=y_pred)
        f1_measure = f1_score(y_true=y_true, y_pred=y_pred)
        false_alarm_rate = fp / (tn + fp)

        print("Number of anomalies in ground truth: {}".format(y_true.sum()))

        print(
            "Precision: {precision}\nRecall: {recall}\nF1: {f1}\nTrue Positives: {tp}\n"
            "True Negatives: {tn}\nFalse Positives: {fp}\nFalse Negatives: {fn}"
            "\nFalse Alarm Rate: {far}".format(
                precision=precision, recall=recall, f1=f1_measure, tp=tp, tn=tn, fp=fp, fn=fn, far=false_alarm_rate
            )
        )

    print("====================================\n")

# end
