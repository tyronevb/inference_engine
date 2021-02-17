"""Script to split a dataset into train, validation and test datasets."""

__author__ = "tyronevb"
__date__ = "2021"


import argparse
import sys

sys.path.append("../")
from utils.hdfs_data_loader import load_HDFS  # noqa

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split dataset into train, validation and test sets. Mainly for HDFS log data that is "
        "acompanied by anomaly labels."
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
        "-a",
        "--anomaly_label_file",
        action="store",
        help="Absolute path to .csv file containing anomaly labels for the given log file. (Ground truth)",
        default=None,
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        action="store",
        help="Absolute path to directory to which to save new datasets. Should include training /",
        required=True,
    )
    parser.add_argument(
        "-t",
        "--train_ratio",
        action="store",
        help="Ratio of data to use for train set",
        type=float,
    )
    parser.add_argument(
        "-v",
        "--validation_ratio",
        action="store",
        help="Ratio of data to use for validation set",
        type=float,
    )
    parser.add_argument(
        "-w", "--window_size", action="store", help="Size of sliding window to use when creating features", type=int
    )
    parser.add_argument(
        "-s",
        "--split",
        action="store",
        help="Specify whether data is to be split sequentially or uniformly. Uniform only possible with labels and for"
        " HDFS when log messages are grouped by sessions",
        choices=["uniform", "sequential"],
        default="uniform",
    )

    args = parser.parse_args()

    log_base_name = args.parsed_log_file.split("/")[-1].split(".")[0]

    (
        (x_train, window_y_train, y_train),
        (x_validation, window_y_validation, y_validation),
        (x_test, window_y_test, y_test),
    ) = load_HDFS(
        log_file=args.parsed_log_file,
        label_file=args.anomaly_label_file,
        window="session",
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
        split_type=args.split,
        save_csv=False,
        window_size=args.window_size,
    )

    # merge x and window y
    x_train["Label"] = window_y_train
    x_validation["Label"] = window_y_validation
    x_test["Label"] = window_y_test

    x_train.to_csv("{out_dir}{base}_x_train.csv".format(out_dir=args.output_dir, base=log_base_name))
    # window_y_train.to_csv("{out_dir}{base}_window_y_train.csv".format(out_dir=args.output_dir, base=log_base_name))
    y_train.to_csv("{out_dir}{base}_y_train.csv".format(out_dir=args.output_dir, base=log_base_name))

    x_validation.to_csv("{out_dir}{base}_x_validation.csv".format(out_dir=args.output_dir, base=log_base_name))
    # window_y_validation.to_csv(
    #    "{out_dir}{base}_window_y_validation.csv".format(out_dir=args.output_dir, base=log_base_name)
    # )
    y_validation.to_csv("{out_dir}{base}_y_validation.csv".format(out_dir=args.output_dir, base=log_base_name))

    x_test.to_csv("{out_dir}{base}_x_test.csv".format(out_dir=args.output_dir, base=log_base_name))
    # window_y_test.to_csv("{out_dir}{base}_window_y_test.csv".format(out_dir=args.output_dir, base=log_base_name))
    y_test.to_csv("{out_dir}{base}_y_test.csv".format(out_dir=args.output_dir, base=log_base_name))

# end
