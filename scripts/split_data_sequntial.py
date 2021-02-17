"""Script to split a dataset into train, validation and test datasets sequentially."""

__author__ = "tyronevb"
__date__ = "2021"


import argparse
import pandas as pd

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
        "-o",
        "--output_dir",
        action="store",
        help="Absolute path to directory to which to save new datasets. Should include training /",
        required=True,
    )
    parser.add_argument(
        "-t", "--train_ratio", action="store", help="Ratio of data to use for train set", type=float, default=0.64
    )
    parser.add_argument(
        "-v",
        "--validation_ratio",
        action="store",
        help="Ratio of data to use for validation set",
        type=float,
        default=0.16,
    )

    args = parser.parse_args()

    log_base_name = args.parsed_log_file.split("/")[-1].split(".")[0]

    df_parsed_log = pd.read_csv(args.parsed_log_file)

    num_train = int(args.train_ratio * df_parsed_log.shape[0])
    x_train = df_parsed_log[0:num_train]

    num_validation = int(args.validation_ratio * df_parsed_log.shape[0])
    x_validation = df_parsed_log[num_train : num_train + num_validation]

    x_test = df_parsed_log[num_train + num_validation :]

    x_train.to_csv("{out_dir}{base}_x_train.csv".format(out_dir=args.output_dir, base=log_base_name))

    x_validation.to_csv("{out_dir}{base}_x_validation.csv".format(out_dir=args.output_dir, base=log_base_name))

    x_test.to_csv("{out_dir}{base}_x_test.csv".format(out_dir=args.output_dir, base=log_base_name))

# end
