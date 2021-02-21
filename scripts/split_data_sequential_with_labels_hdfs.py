"""Script to split an HDFS dataset with labels into train, validation and test datasets sequentially."""

__author__ = "tyronevb"
__date__ = "2021"


import argparse
import pandas as pd
from collections import OrderedDict
import re


def extract_log_key_sequence_by_session(df_parsed_log: pd.DataFrame) -> pd.DataFrame:
    """
    Extract log keys from a parsed log file by session.

    Takes a given parsed log file and sorts log events by session.
    Concept of a session is unique to a system. Not all systems may support
    this concept. This function only supports log generated by the
    Hadoop Distributed File System (HDFS).

    Adapted from: https://github.com/logpai/loglizer

    :param df_parsed_log: parsed log file DataFrame generated by DataMiner
    :return: list of sequenced log keys per session
    """
    # get set of unique log keys in the data set before sampling
    # self.unique_keys = df_parsed_log["EventId"].unique()

    data_dict = OrderedDict()
    # for HDFS, sessions are denoted by the prefix blk_
    # all log keys corresponding to a given blk_ are grouped as a session
    for idx, row in df_parsed_log.iterrows():
        blkId_list = re.findall(r"(blk_-?\d+)", row["Content"])
        blkId_set = set(blkId_list)
        for blk_Id in blkId_set:
            if blk_Id not in data_dict:
                data_dict[blk_Id] = []
            data_dict[blk_Id].append(row["EventId"])
    data_df = pd.DataFrame(list(data_dict.items()), columns=["BlockId", "EventSequence"])
    return data_df


def create_label_file(df_extracted_seq: pd.DataFrame, df_anomaly_labels) -> pd.DataFrame:
    """
    Prepare the anomaly label file for evaluation.

    :param df_extracted_seq: DataFrame containing extracted sequences of events
    :param df_anomaly_labels: DataFrame containing anomaly labels
    :return: DataFrame containing anomaly label and session number mapping
    """
    df_out = df_extracted_seq.reset_index()
    df_out.rename(columns={"index": "session_id"}, inplace=True)
    df_out.drop("EventSequence", 1, inplace=True)
    df_out = pd.merge(df_out, df_anomaly_labels, on=["BlockId"])
    df_out["Label"] = df_out["Label"].apply(lambda x: 1 if x == "Anomaly" else 0)
    df_out.rename(columns={"Label": "Anomaly Label (GT)"}, inplace=True)
    df_out.rename(columns={"index": "session_id"}, inplace=True)

    return df_out


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
        type=str,
        help="Absolute path to anomaly labels, in .csv format. Expected to be session-based for HDFS logs.",
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
    df_anomaly_labels = pd.read_csv(args.anomaly_label_file)

    num_train = int(args.train_ratio * df_parsed_log.shape[0])
    x_train = df_parsed_log[0:num_train]
    anomaly_labels_train = create_label_file(extract_log_key_sequence_by_session(x_train), df_anomaly_labels)

    num_validation = int(args.validation_ratio * df_parsed_log.shape[0])
    x_validation = df_parsed_log[num_train : num_train + num_validation]
    anomaly_labels_validation = create_label_file(extract_log_key_sequence_by_session(x_validation), df_anomaly_labels)

    x_test = df_parsed_log[num_train + num_validation :]
    anomaly_labels_test = create_label_file(extract_log_key_sequence_by_session(x_test), df_anomaly_labels)

    x_train.to_csv("{out_dir}{base}_x_train.csv".format(out_dir=args.output_dir, base=log_base_name))
    anomaly_labels_train.to_csv("{out_dir}{base}_y_train.csv".format(out_dir=args.output_dir, base=log_base_name))

    x_validation.to_csv("{out_dir}{base}_x_validation.csv".format(out_dir=args.output_dir, base=log_base_name))
    anomaly_labels_validation.to_csv(
        "{out_dir}{base}_y_validation.csv".format(out_dir=args.output_dir, base=log_base_name)
    )

    x_test.to_csv("{out_dir}{base}_x_test.csv".format(out_dir=args.output_dir, base=log_base_name))
    anomaly_labels_test.to_csv("{out_dir}{base}_y_test.csv".format(out_dir=args.output_dir, base=log_base_name))

# end
