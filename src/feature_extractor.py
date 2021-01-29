"""
Implements the FeatureExtractor Class.

An instance of the FeatureExtractor is initialised to tune options to extract
features for log file anomaly detection. Parsed log data is then passed to the FeatureExtractor
to extract the features and create a dataset ready to be input to the Anomaly Detection Model.

"""

__author__ = "tyronevb"
__date__ = "2021"

import pandas as pd
import datetime
import yaml
from collections import OrderedDict
import re


class FeatureExtractor(object):
    """Class definition for the FeatureExtractor class."""

    def __init__(
        self,
        sample_by_session: bool = False,
        window_size: int = 10,
        training_mode: bool = True,
        data_transformation: str = None,
        output_dir: str = None,
        name: str = None,
        verbose: bool = False,
    ):
        """
        Initialise FeatureExtractor.

        The FeatureExtractor is initialised with various flags to specify
        options for the FeatureExtractor. The FeatureExtractor extracts and
        creates features from a sequence of log event keys by applying a sliding
        window to the sequence of keys to select a window of log keys as an
        input and the next log key in the sequence as the expected output
        or label.

        :param sample_by_session: Enable sampling of log events by session.
        Only log files generated by the Hadoop Distributed File System
        currently supports this sampling scheme.
        :param window_size: specify the size of the sliding window
        :param training_mode: specify whether the features are being extracted
        for model training or prediction
        :param data_transformation: specify path to yaml file containing previously stored
        data transformation mapping. Required if features are being extracted for prediction
        :param output_dir: path to directory where generated outputs are to be saved. should include trailing /
        :param name: name for *this* feature extractor
        :param verbose: enable printing of various statistics during the feature extraction process
        """
        self.sample_by_session = sample_by_session
        self.window_size = window_size
        self.verbose = verbose
        self.training_mode = training_mode
        self.output_dir = output_dir
        self.name = name

        # check if data transformation is required and provided
        if not self.training_mode and data_transformation is None:
            raise ValueError("Data transformation is required if not in training mode.")

        # load the stored data transformation for predict/detect mode
        if data_transformation is not None and not self.training_mode:
            print("Using transformation mapping stored in {}".format(data_transformation))
            with open(data_transformation, "r") as f:
                self.data_transformation = yaml.load(f, Loader=yaml.FullLoader)

        # set the data_transformation attribute to None if in training mode
        if self.training_mode:
            self.data_transformation = None

        # create attribute for storing set of unique log keys
        self.unique_keys = None

    def extract_features(self, df_parsed_log: pd.DataFrame, save_to_file: bool = False) -> pd.DataFrame:
        """
        Extract features from parsed log file DataFrame.

        Invoke the various methods of the feature extractor to extract and create
        features from the parsed log file DataFrame. The sequence of log keys
        in the parsed log file is broken up into a number of windows consisting of
        sequenced log keys and with the next log key in the sequence being specified as the
        label.

        Features are in the format:
        Input = Window of N log keys
        Output = Next expected log key

        :param df_parsed_log: pandas DataFrame containing the parsed log file
        as ouput by the DataMiner
        :param save_to_file: save feature extracted DataFrame to a csv file
        :return: DataFrame containing a dataset of features ready for input to
        an anomaly detection model
        """
        # get the sequence of log event keys
        if self.sample_by_session:
            # sample log events by session and create a sequence of log event keys per session
            log_key_seq = self.extract_log_key_sequence_by_session(df_parsed_log)
        else:
            # treat the log events as belonging to a single sessions and create a sequence of all log event keys
            log_key_seq = self.extract_log_key_sequence(df_parsed_log)

        # apply a slicing function to create windows of input log keys and expected output log keys
        df_dataset = self.create_features_and_labels(log_key_seq)

        # apply data transformation to the log keys
        if self.training_mode:
            # generate the data transformation if in training mode and then transform the data
            df_dataset_transformed = self.fit_transform(df_dataset)
        else:
            # use a provided data transformation if in predict/detect mode to transform the data
            df_dataset_transformed = self.transform(df_dataset)

        if save_to_file:
            time_now = datetime.datetime.now().strftime("%d-%m-%Y-%Hh%Mm%Ss")
            filename = "{dir}{name}_dataset_with_features_{timestamp}.csv".format(
                dir=self.output_dir, name=self.name, timestamp=time_now
            )
            df_dataset_transformed.to_csv(filename)
            print("\nFeature extracted dataset saved to {}".format(filename))

        return df_dataset_transformed

    def extract_log_key_sequence_by_session(self, df_parsed_log: pd.DataFrame) -> list:
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
        self.unique_keys = df_parsed_log["EventId"].unique()

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

        return list(data_df["EventSequence"])  # only return the sequences of keys per session

    def extract_log_key_sequence(self, df_parsed_log: pd.DataFrame) -> list:
        """
        Extract sequence of log keys from parsed log file.

        Extracts the sequence of all log keys in the parsed log file. Assumes
        a single session for all log events.

        :param df_parsed_log: parsed log file DataFrame generated by DataMiner
        :return: list of sequenced log keys
        """
        df_copy = df_parsed_log.copy(deep=True)
        log_key_seq = df_copy["EventId"]  # log keys are denoted by the "EventId" column in the DataFrame
        self.unique_keys = log_key_seq.unique()  # find the set of all unique log keys in the data set
        return [list(log_key_seq)]  # return a nested list to indicate a single session

    def create_features_and_labels(self, log_key_sequences: list) -> pd.DataFrame:
        """
        Create input features and labels for data set.

        Given a sequence of log event keys create input features and labels
        for training an anomaly detector. Input features are created by selecting
        a window of logs and choosing the next log key in the sequence, after the window,
        as the label.

        The window size is tunable. A window is only valid if it contains the specified number of keys
        and has a valid next log key appearing after it.

        For each session of log sequences, a sliding window is applied to the log sequences to
        select the input features and the label.

        For training, the anomaly detection model will learn the sequence of log events from
        this data set by being trained to predict the next log key in the sequence correctly.

        For anomaly detection, the same scheme is used to create features from the production
        data set. In this context, the anomaly detector specifies the next likely log key given a
        sequence of logs. This is then compared to the actual log key in the data set and an anomaly
        is flagged if they are different.

        :param log_key_sequences: list of sessions of log key sequences
        :return: pandas DataFrame containing [SessionId, EventSequence, Label]
        where EventSequence is the input feature (sequence of log keys) and
        Label is the output/expected log key
        """
        dataset = []  # empty list to store input features and labels
        num_sessions = 0  # counter to track the number of sessions processed

        # process each session in the list of log key sequences individually
        for sess_idx, seq in enumerate(log_key_sequences):
            num_sessions += 1
            sequence_length = len(seq)

            window_start = 0  # initialise sliding window at the base index
            # slide the window across the sequence while there are enough items in the list
            # and extract a window of keys and the next log key
            while (window_start + self.window_size) < sequence_length:
                window = seq[window_start : window_start + self.window_size]
                next_log_key = seq[window_start + self.window_size]
                dataset.append([sess_idx, window, next_log_key])  # append sample to dataset list
                window_start += 1  # slide window by 1 position
            else:
                pass
                # removed this logic on 26/01/2021
                # when the sequence is too short for the window size, pad to window size
                # window = seq[window_start: window_start + self.window_size]
                # padding = ["PAD"] * (self.window_size - len(window))
                # window += padding
                # next_log_key = "PAD"
                # dataset.append([sess_idx, window, next_log_key]) # append sample to dataset list

        # convert dataset list to a pandas DataFrame
        df_dataset = pd.DataFrame(dataset, columns=["SessionId", "EventSequence", "Label"])

        if self.verbose:
            print(
                "\nFeatures and label extraction complete: {num_win} input "
                "samples (windows) of size {win_size} created across "
                "{num_sess} session(s).".format(
                    num_win=df_dataset.shape[0], num_sess=num_sessions, win_size=self.window_size
                )
            )

        return df_dataset

    def fit_transform(self, feature_dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Apply a transformation to the log keys on a data set to be used for training.

        The log keys extracted by the DataMiner are alphanumeric. This method
        applies a transformation to map each unique log key to a unique feature
        representation value.

        In the context of training, the transformation is first derived.

        :param feature_dataset: DataFrame of features
        :return: DataFrame of feature dataset with transformation applied
        """
        # derive the data transformation and feature representation
        # each unique log key is mapped to a unique integer (list of keys used here excludes padding)
        # start at 2 as 0 is used for out-of-vocabulary i.e. the default of dict.get
        # and 1 is used for padding words
        transformation = {log_key: value for value, log_key in enumerate(self.unique_keys, 2)}

        transformation["OOV"] = 0
        transformation["PAD"] = 1

        # set the data transformation attribute
        self.data_transformation = transformation

        # save derived transformation to a yaml file
        timestamp = datetime.datetime.now().strftime("%d-%m-%Y-%Hh%Mm%Ss")
        transformation_filename = "{dir}{name}_log_key_mapping-{timestamp}.yaml".format(
            dir=self.output_dir, name=self.name, timestamp=timestamp
        )

        with open(transformation_filename, "w") as f:
            _ = yaml.dump(transformation, f)

        print("\nTransformation mapping saved to file: {filename}".format(filename=transformation_filename))

        if self.verbose:
            print("\nTransformation mapping: {}".format(self.data_transformation))

        feature_dataset = feature_dataset.copy(deep=True)
        return self._transform(feature_dataset)  # apply the transformation

    def transform(self, feature_dataset: pd.DataFrame) -> pd.DataFrame:
        """

        Apply a transformation to the log keys on a dataset to be used for detection.

        The log keys extracted by the DataMiner are alphanumeric. This method
        applies a transformation to map each unique log key to a unique feature
        representation value.

        In the context of detection, a previously derived transformation,
        from the same log files, is used.

        :param feature_dataset: DataFrame of features
        :return: DataFrame of feature dataset with transformation applied
        """
        # raise error if no transformation was provided
        if self.data_transformation is None:
            raise ValueError("Data transformation dictionary must be provided for test/predict/detect operations")
        else:
            feature_dataset = feature_dataset.copy(deep=True)
            return self._transform(feature_dataset)  # apply the transformation

    def _transform(self, feature_dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Apply data transformation.

        Uses the data transformation stored in the data_transformation
        class attribute to transform a given data set.

        :param feature_dataset: DataFrame of features
        :return: DataFrame of feature dataset with transformation applied
        """
        # apply the transformation by replacing each log key in the data set
        # with its corresponding feature representation in the transformation
        # dictionary

        # apply transformation to input features
        feature_dataset["EventSequence"] = feature_dataset["EventSequence"].map(
            lambda x: [self.data_transformation.get(item, 0) for item in x]
        )

        # apply transformation to label
        feature_dataset["Label"] = feature_dataset["Label"].map(lambda x: self.data_transformation.get(x, 0))

        return feature_dataset


# end
