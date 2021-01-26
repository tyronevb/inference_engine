"""
Implements the InferenceEngine Class.

Description TBD...
"""

__author__ = "tyronevb"
__date__ = "2021"

import datetime
import torch
import pandas as pd
import os
import yaml
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from collections import OrderedDict

from src.feature_extractor import FeatureExtractor
from src.anomaly_detection_model import AnomalyDetectorLSTM
from utils.utils import LogSeqDataset


class InferenceEngine(object):
    """Class definition for the InferenceEngine class"""
    def __init__(self, config_file: str, name: str, device: str = None, verbose: bool = True):
        """
        Initialise InferenceEngine.

        The InferenceEngine class implements the Inference Engine of the Cognitive Debugging Framework.
        This component implements a deep learning model that can be trained to learn relationships between
        a sequence of log events with the aim of identifying and detecting failures within a given log file.

        This class implements methods for training the model and using the model for inference, as well as loading
        the input data into the required data structures.

        :param config_file: path to configuration file containing Inference Engine configuration
        :param name: name for *this* inference engine
        :param device: device to use for training, either "cpu" or "cuda". Auto-detects if None
        :param verbose: flag to enable verbose output and statistics
        """

        # todo: create a directory, and change to the working directory
        # this is where all outputs from *this* inference engine will be stored
        # create a path to a working directory for storing inference engine outputs
        self.path = "{base}_{timestamp}".format(base=name,
                                                timestamp=datetime.datetime.now().strftime("%d-%m-%y_%H_%M_%S"))
        os.mkdir(self.path)

        # change working directory
        # os.chdir(self.path)

        # a name prefix
        self.ie_name = "{base}_{timestamp}_inference_engine".format(base=name,
                                                                    timestamp=datetime.datetime.now().strftime("%d-%m-%y_%H_%M_%S"))
        # check which device to use i.e. CPU or GPU if available
        # or use specified device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda:0")
            else:
                self.device = torch.device("cpu")
        else:
            # use specified device
            self.device = torch.device(device)
        print("Using device: {}".format(self.device))

        # set verbose flag
        self.verbose = verbose

        # get data from config file and initialise class attributes
        # load config file for parameter tuning
        if config_file is not None:
            with open(config_file) as config:
                data = yaml.load(config, Loader=yaml.FullLoader)

        # inference engine parameters
        self.training_mode = True if data["operation_mode"] == "train" else False

        # feature extraction parameters
        self.sample_by_session = data["session_sampling"]
        self.window_size = data["window_size"]
        self.data_transformation = data["data_transformation"]

        # model parameters
        self.input_size = data["input_size"]
        self.hidden_size = data["hidden_size"]
        self.output_size = data["output_size"]
        self.num_lstm_layers = data["num_lstm_layers"]
        self.bidirectional_lstm = data["bidirectional_lstm"]
        self.dropout = data["dropout"]

        if self.training_mode:
            # training parameters
            self.batch_size = data["batch_size"]
            self.learning_rate = data["learning_rate"]
            self.num_epochs = data["num_epochs"]
            self.optimizer = data["optimizer"]["type"]
            self.optimizer_parameters = data["optimizer"]["parameters"]
        else:
            # inference parameters
            self.num_candidates = data["num_candidates"]
            self.model_parameters = data["model_parameters"]

        # instantiate a FeatureExtractor
        self.feature_extractor = FeatureExtractor(sample_by_session=self.sample_by_session,
                                                  window_size=self.window_size,
                                                  training_mode=self.training_mode,
                                                  data_transformation=self.data_transformation,
                                                  verbose=self.verbose)

        # instantiate a Model
        self.model = AnomalyDetectorLSTM(input_size=self.input_size, hidden_size=self.hidden_size,
                                         output_size=self.output_size, num_layers=self.num_lstm_layers,
                                         batch_first=True, dropout=self.dropout,
                                         bidirectional=self.bidirectional_lstm)

        # move model to device
        self.model = self.model.to(self.device)

        # todo: maybe clean this up use logger? do once the ie is complete

        self.log = []
        self.log.append("Inference Engine initialised in {mode} mode\n".format(mode=data["operation_mode"]))
        self.log.append("Feature Extraction Setup: \n")
        self.log.append("\nSession Sampling: {session}\nWindow Size: {window}\n".format(session=self.sample_by_session,
                                                                         window=self.window_size))
        self.log.append("\nSession Sampling: {session}\nWindow Size: {window}\n".format(session=self.sample_by_session,
                                                                        window=self.window_size))
        self.log.append("Model Architecture:\n")
        self.log.append("\nInput Size: {input}\nHidden Size: {hidden}\nOutput Size: {output}\nLSTM Layers: {layers}\nBidrectional: {bidir}\n".format(
            input=self.input_size, hidden=self.hidden_size, output=self.output_size, layers=self.num_lstm_layers,
            bidir=self.bidirectional_lstm))

        self.log.append("Working Directory: {dir}".format(dir=self.path))

        if self.training_mode:
            self.log.append("Training Configuration:\n")
            self.log.append("Epochs: {epochs}\nLearning Rate: {learn}\nOptimizer: {optim}".format(epochs=self.num_epochs, learn=self.learning_rate, optim=self.optimizer))

        else:
            self.log.append("Inference Configuration:\n")
            self.log.append("Number of Valid Candidates: {num_candidates}\nModel Source: {model}".format(num_candidates=self.num_candidates, model=self.model_parameters))

        if self.verbose:
            for line in self.log:
                print(line)
            # print("Inference Engine initialised in {mode} mode\n".format(mode=data["operation_mode"]))
            # print("Feature Extraction Setup: \n")
            # print("\nSession Sampling: {session}\nWindow Size: {window}\n".format(session=self.sample_by_session, window=self.window_size))
            # print("Model Architecture:\n")
            # print("\nInput Size: {input}\nHidden Size: {hidden}\nOutput Size: {output}\nLSTM Layers: {layers}\nBidrectional: {bidir}\n".format(
            #     input=self.input_size, hidden=self.hidden_size, output=self.output_size, layers=self.num_lstm_layers, bidir=self.bidirectional_lstm)
            # )
            # if self.training_mode:
            #     print("Training Configuration:\n")
            #     print("Epochs: {epochs}\nLearning Rate: {learn}\nOptimizer: {optim}".format(epochs=self.num_epochs, learn=self.learning_rate, optim=self.optimizer))
            # else:
            #     print("Inference Configuration:\n")
            #     print("Number of Valid Candidates: {num_candidates}\nModel Source: {model}".format(num_candidates=self.num_candidates, model=self.model_parameters))

    def batch_and_load_data_to_tensors(self, features_dataset: pd.DataFrame) \
            -> DataLoader:
        """
        Split input data into batch and encode as tensors.

        Only used during model training.

        :param features_dataset: DataFrame consisting of feature-extracted, parsed log file
        Must contain "EventSequence" and "Label" columns for each record
        :return: Iterable DataLoader containing input data encoded as tensors
        """

        # split into features and labels and encode as tensors
        # also ensures the dimensions of the input sequence are correct
        dataset = LogSeqDataset(features_dataset["EventSequence"],
                                features_dataset["Label"])

        # split into batches and create an iterable for the data set
        data_loader = DataLoader(dataset,
                                 batch_size=self.batch_size,
                                 shuffle=True,
                                 pin_memory=True)

        print("\n{input_record} input records split into {batches} batches of size {batch_size}".format(
            input_record=len(dataset), batches=len(data_loader), batch_size=self.batch_size
        ))

        return data_loader

    def get_features(self, df_parsed_log: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features from a given parsed log file.

        Uses and instance of FeatureExtractor to extract features from a given
        parsed log dataset.
        :param df_parsed_log: DataFrame containing parsed log file data
        :return: DataFrame containing input features and labels extracted from
        the given dataset
        """

        # extract input features and label pairs from the input data
        feature_df = self.feature_extractor.extract_features(df_parsed_log=df_parsed_log,
                                                             save_to_file=True)
        return feature_df

    def train_model(self, train_data: DataLoader) -> None:
        """
        Train the model.

        Using the provided input training data, train the deep learning model. This method
        implements the forward pass through the network, performs back propagation and
        parameter update. The entire input dataset is used to train the model a number of times.

        After training, the learned model parameters are saved to a file.

        :param train_data: batched, input training data, in an iterable DataLoader object
        :return: None
        """

        # define optimizer to be used - this updates the parameters of the model
        # once the wieghts have been computed
        # two optimizers are supported

        # todo: which parameters of the optimizers are to be tunable?
        if self.optimizer == "adam":
            optimizer = torch.optim.Adam(self.model.parameters(),
                                         lr=self.learning_rate)
        elif self.optimizer == "sgd":
            optimizer = torch.optim.SGD(self.model.parameters(),
                                        lr=self.learning_rate)
        else:
            raise NotImplementedError("Invalid optimization strategy specified")

        # set the model to training mode - all layers to be trained, parameters to be learned
        self.model.train()

        # overall training start time (tick)
        train_start_time = datetime.datetime.now()

        # use the entire training dataset to train the model for the number of epochs
        for epoch in range(self.num_epochs):
            start = datetime.datetime.now().strftime("%H:%M:%S")  # epoch start time
            print("\nTraining Epoch: {epoch} at {start_time}".format(epoch=epoch, start_time=start))

            total_loss_per_epoch = 0.0  # used to accumulate the loss for each epoch

            # zero the gradients so that they don't accumulate
            optimizer.zero_grad()

            # define loss function that will be used to compute the loss
            # CrossEntropyLoss is used as the output is modelled as categorical
            criterion = torch.nn.CrossEntropyLoss()

            # load the data iterable through a progress bar
            data_progress = tqdm(train_data, desc="\n Training Progress:")

            # loop through the batches of input training dataset
            for idx, (batch_logs, batch_labels) in enumerate(data_progress):
                output = self.model(input_data=batch_logs, device=self.device)  # use model to predict output
                loss = criterion(output, batch_labels.to(self.device))  # compute loss between predicted output and labels
                total_loss_per_epoch += float(loss)  # accumulate loss
                loss.backward()  # perform back-propagation and compute gradients
                optimizer.step()  # use gradients to update weights
                optimizer.zero_grad()  # clear existing gradients to prevent accumulation
                print("Epoch: {epoch} Train Loss: {train_loss:.5f}".format(epoch=epoch, train_loss=total_loss_per_epoch/(idx+1)))

        # overall model training time (tock)
        train_end_time = datetime.datetime.now()

        # save the trained model parameters and the optimizer state dict
        data_to_save = {"state_dict": self.model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict()}
        model_path = self.path + "/" + self.ie_name + "_model.pth"
        torch.save(data_to_save, model_path)

        if self.verbose:
            print("\nModel training complete. Total training time: {train_time}".format(train_time=train_end_time-train_start_time))
            print("\nModel parameters saved at: {model_path}".format(model_path=model_path))

    def infer_and_detect_anomalies(self, input_dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Perform inference to predict the next expected log key.

        Used to predict the next expected log key given an input sequence of log keys. The top
        candidates are selected and the actual received next log key is compared. If the actual
        key is not among the top candidates, then an anomaly is flagged.

        A previously trained deep learning model is used.

        This method generates an Anomaly Detection Report that consists of each input sequence,
        the actual key, the set of candidate keys and whether this input - label pair
        is predicted to be an anomaly.

        :param input_dataset: a feature-extracted, parsed log file that is to undergo Log File
        Analysis to detect possible failure events
        :return: DataFrame consisting of a an anomaly detection report showing the predicted candidates,
        actual key and whether a line is flagged as a anomaly
        """

        # load a previously trained model to use for prediction
        self.model.load_state_dict(torch.load(self.model_parameters)["state_dict"])

        # set the model for evaluation mode (nothing to be learned)
        self.model.eval()

        # overall prediction start time (tick)
        start_time = datetime.datetime.now()

        anomalies_detected = 0  # counter for the number of anomalies detected
        anomalies = []  # list to store anomalous events and information

        with torch.no_grad():  # disable gradient decent calculations to reduce memory consumption

            # for each entry in the provided dataset (note this is already parsed and feature extracted)
            for index, record in input_dataset.iterrows():

                tmp_anomaly = OrderedDict()  # temporary dictionary to store information about each record analysed

                input_seq = record["EventSequence"]  # get the input sequence
                actual_label = record["Label"]  # get the actual label (log key after the input sequence)

                # encode input sequence as tensor and move to device
                input_seq = torch.tensor(input_seq, dtype=torch.float).view(
                    -1, self.window_size, self.input_size).to(self.device)

                # encode actual output label as tensor and move to device
                actual_label = torch.tensor(actual_label, dtype=torch.float).view(-1).to(self.device)

                # use the anomaly detection model to predict the next log key for the given input sequence
                output = self.model(input_data=input_seq, device=self.device)

                # select the top candidate next log keys as predicted by the model
                # not that these are sorted in ascending order
                # the last item in the list has the highest probability of being the next key
                # argsort sorts the tensor items in ascending order and returns a list of the indices corresponding to the sorted items
                top_candidates = torch.argsort(output, 1)[0][-self.num_candidates:]

                # save information about record to dictionary for further inspection
                tmp_anomaly["input_seq"] = input_seq.flatten().tolist()
                tmp_anomaly["actual_label"] = actual_label.flatten().tolist()
                tmp_anomaly["candidates"] = top_candidates.flatten().tolist()

                # check if the actual key is among the top candidates
                if actual_label not in top_candidates:
                    # anomaly detected (according to model)
                    tmp_anomaly["anomaly"] = 1  # set anomaly flag for the record
                    anomalies_detected += 1  # increment detection counter
                else:
                    tmp_anomaly["anomaly"] = 0

                anomalies.append(tmp_anomaly.copy())  # append dictionary to list (use copy to prevent reference)

        # prediction end time
        end_time = datetime.datetime.now()

        # save this to to pandas dataframe and write to csv
        report_df = pd.DataFrame(anomalies)
        report_path = self.path + "/" + self.ie_name + "_anomaly_detection_report_raw.csv"
        report_df.to_csv(path_or_buf=report_path)

        if self.verbose:
            print("\nAnomaly Detection Complete. Records analysed: {num_records}\n"
                  "\nNumber of anomalies detected: {num_anomalies}\n"
                  "Total detection time: {detect_time}".format(num_records=len(anomalies), num_anomalies=anomalies_detected,
                                                               detect_time=end_time-start_time))
            print("\nAnomaly Detection Report saved at: {ad_report}".format(ad_report=report_path))

        return report_df

    # todo complete when doing the tuning, testing and evaluation framework
    def evaluate_model(self):
        # check for anomalies: compare prediction to next real log event
        pass
# end
