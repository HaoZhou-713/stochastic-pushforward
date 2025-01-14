import torch
from torch.utils.data import Dataset, DataLoader

class SingleSourceDataset(Dataset):
    """
    A dataset class for single-source time series data.

    Args:
        data (torch.Tensor): The input time series data of shape (num_samples, sequence_length, num_features).
        lookback (int): The number of time steps to look back for input sequences.
        lookahead (int): The number of time steps to predict for output sequences.
    """
    def __init__(self, data, lookback, lookahead):
        self.data = data
        self.lookback = lookback
        self.lookahead = lookahead

    def __len__(self):
        """
        Returns the total number of sequences that can be extracted from the dataset.
        """
        total_length = self.data.shape[1]
        return total_length - self.lookback - self.lookahead

    def __getitem__(self, index):
        """
        Returns a single input-output pair of sequences from the dataset.

        Args:
            index (int): The starting index for the input sequence.

        Returns:
            tuple: A pair of tensors (x0, x1), where
                   x0 is the input sequence of shape (num_samples, lookback, num_features), and
                   x1 is the output sequence of shape (num_samples, lookahead, num_features).
        """
        x0 = self.data[:, index:index + self.lookback, :]
        x1 = self.data[:, index + self.lookback:index + self.lookback + self.lookahead, :]
        return x0, x1

class MultiSourceDataset(Dataset):
    """
    A dataset class for multi-source time series data, allowing weighted sampling from two datasets.

    Args:
        data1 (torch.Tensor): The first input time series dataset of shape (num_samples, sequence_length, num_features).
        data2 (torch.Tensor): The second input time series dataset of shape (num_samples, sequence_length, num_features).
        lookback (int): The number of time steps to look back for input sequences.
        lookahead (int): The number of time steps to predict for output sequences.
        probability (float): The probability of sampling from the first dataset (default: 0.7).
        coef (float): A weight applied to the output sequences when sampled from the second dataset (default: 0.3).
    """
    def __init__(self, data1, data2, lookback, lookahead, probability=0.7, coef=0.3):
        self.data1 = data1
        self.data2 = data2
        self.lookback = lookback
        self.lookahead = lookahead
        self.probability = probability  # Probability of sampling from data1
        self.coef = coef  # Weight applied to data from data2
        # Calculate the minimum valid data length between the two datasets
        self.min_len = min(data1.shape[1] - lookback - lookahead, data2.shape[1] - lookback - lookahead)

    def __len__(self):
        """
        Returns the total number of sequences that can be extracted, considering both datasets.
        """
        return self.min_len*2  # Account for both datasets by alternating sampling

    def __getitem__(self, index):
        """
        Returns a single input-output pair of sequences from one of the datasets, with weights applied as necessary.

        Args:
            index (int): The starting index for the input sequence.

        Returns:
            tuple: A triplet (x0, x1, coef), where
                   x0 is the input sequence of shape (num_samples, lookback, num_features),
                   x1 is the output sequence of shape (num_samples, lookahead, num_features), and
                   coef is the weight applied to the output.
        """
        # Decide whether to sample from data1 or data2 based on the probability
        if torch.rand(1).item() < self.probability:
            data_src_x0 = self.data1
            data_src_x1 = self.data1
            coef = 1  # No weight applied for data1
        else:
            data_src_x0 = self.data2
            data_src_x1 = self.data1
            coef = self.coef  # Weight applied for data2

        # # Alternate index for balancing sampling between the two datasets
        index = index//2

        # Extract input and output sequences
        x0 = data_src_x0[:, index:index + self.lookback, :]
        x1 = data_src_x1[:, index + self.lookback:index + self.lookback + self.lookahead, :]

        return x0, x1, coef
    