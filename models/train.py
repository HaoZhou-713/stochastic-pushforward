import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from ssim import SSIM  # Assuming you have an SSIM module
from . import dataset
from . import models


class StochasticPushForward:
    """
    Encapsulates the training and testing pipeline for a stochastic push-forward model.
    """

    def __init__(self, cae, model, lookback=1, lookahead=1):
        """
        Initialize the class with model components and hyperparameters.

        Args:
            cae: Autoencoder model containing an encoder and decoder. Default trained.
            model: Predictive model for the latent space.
            lookback (int): Number of timesteps for input sequences.
            lookahead (int): Number of timesteps to predict.
        """
        self.cae = cae
        self.encoder = cae.encoder
        self.decoder = cae.decoder
        self.lookback = lookback
        self.lookahead = lookahead
        self.model = model
        self.data1 = None
        self.test_data = None

    def load_data(self, path: str, batch_size: int):
        """
        Load training data and initialize DataLoader. Default: Compressed data.

        Args:
            path (str): Path to the dataset file.
            batch_size (int): Batch size for the DataLoader.
        """
        loaded_data = torch.load(path)
        self.data1 = loaded_data
        self.totalbatch = self.data1.shape[0]
        self.seqlen = self.data1.shape[1]
        self.latent_dim = self.data1.shape[2]
        dataset = dataset.SingleSourceDataset(data=self.data1, lookback=self.lookback, lookahead=self.lookahead)
        self.initial_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        print("The initial dataloader for training data has been generated.")

    def load_test_data(self, path: str):
        """
        Load test dataset. Default: Compressed data.

        Args:
            path (str): Path to the test dataset file.
        """
        self.test_data = torch.load(path)
        print("The test data has been loaded.")

    def get_separate_loss(self, decoded_outputs: torch.Tensor, decoded_targets: torch.Tensor, criterion: nn.Module):
        """
        Compute losses for each time step in the sequence.

        Args:
            decoded_outputs (torch.Tensor): Predicted outputs.
            decoded_targets (torch.Tensor): Ground truth targets.
            criterion (nn.Module): Loss function to compute the error.

        Returns:
            list: Loss values for each time step.
        """
        losses = [
            criterion(decoded_outputs[:, i], decoded_targets[:, i]).item()
            for i in range(decoded_outputs.shape[1])
        ]
        return losses

    def loop_prediction(self, criterion, device, start_step=50, num_steps=49, output_shape=(30, 3, 3, 64, 64)):
        """
        Perform iterative predictions on the test dataset.

        Args:
            criterion: Loss function.
            device: Computation device (CPU or GPU).
            start_step (int): Starting step for prediction.
            num_steps (int): Number of prediction steps.
            output_shape (tuple): Shape to reshape decoded outputs and targets.

        Returns:
            tuple: Accumulated errors, SSIM values, and test results.
        """
        ssim = SSIM(data_range=1.0)
        accumulated_error = []
        ssims = []
        test_results = None

        for i in range(start_step, start_step + num_steps):
            if i == start_step:
                inputs = self.test_data[:, start_step * self.lookback:start_step * self.lookback + self.lookback]
                targets = self.test_data[:, start_step * self.lookback + self.lookback:start_step * self.lookback + self.lookback + self.lookahead]
            else:
                inputs = outputs[:, -self.lookback:].detach()
                targets = self.test_data[:, i * self.lookback + (i - start_step) * self.lookahead:i * self.lookback + (i - start_step + 1) * self.lookahead]

            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = self.model(inputs)
            self.decoder = self.decoder.to(device)

            # Decode outputs and targets
            decoded_outputs = self.decoder(outputs.reshape(-1, self.latent_dim)).reshape(*output_shape)
            decoded_targets = self.decoder(targets.reshape(-1, self.latent_dim)).reshape(*output_shape)

            # Compute separate losses
            losses = self.get_separate_loss(decoded_outputs, decoded_targets, criterion)

            decoded_outputs_np = decoded_outputs.cpu().detach().numpy()
            test_results = (
                decoded_outputs_np if test_results is None else np.concatenate((test_results, decoded_outputs_np), axis=1)
            )

            for j in range(decoded_outputs.shape[1]):
                ssim.update((decoded_outputs[:, j], decoded_targets[:, j]))
                ssims.append(ssim.compute())

            accumulated_error.extend(losses)

        return accumulated_error, ssims, test_results

    def train_lstm_with_stop_condition(self, criterion_training, criterion_testing, probability, coef, max_epochs, target_loss, device, learning_rate=5e-5, depth=2, buffer_update_interval=None):
        """
        Train the LSTM model with a stopping condition based on the target loss.

        Args:
            criterion_training: Training loss function.
            criterion_testing: Testing loss function.
            probability (float): Probability for buffer dataset sampling.
            coef (float): Coefficient for weighted loss.
            max_epochs (int): Maximum training epochs.
            target_loss (float): Loss threshold for early stopping.
            device: Computation device (CPU or GPU).
            learning_rate (float): Learning rate for the optimizer.
            depth (int): Recursion depth for generating buffer data.
            buffer_update_interval (int, optional): Interval for updating buffer data.

        Returns:
            tuple: Trained model, accumulated errors, SSIM values, and test results.
        """
        self.model.to(device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        train_loader = self.initial_dataloader

        print("Begin training")

        for epoch in range(max_epochs):
            if buffer_update_interval and epoch % buffer_update_interval == 0 and depth > 1:
                buffer_data = self.process_and_save_transformed_data(depth - 1)
                multi_dataset = dataset.MultiSourceDataset(self.data1, buffer_data, self.lookback, self.lookahead, probability=probability, coef=coef)
                train_loader = DataLoader(multi_dataset, batch_size=10, shuffle=True)
                print(f"Buffer data updated at epoch {epoch}")
                self.model.to(device)

            self.model.train()
            for x0, x1, *coef in train_loader:
                x0, x1 = x0.to(device), x1.to(device)
                optimizer.zero_grad()
                outputs = self.model(x0)
                loss = criterion_training(outputs, x1).mean()
                loss.backward()
                optimizer.step()

            self.model.eval()
            with torch.no_grad():
                accumulated_error, ssims, test_results = self.loop_prediction(criterion_testing, device)

            print(f"Epoch {epoch + 1}/{max_epochs}, Loss: {loss:.4f}, Accumulated Error: {accumulated_error[-1]:.4f}, SSIM: {ssims[-1]:.4f}")

            if accumulated_error[-1] <= target_loss:
                print("Target loss reached. Stopping training.")
                break

        return self.model, accumulated_error, ssims, test_results

    def process_and_save_transformed_data(self, depth: int) -> torch.Tensor:
        """
        Transform input data recursively using the model for the specified depth on the CPU.

        Args:
            depth (int): Recursion depth for transformation.

        Returns:
            torch.Tensor: Transformed data.
        """
        self.model.to('cpu')
        self.model.eval()

        if self.data1.shape[1] < self.lookahead * (depth + 1) + self.lookback:
            raise ValueError(f"Not enough timesteps in data for depth={depth + 1} prediction.")

        transformed_data = [
            self.data1[:, :self.lookback + (depth - 1) * self.lookahead]
        ]

        for i in range(0, self.data1.shape[1] - self.lookahead * (depth + 1), self.lookahead):
            current_data = self.data1[:, i:i + self.lookback]
            for _ in range(depth):
                current_data = self.model(current_data[:, -self.lookback:]).detach()
            transformed_data.append(current_data)

        return torch.cat(transformed_data, dim=1)
