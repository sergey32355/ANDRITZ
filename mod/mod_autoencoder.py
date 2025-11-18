"""TBC"""

import os
import copy
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
torch.use_deterministic_algorithms(True)

class EarlyStopping:
    """TBC"""

    def __init__(self, patience=5, delta=0.0):
        self.patience = patience
        self.delta = delta
        self.best_loss = float("inf")
        self.best_state = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


class Autoencoder(nn.Module):
    """TBC"""

    def __init__(self, input_dim, latent_dim, n_layers=2):
        super(Autoencoder, self).__init__()
        dims = np.linspace(input_dim, latent_dim, n_layers + 1, dtype=int)
        # Create encoder layers that progressively reduce dimensions
        encoder_layers = []
        decoder_layers = []
        # Add layers based on dims array
        for i in range(n_layers):
            encoder_layers.extend(
                [nn.Linear(dims[i], dims[i + 1]),nn.ReLU()] if i < n_layers - 1 else [nn.Linear(dims[i], dims[i + 1])]
            )
        self.encoder = nn.Sequential(*encoder_layers)
        for i in range(n_layers):
            decoder_layers.extend(
                [nn.Linear(dims[-i-1], dims[-i-2]),nn.ReLU()] if i < n_layers - 1 else [nn.Linear(dims[-i-1], dims[-i-2])]
            )
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, X):
        Z = self.encoder(X)
        out = self.decoder(Z)
        return out


class AutoencoderEstimator(BaseEstimator, TransformerMixin):
    """TBC"""

    def __init__(
            self,
            input_dim,
            random_seed,
            latent_dim = None,
            n_layers = None,
            n_epochs = None,
            lr = None,
            batch_size = None,
            patience = 5,
            val_split = 0.1,
            device = "cpu",
        ):
        # Store parameters
        self.input_dim = input_dim
        self.latent_dim = latent_dim if latent_dim is not None else round(0.5 * input_dim)
        self.n_layers = n_layers if n_layers is not None else (2 if input_dim <= 150 else 3)
        self.n_epochs = n_epochs if n_epochs is not None else (50 if input_dim <= 150 else 100)
        self.lr = lr if lr is not None else (1e-3 if input_dim <= 150 else 5e-4)
        self.batch_size = batch_size if batch_size is not None else (32 if input_dim <= 150 else 64)
        self.patience = patience
        self.val_split = val_split
        self.random_seed = random_seed
        self.device = device# or ("cuda" if torch.cuda.is_available() else "cpu")

    def get_init_weights_path(self, base_folder = os.getcwd()):
        weights_folder = os.path.join(base_folder, "init_weights_ae")
        os.makedirs(weights_folder, exist_ok=True)
        weights_file = os.path.join(weights_folder, f"{self.input_dim}_{self.latent_dim}_{self.n_layers}_{self.random_seed}.pt")
        return weights_file

    def fit(self, X, y=None):

        # Initialize the model
        self.model_ = Autoencoder(input_dim=self.input_dim, latent_dim=self.latent_dim, n_layers=self.n_layers)
        # Load initial weights for deterministic results
        init_weights_ae_path = self.get_init_weights_path()
        try:
            self.model_.load_state_dict(torch.load(init_weights_ae_path))
        except FileNotFoundError:
            torch.save(self.model_.state_dict(), init_weights_ae_path)

        self.criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model_.parameters(), lr=self.lr)
        early_stopper = EarlyStopping(patience=self.patience)

        # Convert input data to torch tensors
        if type(X) == pd.DataFrame:
            X = X.values
        X_tensor = torch.tensor(X, dtype=torch.float32)#, device=self.device)
        dataset = TensorDataset(X_tensor)

        # internal validation split (for early stopping only)
        n_samples = len(dataset)
        val_size = int(self.val_split * n_samples)
        train_size = n_samples - val_size
        train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size], generator = torch.Generator().manual_seed(self.random_seed))
        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=False) 
        # val_loader = DataLoader(val_set, batch_size=self.batch_size, shuffle=False)
        # Extract tensors from torch.utils.data.Subset
        val_set = torch.cat([item[0].unsqueeze(0) for item in val_set], dim=0)

        # Training loop
        # self.model_.to(self.device, non_blocking=True)
        self.model_.train()
        for epoch in range(self.n_epochs):

            for (x,) in train_loader:
                x = x.to(self.device)
                optimizer.zero_grad()
                loss = self.criterion(self.model_(x), x)
                loss.backward()
                optimizer.step()

            # validation for early stopping
            val_loss = self.compute_reconstruction_error(val_set)
            early_stopper(val_loss, self.model_)
            if early_stopper.early_stop:
                print(f"Early stopping triggered at epoch {epoch+1}/{self.n_epochs}. Best loss: {early_stopper.best_loss:.5f}")
                break
        self.model_.load_state_dict(early_stopper.best_state)

    # Anomaly scoring function
    def compute_reconstruction_error(self, X, aggregate=True):
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32)
        self.model_.eval()
        with torch.no_grad():
            recon = self.model_(X)
            if aggregate:
                error = self.criterion(recon, X).detach()
            else:
                error = torch.mean((X - recon) ** 2, dim=1).detach()
        return error
    
    def predict_proba(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        error = self.compute_reconstruction_error(X, aggregate=False)
        # Convert to pseudo-probabilities (not calibrated!) using sigmoid
        # Higher reconstruction error -> higher probability of being anomalous
        anomaly_proba = 2*torch.sigmoid(error)-1
        try:
            np_proba = anomaly_proba.numpy()
        except TypeError:
            np_proba = anomaly_proba.cpu().numpy()
        return np_proba
    
    def score(self, X, y=None):
        # sklearn assumes "higher is better"
        # so return negative reconstruction error
        neg_score = self.compute_reconstruction_error(X)
        # print("Loss:", neg_score, "Shape of X:", str(X.shape))
        return -neg_score
        
    # def predict(self, X, rec_error_threshold_factor=2.0):
    #     X_tensor = torch.tensor(X.values, dtype=torch.float32, device=self.device)
    #     errors = self.compute_reconstruction_error(X_tensor)
    #     mean_train_rec_error = torch.mean(self.compute_reconstruction_error(self.X_train)).item()
    #     # Convert to -1 (anomaly) and 1 (normal)
    #     predictions = (errors / mean_train_rec_error < rec_error_threshold_factor).int() * 2 - 1
    #     try:
    #         np_preds = predictions.numpy()
    #     except TypeError:
    #         np_preds = predictions.cpu().numpy()
    #     return np_preds

    # # Convert everything to Tensor
    # if isinstance(X, np.ndarray):
    #     X = torch.tensor(X, dtype=torch.float32)
    # elif isinstance(X, pd.DataFrame):
    #     X = torch.tensor(X.values, dtype=torch.float32)
    # elif isinstance(X, torch.utils.data.DataLoader):
    #     X = torch.cat([batch[0] for batch in X], dim=0)
    # elif isinstance(X, torch.utils.data.Subset):
    #     X = torch.cat([item[0].unsqueeze(0) for item in X], dim=0)