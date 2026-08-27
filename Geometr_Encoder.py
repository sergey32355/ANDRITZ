import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, TensorDataset

"""
If your data are X ∈ R^(N×M) with rows as observations, I would consider a geometric anomaly detector a very good complement to the Transformer/GAN approach.
The central idea is:
Learn the geometry/manifold occupied by normal samples, then score a point according to how far it lies from that geometry.
There are several ways to do this. For your setup, I'd recommend a Deep Geometric Anomaly Detector (DGAD) combining:
a neural embedding,
local geometry of normal samples,
distance to the normal manifold,
local density,
optionally, curvature.
This gives a detector that is quite different from reconstruction-based Transformers.
"""

"""

X [N × M]
     │
     ▼
┌──────────────┐
│ Neural       │
│ embedding    │
│ fθ(x)        │
└──────┬───────┘
       │
       ▼
 Z [N × D]
       │
       ├───────────────┐
       │               │
       ▼               ▼
   k-NN graph      Local PCA
       │               │
       ▼               ▼
 local density     tangent space
       │               │
       └───────┬───────┘
               ▼
       geometric score

"""

# ============================================================
# Device utility
# ============================================================

def resolve_device(device=None):
    """
    Resolve requested device.

    Parameters
    ----------
    device : None, str, or torch.device
        None:
            Automatically select CUDA if available, otherwise CPU.

        "cuda":
            Use CUDA. Raises an error if CUDA is unavailable.

        "cpu":
            Force CPU.

        "cuda:0", "cuda:1", ...
            Use a specific CUDA device.

    Returns
    -------
    torch.device
    """

    if device is None:
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    device = torch.device(device)

    if device.type == "cuda":

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested, but CUDA "
                "is not available."
            )

        if device.index is not None:
            if device.index >= torch.cuda.device_count():
                raise RuntimeError(
                    f"CUDA device {device.index} "
                    f"does not exist. "
                    f"Available devices: "
                    f"{torch.cuda.device_count()}"
                )

    return device


class GeometricEncoder(nn.Module):

    def __init__(
        self,
        n_features,
        latent_dim=16,
    ):
        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(
                n_features,
                128,
            ),

            nn.LayerNorm(128),
            nn.GELU(),

            nn.Linear(
                128,
                64,
            ),

            nn.LayerNorm(64),
            nn.GELU(),

            nn.Linear(
                64,
                latent_dim,
            ),
        )

    def forward(self, x):

        return self.net(x)

@torch.no_grad()
def build_knn_graph(
    X,
    k=10,
    device=None,
):
    """
    Build kNN graph in the original feature space.

    Parameters
    ----------
    X : np.ndarray
        Shape [N, M]

    k : int
        Number of neighbors.

    device : str or torch.device
        "cuda", "cpu", or None.

    Returns
    -------
    torch.Tensor
        Neighbor indices [N, k]
    """

    device = resolve_device(device)

    X_tensor = torch.as_tensor(
        X,
        dtype=torch.float32,
        device=device,
    )

    N = X_tensor.shape[0]

    if k >= N:
        raise ValueError(
            f"k={k} must be smaller than "
            f"number of samples N={N}"
        )

    distances = torch.cdist(
        X_tensor,
        X_tensor,
    )

    # Don't select point itself
    distances.fill_diagonal_(
        float("inf")
    )

    knn = torch.topk(
        distances,
        k=k,
        largest=False,
    ).indices

    return knn



def train_geometric_encoder(
    X_normal,
    latent_dim=16,
    k=10,
    epochs=100,
    learning_rate=1e-3,
    device=None,
):
    """
    Train deep geometric embedding using
    normal samples only.
    """

    device = resolve_device(device)

    print(
        f"Training geometric encoder on {device}"
    )

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(
                device
            )
        )

    X_tensor = torch.as_tensor(
        X_normal,
        dtype=torch.float32,
        device=device,
    )

    n_samples, n_features = (
        X_normal.shape
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = GeometricEncoder(
        n_features=n_features,
        latent_dim=latent_dim,
    ).to(device)

    # --------------------------------------------------------
    # Original-space kNN graph
    # --------------------------------------------------------

    print("Building normal kNN graph...")

    knn = build_knn_graph(
        X_normal,
        k=k,
        device=device,
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4,
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    for epoch in range(epochs):

        model.train()

        # Encode all normal samples
        z = model(X_tensor)

        # [N, k, D]
        z_neighbors = z[knn]

        # [N, 1, D]
        z_center = z.unsqueeze(1)

        # ----------------------------------------------------
        # Local geometry loss
        # ----------------------------------------------------

        local_distances = (
            z_center - z_neighbors
        ).pow(2).sum(dim=-1)

        local_loss = (
            local_distances.mean()
        )

        # ----------------------------------------------------
        # Prevent embedding collapse
        # ----------------------------------------------------

        std = torch.sqrt(
            z.var(
                dim=0,
                unbiased=False,
            )
            + 1e-4
        )

        variance_loss = F.relu(
            1.0 - std
        ).mean()

        # ----------------------------------------------------
        # Total loss
        # ----------------------------------------------------

        loss = (
            local_loss
            + 0.1 * variance_loss
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            5.0,
        )

        optimizer.step()

        if epoch % 10 == 0:

            print(
                f"Epoch {epoch:4d}/{epochs} | "
                f"Loss={loss.item():.6f} | "
                f"Local={local_loss.item():.6f}"
            )

    return model


@torch.no_grad()
def encode_data(
    model,
    X,
    device=None,
    batch_size=1024,
):
    """
    Encode X using geometric model.
    """

    device = resolve_device(device)

    model = model.to(device)
    model.eval()

    dataset = TensorDataset(
        torch.as_tensor(
            X,
            dtype=torch.float32,
        )
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    embeddings = []

    for (x,) in loader:

        x = x.to(
            device,
            non_blocking=(
                device.type == "cuda"
            ),
        )

        z = model(x)

        embeddings.append(
            z.cpu()
        )

    return torch.cat(
        embeddings,
        dim=0,
    )

@torch.no_grad()
def geometric_score(
    model,
    X,
    Z_normal,
    k=10,
    device=None,
    batch_size=512,
):
    """
    Geometric anomaly score based on
    distance to k nearest normal points.
    """

    device = resolve_device(device)

    model = model.to(device)
    model.eval()

    Z_normal = Z_normal.to(
        device
    )

    dataset = TensorDataset(
        torch.as_tensor(
            X,
            dtype=torch.float32,
        )
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    scores = []

    for (x,) in loader:

        x = x.to(
            device,
            non_blocking=(
                device.type == "cuda"
            ),
        )

        # --------------------------------------------
        # Project into latent space
        # --------------------------------------------

        z = model(x)

        # --------------------------------------------
        # Distance to normal manifold
        # --------------------------------------------

        distances = torch.cdist(
            z,
            Z_normal,
        )

        knn_distances = torch.topk(
            distances,
            k=k,
            largest=False,
        ).values

        # --------------------------------------------
        # Average kNN distance
        # --------------------------------------------

        score = (
            knn_distances
            .pow(2)
            .mean(dim=1)
        )

        scores.append(
            score.cpu()
        )

    return torch.cat(
        scores
    ).numpy()

#************************************************************************************
#************************************************************************************
#this is everything in one class

class GeometricAnomalyDetector:

    def __init__(
        self,
        latent_dim=16,
        k=10,
        device=None,
    ):

        self.device = resolve_device(
            device
        )

        self.latent_dim = latent_dim
        self.k = k

        self.model = None
        self.Z_normal = None

    def fit(
        self,
        X_normal,
        epochs=100,
        learning_rate=1e-3,
    ):

        self.model = (
            train_geometric_encoder(
                X_normal=X_normal,
                latent_dim=self.latent_dim,
                k=self.k,
                epochs=epochs,
                learning_rate=learning_rate,
                device=self.device,
            )
        )

        self.Z_normal = encode_data(
            self.model,
            X_normal,
            device=self.device,
        )

        return self

    def score(
        self,
        X,
    ):

        if self.model is None:
            raise RuntimeError(
                "Detector has not been fitted."
            )

        return geometric_score(
            model=self.model,
            X=X,
            Z_normal=self.Z_normal,
            k=self.k,
            device=self.device,
        )