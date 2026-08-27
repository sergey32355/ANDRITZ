
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

#Learn a latent geometry
#
#
#
"""
X
│
▼
Encoder fθ
│
▼
Z
│
├── kNN graph
├── local density
├── tangent space
└── anomaly distance
"""

# ============================================================
# Device handling
# ============================================================

def resolve_device(device=None):
    """
    Parameters
    ----------
    device : None, str, or torch.device
        None       -> automatically select CUDA if available,
                      otherwise CPU.
        "auto"     -> same as None.
        "cuda"     -> use CUDA.
        "cuda:0"   -> use specific GPU.
        "cpu"      -> force CPU.

    Returns
    -------
    torch.device
    """

    if device is None or device == "auto":
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
            n_gpus = torch.cuda.device_count()

            if device.index >= n_gpus:
                raise RuntimeError(
                    f"{device} does not exist. "
                    f"Available GPUs: {n_gpus}"
                )

    return device

#Self-deforming geometric model

class GeometricEncoder(nn.Module):

    def __init__(
        self,
        n_features,
        latent_dim=16,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.LayerNorm(128),
            nn.GELU(),

            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),

            nn.Linear(64, latent_dim),
        )

    def forward(self, x):
        return self.net(x)


class DeformationField(nn.Module):

    def __init__(
        self,
        latent_dim,
        hidden_dim=64,
        max_deformation=1.0,
    ):
        super().__init__()

        self.max_deformation = max_deformation

        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),

            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, z):

        delta = self.net(z)

        # Limit deformation magnitude
        delta = torch.tanh(delta)

        delta = (
            self.max_deformation
            * delta
        )

        return z + delta

#Detector class
class SelfDeformingGeometricAnomalyDetector:

    def __init__(
        self,
        latent_dim=16,
        k=10,
        max_deformation=1.0,
        device=None,
    ):
        """
        Self-deforming geometric anomaly detector.

        Parameters
        ----------
        latent_dim : int
            Dimension of learned geometric space.

        k : int
            Number of nearest normal neighbors.

        max_deformation : float
            Maximum latent deformation.

        device : None, str, torch.device
            None / "auto" -> CUDA if available,
                              otherwise CPU.
            "cuda"      -> GPU.
            "cuda:0"    -> specific GPU.
            "cpu"       -> CPU.
        """

        self.device = resolve_device(device)

        self.latent_dim = latent_dim
        self.k = k
        self.max_deformation = max_deformation

        self.encoder = None
        self.deformation = None

        self.Z_normal = None
        self.Z_normal_deformed = None

        self.is_fitted = False

        print(
            f"Using device: {self.device}"
        )

        if self.device.type == "cuda":
            print(
                "GPU:",
                torch.cuda.get_device_name(
                    self.device
                )
            )

    #Fit the normal geometry
    def fit_normal(self,
                   X_normal,
                   epochs=100,
                   learning_rate=1e-3,
                  ):
        """
        Learn the initial normal geometry.

        X_normal:
            [N, M]
        """

        X = torch.as_tensor(
            X_normal,
            dtype=torch.float32,
            device=self.device,
        )

        n_samples, n_features = X.shape

        if self.k >= n_samples:
            raise ValueError(
                f"k={self.k} must be smaller "
                f"than N={n_samples}"
            )

        # ----------------------------------------------------
        # Models
        # ----------------------------------------------------

        self.encoder = GeometricEncoder(
            n_features=n_features,
            latent_dim=self.latent_dim,
        ).to(self.device)

        self.deformation = DeformationField(
            latent_dim=self.latent_dim,
            max_deformation=self.max_deformation,
        ).to(self.device)

        optimizer = torch.optim.AdamW(
            self.encoder.parameters(),
            lr=learning_rate,
            weight_decay=1e-4,
        )

        # ----------------------------------------------------
        # Original-space kNN
        # ----------------------------------------------------

        with torch.no_grad():

            distances = torch.cdist(
                X,
                X,
            )

            distances.fill_diagonal_(
                float("inf")
            )

            knn = torch.topk(
                distances,
                k=self.k,
                largest=False,
            ).indices

        # ----------------------------------------------------
        # Train encoder
        # ----------------------------------------------------

        for epoch in range(epochs):

            self.encoder.train()

            Z = self.encoder(X)

            Z_neighbors = Z[knn]

            Z_center = Z.unsqueeze(1)

            # Preserve local geometry
            local_loss = (
                Z_center - Z_neighbors
            ).pow(2).sum(dim=-1).mean()

            # Prevent collapse
            std = torch.sqrt(
                Z.var(
                    dim=0,
                    unbiased=False,
                )
                + 1e-4
            )

            variance_loss = F.relu(
                1.0 - std
            ).mean()

            loss = (
                local_loss
                + 0.1 * variance_loss
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.encoder.parameters(),
                5.0,
            )

            optimizer.step()

            if epoch % 10 == 0:

                print(
                    f"[Geometry] "
                    f"Epoch {epoch:4d}/{epochs} | "
                    f"Loss={loss.item():.6f}"
                )

        # ----------------------------------------------------
        # Save normal latent representation
        # ----------------------------------------------------

        self.encoder.eval()

        with torch.no_grad():

            self.Z_normal = self.encoder(X)

        self.is_fitted = True

        return self

    #Self-deformation using labeled anomalies
    def deform(
            self,
            X_normal,
            X_anomaly,
            epochs=100,
            learning_rate=1e-3,
            margin=2.0,
            lambda_geometry=1.0,
            lambda_separation=1.0,
            lambda_deformation=0.1,
        ):
        """
        Learn deformation using labeled anomalies.

        Normal samples define the manifold.
        Anomaly samples are pushed away from it.
        """

        if not self.is_fitted:
            raise RuntimeError(
                "Call fit_normal() first."
            )

        Xn = torch.as_tensor(
            X_normal,
            dtype=torch.float32,
            device=self.device,
        )

        Xa = torch.as_tensor(
            X_anomaly,
            dtype=torch.float32,
            device=self.device,
        )

        self.encoder.eval()

        # Freeze encoder initially.
        # The deformation learns how to reshape
        # the existing geometry.
        for p in self.encoder.parameters():
            p.requires_grad = False

        Zn = self.encoder(Xn)
        Za = self.encoder(Xa)

        optimizer = torch.optim.AdamW(
            self.deformation.parameters(),
            lr=learning_rate,
        )

        for epoch in range(epochs):

            self.deformation.train()

            Zn_def = self.deformation(Zn)
            Za_def = self.deformation(Za)

            # =================================================
            # 1. Preserve normal geometry
            # =================================================

            D_before = torch.cdist(
                Zn,
                Zn,
            )

            D_after = torch.cdist(
                Zn_def,
                Zn_def,
            )

            geometry_loss = F.mse_loss(
                D_after,
                D_before.detach(),
            )

            # =================================================
            # 2. Push anomalies away
            # =================================================

            D_anomaly = torch.cdist(
                Za_def,
                Zn_def,
            )

            nearest_distance = (
                D_anomaly.min(dim=1).values
            )

            separation_loss = F.relu(
                margin - nearest_distance
            ).pow(2).mean()

            # =================================================
            # 3. Avoid excessive deformation
            # =================================================

            delta_normal = (
                Zn_def - Zn
            )

            delta_anomaly = (
                Za_def - Za
            )

            deformation_loss = (
                delta_normal.pow(2).mean()
                +
                delta_anomaly.pow(2).mean()
            )

            # =================================================
            # Total
            # =================================================

            loss = (

                lambda_geometry
                * geometry_loss

                +

                lambda_separation
                * separation_loss

                +

                lambda_deformation
                * deformation_loss
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.deformation.parameters(),
                5.0,
            )

            optimizer.step()

            if epoch % 10 == 0:

                print(
                    f"[Deformation] "
                    f"Epoch {epoch:4d}/{epochs} | "
                    f"Loss={loss.item():.6f} | "
                    f"Geometry="
                    f"{geometry_loss.item():.4f} | "
                    f"Separation="
                    f"{separation_loss.item():.4f}"
                )

        # ----------------------------------------------------
        # Save deformed normal geometry
        # ----------------------------------------------------

        self.deformation.eval()

        with torch.no_grad():

            self.Z_normal_deformed = (
                self.deformation(
                    self.Z_normal
                )
            )

        return self

    #Anomaly score
    @torch.no_grad()
    def score(
            self,
            X,
        ):
        """
        Calculate geometric anomaly scores.

        Larger score = more anomalous.
        """

        if not self.is_fitted:
            raise RuntimeError(
                "Detector has not been fitted."
            )

        self.encoder.eval()
        self.deformation.eval()

        X = torch.as_tensor(
            X,
            dtype=torch.float32,
            device=self.device,
        )

        Z = self.encoder(X)

        Z_def = self.deformation(Z)

        distances = torch.cdist(
            Z_def,
            self.Z_normal_deformed,
        )

        k = min(
            self.k,
            self.Z_normal_deformed.shape[0],
        )

        nearest_distances = torch.topk(
            distances,
            k=k,
            largest=False,
        ).values

        score = (
            nearest_distances.pow(2)
            .mean(dim=1)
        )

        return score.cpu().numpy()