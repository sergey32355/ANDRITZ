import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



#Self-Deforming Hyperbolic Anomaly Detector (SD-HAD)
"""
                     INPUT X
                        │
                        ▼
                  Neural Encoder
                        │
                        ▼
                 Hyperbolic map
                        │
                        ▼
                  z ∈ H^D
                        │
              ┌─────────┴─────────┐
              │                   │
              ▼                   ▼
        Normal manifold      Local geometry
              │                   │
              └─────────┬─────────┘
                        │
                        ▼
              Hyperbolic deformation
                        ▲
                        │
                 labeled anomalies
                        │
                        ▼
              maximize H-distance
                 from normals

#**********************************************************
#**********************************************************
#**********************************************************


Euclidean

             ● ● ●
          ● ● ● ● ●
        ● ● ● ● ● ● ●
          ● ● ● ● ●


Hyperbolic

                 ●
              ●     ●
           ●           ●
        ●                 ●
     ●                       ●
"""

#*************************************************************************
#Hyperbolic self-deformation

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# DEVICE
# ============================================================

def resolve_device(device=None):

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
                "CUDA was requested but CUDA is not available."
            )

        if device.index is not None:
            n_gpus = torch.cuda.device_count()

            if device.index >= n_gpus:
                raise RuntimeError(
                    f"{device} does not exist. "
                    f"Available GPUs: {n_gpus}"
                )

    return device


# ============================================================
# SAFE TENSOR UTILITIES
# ============================================================

def unwrap_tensor(x, name="value"):
    """
    Convert common tuple-returning PyTorch operations into
    the actual values tensor.

    This protects against code such as:

        x = torch.min(tensor, dim=1)

    which returns:

        (values, indices)
    """

    if isinstance(x, tuple):

        if len(x) == 0:
            raise ValueError(
                f"{name} is an empty tuple."
            )

        # Common case:
        # torch.min(..., dim=...)
        # torch.max(..., dim=...)
        # torch.topk(...)
        x = x[0]

    if not torch.is_tensor(x):

        x = torch.as_tensor(
            x,
            dtype=torch.float32,
        )

    return x


def scalar_tensor(
    x,
    device,
    dtype=torch.float32,
    name="value",
):
    """
    Convert x to a scalar tensor.

    Handles:
        float
        int
        numpy scalar
        torch scalar tensor
        (value, indices) tuple
        one-element tuple/list
    """

    # --------------------------------------------------------
    # Unwrap tuple
    # --------------------------------------------------------

    if isinstance(x, tuple):

        if len(x) == 0:
            raise ValueError(
                f"{name} is an empty tuple."
            )

        # If this is something like:
        #
        # (values, indices)
        #
        # use values.
        x = x[0]

    # --------------------------------------------------------
    # Convert to tensor
    # --------------------------------------------------------

    if not torch.is_tensor(x):

        x = torch.as_tensor(
            x,
            dtype=dtype,
            device=device,
        )

    else:

        x = x.to(
            device=device,
            dtype=dtype,
        )

    # --------------------------------------------------------
    # Remove dimensions of size 1
    # --------------------------------------------------------

    x = x.squeeze()

    # --------------------------------------------------------
    # Require scalar
    # --------------------------------------------------------

    if x.numel() != 1:

        raise ValueError(
            f"{name} must be a scalar, "
            f"but received shape {tuple(x.shape)} "
            f"with {x.numel()} elements."
        )

    return x.reshape(())


# ============================================================
# POINCARE BALL
# ============================================================

def project_ball(
    x,
    curvature=1.0,
    eps=1e-5,
):
    """
    Project x into the Poincare ball.

    Ball radius:

        R = 1 / sqrt(curvature)
    """

    if curvature <= 0:
        raise ValueError(
            "curvature must be > 0."
        )

    radius = (
        1.0
        / np.sqrt(curvature)
    )

    max_norm = radius - eps

    norm = torch.linalg.norm(
        x,
        dim=-1,
        keepdim=True,
    )

    scale = torch.clamp(
        max_norm
        / norm.clamp_min(eps),
        max=1.0,
    )

    return x * scale


def mobius_add(
    x,
    y,
    curvature=1.0,
    eps=1e-7,
):
    """
    Mobius addition in the Poincare ball.

    Supports broadcasting.

    x : [..., D]
    y : [..., D]
    """

    c = curvature

    x2 = (
        x * x
    ).sum(
        dim=-1,
        keepdim=True,
    )

    y2 = (
        y * y
    ).sum(
        dim=-1,
        keepdim=True,
    )

    xy = (
        x * y
    ).sum(
        dim=-1,
        keepdim=True,
    )

    numerator = (
        (1.0 + 2.0 * c * xy + c * y2)
        * x
        +
        (1.0 - c * x2)
        * y
    )

    denominator = (
        1.0
        + 2.0 * c * xy
        + c * c * x2 * y2
    )

    result = (
        numerator
        /
        denominator.clamp_min(eps)
    )

    return result


# ============================================================
# POINCARE DISTANCE
# ============================================================

def poincare_distance(
    x,
    y,
    curvature=1.0,
    eps=1e-7,
):
    """
    Hyperbolic distance between corresponding points.
    """

    diff = mobius_add(
        -x,
        y,
        curvature=curvature,
    )

    norm = torch.linalg.norm(
        diff,
        dim=-1,
    )

    sqrt_c = np.sqrt(
        curvature
    )

    argument = (
        sqrt_c * norm
    ).clamp(
        min=0.0,
        max=1.0 - eps,
    )

    return (
        2.0 / sqrt_c
    ) * torch.atanh(
        argument
    )


def pairwise_hyperbolic_distance(
    x,
    y,
    curvature=1.0,
    eps=1e-7,
):
    """
    Pairwise hyperbolic distance.

    x : [N, D]
    y : [M, D]

    Returns:

        [N, M]
    """

    # --------------------------------------------------------
    # Make sure x/y are tensors
    # --------------------------------------------------------

    x = unwrap_tensor(
        x,
        name="x",
    )

    y = unwrap_tensor(
        y,
        name="y",
    )

    if x.ndim != 2:
        raise ValueError(
            f"x must have shape [N, D], "
            f"got {tuple(x.shape)}"
        )

    if y.ndim != 2:
        raise ValueError(
            f"y must have shape [M, D], "
            f"got {tuple(y.shape)}"
        )

    # --------------------------------------------------------
    # Broadcasting
    # --------------------------------------------------------

    x_expanded = (
        x[:, None, :]
    )

    y_expanded = (
        y[None, :, :]
    )

    diff = mobius_add(
        -x_expanded,
        y_expanded,
        curvature=curvature,
    )

    norm = torch.linalg.norm(
        diff,
        dim=-1,
    )

    sqrt_c = np.sqrt(
        curvature
    )

    argument = (
        sqrt_c * norm
    ).clamp(
        min=0.0,
        max=1.0 - eps,
    )

    distances = (
        2.0 / sqrt_c
    ) * torch.atanh(
        argument
    )

    # --------------------------------------------------------
    # Safety check
    # --------------------------------------------------------

    if not torch.is_tensor(distances):

        raise TypeError(
            "pairwise_hyperbolic_distance "
            "did not return a Tensor."
        )

    return distances


# ============================================================
# HYPERBOLIC ENCODER
# ============================================================

class HyperbolicEncoder(nn.Module):

    def __init__(
        self,
        n_features,
        latent_dim=16,
        curvature=1.0,
    ):

        super().__init__()

        self.curvature = curvature

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

        z = self.net(x)

        z = torch.tanh(z)

        radius = (
            1.0
            / np.sqrt(
                self.curvature
            )
        )

        z = (
            0.9
            * radius
            * z
        )

        z = project_ball(
            z,
            curvature=self.curvature,
        )

        return z


# ============================================================
# SELF-DEFORMATION NETWORK
# ============================================================

class HyperbolicDeformation(
    nn.Module
):

    def __init__(
        self,
        latent_dim,
        hidden_dim=64,
        max_deformation=0.2,
        curvature=1.0,
    ):

        super().__init__()

        self.max_deformation = (
            max_deformation
        )

        self.curvature = curvature

        self.net = nn.Sequential(

            nn.Linear(
                latent_dim,
                hidden_dim,
            ),

            nn.LayerNorm(hidden_dim),

            nn.GELU(),

            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),

            nn.GELU(),

            nn.Linear(
                hidden_dim,
                latent_dim,
            ),
        )

    def forward(self, z):

        # ----------------------------------------------------
        # Learn deformation
        # ----------------------------------------------------

        delta = self.net(z)

        delta = torch.tanh(
            delta
        )

        delta = (
            self.max_deformation
            * delta
        )

        # ----------------------------------------------------
        # Move point using Mobius addition
        # ----------------------------------------------------

        z_new = mobius_add(
            z,
            delta,
            curvature=self.curvature,
        )

        # ----------------------------------------------------
        # Keep inside ball
        # ----------------------------------------------------

        z_new = project_ball(
            z_new,
            curvature=self.curvature,
        )

        return z_new


# ============================================================
# NORMAL GEOMETRY LOSS
# ============================================================

def normal_geometry_loss(
    z_normal,
    z_normal_deformed,
    curvature=1.0,
    max_points=512,
):
    """
    Preserve relative hyperbolic geometry
    of normal observations.
    """

    n = z_normal.shape[0]

    # --------------------------------------------------------
    # Subsample for O(N^2) distance calculation
    # --------------------------------------------------------

    if n > max_points:

        indices = torch.randperm(
            n,
            device=z_normal.device,
        )[:max_points]

        z_normal = (
            z_normal[
                indices
            ]
        )

        z_normal_deformed = (
            z_normal_deformed[
                indices
            ]
        )

    # --------------------------------------------------------
    # Original distances
    # --------------------------------------------------------

    D_before = (
        pairwise_hyperbolic_distance(
            z_normal,
            z_normal,
            curvature=curvature,
        )
    )

    # --------------------------------------------------------
    # Deformed distances
    # --------------------------------------------------------

    D_after = (
        pairwise_hyperbolic_distance(
            z_normal_deformed,
            z_normal_deformed,
            curvature=curvature,
        )
    )

    # --------------------------------------------------------
    # Preserve geometry
    # --------------------------------------------------------

    return F.mse_loss(
        D_after,
        D_before.detach(),
    )


# ============================================================
# ANOMALY SEPARATION LOSS
# ============================================================

def hyperbolic_separation_loss(
    z_anomaly,
    z_normal,
    curvature=1.0,
    margin=2.0,
):
    """
    Push anomalies away from the normal manifold.

    For each anomaly:

        d_min = min_j d(anomaly, normal_j)

    Objective:

        max(0, margin - d_min)^2

    This implementation explicitly protects against
    tuple-valued `margin` and tuple-valued distance outputs.
    """

    # --------------------------------------------------------
    # Ensure margin is a scalar Tensor
    # --------------------------------------------------------

    margin_tensor = scalar_tensor(
        margin,
        device=z_anomaly.device,
        dtype=z_anomaly.dtype,
        name="margin",
    )

    # --------------------------------------------------------
    # Calculate pairwise distances
    # --------------------------------------------------------

    distances = (
        pairwise_hyperbolic_distance(
            z_anomaly,
            z_normal,
            curvature=curvature,
        )
    )

    # --------------------------------------------------------
    # HARD SAFETY CHECK
    # --------------------------------------------------------

    if isinstance(distances, tuple):

        distances = distances[0]

    if not torch.is_tensor(distances):

        raise TypeError(
            "distances must be a Tensor, "
            f"got {type(distances)}"
        )

    if distances.ndim != 2:

        raise ValueError(
            "Distance matrix must have "
            f"2 dimensions, got {distances.ndim}."
        )

    # --------------------------------------------------------
    # Nearest normal sample
    #
    # torch.amin ALWAYS returns only values.
    # --------------------------------------------------------

    nearest_distance = torch.amin(
        distances,
        dim=1,
    )

    # --------------------------------------------------------
    # HARD SAFETY CHECK
    # --------------------------------------------------------

    if isinstance(nearest_distance, tuple):

        nearest_distance = (
            nearest_distance[0]
        )

    if not torch.is_tensor(
        nearest_distance
    ):

        nearest_distance = torch.as_tensor(
            nearest_distance,
            dtype=z_anomaly.dtype,
            device=z_anomaly.device,
        )

    # --------------------------------------------------------
    # Debug information if something unexpected occurs
    # --------------------------------------------------------

    if not torch.is_tensor(
        margin_tensor
    ):

        raise TypeError(
            "margin_tensor is not a Tensor."
        )

    if not torch.is_tensor(
        nearest_distance
    ):

        raise TypeError(
            "nearest_distance is not a Tensor."
        )

    # --------------------------------------------------------
    # Margin violation
    #
    # IMPORTANT:
    # We use torch.clamp rather than F.relu.
    # Both are mathematically equivalent here.
    # --------------------------------------------------------

    violation = (
        margin_tensor
        - nearest_distance
    )

    violation = torch.clamp(
        violation,
        min=0.0,
    )

    # --------------------------------------------------------
    # Final separation loss
    # --------------------------------------------------------

    loss = (
        violation
        .pow(2)
        .mean()
    )

    return loss


# ============================================================
# DEFORMATION REGULARIZATION
# ============================================================

def deformation_regularization(
    z,
    z_deformed,
):

    delta = (
        z_deformed
        - z
    )

    return (
        delta
        .pow(2)
        .mean()
    )


# ============================================================
# COMPLETE DEFORMATION LOSS
# ============================================================

def total_deformation_loss(
    z_normal,
    z_anomaly,
    z_normal_deformed,
    z_anomaly_deformed,
    curvature=1.0,
    margin=2.0,
    lambda_geometry=1.0,
    lambda_separation=1.0,
    lambda_deformation=0.1,
):
    """
    Total loss:

        L =
            lambda_geometry * L_geometry
            +
            lambda_separation * L_separation
            +
            lambda_deformation * L_deformation
    """

    # --------------------------------------------------------
    # Normal geometry
    # --------------------------------------------------------

    L_geometry = (
        normal_geometry_loss(
            z_normal,
            z_normal_deformed,
            curvature=curvature,
        )
    )

    # --------------------------------------------------------
    # Anomaly separation
    # --------------------------------------------------------

    L_separation = (
        hyperbolic_separation_loss(
            z_anomaly,
            z_normal,
            curvature=curvature,
            margin=margin,
        )
    )

    # --------------------------------------------------------
    # Deformation penalty
    # --------------------------------------------------------

    L_deformation = (

        deformation_regularization(
            z_normal,
            z_normal_deformed,
        )

        +

        deformation_regularization(
            z_anomaly,
            z_anomaly_deformed,
        )
    )

    # --------------------------------------------------------
    # Total
    # --------------------------------------------------------

    loss = (

        lambda_geometry
        * L_geometry

        +

        lambda_separation
        * L_separation

        +

        lambda_deformation
        * L_deformation
    )

    return (
        loss,
        L_geometry,
        L_separation,
        L_deformation,
    )


# ============================================================
# MAIN DETECTOR
# ============================================================

class HyperbolicSelfDeformingDetector:

    def __init__(
        self,
        latent_dim=16,
        curvature=1.0,
        k=10,
        max_deformation=0.2,
        device=None,
    ):

        self.device = resolve_device(
            device
        )

        self.latent_dim = latent_dim

        self.curvature = curvature

        self.k = k

        self.max_deformation = (
            max_deformation
        )

        self.encoder = None

        self.deformation = None

        self.Z_normal = None

        self.Z_normal_deformed = None

        self.fitted = False

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


    # ========================================================
    # FIT
    # ========================================================

    def fit(
        self,
        X_normal,
        X_anomaly,
        epochs_encoder=100,
        epochs_deformation=100,
        learning_rate=1e-3,
        margin=2.0,
        lambda_geometry=1.0,
        lambda_separation=1.0,
        lambda_deformation=0.1,
    ):

        # ----------------------------------------------------
        # Convert input
        # ----------------------------------------------------

        X_normal = np.asarray(
            X_normal,
            dtype=np.float32,
        )

        X_anomaly = np.asarray(
            X_anomaly,
            dtype=np.float32,
        )

        # ----------------------------------------------------
        # Check input
        # ----------------------------------------------------

        if X_normal.ndim != 2:

            raise ValueError(
                "X_normal must have shape [N, M]."
            )

        if X_anomaly.ndim != 2:

            raise ValueError(
                "X_anomaly must have shape [N, M]."
            )

        if (
            X_normal.shape[1]
            != X_anomaly.shape[1]
        ):

            raise ValueError(
                "X_normal and X_anomaly "
                "must have the same number of features."
            )

        if X_normal.shape[0] < 2:

            raise ValueError(
                "At least two normal samples are required."
            )

        if X_anomaly.shape[0] < 1:

            raise ValueError(
                "At least one anomaly sample is required."
            )

        # ----------------------------------------------------
        # Normalize margin BEFORE training
        #
        # This is important because it guarantees that
        # `margin` can never be a tuple later.
        # ----------------------------------------------------

        margin = float(
            scalar_tensor(
                margin,
                device=self.device,
                name="margin",
            ).item()
        )

        if margin <= 0:

            raise ValueError(
                f"margin must be > 0, got {margin}"
            )

        print(
            f"Hyperbolic margin = {margin:.4f}"
        )

        # ----------------------------------------------------
        # Move arrays to device
        # ----------------------------------------------------

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

        n_features = (
            Xn.shape[1]
        )

        # ====================================================
        # STAGE 1
        # ====================================================

        print()
        print(
            "=========================================="
        )
        print(
            "Stage 1: Learning normal hyperbolic geometry"
        )
        print(
            "=========================================="
        )

        self.encoder = (
            HyperbolicEncoder(
                n_features=n_features,
                latent_dim=self.latent_dim,
                curvature=self.curvature,
            )
            .to(self.device)
        )

        optimizer = torch.optim.AdamW(
            self.encoder.parameters(),
            lr=learning_rate,
            weight_decay=1e-4,
        )

        # ----------------------------------------------------
        # Euclidean kNN graph on input data
        # ----------------------------------------------------

        with torch.no_grad():

            original_distances = (
                torch.cdist(
                    Xn,
                    Xn,
                )
            )

            original_distances.fill_diagonal_(
                float("inf")
            )

            k_graph = min(
                self.k,
                Xn.shape[0] - 1,
            )

            knn_result = torch.topk(
                original_distances,
                k=k_graph,
                largest=False,
                dim=1,
            )

            # Explicitly take values/indices correctly
            knn = knn_result.indices

        # ----------------------------------------------------
        # Train encoder
        # ----------------------------------------------------

        for epoch in range(
            epochs_encoder
        ):

            self.encoder.train()

            Z = self.encoder(
                Xn
            )

            Z_neighbors = Z[
                knn
            ]

            local_loss = (
                Z.unsqueeze(1)
                -
                Z_neighbors
            ).pow(2).sum(
                dim=-1
            ).mean()

            # Prevent latent collapse
            std = torch.sqrt(
                Z.var(
                    dim=0,
                    unbiased=False,
                )
                + 1e-4
            )

            variance_loss = (
                F.relu(
                    1.0 - std
                ).mean()
            )

            loss = (
                local_loss
                +
                0.1 * variance_loss
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.encoder.parameters(),
                max_norm=5.0,
            )

            optimizer.step()

            if (
                epoch % 10 == 0
                or
                epoch == epochs_encoder - 1
            ):

                print(
                    f"Epoch "
                    f"{epoch + 1:4d}/"
                    f"{epochs_encoder} | "
                    f"Loss="
                    f"{loss.item():.6f}"
                )

        # ----------------------------------------------------
        # Encode data
        # ----------------------------------------------------

        self.encoder.eval()

        with torch.no_grad():

            self.Z_normal = (
                self.encoder(
                    Xn
                )
            )

            Z_anomaly = (
                self.encoder(
                    Xa
                )
            )

        # ====================================================
        # STAGE 2
        # ====================================================

        print()
        print(
            "=========================================="
        )
        print(
            "Stage 2: Self-deforming hyperbolic geometry"
        )
        print(
            "=========================================="
        )

        self.deformation = (
            HyperbolicDeformation(
                latent_dim=self.latent_dim,
                hidden_dim=64,
                max_deformation=self.max_deformation,
                curvature=self.curvature,
            )
            .to(self.device)
        )

        # Freeze encoder
        for parameter in (
            self.encoder.parameters()
        ):

            parameter.requires_grad = False

        optimizer = torch.optim.AdamW(
            self.deformation.parameters(),
            lr=learning_rate,
        )

        # ----------------------------------------------------
        # Train deformation
        # ----------------------------------------------------

        for epoch in range(
            epochs_deformation
        ):

            self.deformation.train()

            Z_normal_def = (
                self.deformation(
                    self.Z_normal
                )
            )

            Z_anomaly_def = (
                self.deformation(
                    Z_anomaly
                )
            )

            (
                loss,
                L_geometry,
                L_separation,
                L_deformation,
            ) = total_deformation_loss(

                z_normal=self.Z_normal,

                z_anomaly=Z_anomaly,

                z_normal_deformed=(
                    Z_normal_def
                ),

                z_anomaly_deformed=(
                    Z_anomaly_def
                ),

                curvature=self.curvature,

                margin=margin,

                lambda_geometry=(
                    lambda_geometry
                ),

                lambda_separation=(
                    lambda_separation
                ),

                lambda_deformation=(
                    lambda_deformation
                ),
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.deformation.parameters(),
                max_norm=5.0,
            )

            optimizer.step()

            if (
                epoch % 10 == 0
                or
                epoch == epochs_deformation - 1
            ):

                print(
                    f"Epoch "
                    f"{epoch + 1:4d}/"
                    f"{epochs_deformation} | "
                    f"Loss={loss.item():.6f} | "
                    f"Geometry={L_geometry.item():.6f} | "
                    f"Separation={L_separation.item():.6f} | "
                    f"Deformation={L_deformation.item():.6f}"
                )

        # ----------------------------------------------------
        # Save final normal manifold
        # ----------------------------------------------------

        self.deformation.eval()

        with torch.no_grad():

            self.Z_normal_deformed = (
                self.deformation(
                    self.Z_normal
                )
            )

        self.fitted = True

        print()
        print(
            "Training completed."
        )

        return self


    # ========================================================
    # ANOMALY SCORE
    # ========================================================

    @torch.no_grad()
    def score(
        self,
        X,
        batch_size=512,
    ):

        if not self.fitted:

            raise RuntimeError(
                "Detector has not been fitted."
            )

        X = np.asarray(
            X,
            dtype=np.float32,
        )

        if X.ndim != 2:

            raise ValueError(
                "X must have shape [N, M]."
            )

        self.encoder.eval()

        self.deformation.eval()

        scores = []

        for start in range(
            0,
            len(X),
            batch_size,
        ):

            end = min(
                start + batch_size,
                len(X),
            )

            X_batch = torch.as_tensor(
                X[start:end],
                dtype=torch.float32,
                device=self.device,
            )

            # ------------------------------------------------
            # Encode
            # ------------------------------------------------

            Z = self.encoder(
                X_batch
            )

            # ------------------------------------------------
            # Deform
            # ------------------------------------------------

            Z_def = self.deformation(
                Z
            )

            # ------------------------------------------------
            # Distance to normal manifold
            # ------------------------------------------------

            distances = (
                pairwise_hyperbolic_distance(
                    Z_def,
                    self.Z_normal_deformed,
                    curvature=self.curvature,
                )
            )

            # ------------------------------------------------
            # k nearest normal points
            # ------------------------------------------------

            k = min(
                self.k,
                self.Z_normal_deformed.shape[0],
            )

            topk_result = torch.topk(
                distances,
                k=k,
                largest=False,
                dim=1,
            )

            nearest = (
                topk_result.values
            )

            # ------------------------------------------------
            # Average squared hyperbolic distance
            # ------------------------------------------------

            batch_scores = (
                nearest
                .pow(2)
                .mean(dim=1)
            )

            scores.append(
                batch_scores.cpu()
            )

        if len(scores) == 0:

            return np.empty(
                0,
                dtype=np.float32,
            )

        return torch.cat(
            scores,
            dim=0,
        ).numpy()


    # ========================================================
    # PREDICT
    # ========================================================

    def predict(
        self,
        X,
        threshold,
    ):

        scores = self.score(
            X
        )

        threshold = float(
            threshold
        )

        return (
            scores >= threshold
        ).astype(
            np.int64
        )


    # ========================================================
    # TRANSFORM
    # ========================================================

    @torch.no_grad()
    def transform(
        self,
        X,
    ):

        if not self.fitted:

            raise RuntimeError(
                "Detector has not been fitted."
            )

        X = torch.as_tensor(
            X,
            dtype=torch.float32,
            device=self.device,
        )

        self.encoder.eval()

        self.deformation.eval()

        Z = self.encoder(
            X
        )

        Z = self.deformation(
            Z
        )

        return Z.cpu().numpy()


# ============================================================
# EXAMPLE
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Replace these with your own arrays
    # --------------------------------------------------------

    rng = np.random.default_rng(
        42
    )

    N_normal = 1000

    N_anomaly = 100

    N_features = 32

    X_normal = (
        rng.normal(
            0.0,
            1.0,
            size=(
                N_normal,
                N_features,
            ),
        )
        .astype(
            np.float32
        )
    )

    X_anomaly = (
        rng.normal(
            3.0,
            1.0,
            size=(
                N_anomaly,
                N_features,
            ),
        )
        .astype(
            np.float32
        )
    )

    X_test = np.concatenate(
        [
            X_normal[:100],
            X_anomaly[:50],
        ],
        axis=0,
    )

    # --------------------------------------------------------
    # Detector
    #
    # None / "auto":
    # CUDA if available, otherwise CPU
    # --------------------------------------------------------

    detector = (
        HyperbolicSelfDeformingDetector(
            latent_dim=16,
            curvature=1.0,
            k=10,
            max_deformation=0.2,
            device=None,
        )
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    detector.fit(

        X_normal=X_normal,

        X_anomaly=X_anomaly,

        epochs_encoder=50,

        epochs_deformation=50,

        learning_rate=1e-3,

        margin=2.0,

        lambda_geometry=1.0,

        lambda_separation=1.0,

        lambda_deformation=0.1,
    )

    # --------------------------------------------------------
    # Scores
    # --------------------------------------------------------

    scores = detector.score(
        X_test
    )

    print()
    print(
        "Anomaly scores:"
    )

    print(
        scores[:20]
    )

    # --------------------------------------------------------
    # Threshold based on normal training distribution
    # --------------------------------------------------------

    normal_scores = (
        detector.score(
            X_normal
        )
    )

    threshold = np.percentile(
        normal_scores,
        99,
    )

    print()
    print(
        "Threshold:",
        threshold,
    )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    predictions = (
        detector.predict(
            X_test,
            threshold,
        )
    )

    print()
    print(
        "Predictions:"
    )

    print(
        predictions
    )

    # --------------------------------------------------------
    # Hyperbolic latent representation
    # --------------------------------------------------------

    Z_test = (
        detector.transform(
            X_test
        )
    )

    print()
    print(
        "Hyperbolic representation shape:",
        Z_test.shape,
    )


#THIS VERSION IS BLOCK INDEPENDENT

# ============================================================
# Device
# ============================================================
"""
def resolve_device(device=None):

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
                "CUDA requested but unavailable."
            )

    return device

#Poincaré ball operations
def mobius_add(
    x,
    y,
    c=1.0,
    eps=1e-7,
):
    
    #Möbius addition on the Poincaré ball.
    

    x2 = (x * x).sum(
        dim=-1,
        keepdim=True,
    )

    y2 = (y * y).sum(
        dim=-1,
        keepdim=True,
    )

    xy = (x * y).sum(
        dim=-1,
        keepdim=True,
    )

    numerator = (
        (1 + 2 * c * xy + c * y2) * x
        +
        (1 - c * x2) * y
    )

    denominator = (
        1
        + 2 * c * xy
        + c**2 * x2 * y2
    )

    return numerator / (
        denominator.clamp_min(eps)
    )

def project_ball(
    x,
    c=1.0,
    eps=1e-5,
):
    
    #Project points into the valid Poincaré ball.
    

    max_norm = (
        1.0 / np.sqrt(c)
    ) - eps

    norm = torch.linalg.norm(
        x,
        dim=-1,
        keepdim=True,
    )

    scale = torch.clamp(
        max_norm / norm.clamp_min(eps),
        max=1.0,
    )

    return x * scale

#Hyperbolic distance
def mobius_neg(
    x,
):
    return -x


def poincare_distance(
    x,
    y,
    c=1.0,
    eps=1e-7,
):
    
    #Hyperbolic distance in the Poincaré ball.
    

    diff = mobius_add(
        -x,
        y,
        c=c,
    )

    norm = torch.linalg.norm(
        diff,
        dim=-1,
    )

    sqrt_c = np.sqrt(c)

    argument = (
        sqrt_c * norm
    ).clamp(
        max=1.0 - eps
    )

    distance = (
        2.0 / sqrt_c
    ) * torch.atanh(
        argument
    )

    return distance

#Hyperbolic encoder
class HyperbolicEncoder(nn.Module):

    def __init__(
        self,
        n_features,
        latent_dim=16,
        curvature=1.0,
    ):
        super().__init__()

        self.curvature = curvature

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

        z = self.net(x)

        # Map unconstrained Euclidean
        # output into the Poincaré ball.
        z = torch.tanh(z)

        z = (
            z
            / np.sqrt(self.curvature)
            * 0.95
        )

        return project_ball(
            z,
            c=self.curvature,
        )

#Hyperbolic deformation field
class HyperbolicDeformation(
    nn.Module
):

    def __init__(
        self,
        latent_dim,
        hidden_dim=64,
        max_deformation=0.2,
        curvature=1.0,
    ):
        super().__init__()

        self.max_deformation = (
            max_deformation
        )

        self.curvature = curvature

        self.net = nn.Sequential(

            nn.Linear(
                latent_dim,
                hidden_dim,
            ),

            nn.LayerNorm(hidden_dim),
            nn.GELU(),

            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),

            nn.GELU(),

            nn.Linear(
                hidden_dim,
                latent_dim,
            ),
        )

    def forward(self, z):

        delta = self.net(z)

        delta = torch.tanh(
            delta
        )

        delta = (
            self.max_deformation
            * delta
        )

        # Hyperbolic displacement
        z_new = mobius_add(
            z,
            delta,
            c=self.curvature,
        )

        return project_ball(
            z_new,
            c=self.curvature,
        )

def hyperbolic_separation_loss(
    z_anomaly,
    z_normal,
    margin=2.0,
    curvature=1.0,
):
    
    #Push anomalies away from the
    #closest normal points.
    

    distances = torch.cdist(
        z_anomaly,
        z_normal,
    )

    # This is only an approximation.
    # For true hyperbolic distance use
    # pairwise Poincaré distance below.

def pairwise_hyperbolic_distance(
    x,
    y,
    curvature=1.0,
):
    
    #Pairwise hyperbolic distance matrix.
    #x: [N, D]
    #y: [M, D]
    #returns:
    #    [N, M]
    

    x_exp = x[:, None, :]
    y_exp = y[None, :, :]

    diff = mobius_add(
        -x_exp,
        y_exp,
        c=curvature,
    )

    norm = torch.linalg.norm(
        diff,
        dim=-1,
    )

    sqrt_c = np.sqrt(curvature)

    argument = (
        sqrt_c * norm
    ).clamp(
        max=1.0 - 1e-7
    )

    return (
        2.0 / sqrt_c
    ) * torch.atanh(
        argument
    )

def hyperbolic_separation_loss(
    z_anomaly,
    z_normal,
    margin=2.0,
    curvature=1.0,
):

    distances = (
        pairwise_hyperbolic_distance(
            z_anomaly,
            z_normal,
            curvature=curvature,
        )
    )

    nearest_distance = (
        distances.min(
            dim=1
        ).values
    )

    loss = F.relu(
        margin - nearest_distance
    ).pow(2).mean()

    return loss

#Preserve normal hyperbolic geometry

def normal_geometry_loss(
    z_normal,
    z_normal_deformed,
    curvature=1.0,
):
    
    #Preserve pairwise hyperbolic geometry of normal samples.
    

    D_before = (
        pairwise_hyperbolic_distance(
            z_normal,
            z_normal,
            curvature=curvature,
        )
    )

    D_after = (
        pairwise_hyperbolic_distance(
            z_normal_deformed,
            z_normal_deformed,
            curvature=curvature,
        )
    )

    return F.mse_loss(
        D_after,
        D_before.detach(),
    )

#Deformation regularization

def deformation_regularization(
    z,
    z_deformed,
):
    delta = (
        z_deformed - z
    )

    return delta.pow(2).mean()

#Complete self-deformation objective

def total_hyperbolic_loss(
    z_normal,
    z_anomaly,
    z_normal_deformed,
    z_anomaly_deformed,
    curvature=1.0,
    margin=2.0,
    lambda_geometry=1.0,
    lambda_separation=1.0,
    lambda_deformation=0.1,
):

    # --------------------------------------------------------
    # Normal geometry
    # --------------------------------------------------------

    L_geometry = normal_geometry_loss(
        z_normal,
        z_normal_deformed,
        curvature=curvature,
    )

    # --------------------------------------------------------
    # Anomaly separation
    # --------------------------------------------------------

    L_separation = (
        hyperbolic_separation_loss(
            z_anomaly_deformed,
            z_normal_deformed,
            margin=margin,
            curvature=curvature,
        )
    )

    # --------------------------------------------------------
    # Regularization
    # --------------------------------------------------------

    L_deformation = (
        deformation_regularization(
            z_normal,
            z_normal_deformed,
        )
        +
        deformation_regularization(
            z_anomaly,
            z_anomaly_deformed,
        )
    )

    # --------------------------------------------------------
    # Total
    # --------------------------------------------------------

    loss = (

        lambda_geometry
        * L_geometry

        +

        lambda_separation
        * L_separation

        +

        lambda_deformation
        * L_deformation
    )

    return (
        loss,
        L_geometry,
        L_separation,
        L_deformation,
    )

#Training

class HyperbolicSelfDeformingDetector:

    def __init__(
        self,
        latent_dim=16,
        k=10,
        curvature=1.0,
        max_deformation=0.2,
        device=None,
    ):

        self.device = resolve_device(
            device
        )

        self.latent_dim = latent_dim
        self.k = k
        self.curvature = curvature

        self.encoder = None
        self.deformation = None

        self.Z_normal = None
        self.Z_normal_deformed = None

        print(
            f"Device: {self.device}"
        )

        if self.device.type == "cuda":

            print(
                "GPU:",
                torch.cuda.get_device_name(
                    self.device
                )
            )


    def fit(
        self,
        X_normal,
        X_anomaly,
        epochs_encoder=100,
        epochs_deformation=100,
        learning_rate=1e-3,
        margin=2.0,
        lambda_geometry=1.0,
        lambda_separation=1.0,
        lambda_deformation=0.1,
        ):

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

        n_features = Xn.shape[1]

        # ====================================================
        # Encoder
        # ====================================================

        self.encoder = (
            HyperbolicEncoder(
                n_features=n_features,
                latent_dim=self.latent_dim,
                curvature=self.curvature,
            )
            .to(self.device)
        )

        optimizer = torch.optim.AdamW(
            self.encoder.parameters(),
            lr=learning_rate,
            weight_decay=1e-4,
        )

        # ====================================================
        # Train normal geometry
        # ====================================================

        for epoch in range(
            epochs_encoder
        ):

            self.encoder.train()

            Z = self.encoder(Xn)

            # Euclidean proxy for local
            # neighbor preservation during
            # initial representation learning.
            distances = torch.cdist(
                Xn,
                Xn,
            )

            distances.fill_diagonal_(
                float("inf")
            )

            knn = torch.topk(
                distances,
                k=self.k,
                largest=False,
            ).indices

            Z_neighbors = Z[knn]

            local_loss = (
                Z.unsqueeze(1)
                - Z_neighbors
            ).pow(2).sum(
                dim=-1
            ).mean()

            optimizer.zero_grad(
                set_to_none=True
            )

            local_loss.backward()

            optimizer.step()

            if epoch % 10 == 0:

                print(
                    f"[Encoder] "
                    f"{epoch:4d}/"
                    f"{epochs_encoder} "
                    f"loss="
                    f"{local_loss.item():.6f}"
                )

        # ====================================================
        # Initial latent representation
        # ====================================================

        self.encoder.eval()

        with torch.no_grad():

            self.Z_normal = (
                self.encoder(Xn)
            )

            Z_anomaly = (
                self.encoder(Xa)
            )

        # ====================================================
        # Deformation field
        # ====================================================

        self.deformation = (
            HyperbolicDeformation(
                latent_dim=self.latent_dim,
                curvature=self.curvature,
            )
            .to(self.device)
        )

        optimizer = torch.optim.AdamW(
            self.deformation.parameters(),
            lr=learning_rate,
        )

        # Freeze encoder
        for p in self.encoder.parameters():
            p.requires_grad = False

        # ====================================================
        # Self deformation
        # ====================================================

        for epoch in range(
            epochs_deformation
        ):

            self.deformation.train()

            Z_normal_def = (
                self.deformation(
                    self.Z_normal
                )
            )

            Z_anomaly_def = (
                self.deformation(
                    Z_anomaly
                )
            )

            (
                loss,
                L_geo,
                L_sep,
                L_def,
            ) = total_hyperbolic_loss(
                self.Z_normal,
                Z_anomaly,
                Z_normal_def,
                Z_anomaly_def,
                curvature=self.curvature,
                margin=margin,
                lambda_geometry=lambda_geometry,
                lambda_separation=lambda_separation,
                lambda_deformation=lambda_deformation,
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
                    f"{epoch:4d}/"
                    f"{epochs_deformation} "
                    f"loss={loss.item():.6f} "
                    f"geo={L_geo.item():.4f} "
                    f"sep={L_sep.item():.4f}"
                )

        # ====================================================
        # Store final normal manifold
        # ====================================================

        self.deformation.eval()

        with torch.no_grad():

            self.Z_normal_deformed = (
                self.deformation(
                    self.Z_normal
                )
            )

        return self

    #Anomaly scoring
    @torch.no_grad()
    def score(self,
              X,
              ):

        if self.encoder is None:
            raise RuntimeError(
                "Detector has not been fitted."
            )

        X = torch.as_tensor(
            X,
            dtype=torch.float32,
            device=self.device,
        )

        self.encoder.eval()
        self.deformation.eval()

        Z = self.encoder(X)

        Z_def = self.deformation(Z)

        distances = (
            pairwise_hyperbolic_distance(
                Z_def,
                self.Z_normal_deformed,
                curvature=self.curvature,
            )
        )

        k = min(
            self.k,
            self.Z_normal_deformed.shape[0],
        )

        nearest = torch.topk(
            distances,
            k=k,
            largest=False,
        ).values

        scores = (
            nearest.pow(2)
            .mean(dim=1)
        )

        return scores.cpu().numpy()
"""