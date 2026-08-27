import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, TensorDataset


#****************************************************************************************
#****************************************************************************************
#TRANSOFRMER GAN ANOMALY DETECTOR

#   CLASSIFIER OBJECT

#two-stage semi-supervised anomaly framework:

#Transformer anomaly detector is trained only on normal samples.
#Freeze the Transformer.
#Train a GAN + contrastive representation network using both normal and labeled anomalous samples.
#The GAN generator is explicitly encouraged to produce samples with a high Transformer anomaly score, while the discriminator prevents it from generating arbitrary garbage.
#A contrastive loss makes normal/anomaly representations well separated.
#The final anomaly score combines the Transformer score and the learned contrastive score.

"""
  ┌─────────────────────────┐
                    │ Normal training signals │
                    └────────────┬────────────┘
                                 │
                                 ▼
                       Transformer Encoder
                                 │
                                 ▼
                        Normal representation
                                 │
                                 ▼
                       anomaly score S_T(x)
                                 │
                    ┌────────────┴────────────┐
                    │                         │
                    │       FROZEN            │
                    │                         │
                    └─────────────────────────┘


    Normal + labeled anomalies
              │
              ▼
      Contrastive Encoder
              │
              ▼
         embedding z
          /       \
         /         \
      normal      anomaly
         \         /
          \       /
           contrastive
              loss


        Normal samples
              │
              ▼
          Generator G
              │
              ▼
        synthetic anomaly
              │
       ┌──────┴────────┐
       ▼               ▼
Transformer score   GAN discriminator
       │               │
       │               │
       └───────┬───────┘
               ▼
        Generator objective

     maximize anomaly score
     + remain realistic
     + remain different from normal
     """

# ============================================================
# 1. Automatic CPU / CUDA selection
# ============================================================

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")

        print("=" * 60)
        print("Using CUDA / GPU")
        print("GPU:", torch.cuda.get_device_name(0))
        print("CUDA:", torch.version.cuda)
        print("=" * 60)

        # Faster matrix operations on modern NVIDIA GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    else:
        device = torch.device("cpu")

        print("=" * 60)
        print("CUDA not available -> using CPU")
        print("=" * 60)

    return device


DEVICE = get_device()


# ============================================================
# 2. Transformer one-class anomaly detector
#
# IMPORTANT:
# This model is trained ONLY using normal samples.
# ============================================================



class TransformerAnomalyDetector(nn.Module):

    def __init__(
        self,
        n_features,
        d_model=64,
        n_heads=4,
        n_layers=3,
        dim_feedforward=128,
        dropout=0.1,
    ):
        super().__init__()

        self.n_features = n_features

        # Each scalar feature becomes a Transformer token
        self.feature_embedding = nn.Linear(
            1,
            d_model,
        )

        # Learnable feature identity
        self.feature_id = nn.Parameter(
            torch.randn(
                1,
                n_features,
                d_model,
            )
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
        )

        # Reconstruction head
        self.decoder = nn.Sequential(
            nn.Linear(
                d_model,
                d_model,
            ),
            nn.GELU(),

            nn.Linear(
                d_model,
                1,
            ),
        )

    def encode(self, x):

        # x: [B, M]

        h = x.unsqueeze(-1)

        # [B, M, 1] -> [B, M, D]
        h = self.feature_embedding(h)

        h = h + self.feature_id

        h = self.encoder(h)

        return h

    def forward(self, x):

        h = self.encode(x)

        x_hat = self.decoder(h)

        return x_hat.squeeze(-1)

    def anomaly_score(self, x):

        x_hat = self.forward(x)

        # Reconstruction error for each sample
        score = (
            (x - x_hat) ** 2
        ).mean(dim=1)

        return score


# ============================================================
# 3. Train Transformer ONLY on normal data
# ============================================================

def train_transformer(
    X_normal,
    epochs=100,
    batch_size=512,
    learning_rate=1e-3,
):

    n_features = X_normal.shape[1]

    model = TransformerAnomalyDetector(
        n_features=n_features,
        d_model=64,
        n_heads=4,
        n_layers=3,
        dim_feedforward=128,
        dropout=0.1,
    ).to(DEVICE)

    dataset = TensorDataset(
        torch.tensor(
            X_normal,
            dtype=torch.float32,
        )
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=(DEVICE.type == "cuda"),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4,
    )

    for epoch in range(epochs):

        model.train()

        total_loss = 0.0
        total_samples = 0

        for (x,) in loader:

            x = x.to(
                DEVICE,
                non_blocking=(DEVICE.type == "cuda"),
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            x_hat = model(x)

            loss = F.mse_loss(
                x_hat,
                x,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                5.0,
            )

            optimizer.step()

            total_loss += (
                loss.item()
                * x.shape[0]
            )

            total_samples += x.shape[0]

        avg_loss = (
            total_loss
            / total_samples
        )

        if epoch % 10 == 0:

            print(
                f"[Transformer] "
                f"Epoch {epoch:4d}/{epochs} | "
                f"Loss = {avg_loss:.6f}"
            )

    return model


# ============================================================
# 4. Contrastive encoder
#
# Uses NORMAL + LABELED ANOMALY samples.
# ============================================================

class ContrastiveEncoder(nn.Module):

    def __init__(
        self,
        n_features,
        embedding_dim=64,
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
                embedding_dim,
            ),
        )

    def forward(self, x):

        z = self.net(x)

        return F.normalize(
            z,
            p=2,
            dim=-1,
        )


# ============================================================
# 5. Supervised contrastive loss
# ============================================================

def supervised_contrastive_loss(
    embeddings,
    labels,
    temperature=0.1,
):

    B = embeddings.shape[0]

    similarity = (
        embeddings @ embeddings.T
    ) / temperature

    # Remove self-similarity
    self_mask = torch.eye(
        B,
        dtype=torch.bool,
        device=embeddings.device,
    )

    similarity = similarity.masked_fill(
        self_mask,
        -1e9,
    )

    # Same class = positive pair
    positive_mask = (
        labels[:, None]
        ==
        labels[None, :]
    )

    positive_mask = (
        positive_mask
        & ~self_mask
    )

    log_prob = (
        similarity
        -
        torch.logsumexp(
            similarity,
            dim=1,
            keepdim=True,
        )
    )

    positive_count = (
        positive_mask.sum(dim=1)
    )

    valid = positive_count > 0

    loss = -(
        log_prob
        * positive_mask
    ).sum(dim=1)

    loss = loss / (
        positive_count + 1e-8
    )

    return loss[valid].mean()


# ============================================================
# 6. Generator
#
# Takes a NORMAL sample + noise and creates
# a synthetic anomaly.
# ============================================================

class AnomalyGenerator(nn.Module):

    def __init__(
        self,
        n_features,
        noise_dim=32,
    ):
        super().__init__()

        self.noise_dim = noise_dim

        self.net = nn.Sequential(

            nn.Linear(
                n_features + noise_dim,
                128,
            ),

            nn.LayerNorm(128),
            nn.GELU(),

            nn.Linear(
                128,
                128,
            ),

            nn.GELU(),

            nn.Linear(
                128,
                n_features,
            ),
        )

    def forward(
        self,
        x,
        noise,
    ):

        h = torch.cat(
            [x, noise],
            dim=1,
        )

        # Generate perturbation
        delta = self.net(h)

        # Synthetic anomaly
        x_fake = x + delta

        return x_fake


# ============================================================
# 7. GAN discriminator
# ============================================================

class AnomalyDiscriminator(nn.Module):

    def __init__(
        self,
        n_features,
    ):
        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(
                n_features,
                128,
            ),

            nn.LayerNorm(128),
            nn.LeakyReLU(0.2),

            nn.Linear(
                128,
                64,
            ),

            nn.LeakyReLU(0.2),

            nn.Linear(
                64,
                1,
            ),
        )

    def forward(self, x):

        return self.net(x)


# ============================================================
# 8. Train GAN + Contrastive model
# ============================================================

def train_gan_contrastive(
    transformer,
    X_normal,
    X_anomaly,

    epochs=100,
    batch_size=256,

    noise_dim=32,

    learning_rate=2e-4,

    lambda_anomaly=1.0,
    lambda_contrastive=1.0,
    lambda_perturb=0.05,
    lambda_embedding=1.0,
):

    n_features = X_normal.shape[1]

    # --------------------------------------------------------
    # Models
    # --------------------------------------------------------

    generator = AnomalyGenerator(
        n_features=n_features,
        noise_dim=noise_dim,
    ).to(DEVICE)

    discriminator = AnomalyDiscriminator(
        n_features=n_features,
    ).to(DEVICE)

    contrastive_encoder = ContrastiveEncoder(
        n_features=n_features,
        embedding_dim=64,
    ).to(DEVICE)

    # --------------------------------------------------------
    # Transformer is frozen
    # --------------------------------------------------------

    transformer.eval()

    for p in transformer.parameters():
        p.requires_grad = False

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    normal_dataset = TensorDataset(
        torch.tensor(
            X_normal,
            dtype=torch.float32,
        )
    )

    anomaly_dataset = TensorDataset(
        torch.tensor(
            X_anomaly,
            dtype=torch.float32,
        )
    )

    normal_loader = DataLoader(
        normal_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=(DEVICE.type == "cuda"),
    )

    anomaly_loader = DataLoader(
        anomaly_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=(DEVICE.type == "cuda"),
    )

    # --------------------------------------------------------
    # Optimizers
    # --------------------------------------------------------

    optimizer_G = torch.optim.AdamW(
        generator.parameters(),
        lr=learning_rate,
        betas=(0.5, 0.999),
    )

    optimizer_D = torch.optim.AdamW(
        discriminator.parameters(),
        lr=learning_rate,
        betas=(0.5, 0.999),
    )

    optimizer_C = torch.optim.AdamW(
        contrastive_encoder.parameters(),
        lr=learning_rate,
    )

    bce = nn.BCEWithLogitsLoss()

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    for epoch in range(epochs):

        generator.train()
        discriminator.train()
        contrastive_encoder.train()

        anomaly_iter = iter(
            anomaly_loader
        )

        for (x_normal,) in normal_loader:

            try:
                (
                    x_real_anomaly,
                ) = next(anomaly_iter)

            except StopIteration:

                anomaly_iter = iter(
                    anomaly_loader
                )

                (
                    x_real_anomaly,
                ) = next(anomaly_iter)

            # ------------------------------------------------
            # Move to CPU or CUDA automatically
            # ------------------------------------------------

            x_normal = x_normal.to(
                DEVICE,
                non_blocking=(
                    DEVICE.type == "cuda"
                ),
            )

            x_real_anomaly = (
                x_real_anomaly.to(
                    DEVICE,
                    non_blocking=(
                        DEVICE.type == "cuda"
                    ),
                )
            )

            B = x_normal.shape[0]

            # =================================================
            # A. Contrastive learning
            # =================================================

            optimizer_C.zero_grad(
                set_to_none=True
            )

            z_normal = contrastive_encoder(
                x_normal
            )

            z_anomaly = contrastive_encoder(
                x_real_anomaly
            )

            z_all = torch.cat(
                [
                    z_normal,
                    z_anomaly,
                ],
                dim=0,
            )

            labels = torch.cat(
                [
                    torch.zeros(
                        B,
                        device=DEVICE,
                    ),

                    torch.ones(
                        B,
                        device=DEVICE,
                    ),
                ],
                dim=0,
            ).long()

            loss_contrastive = (
                supervised_contrastive_loss(
                    z_all,
                    labels,
                )
            )

            loss_contrastive.backward()

            optimizer_C.step()

            # =================================================
            # B. Generate synthetic anomalies
            # =================================================

            noise = torch.randn(
                B,
                noise_dim,
                device=DEVICE,
            )

            x_fake = generator(
                x_normal,
                noise,
            )

            # =================================================
            # C. Train discriminator
            # =================================================

            optimizer_D.zero_grad(
                set_to_none=True
            )

            real_logits = discriminator(
                x_real_anomaly
            )

            fake_logits = discriminator(
                x_fake.detach()
            )

            loss_D_real = bce(
                real_logits,
                torch.ones_like(
                    real_logits
                ),
            )

            loss_D_fake = bce(
                fake_logits,
                torch.zeros_like(
                    fake_logits
                ),
            )

            loss_D = (
                loss_D_real
                + loss_D_fake
            )

            loss_D.backward()

            optimizer_D.step()

            # =================================================
            # D. Train generator
            # =================================================

            optimizer_G.zero_grad(
                set_to_none=True
            )

            noise = torch.randn(
                B,
                noise_dim,
                device=DEVICE,
            )

            x_fake = generator(
                x_normal,
                noise,
            )

            # -------------------------------------------------
            # GAN loss
            #
            # Generator wants to look like a REAL anomaly
            # -------------------------------------------------

            fake_logits = discriminator(
                x_fake
            )

            loss_G_gan = bce(
                fake_logits,
                torch.ones_like(
                    fake_logits
                ),
            )

            # -------------------------------------------------
            # Transformer anomaly loss
            #
            # We WANT high anomaly score.
            # Therefore negative score.
            # -------------------------------------------------

            transformer_score = (
                transformer.anomaly_score(
                    x_fake
                )
            )

            loss_G_anomaly = (
                -transformer_score.mean()
            )

            # -------------------------------------------------
            # Perturbation regularization
            #
            # Prevent completely arbitrary garbage.
            # -------------------------------------------------

            delta = (
                x_fake - x_normal
            )

            loss_G_perturb = (
                delta.pow(2).mean()
            )

            # -------------------------------------------------
            # Contrastive anomaly embedding
            #
            # Generated anomaly should be closer to
            # REAL anomaly embeddings than NORMAL embeddings.
            # -------------------------------------------------

            z_fake = contrastive_encoder(
                x_fake
            )

            # Real anomaly centroid
            with torch.no_grad():

                z_real_anomaly = (
                    contrastive_encoder(
                        x_real_anomaly
                    )
                )

                anomaly_centroid = (
                    z_real_anomaly.mean(
                        dim=0,
                        keepdim=True,
                    )
                )

                anomaly_centroid = F.normalize(
                    anomaly_centroid,
                    dim=-1,
                )

            fake_anomaly_similarity = (
                z_fake
                @ anomaly_centroid.T
            ).squeeze(1)

            loss_G_embedding = (
                -fake_anomaly_similarity.mean()
            )

            # -------------------------------------------------
            # Total generator loss
            # -------------------------------------------------

            loss_G = (

                loss_G_gan

                + lambda_anomaly
                * loss_G_anomaly

                + lambda_perturb
                * loss_G_perturb

                + lambda_embedding
                * loss_G_embedding
            )

            loss_G.backward()

            torch.nn.utils.clip_grad_norm_(
                generator.parameters(),
                5.0,
            )

            optimizer_G.step()

        # =====================================================
        # Logging
        # =====================================================

        if epoch % 10 == 0:

            print(
                f"[GAN] "
                f"Epoch {epoch:4d}/{epochs} | "
                f"D={loss_D.item():.4f} | "
                f"G={loss_G.item():.4f} | "
                f"Contrastive="
                f"{loss_contrastive.item():.4f} | "
                f"Transformer anomaly="
                f"{transformer_score.mean().item():.4f}"
            )

    return (
        generator,
        discriminator,
        contrastive_encoder,
    )

@torch.no_grad()
def get_normal_centroid(
    encoder,
    X_normal,
    batch_size=1024,
):

    encoder.eval()

    dataset = TensorDataset(
        torch.tensor(
            X_normal,
            dtype=torch.float32,
        )
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    embeddings = []

    for (x,) in loader:

        x = x.to(
            DEVICE,
            non_blocking=(
                DEVICE.type == "cuda"
            ),
        )

        z = encoder(x)

        embeddings.append(z)

    z = torch.cat(
        embeddings,
        dim=0,
    )

    centroid = z.mean(
        dim=0,
        keepdim=True,
    )

    centroid = F.normalize(
        centroid,
        dim=-1,
    )

    return centroid

@torch.no_grad()
def anomaly_score(
    transformer,
    contrastive_encoder,
    X,
    normal_centroid,
    alpha=0.7,
    batch_size=512,
):

    transformer.eval()
    contrastive_encoder.eval()

    dataset = TensorDataset(
        torch.tensor(
            X,
            dtype=torch.float32,
        )
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=(DEVICE.type == "cuda"),
    )

    scores = []

    for (x,) in loader:

        x = x.to(
            DEVICE,
            non_blocking=(
                DEVICE.type == "cuda"
            ),
        )

        # --------------------------------------------
        # Transformer score
        # --------------------------------------------

        s_transformer = (
            transformer.anomaly_score(x)
        )

        # --------------------------------------------
        # Contrastive score
        # --------------------------------------------

        z = contrastive_encoder(x)

        similarity = (
            z @ normal_centroid.T
        ).squeeze(1)

        s_contrastive = (
            1.0 - similarity
        )

        # --------------------------------------------
        # Combined score
        # --------------------------------------------

        score = (
            alpha * s_transformer
            + (1.0 - alpha)
            * s_contrastive
        )

        scores.append(
            score.cpu()
        )

    return torch.cat(
        scores
    ).numpy()


