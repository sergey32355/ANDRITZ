"""
The complete pipeline below does:

Accepts a 2D NumPy array.
Standardizes the features.
Trains a Transformer using masked-feature reconstruction.
Calculates per-feature anomaly scores.
Calibrates the anomaly threshold from normal training data.
Detects anomalies.
Creates an evidence JSON object.
Loads a pretrained Qwen instruct model.
Sends the evidence to the LLM.
Produces a human-readable explanation and structured JSON explanation.
Automatically selects NVIDIA CUDA → Apple MPS → CPU.


Transformer-based anomaly detector for data shaped:

    [num_samples, num_features, feature_dim]

For a single observation:

    X.shape == [N, M]

where:
    N = number of features
    M = dimensionality of each feature

The model:
    1. Projects each feature vector into transformer space
    2. Uses a Transformer encoder across features
    3. Learns to reconstruct masked features
    4. Uses reconstruction error as the primary anomaly signal
    5. Produces per-feature anomaly contributions
    6. Calibrates an anomaly threshold from normal validation data
    7. Creates an LLM-ready JSON explanation payload

Device selection:
    CUDA GPU -> Apple MPS -> CPU

To force a particular CUDA GPU:
    CUDA_DEVICE=1 python anomaly_detector.py

or change CUDA_DEVICE below.
"""
"""
===============================================================
COMPLETE TRANSFORMER + LLM ANOMALY DETECTION PIPELINE
===============================================================

INPUT FORMAT
------------

The input is a standard 2D NumPy array:

    X.shape = [num_samples, num_features]

Example:

    X.shape = (10000, 50)

where:

    10000 = observations
    50    = features

Each scalar feature becomes one Transformer token.

Pipeline:

    NumPy X
       |
       v
    Standardization
       |
       v
    Transformer
       |
       +---- masked feature reconstruction
       |
       v
    Per-feature reconstruction error
       |
       v
    Global anomaly score
       |
       v
    Threshold calibration
       |
       v
    Anomaly explanation JSON
       |
       v
    Pretrained LLM
       |
       v
    Human-readable explanation


INSTALL
-------

    pip install -U numpy torch transformers accelerate


LLM
---

Default:

    Qwen/Qwen3-4B-Instruct-2507

You can change MODEL_NAME below.


GPU
---

Automatic:

    CUDA GPU -> Apple MPS -> CPU

To use a particular NVIDIA GPU:

    CUDA_DEVICE=1 python anomaly_pipeline.py

or:

    CUDA_DEVICE=0 python anomaly_pipeline.py
"""

"""
====================================================================
COMPLETE TABULAR ANOMALY DETECTION + LLM EXPLANATION PIPELINE
====================================================================

INPUT
-----

X_train:
    2D NumPy array containing NORMAL observations.

    Shape:
        [num_samples, num_features]

X_test:
    2D NumPy array containing observations to analyze.

    Shape:
        [num_samples, num_features]


ARCHITECTURE
------------

Each scalar feature is treated as a Transformer token.

Example:

    X.shape = (10000, 50)

    sample:
        [f1, f2, f3, ..., f50]

becomes:

    token_1 = f1
    token_2 = f2
    token_3 = f3
    ...
    token_50 = f50

The Transformer learns to reconstruct masked features.

Anomaly score:

    mean(
        (observed_feature - reconstructed_feature)^2
    )

The threshold is calibrated from normal training data.

The LLM receives ONLY the detector evidence and explains
the result. The LLM does NOT calculate the anomaly score.


INSTALL
-------

    pip install -U numpy torch transformers accelerate


DEFAULT LLM
-----------

    Qwen/Qwen3-4B-Instruct-2507


GPU
---

Automatic selection:

    NVIDIA CUDA
        |
        v
    Apple MPS
        |
        v
    CPU

To select a specific NVIDIA GPU:

    CUDA_DEVICE=1 python Transformer_LLM.py
"""


import os
import json
import numpy as np

import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)


# ====================================================================
# 1. REPRODUCIBILITY
# ====================================================================

SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ====================================================================
# 2. DEVICE SELECTION
# ====================================================================

def select_device():

    # ---------------------------------------------------------------
    # NVIDIA CUDA
    # ---------------------------------------------------------------

    if torch.cuda.is_available():

        gpu_id = int(
            os.environ.get(
                "CUDA_DEVICE",
                "0",
            )
        )

        num_gpus = torch.cuda.device_count()

        if gpu_id >= num_gpus:

            raise ValueError(
                f"CUDA_DEVICE={gpu_id}, "
                f"but only {num_gpus} GPU(s) are available."
            )

        print(
            f"Using CUDA GPU {gpu_id}: "
            f"{torch.cuda.get_device_name(gpu_id)}"
        )

        return torch.device(
            f"cuda:{gpu_id}"
        )

    # ---------------------------------------------------------------
    # Apple Silicon MPS
    # ---------------------------------------------------------------

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):

        print(
            "Using Apple MPS GPU"
        )

        return torch.device(
            "mps"
        )

    # ---------------------------------------------------------------
    # CPU
    # ---------------------------------------------------------------

    print(
        "Using CPU"
    )

    return torch.device(
        "cpu"
    )


DEVICE = select_device()


# ====================================================================
# 3. FEATURE STANDARDIZER
# ====================================================================

class FeatureStandardizer:

    """
    Standardizes every feature independently.

        x_scaled = (x - mean) / std

    IMPORTANT:
        Fit ONLY on normal training data.
    """

    def __init__(self):

        self.mean = None
        self.std = None


    def fit(self, X):

        X = np.asarray(
            X,
            dtype=np.float32,
        )

        if X.ndim != 2:

            raise ValueError(
                "X must be a 2D NumPy array "
                "with shape [samples, features]."
            )

        self.mean = X.mean(
            axis=0,
            keepdims=True,
        )

        self.std = X.std(
            axis=0,
            keepdims=True,
        )

        # Prevent division by zero for constant features.

        self.std[
            self.std < 1e-8
        ] = 1.0

        return self


    def transform(self, X):

        X = np.asarray(
            X,
            dtype=np.float32,
        )

        if X.ndim != 2:

            raise ValueError(
                "X must be a 2D NumPy array "
                "with shape [samples, features]."
            )

        return (
            X - self.mean
        ) / self.std


    def fit_transform(self, X):

        self.fit(X)

        return self.transform(X)


# ====================================================================
# 4. TRANSFORMER ANOMALY DETECTOR
# ====================================================================

class TransformerAnomalyDetector(
    nn.Module
):

    """
    Transformer-based anomaly detector for tabular data.

    Input:

        x.shape = [batch, num_features]

    Each scalar feature is converted into a Transformer token.

    The model is trained using masked feature reconstruction.
    """

    def __init__(
        self,

        num_features,

        d_model=128,

        nhead=8,

        num_layers=4,

        dim_feedforward=512,

        dropout=0.1,
    ):

        super().__init__()

        self.num_features = num_features

        self.d_model = d_model

        # ------------------------------------------------------------
        # Scalar feature -> embedding
        # ------------------------------------------------------------

        self.feature_projection = nn.Sequential(

            nn.Linear(
                1,
                d_model,
            ),

            nn.LayerNorm(
                d_model
            ),

            nn.GELU(),
        )

        # ------------------------------------------------------------
        # Feature identity embedding
        #
        # This tells the Transformer which feature a token
        # represents.
        # ------------------------------------------------------------

        self.feature_embedding = nn.Embedding(

            num_features,

            d_model,
        )

        # ------------------------------------------------------------
        # CLS token representing the whole observation
        # ------------------------------------------------------------

        self.cls_token = nn.Parameter(

            torch.randn(
                1,
                1,
                d_model,
            ) * 0.02
        )

        # ------------------------------------------------------------
        # Transformer encoder
        # ------------------------------------------------------------

        encoder_layer = (
            nn.TransformerEncoderLayer(

                d_model=d_model,

                nhead=nhead,

                dim_feedforward=dim_feedforward,

                dropout=dropout,

                activation="gelu",

                batch_first=True,

                norm_first=True,
            )
        )

        self.encoder = (
            nn.TransformerEncoder(

                encoder_layer,

                num_layers=num_layers,
            )
        )

        self.final_norm = nn.LayerNorm(
            d_model
        )

        # ------------------------------------------------------------
        # Reconstruction head
        # ------------------------------------------------------------

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

        # ------------------------------------------------------------
        # Optional global anomaly head.
        #
        # This is not used for unsupervised training.
        # It can be useful later if labeled anomalies are available.
        # ------------------------------------------------------------

        self.anomaly_head = nn.Sequential(

            nn.Linear(
                d_model,
                d_model // 2,
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                d_model // 2,
                1,
            ),
        )


    def forward(
        self,
        x,
        mask=None,
    ):

        """
        Parameters
        ----------

        x:
            [batch, num_features]

        mask:
            [batch, num_features]

            True means the feature is masked.
        """

        batch_size, num_features = (
            x.shape
        )

        if num_features != self.num_features:

            raise ValueError(
                f"Expected {self.num_features} "
                f"features but received "
                f"{num_features}."
            )

        # ------------------------------------------------------------
        # [B, N]
        #
        # ->
        #
        # [B, N, 1]
        # ------------------------------------------------------------

        x_tokens = (
            x.unsqueeze(-1)
        )

        # ------------------------------------------------------------
        # Scalar -> d_model
        #
        # [B, N, 1]
        #
        # ->
        #
        # [B, N, d_model]
        # ------------------------------------------------------------

        tokens = (
            self.feature_projection(
                x_tokens
            )
        )

        # ------------------------------------------------------------
        # Feature identity embeddings
        # ------------------------------------------------------------

        feature_ids = torch.arange(

            num_features,

            device=x.device,
        )

        feature_embeddings = (
            self.feature_embedding(
                feature_ids
            )
        )

        tokens = (
            tokens
            + feature_embeddings.unsqueeze(0)
        )

        # ------------------------------------------------------------
        # Mask selected features
        # ------------------------------------------------------------

        if mask is not None:

            tokens = tokens.masked_fill(

                mask.unsqueeze(-1),

                0.0,
            )

        # ------------------------------------------------------------
        # Add CLS token
        # ------------------------------------------------------------

        cls = (
            self.cls_token.expand(
                batch_size,
                -1,
                -1,
            )
        )

        tokens = torch.cat(

            [
                cls,
                tokens,
            ],

            dim=1,
        )

        # ------------------------------------------------------------
        # Transformer
        # ------------------------------------------------------------

        encoded = (
            self.encoder(
                tokens
            )
        )

        encoded = (
            self.final_norm(
                encoded
            )
        )

        # ------------------------------------------------------------
        # Global observation representation
        # ------------------------------------------------------------

        cls_embedding = (
            encoded[
                :,
                0,
            ]
        )

        # ------------------------------------------------------------
        # Feature representations
        # ------------------------------------------------------------

        feature_embeddings = (
            encoded[
                :,
                1:,
            ]
        )

        # ------------------------------------------------------------
        # Reconstruct features
        # ------------------------------------------------------------

        reconstruction = (
            self.decoder(
                feature_embeddings
            )
            .squeeze(-1)
        )

        # ------------------------------------------------------------
        # Optional global anomaly output
        # ------------------------------------------------------------

        anomaly_logit = (
            self.anomaly_head(
                cls_embedding
            )
            .squeeze(-1)
        )

        return {

            "reconstruction":
                reconstruction,

            "anomaly_logit":
                anomaly_logit,

            "embedding":
                cls_embedding,

            "feature_embeddings":
                feature_embeddings,
        }


# ====================================================================
# 5. RANDOM FEATURE MASKING
# ====================================================================

def random_feature_mask(
    batch_size,
    num_features,
    mask_ratio,
    device,
):

    mask = (
        torch.rand(

            batch_size,

            num_features,

            device=device,
        )
        < mask_ratio
    )

    # Guarantee at least one masked feature per observation.

    for i in range(batch_size):

        if not mask[i].any():

            random_index = torch.randint(

                0,

                num_features,

                (1,),

                device=device,
            )

            mask[
                i,
                random_index
            ] = True

    return mask


# ====================================================================
# 6. MASKED RECONSTRUCTION LOSS
# ====================================================================

def masked_reconstruction_loss(
    reconstruction,
    target,
    mask,
):

    # ---------------------------------------------------------------
    # Per-feature squared reconstruction error
    # ---------------------------------------------------------------

    feature_error = (
        reconstruction - target
    ) ** 2

    # ---------------------------------------------------------------
    # Train only on masked features.
    # ---------------------------------------------------------------

    masked_error = (
        feature_error[mask]
    )

    if masked_error.numel() == 0:

        return feature_error.mean()

    return masked_error.mean()


# ====================================================================
# 7. TRAIN TRANSFORMER
# ====================================================================

def train_model(
    model,
    X_train,

    epochs=30,

    batch_size=128,

    learning_rate=1e-4,

    weight_decay=1e-4,

    mask_ratio=0.15,
):

    X_train = np.asarray(
        X_train,
        dtype=np.float32,
    )

    if X_train.ndim != 2:

        raise ValueError(
            "X_train must be 2D."
        )

    dataset = TensorDataset(

        torch.from_numpy(
            X_train
        )
    )

    loader = DataLoader(

        dataset,

        batch_size=batch_size,

        shuffle=True,

        pin_memory=(
            DEVICE.type == "cuda"
        ),
    )

    optimizer = torch.optim.AdamW(

        model.parameters(),

        lr=learning_rate,

        weight_decay=weight_decay,
    )

    print()
    print(
        "Starting Transformer training..."
    )

    for epoch in range(
        epochs
    ):

        model.train()

        total_loss = 0.0

        for (
            x,
        ) in loader:

            x = x.to(

                DEVICE,

                non_blocking=True,
            )

            mask = (
                random_feature_mask(

                    batch_size=x.size(0),

                    num_features=x.size(1),

                    mask_ratio=mask_ratio,

                    device=DEVICE,
                )
            )

            output = (
                model(
                    x,
                    mask=mask,
                )
            )

            loss = (
                masked_reconstruction_loss(

                    output[
                        "reconstruction"
                    ],

                    x,

                    mask,
                )
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(

                model.parameters(),

                max_norm=1.0,
            )

            optimizer.step()

            total_loss += (
                loss.item()
            )

        avg_loss = (
            total_loss
            / len(loader)
        )

        print(
            f"Epoch "
            f"{epoch + 1:03d}/{epochs} "
            f"| loss = "
            f"{avg_loss:.6f}"
        )

    print(
        "Training complete."
    )


# ====================================================================
# 8. COMPUTE ANOMALY SCORES
# ====================================================================

@torch.no_grad()
def compute_anomaly_scores(
    model,
    X,
    batch_size=256,
):

    model.eval()

    X = np.asarray(
        X,
        dtype=np.float32,
    )

    if X.ndim == 1:

        X = X.reshape(
            1,
            -1,
        )

    if X.ndim != 2:

        raise ValueError(
            "X must be 2D."
        )

    tensor = torch.from_numpy(
        X
    )

    loader = DataLoader(

        TensorDataset(
            tensor
        ),

        batch_size=batch_size,

        shuffle=False,
    )

    all_scores = []

    all_feature_errors = []

    all_reconstructions = []

    all_embeddings = []

    for (
        x,
    ) in loader:

        x = x.to(
            DEVICE
        )

        output = model(
            x
        )

        reconstruction = (
            output[
                "reconstruction"
            ]
        )

        # -----------------------------------------------------------
        # Feature-level reconstruction error
        # -----------------------------------------------------------

        feature_errors = (
            reconstruction - x
        ) ** 2

        # -----------------------------------------------------------
        # Global anomaly score
        # -----------------------------------------------------------

        scores = (
            feature_errors.mean(
                dim=1
            )
        )

        all_scores.append(
            scores.cpu()
        )

        all_feature_errors.append(
            feature_errors.cpu()
        )

        all_reconstructions.append(
            reconstruction.cpu()
        )

        all_embeddings.append(
            output[
                "embedding"
            ].cpu()
        )

    return {

        "score":
            torch.cat(
                all_scores
            ),

        "feature_errors":
            torch.cat(
                all_feature_errors
            ),

        "reconstruction":
            torch.cat(
                all_reconstructions
            ),

        "embedding":
            torch.cat(
                all_embeddings
            ),
    }


# ====================================================================
# 9. CALIBRATE THRESHOLD
# ====================================================================

def calibrate_threshold(
    model,
    X_normal,
    percentile=99.5,
):

    result = (
        compute_anomaly_scores(

            model,

            X_normal,
        )
    )

    scores = (
        result[
            "score"
        ]
    )

    threshold = torch.quantile(

        scores,

        percentile / 100.0,
    )

    print()
    print(
        f"Normal score mean: "
        f"{scores.mean().item():.6f}"
    )

    print(
        f"Normal score std:  "
        f"{scores.std().item():.6f}"
    )

    print(
        f"Threshold "
        f"({percentile} percentile): "
        f"{threshold.item():.6f}"
    )

    return threshold.item()


# ====================================================================
# 10. CREATE DETECTOR EVIDENCE FOR LLM
# ====================================================================

def create_explanation(
    result,
    sample_index,
    threshold,

    feature_names=None,

    original_values=None,

    top_k=10,
):

    score = float(

        result[
            "score"
        ][sample_index]
    )

    feature_errors = (

        result[
            "feature_errors"
        ][sample_index]
    )

    num_features = len(
        feature_errors
    )

    k = min(
        top_k,
        num_features,
    )

    values, indices = torch.topk(

        feature_errors,

        k=k,
    )

    top_features = []

    for (
        feature_idx,
        error,
    ) in zip(

        indices.tolist(),

        values.tolist(),
    ):

        # -----------------------------------------------------------
        # Feature name
        # -----------------------------------------------------------

        if feature_names is not None:

            feature_name = (
                feature_names[
                    feature_idx
                ]
            )

        else:

            feature_name = (
                f"feature_{feature_idx}"
            )

        # -----------------------------------------------------------
        # Reconstructed value
        # -----------------------------------------------------------

        reconstructed = float(

            result[
                "reconstruction"
            ][
                sample_index,
                feature_idx
            ]
        )

        item = {

            "feature":
                feature_name,

            "feature_index":
                feature_idx,

            "reconstruction_error":
                float(error),

            "reconstructed_value":
                reconstructed,
        }

        # -----------------------------------------------------------
        # Original observed value
        # -----------------------------------------------------------

        if original_values is not None:

            item[
                "observed_value"
            ] = float(

                original_values[
                    sample_index,
                    feature_idx
                ]
            )

        top_features.append(
            item
        )

    # ---------------------------------------------------------------
    # Relative contribution
    # ---------------------------------------------------------------

    total_error = (
        feature_errors.sum().item()
    )

    for item in top_features:

        if total_error > 0:

            item[
                "relative_contribution"
            ] = (

                item[
                    "reconstruction_error"
                ]

                / total_error
            )

        else:

            item[
                "relative_contribution"
            ] = 0.0

    return {

        "anomaly_score":
            score,

        "threshold":
            float(threshold),

        "distance_above_threshold":
            float(
                score - threshold
            ),

        "is_anomaly":
            bool(
                score > threshold
            ),

        "top_contributing_features":
            top_features,
    }


# ====================================================================
# 11. LLM EXPLAINER
# ====================================================================

class AnomalyExplainer:

    """
    Uses a pretrained instruction-following LLM to turn
    detector evidence into a human-readable explanation.
    """

    def __init__(

        self,

        model_name=(
            "Qwen/Qwen3-4B-Instruct-2507"
        ),

        max_new_tokens=400,
    ):

        self.model_name = (
            model_name
        )

        self.max_new_tokens = (
            max_new_tokens
        )

        print()
        print(
            "Loading LLM:"
        )

        print(
            model_name
        )

        # -----------------------------------------------------------
        # Tokenizer
        # -----------------------------------------------------------

        self.tokenizer = (
            AutoTokenizer.from_pretrained(
                model_name
            )
        )

        # -----------------------------------------------------------
        # Model
        #
        # device_map="auto" allows Accelerate to place the
        # model automatically.
        # -----------------------------------------------------------

        self.model = (
            AutoModelForCausalLM.from_pretrained(

                model_name,

                torch_dtype="auto",

                device_map="auto",
            )
        )

        self.model.eval()

        print(
            "LLM loaded."
        )


    # =================================================================
    # SYSTEM PROMPT
    # =================================================================

    @staticmethod
    def system_prompt():

        # -------------------------------------------------------------
        # IMPORTANT:
        #
        # This is deliberately NOT a triple-quoted string.
        #
        # This avoids the unterminated triple-quoted f-string
        # problem from the previous version.
        # -------------------------------------------------------------

        return (
            "You are an anomaly-analysis assistant helping "
            "an engineer understand the output of a "
            "machine-learning anomaly detector.\n\n"

            "Your job is to explain the detector's numerical "
            "evidence.\n\n"

            "IMPORTANT RULES:\n\n"

            "1. The anomaly detector determines whether an "
            "observation is anomalous. Do not override its "
            "decision.\n\n"

            "2. Use ONLY the information contained in the "
            "detector output.\n\n"

            "3. Do NOT invent a root cause.\n\n"

            "4. Do NOT claim that a feature caused the anomaly "
            "unless causal evidence is explicitly provided.\n\n"

            "5. A high reconstruction error means that the "
            "observed feature is inconsistent with patterns "
            "learned from normal observations.\n\n"

            "6. Distinguish between observed evidence, "
            "interpretation, and possible hypotheses.\n\n"

            "7. If the evidence is insufficient to determine "
            "the cause, explicitly say so.\n\n"

            "8. Focus on the most important contributing "
            "features.\n\n"

            "9. Be concise and technically precise.\n\n"

            "10. Never invent feature values, statistics, "
            "relationships, or domain-specific information."
        )


    # =================================================================
    # BUILD TEXT PROMPT
    # =================================================================

    @staticmethod
    def build_prompt(
        explanation_payload,
    ):

        # -------------------------------------------------------------
        # Convert detector output into JSON.
        # -------------------------------------------------------------

        detector_json = json.dumps(

            explanation_payload,

            indent=2,

            ensure_ascii=False,
        )

        # -------------------------------------------------------------
        # IMPORTANT:
        #
        # No triple-quoted f-string.
        #
        # JSON is simply concatenated into the prompt.
        # -------------------------------------------------------------

        prompt = (

            "Analyze the following anomaly detector result.\n\n"

            "DETECTOR OUTPUT:\n\n"

            "JSON:\n"

            + detector_json

            + "\n\n"

            "Provide your answer using exactly this structure:\n\n"

            "Anomaly status:\n"

            "<normal or anomaly>\n\n"

            "Summary:\n"

            "<one or two concise sentences>\n\n"

            "Main contributing features:\n"

            "- <feature>: <explanation>\n"

            "- <feature>: <explanation>\n"

            "- ...\n\n"

            "Interpretation:\n"

            "<what the detector evidence suggests>\n\n"

            "Limitations:\n"

            "<what cannot be concluded from this evidence>\n\n"

            "IMPORTANT:\n"

            "Do not invent a physical or business root cause.\n"

            "Use only information provided by the detector."
        )

        return prompt


    # =================================================================
    # GENERATE TEXT EXPLANATION
    # =================================================================

    @torch.no_grad()
    def explain(

        self,

        explanation_payload,

        temperature=0.2,
    ):

        user_prompt = (
            self.build_prompt(
                explanation_payload
            )
        )

        messages = [

            {
                "role":
                    "system",

                "content":
                    self.system_prompt(),
            },

            {
                "role":
                    "user",

                "content":
                    user_prompt,
            },
        ]

        # -------------------------------------------------------------
        # Use the model's native chat template.
        # -------------------------------------------------------------

        inputs = (
            self.tokenizer.apply_chat_template(

                messages,

                add_generation_prompt=True,

                tokenize=True,

                return_dict=True,

                return_tensors="pt",
            )
        )

        # -------------------------------------------------------------
        # Put input tensors on the model's device.
        # -------------------------------------------------------------

        inputs = {

            key:
                value.to(
                    self.model.device
                )

            for key, value
            in inputs.items()
        }

        # -------------------------------------------------------------
        # Generate response.
        # -------------------------------------------------------------

        outputs = (
            self.model.generate(

                **inputs,

                max_new_tokens=(
                    self.max_new_tokens
                ),

                do_sample=True,

                temperature=temperature,

                top_p=0.9,

                repetition_penalty=1.05,
            )
        )

        # -------------------------------------------------------------
        # Remove prompt tokens from generated output.
        # -------------------------------------------------------------

        input_length = (
            inputs[
                "input_ids"
            ].shape[-1]
        )

        generated_tokens = (
            outputs[
                0,
                input_length:
            ]
        )

        response = (
            self.tokenizer.decode(

                generated_tokens,

                skip_special_tokens=True,
            )
        )

        return response.strip()


    # =================================================================
    # JSON SYSTEM PROMPT
    # =================================================================

    @staticmethod
    def json_system_prompt():

        return (

            "You are an anomaly-analysis assistant.\n\n"

            "Convert machine-learning anomaly detector "
            "evidence into a concise structured explanation.\n\n"

            "STRICT RULES:\n\n"

            "- Use ONLY the supplied detector evidence.\n"

            "- Never invent a root cause.\n"

            "- Never invent feature values.\n"

            "- Never invent statistics.\n"

            "- Never claim causality without evidence.\n"

            '- If the cause is unknown, explicitly say "unknown".\n\n'

            "Return ONLY valid JSON.\n\n"

            "Required schema:\n"

            "{\n"

            '  "status": "anomaly" or "normal",\n'

            '  "summary": "string",\n'

            '  "evidence": ["string"],\n'

            '  "interpretation": "string",\n'

            '  "limitations": "string"\n'

            "}"
        )


    # =================================================================
    # BUILD JSON PROMPT
    # =================================================================

    @staticmethod
    def build_json_prompt(
        explanation_payload,
    ):

        detector_json = json.dumps(

            explanation_payload,

            indent=2,

            ensure_ascii=False,
        )

        # -------------------------------------------------------------
        # Again, deliberately avoid triple-quoted f-strings.
        # -------------------------------------------------------------

        prompt = (

            "Convert the following anomaly detector output "

            "into the required JSON format.\n\n"

            "DETECTOR OUTPUT:\n\n"

            "JSON:\n"

            + detector_json

            + "\n\n"

            "Return ONLY valid JSON.\n\n"

            "Required schema:\n"

            "{\n"

            '  "status": "anomaly" or "normal",\n'

            '  "summary": "string",\n'

            '  "evidence": ["string"],\n'

            '  "interpretation": "string",\n'

            '  "limitations": "string"\n'

            "}\n\n"

            "Do not add Markdown.\n"

            "Do not add ```json.\n"

            "Do not add text outside the JSON.\n"

            "Do not invent a root cause."
        )

        return prompt


    # =================================================================
    # GENERATE STRUCTURED JSON EXPLANATION
    # =================================================================

    @torch.no_grad()
    def explain_json(

        self,

        explanation_payload,
    ):

        user_prompt = (
            self.build_json_prompt(
                explanation_payload
            )
        )

        messages = [

            {
                "role":
                    "system",

                "content":
                    self.json_system_prompt(),
            },

            {
                "role":
                    "user",

                "content":
                    user_prompt,
            },
        ]

        # -------------------------------------------------------------
        # Tokenize
        # -------------------------------------------------------------

        inputs = (
            self.tokenizer.apply_chat_template(

                messages,

                add_generation_prompt=True,

                tokenize=True,

                return_dict=True,

                return_tensors="pt",
            )
        )

        inputs = {

            key:
                value.to(
                    self.model.device
                )

            for key, value
            in inputs.items()
        }

        # -------------------------------------------------------------
        # Deterministic generation
        # -------------------------------------------------------------

        outputs = (
            self.model.generate(

                **inputs,

                max_new_tokens=500,

                do_sample=False,
            )
        )

        # -------------------------------------------------------------
        # Remove prompt tokens
        # -------------------------------------------------------------

        input_length = (
            inputs[
                "input_ids"
            ].shape[-1]
        )

        generated_tokens = (
            outputs[
                0,
                input_length:
            ]
        )

        response = (
            self.tokenizer.decode(

                generated_tokens,

                skip_special_tokens=True,
            )
        ).strip()

        # -------------------------------------------------------------
        # Remove accidental Markdown fences.
        # -------------------------------------------------------------

        if response.startswith(
            "```json"
        ):

            response = response[
                len("```json"):
            ].strip()

            if response.endswith(
                "```"
            ):

                response = response[
                    :-3
                ].strip()

        elif response.startswith(
            "```"
        ):

            response = response[
                len("```"):
            ].strip()

            if response.endswith(
                "```"
            ):

                response = response[
                    :-3
                ].strip()

        # -------------------------------------------------------------
        # Parse JSON
        # -------------------------------------------------------------

        try:

            return json.loads(
                response
            )

        except json.JSONDecodeError:

            print(
                "WARNING: LLM did not return valid JSON."
            )

            return {

                "raw_response":
                    response
            }


# ====================================================================
# 12. COMPLETE PIPELINE
# ====================================================================

def run_pipeline(

    X_train,

    X_test=None,

    feature_names=None,

    transformer_epochs=30,

    threshold_percentile=99.5,

    top_k=10,

    load_llm=True,
):

    """
    Complete pipeline.

    Parameters
    ----------

    X_train:
        Normal training data.

        Shape:
            [samples, features]

    X_test:
        Data to evaluate.

        Shape:
            [samples, features]

    feature_names:
        Optional list containing one name per feature.

    transformer_epochs:
        Number of Transformer training epochs.

    threshold_percentile:
        Percentile of normal scores used to determine
        the anomaly threshold.

    top_k:
        Number of most important features sent to the LLM.

    load_llm:
        Whether to load the LLM and generate explanations.
    """

    # =================================================================
    # VALIDATE TRAINING DATA
    # =================================================================

    X_train = np.asarray(

        X_train,

        dtype=np.float32,
    )

    if X_train.ndim != 2:

        raise ValueError(

            "X_train must be a 2D NumPy array "
            "[samples, features]."
        )

    num_samples, num_features = (
        X_train.shape
    )

    print()
    print(
        f"Training data shape: "
        f"{X_train.shape}"
    )

    # =================================================================
    # FEATURE NAMES
    # =================================================================

    if feature_names is None:

        feature_names = [

            f"feature_{i}"

            for i in range(
                num_features
            )
        ]

    if len(feature_names) != num_features:

        raise ValueError(

            "feature_names must contain "
            "one name per feature."
        )

    # =================================================================
    # STANDARDIZATION
    # =================================================================

    scaler = (
        FeatureStandardizer()
    )

    X_train_scaled = (
        scaler.fit_transform(
            X_train
        )
    )

    # =================================================================
    # CREATE TRANSFORMER
    # =================================================================

    transformer = (
        TransformerAnomalyDetector(

            num_features=num_features,

            d_model=128,

            nhead=8,

            num_layers=4,

            dim_feedforward=512,

            dropout=0.1,
        )
    ).to(
        DEVICE
    )

    number_parameters = sum(

        p.numel()

        for p in transformer.parameters()
    )

    print(
        f"Transformer parameters: "
        f"{number_parameters:,}"
    )

    # =================================================================
    # TRAIN
    # =================================================================

    train_model(

        model=transformer,

        X_train=X_train_scaled,

        epochs=transformer_epochs,

        batch_size=128,

        learning_rate=1e-4,

        weight_decay=1e-4,

        mask_ratio=0.15,
    )

    # =================================================================
    # CALIBRATE THRESHOLD
    # =================================================================

    threshold = (
        calibrate_threshold(

            model=transformer,

            X_normal=X_train_scaled,

            percentile=threshold_percentile,
        )
    )

    # =================================================================
    # IF NO TEST DATA
    # =================================================================

    if X_test is None:

        return {

            "model":
                transformer,

            "scaler":
                scaler,

            "threshold":
                threshold,

            "feature_names":
                feature_names,
        }

    # =================================================================
    # VALIDATE TEST DATA
    # =================================================================

    X_test = np.asarray(

        X_test,

        dtype=np.float32,
    )

    if X_test.ndim != 2:

        raise ValueError(

            "X_test must be a 2D NumPy array."
        )

    if X_test.shape[1] != num_features:

        raise ValueError(

            "X_test must have the same "
            "number of features as X_train."
        )

    # =================================================================
    # SCALE TEST DATA
    # =================================================================

    X_test_scaled = (
        scaler.transform(
            X_test
        )
    )

    # =================================================================
    # SCORE TEST DATA
    # =================================================================

    result = (
        compute_anomaly_scores(

            model=transformer,

            X=X_test_scaled,
        )
    )

    scores = (
        result[
            "score"
        ].numpy()
    )

    anomaly_flags = (
        scores > threshold
    )

    # =================================================================
    # PRINT DETECTION RESULTS
    # =================================================================

    print()
    print(
        "=" * 70
    )

    print(
        "ANOMALY DETECTION RESULTS"
    )

    print(
        "=" * 70
    )

    for i, score in enumerate(
        scores
    ):

        status = (

            "ANOMALY"

            if anomaly_flags[i]

            else "normal"
        )

        print(

            f"Sample {i:04d}: "

            f"score={score:.6f} "

            f"threshold={threshold:.6f} "

            f"-> {status}"
        )

    # =================================================================
    # LOAD LLM
    # =================================================================

    explainer = None

    if load_llm:

        explainer = (
            AnomalyExplainer(

                model_name=(
                    "Qwen/Qwen3-4B-Instruct-2507"
                ),

                max_new_tokens=400,
            )
        )

    # =================================================================
    # CREATE EXPLANATIONS
    # =================================================================

    explanations = []

    for sample_index in range(
        len(X_test)
    ):

        # -------------------------------------------------------------
        # Detector evidence
        # -------------------------------------------------------------

        evidence = (
            create_explanation(

                result=result,

                sample_index=sample_index,

                threshold=threshold,

                feature_names=feature_names,

                original_values=X_test,

                top_k=top_k,
            )
        )

        # -------------------------------------------------------------
        # LLM explanation
        # -------------------------------------------------------------

        if explainer is not None:

            print()
            print(
                "=" * 70
            )

            print(
                f"SAMPLE {sample_index}"
            )

            print(
                "=" * 70
            )

            print()
            print(
                "DETECTOR EVIDENCE"
            )

            print(
                "-" * 70
            )

            print(
                json.dumps(

                    evidence,

                    indent=2,

                    ensure_ascii=False,
                )
            )

            # ---------------------------------------------------------
            # Human-readable explanation
            # ---------------------------------------------------------

            print()
            print(
                "GENERATING LLM EXPLANATION..."
            )

            text_explanation = (
                explainer.explain(
                    evidence
                )
            )

            print()
            print(
                "LLM EXPLANATION"
            )

            print(
                "-" * 70
            )

            print(
                text_explanation
            )

            # ---------------------------------------------------------
            # Structured JSON explanation
            # ---------------------------------------------------------

            structured = (
                explainer.explain_json(
                    evidence
                )
            )

            print()
            print(
                "STRUCTURED LLM RESULT"
            )

            print(
                "-" * 70
            )

            print(
                json.dumps(

                    structured,

                    indent=2,

                    ensure_ascii=False,
                )
            )

            evidence[
                "llm_explanation"
            ] = text_explanation

            evidence[
                "llm_structured"
            ] = structured

        explanations.append(
            evidence
        )

    # =================================================================
    # RETURN EVERYTHING
    # =================================================================

    return {

        "model":
            transformer,

        "scaler":
            scaler,

        "threshold":
            threshold,

        "scores":
            scores,

        "is_anomaly":
            anomaly_flags,

        "explanations":
            explanations,

        "feature_names":
            feature_names,
    }


# ====================================================================
# 13. COMPLETE EXAMPLE
# ====================================================================

if __name__ == "__main__":

    print()
    print(
        "=" * 70
    )

    print(
        "TRANSFORMER ANOMALY DETECTOR + LLM"
    )

    print(
        "=" * 70
    )


    # =================================================================
    # EXAMPLE DATA
    # =================================================================

    # ---------------------------------------------------------------
    # In a real application replace this with:
    #
    # X_train = your_normal_training_data
    # X_test  = your_data_to_check
    #
    # Both must be NumPy arrays:
    #
    # X_train.shape = [samples, features]
    # X_test.shape  = [samples, features]
    # ---------------------------------------------------------------

    NUM_TRAIN_SAMPLES = 3000

    NUM_TEST_SAMPLES = 10

    NUM_FEATURES = 50


    # =================================================================
    # CREATE SYNTHETIC NORMAL TRAINING DATA
    # =================================================================

    X_train = np.random.normal(

        loc=0.0,

        scale=1.0,

        size=(

            NUM_TRAIN_SAMPLES,

            NUM_FEATURES,
        ),
    ).astype(
        np.float32
    )


    # =================================================================
    # CREATE SYNTHETIC TEST DATA
    # =================================================================

    X_test = np.random.normal(

        loc=0.0,

        scale=1.0,

        size=(

            NUM_TEST_SAMPLES,

            NUM_FEATURES,
        ),
    ).astype(
        np.float32
    )


    # =================================================================
    # INJECT ANOMALY
    # =================================================================

    # Make sample 0 anomalous.

    X_test[
        0,
        17
    ] += 8.0

    X_test[
        0,
        3
    ] -= 7.0

    X_test[
        0,
        22
    ] += 5.0


    # =================================================================
    # FEATURE NAMES
    # =================================================================

    feature_names = [

        f"feature_{i:02d}"

        for i in range(
            NUM_FEATURES
        )
    ]

    # Give some features meaningful names.

    feature_names[3] = (
        "pressure"
    )

    feature_names[17] = (
        "temperature"
    )

    feature_names[22] = (
        "vibration"
    )


    # =================================================================
    # RUN COMPLETE PIPELINE
    # =================================================================

    pipeline = run_pipeline(

        X_train=X_train,

        X_test=X_test,

        feature_names=feature_names,

        transformer_epochs=20,

        threshold_percentile=99.5,

        top_k=10,

        load_llm=True,
    )


    # =================================================================
    # FINAL RESULTS
    # =================================================================

    print()
    print()
    print(
        "=" * 70
    )

    print(
        "FINAL RESULTS"
    )

    print(
        "=" * 70
    )

    print()

    print(
        "Threshold:",
        pipeline[
            "threshold"
        ]
    )

    print()

    print(
        "Scores:"
    )

    for i, score in enumerate(
        pipeline[
            "scores"
        ]
    ):

        status = (

            "ANOMALY"

            if pipeline[
                "is_anomaly"
            ][i]

            else "normal"
        )

        print(

            f"  Sample {i:04d}: "

            f"{score:.6f} "

            f"-> {status}"
        )


    # =================================================================
    # SHOW FIRST SAMPLE EXPLANATION
    # =================================================================

    print()
    print(
        "=" * 70
    )

    print(
        "FIRST SAMPLE EVIDENCE"
    )

    print(
        "=" * 70
    )

    print(

        json.dumps(

            pipeline[
                "explanations"
            ][0],

            indent=2,

            ensure_ascii=False,
        )
    )
