"""TransformerConfig -- vendored verbatim from ts_proba_cuda.

Source: ts_benchmark/baselines/duet/duet_prob.py, lines 73-174
        at commit a8de694266a629124687a8f2b9fcfdba15a3590c.

The rest of duet_prob.py is the Optuna/tensorboard training harness and is not
needed for inference, so only this class was extracted. The class body below is
byte-for-byte identical to upstream; do not edit it -- see PROVENANCE.md.
"""

class TransformerConfig:
    """
    Configuration holder: merges the defaults below with the checkpoint's config_dict.
    Extracted from the upstream trainer module so inference does not have to import it.
    """
    def __init__(self, **kwargs):
        defaults = {
            # --- Core Architecture ---
            "d_model": 512, "d_ff": 2048, "n_heads": 8, "e_layers": 2,
            "factor": 3, "activation": "gelu", "dropout": 0.1, "fc_dropout": 0.1,
            "output_attention": False,

            # --- MoE Parameters (General) ---
            "noisy_gating": True, "hidden_size": 256,
            "loss_coef": 1.0, # MoE loss coefficient

            # --- MoE Parameters (Expert Configuration) ---
            # Hybrid expert mixture: linear experts plus reservoir (ESN) experts.
            "num_linear_experts": 2,
            "num_univariate_esn_experts": 1,
            "num_multivariate_esn_experts": 1,
            "k": 2,              # Default, will be overwritten below

            # --- ESN Expert Default Parameters ---
            # Univariate ESN
            "reservoir_size_uni": 256,
            "spectral_radius_uni": 0.99,
            "sparsity_uni": 0.1,
            "leak_rate_uni": 1.0,
            "input_scaling_uni": 1.0,

            # Multivariate ESN
            "reservoir_size_multi": 256,
            "spectral_radius_multi": 0.99,
            "sparsity_multi": 0.1,
            "leak_rate_multi": 1.0,
            "input_scaling_multi": 0.5, # Already separate

            # --- NEW: ESN Readout Regularization ---
            "esn_uni_weight_decay": 0.0,
            "esn_multi_weight_decay": 0.0,

            # --- Training / Optimization ---
            "lr": 1e-4,
            "lradj": "cosine_warmup", "num_epochs": 100,
            "accumulation_steps": 1,  # gradient accumulation (training only)
            "batch_size": 128, "patience": 10,
            "num_workers": 4,

            # --- NEW: Tier 2 Training Strategies ---
            "use_agc": False,       # Use Adaptive Gradient Clipping
            "agc_lambda": 0.01,     # Clipping factor for AGC

            # --- Data & Miscellaneous ---
            "moving_avg": 25, "CI": False, "freq": "h",
            "quantiles": [0.1, 0.5, 0.9],  # overridden at inference time
            "norm_mode": "subtract_median", # Preferred normalization mode

            # --- NEW: Projection Head Configuration ---
            "projection_head_layers": 0,      # Default to 0 for original behavior (single linear layer)
            "projection_head_dim_factor": 2,  # Hidden dim = in_features / factor
            "projection_head_dropout": 0.1,

            # --- NEW: Interim Validation ---
            "interim_validation_seconds": None, # Default: disabled. Set to e.g. 300 for 5-min validation.

            # --- NEW: Performance Profiling ---
            "profile_epoch": None, # Set to an epoch number (e.g., 2) to enable profiling for that epoch.
            # How often to refresh the tqdm progress bar (every n-th batch).
            "tqdm_update_freq": 10,
            # Minimum seconds between tqdm bar updates, to reduce I/O spam.
            "tqdm_min_interval": 1.0,
        }

        for key, value in defaults.items():
            setattr(self, key, value)

        for key, value in kwargs.items():
            setattr(self, key, value)

        # Abgeleitete Werte
        if hasattr(self, 'seq_len'):
            # Expected by some sub-modules even though inference never varies them.
            self.input_size = self.seq_len
            self.label_len = self.seq_len // 2
        else:
            raise AttributeError("config_dict must contain 'seq_len'")

        if hasattr(self, 'horizon'):
            self.pred_len = self.horizon
        else:
            raise AttributeError("config_dict must contain 'horizon'")

        # 'k' muss kleiner oder gleich der Gesamtanzahl Experten sein.
        # Wir setzen es hier sicherheitshalber nach der Experten-Definition.
        total_experts = (getattr(self, "num_linear_experts", 0) +
                         getattr(self, "num_univariate_esn_experts", 0) +
                         getattr(self, "num_multivariate_esn_experts", 0))
        if total_experts > 0:
            self.k = min(getattr(self, "k", 1), total_experts)

