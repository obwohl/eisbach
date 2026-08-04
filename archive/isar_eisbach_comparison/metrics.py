import numpy as np

def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def crps(y_true, y_pred_quantiles, quantiles):
    """
    Computes an approximation of CRPS using the pinball loss across all quantiles.
    To be mathematically correct for non-uniformly spaced quantiles, we must integrate
    using the trapezoidal rule (or similar weighted integration) over the quantile levels.
    """
    crps_vals = []

    for t in range(len(y_true)):
        y_t = y_true[t]

        # Calculate pinball loss for each quantile
        pinball_losses = []
        valid_quantiles = []
        for idx, q in enumerate(quantiles):
            q_pred = y_pred_quantiles[idx, t]
            if np.isnan(q_pred): continue

            if y_t >= q_pred:
                loss = (y_t - q_pred) * q
            else:
                loss = (q_pred - y_t) * (1 - q)

            # Factor 2 so it is directly comparable to absolute error (CRPS ~ MAE for point forecasts)
            pinball_losses.append(2 * loss)
            valid_quantiles.append(q)

        if not pinball_losses:
            continue

        # Integrate using trapezoidal rule: sum of 0.5 * (loss_i + loss_{i+1}) * (q_{i+1} - q_i)
        # We need to span the full [0, 1] interval. We assume loss approaches MAE near median
        # and tapers off at edges. A standard approximation is to use numpy's trapz over the given quantiles.
        integrated_loss = np.trapz(pinball_losses, valid_quantiles)

        crps_vals.append(integrated_loss)

    return np.mean(crps_vals)

def mean_interval_score(y_true, lower_bounds, upper_bounds, alpha):
    scores = []
    for t in range(len(y_true)):
        y_t = y_true[t]
        l = lower_bounds[t]
        u = upper_bounds[t]

        if np.isnan(l) or np.isnan(u): continue

        score = (u - l)
        if y_t < l:
            score += (2 / alpha) * (l - y_t)
        elif y_t > u:
            score += (2 / alpha) * (y_t - u)
        scores.append(score)

    return np.mean(scores)
