# Chronos-2 Forecasting Experiments

## Model Architecture and Inference Strategy
Chronos-2 is a state-of-the-art time series forecasting foundation model by Amazon.

Based on my research into the Chronos-2 paper ([https://arxiv.org/abs/2510.15821](https://arxiv.org/abs/2510.15821)) and inspection of its source code:

> "Chronos-2 is an encoder-only transformer model which closely follows the design of the T5 encoder... The decoder then uses these representations to directly generate quantile forecasts across multiple future steps—a method known as direct multi-step forecasting."

* The model is an **encoder-only transformer** that relies on a direct **multi-step** forecasting approach (via a Quantile Head).
* Instead of being purely autoregressive step-by-step like language models, it predicts multiple future steps simultaneously using "patches".
* The model chunks historical data into patches and directly generates quantile forecasts across multiple future steps for a given `max_output_patches`.
* For very long horizons that exceed the maximum patch limit, Chronos-2 uses a **long horizon heuristic**: it generates up to `max_output_patches` (default 64 patches * 16 steps = 1024 steps), appends specific predicted quantiles to the historical context, and then autoregressively unrolls the next chunk.
* The model natively produces **quantile predictions** (e.g., 0.1, 0.5, 0.9) directly through its quantile head (Eq 4 in the paper), not by sampling a probability distribution like a language model with a temperature. Thus, it is **deterministic** when generating a point forecast (like median/0.5 quantile).
* "Temperature" or probabilistic sampling the way language models do it does not apply to this model architecture natively for its quantile output. Instead, it directly outputs the predicted quantiles.

## Objective
The goal is to test how Chronos-2 performs when we force it into an step-by-step autoregressive mode (predicting 1 hour at a time and feeding it back) compared to its native multi-step prediction, specifically for the Eisbach water temperature data over a 4-day (96 hours) horizon. We want to see if this forced autoregressive mode can create realistic scenarios and how they compare to the native quantile predictions.

## Determinism
By running the same inputs multiple times, we can see that the model's outputs are completely deterministic. Chronos-2 predicts fixed quantiles out of the model's layers without any sampling process (like standard LMs with next-token prediction via temperature).

However, if we want to create diverse *scenarios* manually by autoregressively sampling from the predictive *distribution* (e.g. taking a random quantile or adding controlled noise) at each step, we can generate a variety of paths!

We will write a script to generate scenarios.

## Findings on Autoregressive Sampling

As seen in the experiments (and plots), naive autoregressive sampling (predicting 1 step, taking a quantile, feeding it back) completely breaks down for Chronos-2.
The errors accumulate exponentially because the model is trained to condition on *true* historical data that follows natural dynamics, not on its own noisy single-step predictions. The y-axis scales to `1e36` when we try to do this, showing that the model outputs explode out of control ("völlig außer Rand und Band").

Because Chronos-2 is a *quantile regression* model (predicting explicit quantiles directly) rather than an autoregressive probability distribution model (like LLMs which predict next tokens via softmax), there is no native "temperature" that introduces diversity. The model is fully deterministic for a given input.

The conclusion is that trying to force Chronos-2 into an autoregressive scenario generator is not viable. The model's native 96-step forecast is highly robust and accurate, while the autoregressive workaround results in garbage outputs.
