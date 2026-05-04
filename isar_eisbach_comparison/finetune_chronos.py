import pandas as pd
import numpy as np
import torch
import os
from transformers import TrainingArguments
from chronos import BaseChronosPipeline
from chronos.chronos2.dataset import Chronos2Dataset
from chronos.chronos2.trainer import Chronos2Trainer, EvaluateAndSaveFinalStepCallback

def prepare_data():
    csv_path = "isar_eisbach_comparison/isar_eisbach_10_years.csv"
    if not os.path.exists(csv_path):
        return None
    df_combined = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    df_ffill = df_combined.ffill().bfill()
    return df_ffill

def run_finetuning():
    df_ffill = prepare_data()
    if df_ffill is None:
        print("Data not found")
        return

    print("Loading base model...")
    pipeline = BaseChronosPipeline.from_pretrained(
        "amazon/chronos-2",
        device_map="cpu",
        torch_dtype=torch.bfloat16,
    )

    # We will hold out the exact backtesting windows from training!
    # Our backtest windows are the last 10 spaced by 30 days. Max history we need to hold out is roughly the last 350 days.
    # We will hold out the last 10,000 hours (approx 1.1 years) to guarantee no leakage into the 10-window showdown.
    holdout_size = 10000
    train_end_idx = len(df_ffill) - holdout_size

    train_data = df_ffill['wassertemp_eisbach'].iloc[:train_end_idx].values

    context_len = min(1024, pipeline.model_context_length)
    pred_len = min(256, pipeline.model_prediction_length)

    print(f"Dataset Size: {len(train_data)}. Holding out {holdout_size} items.")

    # We split into Train and Validation for proper robust fine-tuning
    val_size = 5000
    actual_train_data = train_data[:-val_size]
    val_data = train_data[-val_size:]

    train_dataset = Chronos2Dataset(
        inputs=[actual_train_data],
        context_length=context_len,
        prediction_length=pred_len,
        batch_size=8,
        output_patch_size=pipeline.model.config.output_patch_size if hasattr(pipeline.model.config, 'output_patch_size') else 1,
        mode="train",
    )

    eval_dataset = Chronos2Dataset(
        inputs=[val_data],
        context_length=context_len,
        prediction_length=pred_len,
        batch_size=8,
        output_patch_size=pipeline.model.config.output_patch_size if hasattr(pipeline.model.config, 'output_patch_size') else 1,
        mode="validation",
    )

    output_dir = "isar_eisbach_comparison/finetuned_chronos2"
    os.makedirs(output_dir, exist_ok=True)

    # Implementing robust finetuning logic
    # With early stopping via load_best_model_at_end using evaluation loss
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=1e-4,
        num_train_epochs=1,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        max_steps=500, # A robust amount of steps for actual fine-tuning without taking forever
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_num_workers=0,
    )

    trainer = Chronos2Trainer(
        model=pipeline.inner_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[EvaluateAndSaveFinalStepCallback()],
    )

    print("Starting Fine-Tuning...")
    trainer.train()

    print("Saving fine-tuned model...")
    pipeline.inner_model.save_pretrained(output_dir)
    if hasattr(pipeline.model, 'config'):
        pipeline.model.config.save_pretrained(output_dir)
    print("Fine-tuning complete.")

if __name__ == "__main__":
    run_finetuning()
