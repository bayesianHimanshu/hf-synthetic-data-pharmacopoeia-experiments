from __future__ import annotations

import csv
from pathlib import Path

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from pharma_slm.telemetry import get_meter


class OtelMetricsCallback(TrainerCallback):
    """Emits training loss and learning rate as OpenTelemetry gauge metrics.
    """

    def __init__(self) -> None:
        meter = get_meter(__name__)
        self._loss_gauge = meter.create_gauge(
            "training.loss",
            description="Training loss at each log step",
            unit="1",
        )
        self._lr_gauge = meter.create_gauge(
            "training.learning_rate",
            description="Learning rate at each log step",
            unit="1",
        )

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict | None = None,
        **kwargs,
    ) -> None:
        if logs is None:
            return
        attrs = {"step": str(state.global_step)}
        try:
            if "loss" in logs:
                self._loss_gauge.set(logs["loss"], attrs)
            if "learning_rate" in logs:
                self._lr_gauge.set(logs["learning_rate"], attrs)
        except Exception:
            pass


class PlottingCallback(TrainerCallback):
    """Accumulates loss and LR per log step; writes plots and CSV at train end.
    """

    def __init__(self, plots_dir: str, csv_path: str) -> None:
        self._plots_dir = Path(plots_dir)
        self._csv_path = Path(csv_path)
        self._rows: list[dict] = []

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict | None = None,
        **kwargs,
    ) -> None:
        if logs is None:
            return
        row: dict = {"step": state.global_step}
        if "loss" in logs:
            row["loss"] = logs["loss"]
        if "learning_rate" in logs:
            row["learning_rate"] = logs["learning_rate"]
        if len(row) > 1:  # at least one metric besides step
            self._rows.append(row)

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        if not self._rows:
            return

        self._plots_dir.mkdir(parents=True, exist_ok=True)
        self._csv_path.parent.mkdir(parents=True, exist_ok=True)

        # Write CSV
        fieldnames = list(self._rows[0].keys())
        with open(self._csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self._rows)
        print(f"Training metrics CSV saved to {self._csv_path}")

        # Plots
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        steps = [r["step"] for r in self._rows]

        # Loss curve
        losses = [r.get("loss") for r in self._rows]
        if any(v is not None for v in losses):
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(steps, losses, color="steelblue", linewidth=1.5)
            ax.set_xlabel("Step")
            ax.set_ylabel("Loss")
            ax.set_title("Training Loss")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            path = self._plots_dir / "loss_curve.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Loss curve saved to {path}")

        # LR schedule
        lrs = [r.get("learning_rate") for r in self._rows]
        if any(v is not None for v in lrs):
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(steps, lrs, color="tomato", linewidth=1.5)
            ax.set_xlabel("Step")
            ax.set_ylabel("Learning Rate")
            ax.set_title("Learning Rate Schedule")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            path = self._plots_dir / "lr_schedule.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"LR schedule saved to {path}")
