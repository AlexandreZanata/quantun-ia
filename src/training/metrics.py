import json
from datetime import datetime
from pathlib import Path

from src.training.structured_log import log_event

LOGS_PATH = Path("logs/experiments.jsonl")
LOGS_PATH.parent.mkdir(exist_ok=True)


class ExperimentLogger:
    def __init__(self, exp_id: str, model_name: str):
        self.exp_id = exp_id
        self.model_name = model_name
        self.started_at = datetime.now().isoformat()
        self.epochs = []
        log_event(
            "info",
            "experiment started",
            exp_id=exp_id,
            model_name=model_name,
            started_at=self.started_at,
        )

    def log(self, epoch: int, **metrics):
        self.epochs.append({"epoch": epoch, **metrics})
        log_event("info", "epoch metrics", exp_id=self.exp_id, model_name=self.model_name, epoch=epoch, **metrics)

    def finish(self, elapsed_seconds: float, **extra):
        train_acc = self.epochs[-1].get("accuracy", None) if self.epochs else None
        train_loss = self.epochs[-1].get("loss", None) if self.epochs else None
        test_accuracy = extra.get("test_accuracy")
        record = {
            "exp_id": self.exp_id,
            "model_name": self.model_name,
            "started_at": self.started_at,
            "elapsed_s": elapsed_seconds,
            "final_acc": train_acc,
            "final_loss": train_loss,
            "test_accuracy": test_accuracy,
            "test_loss": extra.get("test_loss"),
            "eval_set": extra.get("eval_set", "holdout_test" if test_accuracy is not None else "train"),
            "n_epochs": len(self.epochs),
            "history": self.epochs,
            **extra,
        }
        with open(LOGS_PATH, "a") as f:
            f.write(json.dumps(record) + "\n")
        log_event(
            "info",
            "experiment finished",
            exp_id=self.exp_id,
            model_name=self.model_name,
            elapsed_s=elapsed_seconds,
            final_acc=train_acc,
            test_accuracy=test_accuracy,
            eval_set=record["eval_set"],
        )
        return record


def load_all_experiments():
    records = []
    if LOGS_PATH.exists():
        with open(LOGS_PATH) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records
