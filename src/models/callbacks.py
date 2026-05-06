import json
import time
from transformers import TrainerCallback

from .config import NUM_EPOCHS, PROGRESS_FILE


class TrainingMonitor:
    def __init__(self):
        self.epoch_metrics = []
        self.step_metrics = []
        self.start_time = None

    def on_train_start(self):
        self.start_time = time.time()
        print("\n" + "=" * 60)
        print("TRAINING STARTED")
        print("=" * 60)
        self._write_progress({"status": "starting", "epoch": 0, "step": 0, "loss": 0})

    def on_epoch_end(self, epoch, metrics):
        elapsed = time.time() - self.start_time
        epoch_data = {
            "epoch": epoch,
            "train_loss": metrics.get("train_loss", 0),
            "eval_loss": metrics.get("eval_loss", 0),
            "eval_accuracy": metrics.get("eval_accuracy", 0),
            "learning_rate": metrics.get("learning_rate", 0),
            "elapsed_time": elapsed,
        }
        self.epoch_metrics.append(epoch_data)
        self._write_progress({
            "status": "running",
            "epoch": epoch,
            "total_epochs": NUM_EPOCHS,
            "loss": metrics.get("train_loss", 0),
            "eval_loss": metrics.get("eval_loss", 0),
            "eval_accuracy": metrics.get("eval_accuracy", 0),
            "elapsed_time": elapsed
        })

    def on_train_end(self):
        total_time = time.time() - self.start_time
        print("\n" + "=" * 60)
        print(f"TRAINING COMPLETED in {total_time:.2f} seconds")
        print("=" * 60)
        self._write_progress({"status": "running", "elapsed_time": total_time, "details": "TinyBERT Training Complete. Moving to Evaluation..."})

    def _write_progress(self, data):
        try:
            with open(PROGRESS_FILE, "w") as f:
                json.dump(data, f)
        except Exception:
            pass


class ProgressCallback(TrainerCallback):
    def __init__(self, monitor: TrainingMonitor):
        self.monitor = monitor
        self.current_loss = 0
        self.total_steps = 0
        self.train_start_time = None

    def on_init_end(self, args, state, control, **kwargs):
        self.total_steps = state.max_steps if hasattr(state, 'max_steps') and state.max_steps else 0

    def on_train_begin(self, args, state, control, **kwargs):
        self.train_start_time = time.time()
        self.monitor.on_train_start()
        self._emit_json({"progress": 0, "status": "starting", "epoch": 0, "step": 0, "loss": 0})

    def on_train_end(self, args, state, control, **kwargs):
        self.monitor.on_train_end()
        elapsed = time.time() - self.train_start_time if self.train_start_time else 0
        self._emit_json({
            "progress": 100,
            "status": "complete",
            "epoch": state.epoch,
            "step": state.global_step,
            "loss": self.current_loss,
            "elapsed_time": round(elapsed, 2)
        })

    def on_epoch_begin(self, args, state, control, **kwargs):
        pass

    def on_epoch_end(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            self.monitor.on_epoch_end(state.epoch, metrics)
            for key in metrics:
                if "train_loss" in key or key == "loss":
                    self.current_loss = metrics[key]

    def on_step_begin(self, args, state, control, **kwargs):
        pass

    def on_step_end(self, args, state, control, **kwargs):
        progress = self._calculate_progress(state)
        self._emit_json({
            "progress": progress,
            "status": "training",
            "epoch": int(state.epoch) if state.epoch is not None else 0,
            "total_epochs": NUM_EPOCHS,
            "step": state.global_step,
            "total_steps": self.total_steps,
            "loss": round(self.current_loss, 6),
        })

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            if "loss" in logs:
                self.current_loss = logs["loss"]
            progress = self._calculate_progress(state)
            self._emit_json({
                "progress": progress,
                "status": "training",
                "epoch": int(state.epoch) if state.epoch is not None else 0,
                "total_epochs": NUM_EPOCHS,
                "step": state.global_step,
                "total_steps": self.total_steps,
                "loss": round(self.current_loss, 6),
                "learning_rate": logs.get("learning_rate", 0),
            })

    def _calculate_progress(self, state):
        if self.total_steps == 0:
            return 0
        current_epoch = state.epoch if state.epoch is not None else 0
        current_step = state.global_step
        total_epochs = NUM_EPOCHS
        if total_epochs == 0:
            return 0
        epoch_progress = current_step / max(self.total_steps / total_epochs, 1)
        overall = ((current_epoch + epoch_progress) / total_epochs) * 100
        return min(100, max(0, overall))

    def _emit_json(self, data):
        try:
            print(json.dumps(data), flush=True)
            with open(PROGRESS_FILE, "w") as f:
                json.dump(data, f)
        except Exception:
            pass