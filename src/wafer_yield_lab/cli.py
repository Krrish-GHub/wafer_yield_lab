"""Small CLI for reproducing the model comparison."""
from .train import train


def main() -> None:
    report = train()
    print(f"Evaluated {report['dataset']['rows']} SECOM samples; metrics saved to reports/metrics.json")
    for name, metrics in report["models"].items():
        print(f"{name:24s} AP={metrics['average_precision']:.3f} "
              f"recall={metrics['recall_fail']:.3f} precision={metrics['precision_fail']:.3f}")
