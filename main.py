import matplotlib
matplotlib.use("Agg")  # Save plots without requiring an interactive GUI.
import matplotlib.pyplot as plt

from stock_utils import (
    OUTPUT_DIR,
    load_and_prepare_data,
    run_experiment,
    save_experiment_outputs,
)


def main():
    df = load_and_prepare_data()
    results = run_experiment(df)
    output_dir = save_experiment_outputs(results)

    print("=" * 60)
    print("STOCK PRICE PREDICTION — EXPERIMENT SUMMARY")
    print("=" * 60)
    print(f"Rows after target creation: {len(df):,}")
    print(f"Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
    print(f"Training samples: {len(results['train']):,}")
    print(f"Testing samples: {len(results['test']):,}")
    print("\nModel comparison (lower MAE/RMSE is better):")
    print(results["metrics"].round(4).to_string(index=False))

    predictions = results["predictions"]
    plt.figure(figsize=(13, 6))
    plt.plot(predictions["Date"], predictions["Actual"], label="Actual", linewidth=2)
    plt.plot(predictions["Date"], predictions["Naive baseline"], label="Naive baseline", alpha=0.8)
    plt.plot(predictions["Date"], predictions["Linear Regression"], label="Linear Regression", alpha=0.85)
    plt.plot(predictions["Date"], predictions["Random Forest"], label="Random Forest", alpha=0.85)
    plt.title("Actual vs Predicted Next-Session Closing Price")
    plt.xlabel("Date")
    plt.ylabel("Price (dataset units)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "actual_vs_predicted.png", dpi=160)
    plt.close()

    # Residual plot for the best model by MAE.
    best_name = results["metrics"].iloc[0]["Model"]
    residuals = predictions["Actual"] - predictions[best_name]
    plt.figure(figsize=(10, 5))
    plt.scatter(predictions[best_name], residuals, alpha=0.55)
    plt.axhline(0, linestyle="--")
    plt.title(f"Residuals — {best_name}")
    plt.xlabel("Predicted price")
    plt.ylabel("Actual - predicted")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "residuals.png", dpi=160)
    plt.close()

    print(f"\nFiles saved in: {output_dir}")
    print("- model_comparison.csv")
    print("- predictions.csv")
    print("- actual_vs_predicted.png")
    print("- residuals.png")
    print("\nNote: the dataset's company/ticker is not verified. These are historical-data experiments, not investment advice.")


if __name__ == "__main__":
    main()
