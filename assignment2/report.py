import matplotlib.pyplot as plt
import pandas as pd

# Load CSV files
files_data = {
    "Baseline": pd.read_csv("results/inference_metrics_codegemma_7b_no_quant.csv"),
    "Quantized": pd.read_csv("results/inference_metrics_codegemma_7b_4bit_quant.csv"),
    "Pruned": pd.read_csv(
        "results/inference_metrics_codegemma_7b_pruned_40percent.csv"
    ),
    "Distilled": pd.read_csv("results/inference_metrics_codegemma_2b_distilled.csv"),
}

# Calculate average emissions
models = list(files_data.keys())
avg_emissions = [df["emissions"].mean() for df in files_data.values()]

# Create bar chart
plt.figure(figsize=(7, 4.5))
bars = plt.bar(models, avg_emissions, color="#1FB8CD", edgecolor="black", linewidth=1.2)

# Add emission text labels on bars
for bar, em in zip(bars, avg_emissions):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{em:.2e}",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="medium",
    )

# Titles and labels
plt.title("Average CO2 Emissions by Model", fontsize=13, pad=12)
plt.xlabel("Model", fontsize=11)
plt.ylabel("Avg CO2 Em (kg)", fontsize=11)
plt.grid(axis="y", linestyle="--", alpha=0.5)

# Save charts
plt.tight_layout()
plt.savefig("results/emission.png", dpi=300)

plt.show()

print("Chart created successfully!")
print("Average emissions by model:")
for model, avg_em in zip(models, avg_emissions):
    print(f"{model}: {avg_em:.6f} kg")


records = []

for model_name, df in files_data.items():
    avg_emission = df["emissions"].mean()
    avg_latency = df["latency"].mean() if "latency" in df.columns else None

    records.append(
        {
            "Model": model_name,
            "Avg_Emissions_kg": avg_emission,
            "Avg_Latency_s": avg_latency,
        }
    )

# Create a summary DataFrame
summary_df = pd.DataFrame(records)


summary_df.to_csv("results/average_metrics.csv", index=False)


print("Average metrics computed successfully!\n")
print(summary_df.to_string(index=False))
print("\nSaved as 'results/average_metrics.csv'")

# Plot Average Latency Bar Chart
plt.figure(figsize=(7, 4.5))

avg_latencies = [df["latency"].mean() for df in files_data.values()]

# Create bar chart
bars = plt.bar(models, avg_latencies, color="#1FB8CD", edgecolor="black", linewidth=1.2)

# Add latency text labels on bars
for bar, latency in zip(bars, avg_latencies):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{latency:.2f}s",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="medium",
    )

# Titles and labels
plt.title("Average Latency by Model", fontsize=13, pad=12)
plt.xlabel("Model", fontsize=11)
plt.ylabel("Avg Latency (s)", fontsize=11)
plt.grid(axis="y", linestyle="--", alpha=0.5)

# Save and show
plt.tight_layout()
plt.savefig("results/latency.png", dpi=300)
plt.show()

print("\nLatency bar chart created successfully!")
print("Average latency by model:")
for model, avg_lat in zip(models, avg_latencies):
    print(f"{model}: {avg_lat:.6f} s")

# Combined Chart: Model Size, Latency, and Accuracy
fig, ax1 = plt.subplots(figsize=(12, 6))

# Collect data
model_sizes_data = []
avg_latencies_data = []
avg_accuracies_data = []

for model_name, df in files_data.items():
    model_size = (
        df["model_size"].iloc[0]
        if "model_size" in df.columns and not df["model_size"].isna().all()
        else 0
    )
    avg_latency = df["latency"].mean() if "latency" in df.columns else 0

    # Calculate accuracy percentage from P/F values
    if "accuracy" in df.columns:
        # Count P's (passes) and calculate percentage
        accuracy_series = df["accuracy"].astype(str)
        total_count = len(accuracy_series)
        pass_count = accuracy_series.str.count("P").sum()
        avg_accuracy = (pass_count / total_count) * 100 if total_count > 0 else 0
    else:
        avg_accuracy = 0

    model_sizes_data.append(model_size)
    avg_latencies_data.append(avg_latency)
    avg_accuracies_data.append(avg_accuracy)

x = range(len(models))
width = 0.25

# Create bars for model size and latency on primary y-axis
bars1 = ax1.bar(
    [i - width for i in x],
    model_sizes_data,
    width,
    label="Model Size (GB)",
    color="#4A90E2",
    edgecolor="black",
    linewidth=1.2,
)
bars2 = ax1.bar(
    [i for i in x],
    avg_latencies_data,
    width,
    label="Avg Latency (s)",
    color="#1FB8CD",
    edgecolor="black",
    linewidth=1.2,
)

ax1.set_xlabel("Model", fontsize=12, fontweight="bold")
ax1.set_ylabel("Model Size (GB) / Latency (s)", fontsize=11)
ax1.set_xticks(x)
ax1.set_xticklabels(models)
ax1.tick_params(axis="y")
ax1.grid(axis="y", linestyle="--", alpha=0.3)

# Create secondary y-axis for accuracy
ax2 = ax1.twinx()
bars3 = ax2.bar(
    [i + width for i in x],
    avg_accuracies_data,
    width,
    label="Accuracy (%)",
    color="#50C878",
    edgecolor="black",
    linewidth=1.2,
)
ax2.set_ylabel("Accuracy (%)", fontsize=11)
ax2.tick_params(axis="y")

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    ax1.text(
        bar.get_x() + bar.get_width() / 2.0,
        height,
        f"{height:.2f}",
        ha="center",
        va="bottom",
        fontsize=9,
    )

for bar in bars2:
    height = bar.get_height()
    ax1.text(
        bar.get_x() + bar.get_width() / 2.0,
        height,
        f"{height:.2f}",
        ha="center",
        va="bottom",
        fontsize=9,
    )

for bar in bars3:
    height = bar.get_height()
    ax2.text(
        bar.get_x() + bar.get_width() / 2.0,
        height,
        f"{height:.1f}%",
        ha="center",
        va="bottom",
        fontsize=9,
    )

ax1.legend(loc="upper left")
ax2.legend(loc="upper right")

plt.tight_layout()
plt.savefig("results/combined_metrics.png", dpi=300)
plt.show()

print("\nCombined metrics chart created successfully!")
print("Saved as 'results/combined_metrics.png'")
print("\nSummary:")
for i, model in enumerate(models):
    print(
        f"{model}: Size={model_sizes_data[i]:.2f} GB, Latency={avg_latencies_data[i]:.2f}s, Accuracy={avg_accuracies_data[i]:.1f}%"
    )

# Generate LaTeX table
print("\n" + "=" * 80)
print("LaTeX Table for Summary Statistics")
print("=" * 80)

latex_table = r"""
\subsection{Summary Statistics}
\begin{table}[htbp]
\centering
\caption{Average CO$_2$ emissions and latency across model variants.}
\begin{tabular}{lcccc}
\toprule
\textbf{Model} & \textbf{CO$_2$ Emissions (kg)} & \textbf{Latency (s)} & \textbf{Size (GB)} & \textbf{Accuracy (\%)}\\
\midrule
"""

for i, model in enumerate(models):
    latex_table += f"{model:<9} & {avg_emissions[i]:.6f} & {avg_latencies_data[i]:.4f} & {model_sizes_data[i]:.3f} & {avg_accuracies_data[i]:.1f}\\\\\n"

latex_table += r"""\bottomrule
\end{tabular}
\label{tab:avg_metrics}
\end{table}
"""

print(latex_table)

# Save LaTeX table to file
with open("results/summary_table.tex", "w") as f:
    f.write(latex_table)

print("LaTeX table saved as 'results/summary_table.tex'")
