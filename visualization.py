
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

BASE_DIR = r"C:\Users\giris\Desktop\cluster_project"

DATA_FILE = os.path.join(
    BASE_DIR, "retail_cleaned_single", "retail_cleaned.csv"
)
SEGMENTS_DIR = os.path.join(BASE_DIR, "customer_segments")
SUMMARY_DIR = os.path.join(BASE_DIR, "cluster_summary")
APRIORI_DIR = os.path.join(BASE_DIR, "apriori_results", "frequent_itemsets")
OUTPUT_DIR = os.path.join(BASE_DIR, "visualizations")

os.makedirs(OUTPUT_DIR, exist_ok=True)

NAVY = "#17365D"
BLUE = "#3478B8"
TEAL = "#20A39E"
SKY = "#77BFEA"
GOLD = "#F2B84B"
PURPLE = "#8878C3"
PALETTE = [BLUE, TEAL, SKY, GOLD, PURPLE]

plt.rcParams.update({
    "figure.facecolor": "#F4F7FB",
    "axes.facecolor": "#FFFFFF",
    "axes.edgecolor": "#D8E1EC",
    "axes.labelcolor": NAVY,
    "text.color": NAVY,
    "xtick.color": NAVY,
    "ytick.color": NAVY,
    "axes.titleweight": "bold",
    "axes.titlesize": 14,
    "font.family": "sans-serif",
    "grid.color": "#E5EBF3",
    "grid.linestyle": "--",
    "grid.alpha": 0.7
})


def load_csv(path):
    if os.path.isfile(path):
        return pd.read_csv(path)

    if os.path.isdir(path):
        files = glob.glob(os.path.join(path, "*.csv"))
        if not files:
            raise FileNotFoundError(f"No CSV files found in {path}")
        return pd.concat(
            [pd.read_csv(file) for file in files],
            ignore_index=True
        )

    raise FileNotFoundError(f"Path not found: {path}")


def finish_plot(filename):
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, filename),
        dpi=300,
        bbox_inches="tight",
        facecolor="#F4F7FB"
    )
    plt.close()
    print("Saved:", filename)


df = load_csv(DATA_FILE)
segments = load_csv(SEGMENTS_DIR)
summary = load_csv(SUMMARY_DIR)

df.columns = df.columns.str.strip()
segments.columns = segments.columns.str.strip()
summary.columns = summary.columns.str.strip()

df["Total_Amount"] = pd.to_numeric(
    df["Total_Amount"], errors="coerce"
)

if "InvoiceDate" in df.columns:
    df["InvoiceDate"] = pd.to_datetime(
        df["InvoiceDate"], errors="coerce"
    )

if "InvoiceDate" in df.columns:
    monthly_sales = (
        df.dropna(subset=["InvoiceDate"])
        .assign(Month=lambda x: x["InvoiceDate"].dt.to_period("M"))
        .groupby("Month")["Total_Amount"]
        .sum()
        .sort_index()
    )

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(
        monthly_sales.index.astype(str),
        monthly_sales.values,
        color=BLUE,
        linewidth=2.5,
        marker="o",
        markersize=4
    )
    ax.fill_between(
        range(len(monthly_sales)),
        monthly_sales.values,
        color=SKY,
        alpha=0.22
    )
    ax.set_title("Monthly Retail Sales Trend", loc="left", pad=15)
    ax.set_xlabel("Month")
    ax.set_ylabel("Total Sales")
    ax.grid(axis="y")
    ax.tick_params(axis="x", rotation=60)
    finish_plot("monthly_sales_trend.png")

if "Cluster" in segments.columns:
    counts = segments["Cluster"].value_counts().sort_index()
    cluster_colors = [
        BLUE if i == 0 else TEAL if i == 1 else PURPLE
        for i in range(len(counts))
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        counts.index.astype(str),
        counts.values,
        color=cluster_colors,
        width=0.58
    )
    ax.bar_label(bars, fmt="%.0f", padding=4, fontsize=10)
    ax.set_title("Customer Distribution by Cluster", loc="left", pad=15)
    ax.set_xlabel("Customer Cluster")
    ax.set_ylabel("Number of Customers")
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    finish_plot("customer_cluster_distribution.png")

for column, title, filename, color in [
    ("Avg_Monetary", "Average Customer Spending by Cluster",
     "average_monetary_by_cluster.png", TEAL),
    ("Avg_Frequency", "Average Purchase Frequency by Cluster",
     "average_frequency_by_cluster.png", BLUE),
    ("Avg_Recency", "Average Recency by Cluster",
     "average_recency_by_cluster.png", PURPLE)
]:
    if {"Cluster", column}.issubset(summary.columns):
        values = pd.to_numeric(summary[column], errors="coerce")
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(
            summary["Cluster"].astype(str),
            values,
            color=PALETTE[:len(summary)],
            width=0.58
        )
        ax.bar_label(bars, fmt="%.2f", padding=4, fontsize=9)
        ax.set_title(title, loc="left", pad=15)
        ax.set_xlabel("Customer Cluster")
        ax.set_ylabel(column.replace("Avg_", "Average "))
        ax.grid(axis="y")
        ax.set_axisbelow(True)
        finish_plot(filename)

if "Description" in df.columns:
    product_sales = (
        df.groupby("Description")["Total_Amount"]
        .sum()
        .nlargest(10)
        .sort_values()
    )

    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.barh(
        product_sales.index.astype(str),
        product_sales.values,
        color=BLUE
    )
    ax.bar_label(
        bars, labels=[f"{v:,.0f}" for v in product_sales.values],
        padding=4, fontsize=8
    )
    ax.set_title("Top 10 Products by Sales Value", loc="left", pad=15)
    ax.set_xlabel("Total Sales")
    ax.set_ylabel("Product")
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    finish_plot("top_10_products_by_sales.png")

if os.path.isdir(APRIORI_DIR) or os.path.isfile(APRIORI_DIR):
    apriori = load_csv(APRIORI_DIR)
    apriori.columns = apriori.columns.str.strip()

    if {"a", "b", "Support"}.issubset(apriori.columns):
        apriori["Support"] = pd.to_numeric(
            apriori["Support"], errors="coerce"
        )
        top_pairs = apriori.nlargest(10, "Support").copy()
        top_pairs["Product Pair"] = (
            top_pairs["a"].astype(str)
            + " + "
            + top_pairs["b"].astype(str)
        )
        top_pairs = top_pairs.sort_values("Support")

        fig, ax = plt.subplots(figsize=(11, 6))
        bars = ax.barh(
            top_pairs["Product Pair"],
            top_pairs["Support"],
            color=TEAL
        )
        ax.bar_label(
            bars,
            labels=[f"{v:.2%}" for v in top_pairs["Support"]],
            padding=4,
            fontsize=8
        )
        ax.set_title("Top Product Pairs by Support", loc="left", pad=15)
        ax.set_xlabel("Support")
        ax.set_ylabel("Product Pair")
        ax.xaxis.set_major_formatter(
            mticker.PercentFormatter(xmax=1)
        )
        ax.grid(axis="x")
        ax.set_axisbelow(True)
        finish_plot("top_product_pairs_by_support.png")
    else:
        print("Apriori chart skipped: expected a, b and Support columns.")
else:
    print("Apriori output folder not found; association chart skipped.")

print("\nAll available visualizations generated.")
print("Output folder:", OUTPUT_DIR)
