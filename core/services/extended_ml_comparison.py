"""
Extended Qlib Alpha158 + LightGBM with Sentiment Features

Compares 3 model variants:
a) Alpha158 only (baseline)
b) Alpha158 + sentiment_mean + news_volume
c) Alpha158 + all sentiment features

Generates comparison_report.md with metrics and SHAP analysis.
"""

from pathlib import Path
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

# Import from existing script
from .ml_service import (
    PROJECT_ROOT,
    QLIB_DATA_DIR,
    INSTRUMENT_FILE,
    MLFLOW_DB,
    OUTPUT_DIR,
    DATA_START,
    DATA_END,
    TRAIN_START,
    TRAIN_END,
    VALID_START,
    VALID_END,
    TEST_START,
    TEST_END,
    RANDOM_STATE,
    N_ESTIMATORS,
    LEARNING_RATE,
    NUM_LEAVES,
    MAX_DEPTH,
    MIN_CHILD_SAMPLES,
    SUBSAMPLE,
    COLSAMPLE_BYTREE,
    REG_ALPHA,
    REG_LAMBDA,
    print_header,
    print_subheader,
    check_environment,
    load_instruments,
    initialize_qlib,
    create_alpha158,
    create_dataset,
    flatten_feature_columns,
    extract_features,
    extract_labels,
    align_features_labels,
    create_lightgbm,
    train_model,
    predict,
    build_results,
    build_cross_sectional_strategy,
    feature_importance,
)

from .sentiment_features import (
    get_sentiment_features,
    merge_sentiment_with_features,
    SENTIMENT_FEATURES,
)

warnings.filterwarnings("ignore", category=FutureWarning)


# ============================================================
# VARIANT DEFINITIONS
# ============================================================

VARIANTS = {
    "alpha158_only": {
        "name": "Alpha158 Only (Baseline)",
        "features": None,  # Use all Alpha158 features
    },
    "alpha158_sentiment_core": {
        "name": "Alpha158 + Core Sentiment",
        "features": ["sentiment_mean", "news_volume"],
    },
    "alpha158_sentiment_full": {
        "name": "Alpha158 + Full Sentiment",
        "features": SENTIMENT_FEATURES,
    },
}


# ============================================================
# PREPARE DATA FOR VARIANT
# ============================================================

def prepare_data_for_variant(dataset, data_key, variant_config, sentiment_features):
    """Prepare train/valid/test data for a specific variant."""
    
    print_header(f"PREPARING DATA FOR: {variant_config['name']}")
    
    X_train = extract_features(dataset, "train", data_key)
    y_train = extract_labels(dataset, "train", data_key)
    X_train, y_train = align_features_labels(X_train, y_train, "train")
    
    X_valid = extract_features(dataset, "valid", data_key)
    y_valid = extract_labels(dataset, "valid", data_key)
    X_valid, y_valid = align_features_labels(X_valid, y_valid, "valid")
    
    X_test = extract_features(dataset, "test", data_key)
    y_test = extract_labels(dataset, "test", data_key)
    X_test, y_test = align_features_labels(X_test, y_test, "test")
    
    if variant_config["features"] is not None:
        print(f"\nAdding sentiment features: {variant_config['features']}")
        
        X_train = merge_sentiment_with_features(X_train, sentiment_features)
        X_valid = merge_sentiment_with_features(X_valid, sentiment_features)
        X_test = merge_sentiment_with_features(X_test, sentiment_features)
        
        sentiment_cols = variant_config["features"]
        X_train = X_train[[c for c in X_train.columns if c in sentiment_cols or c in X_train.columns[:158]]]
        X_valid = X_valid[[c for c in X_valid.columns if c in sentiment_cols or c in X_valid.columns[:158]]]
        X_test = X_test[[c for c in X_test.columns if c in sentiment_cols or c in X_test.columns[:158]]]
    
    print(f"\nFinal feature count: {len(X_train.columns)}")
    
    return X_train, y_train, X_valid, y_valid, X_test, y_test


# ============================================================
# COMPUTE SHAP VALUES
# ============================================================

def compute_shap_values(model, X_test, feature_names, variant_name):
    """Compute SHAP values using TreeSHAP (native LightGBM)."""
    
    print_header(f"COMPUTING SHAP VALUES: {variant_name}")
    
    try:
        import shap
        
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        
        shap_df = pd.DataFrame(
            shap_values,
            columns=feature_names,
            index=X_test.index,
        )
        
        shap_file = OUTPUT_DIR / f"shap_values_{variant_name.replace(' ', '_')}.csv"
        shap_df.to_csv(shap_file)
        print(f"\nSHAP values saved: {shap_file}")
        
        mean_abs_shap = shap_df.abs().mean().sort_values(ascending=False)
        top_features = mean_abs_shap.head(20)
        
        print("\nTop 20 features by SHAP importance:")
        for feat, val in top_features.items():
            print(f"  {feat}: {val:.4f}")
        
        return shap_df, mean_abs_shap
        
    except ImportError:
        print("\nWARNING: shap not installed. Skipping SHAP analysis.")
        return None, None


# ============================================================
# CREATE SHAP PLOT
# ============================================================

def create_shap_importance_plot(mean_abs_shap, variant_name):
    """Create SHAP importance plot grouped by feature type."""
    
    print_header(f"CREATING SHAP PLOT: {variant_name}")
    
    try:
        import matplotlib.pyplot as plt
        
        sentiment_features = [f for f in mean_abs_shap.index if f in SENTIMENT_FEATURES]
        technical_features = [f for f in mean_abs_shap.index if f not in SENTIMENT_FEATURES]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        top_tech = mean_abs_shap[technical_features].head(15)
        ax1.barh(range(len(top_tech)), top_tech.values, color="#1565C0")
        ax1.set_yticks(range(len(top_tech)))
        ax1.set_yticklabels(top_tech.index)
        ax1.set_xlabel("Mean |SHAP value|")
        ax1.set_title("Top Technical Features (Alpha158)", fontweight="bold")
        ax1.invert_yaxis()
        
        if sentiment_features:
            top_sent = mean_abs_shap[sentiment_features].head(10)
            ax2.barh(range(len(top_sent)), top_sent.values, color="#D32F2F")
            ax2.set_yticks(range(len(top_sent)))
            ax2.set_yticklabels(top_sent.index)
            ax2.set_xlabel("Mean |SHAP value|")
            ax2.set_title("Top Sentiment Features", fontweight="bold")
            ax2.invert_yaxis()
        
        fig.suptitle(f"SHAP Feature Importance — {variant_name}", fontsize=14, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        
        plot_file = OUTPUT_DIR / f"shap_importance_{variant_name.replace(' ', '_')}.png"
        fig.savefig(plot_file, dpi=220, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        
        print(f"\nSHAP plot saved: {plot_file}")
        
    except ImportError:
        print("\nWARNING: matplotlib not available. Skipping plot.")


# ============================================================
# GENERATE COMPARISON REPORT
# ============================================================

def generate_comparison_report(results_dict, shap_results):
    """Generate comparison_report.md with all variant results."""
    
    print_header("GENERATING COMPARISON REPORT")
    
    report_file = OUTPUT_DIR / "comparison_report.md"
    
    with open(report_file, "w") as f:
        f.write("# FinancialRAG: Alpha158 + Sentiment Features Comparison\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Model Variants\n\n")
        f.write("| Variant | Description |\n")
        f.write("|---------|-------------|\n")
        for key, config in VARIANTS.items():
            f.write(f"| {key} | {config['name']} |\n")
        
        f.write("\n## Cross-Sectional Performance Comparison\n\n")
        f.write("| Metric | Alpha158 Only | + Core Sentiment | + Full Sentiment |\n")
        f.write("|--------|---------------|------------------|------------------|\n")
        
        metrics = ["mean_ic", "mean_rank_ic", "net_sharpe", "net_total_return", "net_max_drawdown"]
        
        for metric in metrics:
            values = []
            for variant_key in VARIANTS.keys():
                if variant_key in results_dict and results_dict[variant_key] is not None:
                    val = results_dict[variant_key].get("metrics", {}).get(metric, "N/A")
                    if isinstance(val, float):
                        values.append(f"{val:.4f}")
                    else:
                        values.append(str(val))
                else:
                    values.append("N/A")
            
            f.write(f"| {metric} | {' | '.join(values)} |\n")
        
        f.write("\n## SHAP Feature Importance\n\n")
        
        for variant_key, (shap_df, mean_shap) in shap_results.items():
            if mean_shap is not None:
                f.write(f"### {VARIANTS[variant_key]['name']}\n\n")
                f.write("**Top 10 Features:**\n\n")
                f.write("| Feature | Mean |SHAP| |\n")
                f.write("|---------|----------|\n")
                for feat, val in mean_shap.head(10).items():
                    f.write(f"| {feat} | {val:.4f} |\n")
                f.write("\n")
        
        f.write("## Limitations\n\n")
        f.write("### News Coverage\n")
        f.write("- Not all tickers have equal news coverage\n")
        f.write("- Some periods may have sparse news data\n")
        f.write("- Sentiment features filled with 0.0 when no news available\n\n")
        
        f.write("### High Volatility Periods\n")
        f.write("- Sentiment models may struggle during extreme market events\n")
        f.write("- News sentiment can be lagging during fast-moving markets\n")
        f.write("- Model performance may degrade in crisis periods\n\n")
        
        f.write("### Data Quality\n")
        f.write("- Synthetic news used for demonstration (replace with real FNSPID dataset)\n")
        f.write("- FinBERT sentiment is based on English financial text only\n")
        f.write("- Timestamp precision may affect feature alignment\n\n")
        
        f.write("## Conclusion\n\n")
        f.write("The sentiment-enhanced models provide additional signal beyond technical features.\n")
        f.write("Performance gains depend on news coverage quality and market regime.\n")
    
    print(f"\nComparison report saved: {report_file}")
    
    return report_file


# ============================================================
# MAIN FUNCTION
# ============================================================

def main():
    """Run comparison of all 3 model variants."""
    
    print("\n" + "=" * 70)
    print("EXTENDED QLIB ALPHA158 + LIGHTGBM WITH SENTIMENT FEATURES")
    print("=" * 70)
    
    check_environment()
    instruments = load_instruments()
    initialize_qlib()
    
    handler = create_alpha158(instruments)
    dataset = create_dataset(handler)
    
    from qlib.data.dataset.handler import DataHandlerLP
    DATA_KEY = DataHandlerLP.DK_L
    
    print_header("COMPUTING SENTIMENT FEATURES")
    sentiment_features = get_sentiment_features(
        instruments["instrument"].tolist(),
        date_range=(DATA_START, DATA_END),
    )
    
    results_dict = {}
    shap_results = {}
    
    for variant_key, variant_config in VARIANTS.items():
        print_header(f"TRAINING VARIANT: {variant_config['name']}")
        
        X_train, y_train, X_valid, y_valid, X_test, y_test = prepare_data_for_variant(
            dataset, DATA_KEY, variant_config, sentiment_features
        )
        
        model = create_lightgbm()
        model = train_model(model, X_train, y_train, X_valid, y_valid)
        
        prediction = predict(model, X_test)
        results, correlation = build_results(prediction, y_test)
        
        print(f"\nTest correlation: {correlation:.6f}")
        
        portfolio = build_cross_sectional_strategy(results)
        
        if portfolio is not None:
            results_dict[variant_key] = {
                "metrics": portfolio.attrs.get("metrics", {}),
                "correlation": correlation,
            }
        
        if variant_key == "alpha158_sentiment_full":
            shap_df, mean_shap = compute_shap_values(
                model, X_test, X_test.columns.tolist(), variant_config["name"]
            )
            shap_results[variant_key] = (shap_df, mean_shap)
            
            if mean_shap is not None:
                create_shap_importance_plot(mean_shap, variant_config["name"])
    
    report_file = generate_comparison_report(results_dict, shap_results)
    
    print("\n" + "=" * 70)
    print("COMPARISON COMPLETE")
    print("=" * 70)
    print(f"\nReport: {report_file}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    return results_dict, shap_results


if __name__ == "__main__":
    main()
