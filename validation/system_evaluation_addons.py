# validation/system_evaluation_addons.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def generate_additional_metrics(results_df, output_dir='results'):
    os.makedirs(output_dir, exist_ok=True)

    metrics = {}

    # Equity: Coefficient of Variation (CV)
    handovr_dist = results_df['handovr_hospital'].value_counts()
    baseline_dist = results_df['baseline_hospital'].value_counts()
    
    metrics['equity'] = {
        'handovr_cv': float(handovr_dist.std() / handovr_dist.mean()),
        'baseline_cv': float(baseline_dist.std() / baseline_dist.mean())
    }

    # Extreme case analysis
    metrics['extremes'] = {
        'big_losses': int((results_df['time_saved'] < -10).sum()),
        'big_gains': int((results_df['time_saved'] > 30).sum())
    }

    # Borough performance
    borough_summary = results_df.groupby('borough')['time_saved'].agg(['mean', 'count']).reset_index()
    borough_summary.columns = ['Borough', 'MeanTimeSaved', 'IncidentCount']
    borough_summary.to_csv(f"{output_dir}/time_saved_by_borough.csv", index=False)

    return metrics


def plot_borough_performance_heatmap(csv_path='results/time_saved_by_borough.csv', save_path='results/figures/borough_heatmap.png'):
    df = pd.read_csv(csv_path)
    
    # Sort for better color scale
    df_sorted = df.sort_values('MeanTimeSaved', ascending=False)

    plt.figure(figsize=(12, 8))
    sns.barplot(x='MeanTimeSaved', y='Borough', data=df_sorted, palette='coolwarm')
    plt.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    plt.title('Average Time Saved by Borough')
    plt.xlabel('Time Saved (minutes)')
    plt.ylabel('Borough')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()


# Example usage in your main system_evaluation.py:
# from validation.system_evaluation_addons import generate_additional_metrics, plot_borough_performance_heatmap
# extra_metrics = generate_additional_metrics(results_df)
# plot_borough_performance_heatmap()
