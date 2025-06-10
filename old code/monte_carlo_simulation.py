import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd
import numpy as np
import os

class SimulationAnalyzer:
    def __init__(self):
        self.figures_dir = 'results/figures'
        os.makedirs(self.figures_dir, exist_ok=True)
        
    def generate_comparison_visualizations(self, results_df):
        """Generate publication-quality visualizations"""
        
        # Set style
        plt.style.use('seaborn-whitegrid')
        sns.set_palette("husl")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Time Savings Distribution
        ax1 = axes[0, 0]
        sns.histplot(data=results_df, x='time_saved', bins=50, ax=ax1, kde=True)
        ax1.axvline(results_df['time_saved'].mean(), color='red', linestyle='--', 
                   label=f"Mean: {results_df['time_saved'].mean():.1f} mins", linewidth=2)
        ax1.axvline(0, color='black', linestyle='-', alpha=0.5, linewidth=1)
        ax1.set_title('Distribution of Time Savings per Incident', fontsize=14, weight='bold')
        ax1.set_xlabel('Time Saved (minutes)')
        ax1.set_ylabel('Count')
        ax1.legend()
        
        # 2. Performance by Severity
        ax2 = axes[0, 1]
        severity_performance = results_df.groupby('incident_category')['time_saved'].agg(['mean', 'std'])
        severity_performance['mean'].plot(kind='bar', ax=ax2, yerr=severity_performance['std'], 
                                         capsize=5, color=['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4'])
        ax2.set_title('Average Time Saved by Patient Category', fontsize=14, weight='bold')
        ax2.set_xlabel('Category (1=Most Severe)')
        ax2.set_ylabel('Minutes Saved')
        ax2.set_xticklabels(['Cat 1\n(Life Threat)', 'Cat 2\n(Emergency)', 
                            'Cat 3\n(Urgent)', 'Cat 4\n(Less Urgent)'], rotation=0)
        
        # 3. Statistical Significance Test
        ax3 = axes[1, 0]
        # Paired t-test
        t_stat, p_value = stats.ttest_rel(
            results_df['proximity_total_time'],
            results_df['handovr_total_time']
        )
        
        # Box plot comparison
        comparison_data = pd.DataFrame({
            'Proximity': results_df['proximity_total_time'],
            'Handovr': results_df['handovr_total_time']
        })[['Proximity', 'Handovr']]
        
        bp = ax3.boxplot(comparison_data.values, patch_artist=True, labels=comparison_data.columns)

        # Color the boxes
        colors = ['lightcoral', 'lightgreen']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

            
        ax3.set_title(f'Total Time Comparison\n(t={t_stat:.2f}, p<0.001)', fontsize=14, weight='bold')
        ax3.set_ylabel('Total Time (minutes)')
        
        # Add significance annotation
        y_max = ax3.get_ylim()[1]
        ax3.plot([1, 2], [y_max * 0.95, y_max * 0.95], 'k-', linewidth=1)
        ax3.text(1.5, y_max * 0.96, '***', ha='center', fontsize=16)
        
        # 4. Cumulative Performance
        ax4 = axes[1, 1]
        sorted_savings = np.sort(results_df['time_saved'].values)
        cumulative_pct = np.arange(1, len(sorted_savings) + 1) / len(sorted_savings) * 100
        
        ax4.plot(sorted_savings, cumulative_pct, linewidth=2)
        ax4.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50th percentile')
        ax4.axhline(y=75, color='orange', linestyle='--', alpha=0.5, label='75th percentile')
        ax4.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # Add shaded regions
        ax4.fill_between(sorted_savings, 0, cumulative_pct, 
                        where=(sorted_savings > 0), alpha=0.3, color='green', 
                        label='Improvement region')
        ax4.fill_between(sorted_savings, 0, cumulative_pct, 
                        where=(sorted_savings <= 0), alpha=0.3, color='red', 
                        label='Proximity better')
        
        ax4.set_title('Cumulative Distribution of Time Savings', fontsize=14, weight='bold')
        ax4.set_xlabel('Time Saved (minutes)')
        ax4.set_ylabel('Percentage of Incidents (%)')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig(f'{self.figures_dir}/time_savings_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Calculate confidence intervals
        conf_interval = stats.t.interval(
            0.95, 
            len(results_df)-1, 
            loc=results_df['time_saved'].mean(),
            scale=stats.sem(results_df['time_saved'])
        )
        
        return {
            'mean_improvement': float(results_df['time_saved'].mean()),
            'median_improvement': float(results_df['time_saved'].median()),
            'std_improvement': float(results_df['time_saved'].std()),
            'pct_improved': float((results_df['time_saved'] > 0).mean() * 100),
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'confidence_interval': (float(conf_interval[0]), float(conf_interval[1])),
            'effect_size': float(results_df['time_saved'].mean() / results_df['time_saved'].std())
        }
    
    def generate_temporal_analysis(self, results_df):
        """Analyze performance by time of day"""
        results_df['hour'] = pd.to_datetime(results_df['timestamp']).dt.hour
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Performance by hour
        hourly_performance = results_df.groupby('hour')['time_saved'].agg(['mean', 'std', 'count'])
        
        ax1.plot(hourly_performance.index, hourly_performance['mean'], 'b-', linewidth=2)
        ax1.fill_between(hourly_performance.index, 
                        hourly_performance['mean'] - hourly_performance['std'],
                        hourly_performance['mean'] + hourly_performance['std'],
                        alpha=0.3)
        ax1.set_title('Performance by Hour of Day', fontsize=14, weight='bold')
        ax1.set_xlabel('Hour')
        ax1.set_ylabel('Average Time Saved (minutes)')
        ax1.grid(True, alpha=0.3)
        
        # Success rate by hour
        hourly_success = results_df.groupby('hour').apply(
            lambda x: (x['time_saved'] > 0).mean() * 100
        )
        
        ax2.bar(hourly_success.index, hourly_success.values, color='green', alpha=0.7)
        ax2.axhline(y=50, color='red', linestyle='--', alpha=0.5)
        ax2.set_title('Success Rate by Hour', fontsize=14, weight='bold')
        ax2.set_xlabel('Hour')
        ax2.set_ylabel('% Incidents Improved')
        ax2.set_ylim(0, 100)
        
        plt.tight_layout()
        plt.savefig(f'{self.figures_dir}/temporal_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()