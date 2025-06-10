# validation/visualize_results.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

class ValidationVisualizer:
    def __init__(self, results_path='results/system_evaluation_results.csv'):
        """Initialize with results from system evaluation"""
        self.results = pd.read_csv(results_path)
        self.figures_dir = 'results/figures'
        os.makedirs(self.figures_dir, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
    
    def create_all_visualizations(self):
        """Generate all visualizations"""
        self.plot_time_savings_distribution()
        self.plot_performance_by_category()
        self.plot_hospital_selection_comparison()
        self.plot_time_of_day_analysis()
        self.plot_cumulative_performance()
        self.plot_incident_type_performance()
        print(f"All visualizations saved to {self.figures_dir}/")
    
    def plot_time_savings_distribution(self):
        """1. Time Savings Distribution"""
        plt.figure(figsize=(10, 6))
        plt.hist(self.results['time_saved'], bins=30, edgecolor='black', alpha=0.7)
        plt.axvline(self.results['time_saved'].mean(), color='red', linestyle='--', 
                    label=f'Mean: {self.results["time_saved"].mean():.1f} min')
        plt.axvline(0, color='black', linestyle='-', alpha=0.5)
        plt.xlabel('Time Saved (minutes)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Time Savings: Handovr vs Proximity Selection')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{self.figures_dir}/time_savings_distribution.png', dpi=300)
        plt.close()
    
    def plot_performance_by_category(self):
        """2. Performance by Patient Category"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Mean time saved by category
        category_performance = self.results.groupby('category')['time_saved'].agg(['mean', 'std'])
        category_performance['mean'].plot(kind='bar', ax=ax1, yerr=category_performance['std'], 
                                         capsize=5, color=['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4'])
        ax1.set_xlabel('Patient Category')
        ax1.set_ylabel('Mean Time Saved (minutes)')
        ax1.set_title('Average Time Saved by Patient Severity')
        ax1.set_xticklabels(['Cat 1\n(Critical)', 'Cat 2\n(Emergency)', 
                             'Cat 3\n(Urgent)', 'Cat 4\n(Standard)'], rotation=0)
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Success rate by category
        success_rate = self.results.groupby('category')['time_saved'].apply(lambda x: (x > 0).mean() * 100)

        success_rate.plot(kind='bar', ax=ax2, color=['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4'])
        ax2.set_xlabel('Patient Category')
        ax2.set_ylabel('Success Rate (%)')
        ax2.set_title('Percentage of Incidents Improved by Category')
        ax2.set_ylim(0, 100)
        ax2.set_xticklabels(['Cat 1\n(Critical)', 'Cat 2\n(Emergency)', 
                             'Cat 3\n(Urgent)', 'Cat 4\n(Standard)'], rotation=0)
        
        plt.tight_layout()
        plt.savefig(f'{self.figures_dir}/performance_by_category.png', dpi=300)
        plt.close()
    
    def plot_hospital_selection_comparison(self):
        """3. Hospital Selection Comparison"""
        handovr_counts = self.results['handovr_hospital'].value_counts()
        baseline_counts = self.results['baseline_hospital'].value_counts()
        
        # Get top 10 hospitals
        all_hospitals = pd.concat([handovr_counts, baseline_counts]).index.unique()
        top_hospitals = handovr_counts.index[:10] if len(handovr_counts) >= 10 else all_hospitals[:10]
        
        comparison_df = pd.DataFrame({
            'Handovr': [handovr_counts.get(h, 0) for h in top_hospitals],
            'Baseline': [baseline_counts.get(h, 0) for h in top_hospitals]
        }, index=[h.split()[0] + '...' if len(h) > 20 else h for h in top_hospitals])
        
        comparison_df.plot(kind='bar', figsize=(12, 6), color=['#2ca02c', '#d62728'])
        plt.title('Hospital Selection Frequency: Handovr vs Baseline')
        plt.xlabel('Hospital')
        plt.ylabel('Number of Incidents Assigned')
        plt.xticks(rotation=45, ha='right')
        plt.legend(['Handovr', 'Proximity-based'])
        plt.tight_layout()
        plt.savefig(f'{self.figures_dir}/hospital_selection_comparison.png', dpi=300)
        plt.close()
    
    def plot_time_of_day_analysis(self):
        """4. Time of Day Analysis"""
        self.results['hour'] = pd.to_datetime(self.results['timestamp']).dt.hour
        hourly_performance = self.results.groupby('hour')['time_saved'].agg(['mean', 'std', 'count'])
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Average time saved by hour
        ax1.plot(hourly_performance.index, hourly_performance['mean'], 'b-o', linewidth=2, markersize=8)
        ax1.fill_between(hourly_performance.index, 
                        hourly_performance['mean'] - hourly_performance['std'],
                        hourly_performance['mean'] + hourly_performance['std'],
                        alpha=0.3)
        ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax1.set_xlabel('Hour of Day')
        ax1.set_ylabel('Average Time Saved (minutes)')
        ax1.set_title('Performance by Time of Day')
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(range(0, 24))
        
        # Number of incidents by hour
        ax2.bar(hourly_performance.index, hourly_performance['count'], alpha=0.7, color='orange')
        ax2.set_xlabel('Hour of Day')
        ax2.set_ylabel('Number of Incidents')
        ax2.set_title('Incident Distribution by Hour')
        ax2.set_xticks(range(0, 24))
        
        plt.tight_layout()
        plt.savefig(f'{self.figures_dir}/time_of_day_analysis.png', dpi=300)
        plt.close()
    
    def plot_cumulative_performance(self):
        """5. Cumulative Performance"""
        sorted_savings = np.sort(self.results['time_saved'])
        cumulative = np.arange(1, len(sorted_savings) + 1) / len(sorted_savings) * 100
        
        plt.figure(figsize=(10, 6))
        plt.plot(sorted_savings, cumulative, linewidth=3, color='#1f77b4')
        plt.axvline(x=0, color='red', linestyle='--', alpha=0.5, label='Break-even')
        plt.axhline(y=50, color='gray', linestyle=':', alpha=0.5)
        
        # Fill areas
        plt.fill_between(sorted_savings[sorted_savings > 0], 
                        cumulative[sorted_savings > 0], 
                        100, alpha=0.3, color='green', label='Improvement region')
        plt.fill_between(sorted_savings[sorted_savings <= 0], 
                        0, 
                        cumulative[sorted_savings <= 0], 
                        alpha=0.3, color='red', label='Baseline better')
        
        plt.xlabel('Time Saved (minutes)')
        plt.ylabel('Cumulative Percentage (%)')
        plt.title('Cumulative Distribution of Time Savings')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Add annotations
        pct_improved = (self.results['time_saved'] > 0).mean() * 100
        plt.annotate(f'{pct_improved:.1f}% improved', 
                     xy=(0, pct_improved), xytext=(10, pct_improved-10),
                     arrowprops=dict(arrowstyle='->', color='red'),
                     fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.figures_dir}/cumulative_performance.png', dpi=300)
        plt.close()
    
    def plot_incident_type_performance(self):
        """6. Incident Type Performance"""
        plt.figure(figsize=(12, 6))
        
        # Create box plot
        incident_types = self.results['incident_type'].unique()
        data_to_plot = [self.results[self.results['incident_type'] == it]['time_saved'].values 
                       for it in incident_types]
        
        bp = plt.boxplot(data_to_plot, labels=incident_types, patch_artist=True)
        
        # Color the boxes
        colors = plt.cm.Set3(np.linspace(0, 1, len(incident_types)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        plt.xlabel('Incident Type')
        plt.ylabel('Time Saved (minutes)')
        plt.title('Time Savings Distribution by Incident Type')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(f'{self.figures_dir}/incident_type_performance.png', dpi=300)
        plt.close()
    
    def generate_summary_statistics(self):
        """Generate summary statistics table"""
        summary = {
            'Total Incidents': len(self.results),
            'Mean Time Saved (min)': self.results['time_saved'].mean(),
            'Median Time Saved (min)': self.results['time_saved'].median(),
            'Std Dev (min)': self.results['time_saved'].std(),
            'Success Rate (%)': (self.results['time_saved'] > 0).mean() * 100,
            'Total Time Saved (hours)': self.results['time_saved'].sum() / 60,
            'Best Performance (min)': self.results['time_saved'].max(),
            'Worst Performance (min)': self.results['time_saved'].min()
        }
        
        summary_df = pd.DataFrame(summary.items(), columns=['Metric', 'Value'])
        summary_df.to_csv(f'{self.figures_dir}/summary_statistics.csv', index=False)
        
        print("\nSummary Statistics:")
        print("-" * 40)
        for metric, value in summary.items():
            print(f"{metric}: {value:.2f}")
        
        return summary_df

# To use this:
if __name__ == "__main__":
    visualizer = ValidationVisualizer('results/system_evaluation_results.csv')
    visualizer.create_all_visualizations()
    visualizer.generate_summary_statistics()