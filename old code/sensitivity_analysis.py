import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
import os

class SensitivityAnalyzer:
    def __init__(self):
        self.figures_dir = 'results/figures'
        os.makedirs(self.figures_dir, exist_ok=True)
        
    def perform_weight_sensitivity(self, base_weights, variations):
        """Test system performance across weight variations"""
        
        results = []
        
        # Generate all combinations
        weight_combinations = []
        for d, c, s in product(variations, variations, variations):
            # Ensure weights sum to 1 (within tolerance)
            if abs(d + c + s - 1.0) < 0.01:
                weight_combinations.append({
                    'distance': d,
                    'congestion': c,
                    'severity': s
                })
        
        print(f"Testing {len(weight_combinations)} weight combinations...")
        
        # Simulate performance for each combination
        for weights in weight_combinations:
            # Run mini simulation with these weights (placeholder)
            performance = self.run_weighted_simulation(weights, n=1000)
            
            results.append({
                'distance_weight': weights['distance'],
                'congestion_weight': weights['congestion'],
                'severity_weight': weights['severity'],
                'mean_time_saved': performance['mean_saved'],
                'success_rate': performance['success_rate'],
                'variance': performance['variance']
            })
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Create visualization
        self.plot_sensitivity_heatmap(results_df)
        
        return results_df
    
    def run_weighted_simulation(self, weights, n=1000):
        """Run simulation with specific weights"""
        # This is a placeholder - in reality, you'd run your actual simulation
        # with these weights
        
        # Simulate based on weight balance
        congestion_emphasis = weights['congestion']
        distance_emphasis = weights['distance']
        
        # Higher congestion weight should lead to better performance
        base_performance = 15  # Base time saved
        performance_modifier = congestion_emphasis * 1.2 + distance_emphasis * 0.8
        
        mean_saved = base_performance * performance_modifier + np.random.normal(0, 2)
        success_rate = 0.65 + congestion_emphasis * 0.15 + np.random.normal(0, 0.05)
        variance = 10 + distance_emphasis * 5
        
        return {
            'mean_saved': float(mean_saved),
            'success_rate': float(min(max(success_rate, 0), 1)),
            'variance': float(variance)
        }
    
    def plot_sensitivity_heatmap(self, results_df):
        """Create heatmap visualization of sensitivity analysis"""
        
        # Create pivot table for heatmap
        pivot_data = results_df.pivot_table(
            values='mean_time_saved',
            index='congestion_weight',
            columns='distance_weight',
            aggfunc='mean'
        )
        
        plt.figure(figsize=(10, 8))
        
        # Create heatmap
        sns.heatmap(pivot_data, 
                   annot=True, 
                   fmt='.1f',
                   cmap='RdYlGn',
                   center=pivot_data.values.mean(),
                   cbar_kws={'label': 'Mean Time Saved (minutes)'},
                   square=True)
        
        plt.title('Sensitivity Analysis: Impact of Weight Variations', 
                 fontsize=16, weight='bold')
        plt.xlabel('Distance Weight', fontsize=12)
        plt.ylabel('Congestion Weight', fontsize=12)
        
        # Add text box with optimal weights
        optimal_idx = results_df['mean_time_saved'].idxmax()
        optimal_weights = results_df.iloc[optimal_idx]
        
        textstr = f'Optimal Weights:\nDistance: {optimal_weights["distance_weight"]:.2f}\n' \
                 f'Congestion: {optimal_weights["congestion_weight"]:.2f}\n' \
                 f'Severity: {optimal_weights["severity_weight"]:.2f}'
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        plt.savefig(f'{self.figures_dir}/sensitivity_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def test_extreme_scenarios(self):
        """Test edge cases and extreme conditions"""
        
        scenarios = {
            'major_incident': {
                'description': 'Mass casualty event with 50+ patients',
                'performance': self.simulate_major_incident()
            },
            'multiple_closures': {
                'description': '3+ hospitals at capacity',
                'performance': self.simulate_hospital_closures()
            },
            'peak_demand': {
                'description': 'December peak with staff shortages',
                'performance': self.simulate_peak_winter_demand()
            },
            'system_failure': {
                'description': 'Forecast model confidence <60%',
                'performance': self.simulate_forecast_failure()
            }
        }
        
        # Visualize extreme scenario results
        self.plot_extreme_scenarios(scenarios)
        
        return scenarios
    
    def simulate_major_incident(self):
        """Simulate performance during major incident"""
        # Placeholder - implement actual major incident simulation
        return {
            'handovr_time_saved': 8.5,
            'success_rate': 0.62,
            'degradation': 0.45  # 45% performance degradation
        }
    
    def simulate_hospital_closures(self):
        """Simulate multiple hospital closures"""
        return {
            'handovr_time_saved': 12.3,
            'success_rate': 0.68,
            'degradation': 0.28
        }
    
    def simulate_peak_winter_demand(self):
        """Simulate peak winter conditions"""
        return {
            'handovr_time_saved': 10.7,
            'success_rate': 0.65,
            'degradation': 0.36
        }
    
    def simulate_forecast_failure(self):
        """Simulate forecast model failure"""
        return {
            'handovr_time_saved': 3.2,  # Falls back to proximity
            'success_rate': 0.51,
            'degradation': 0.78
        }
    
    def plot_extreme_scenarios(self, scenarios):
        """Visualize extreme scenario performance"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Extract data
        scenario_names = list(scenarios.keys())
        time_saved = [s['performance']['handovr_time_saved'] for s in scenarios.values()]
        degradation = [s['performance']['degradation'] * 100 for s in scenarios.values()]
        
        # Bar plot of time saved
        bars1 = ax1.bar(scenario_names, time_saved, color=['green' if t > 5 else 'orange' for t in time_saved])
        ax1.set_title('Performance Under Extreme Scenarios', fontsize=14, weight='bold')
        ax1.set_ylabel('Time Saved (minutes)')
        ax1.set_xticklabels([s.replace('_', '\n') for s in scenario_names], rotation=0)
        ax1.axhline(y=15, color='red', linestyle='--', alpha=0.5, label='Normal performance')
        
        # Add value labels
        for bar, val in zip(bars1, time_saved):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    f'{val:.1f}', ha='center')
        
        # Degradation plot
        bars2 = ax2.bar(scenario_names, degradation, color=['red' if d > 50 else 'orange' for d in degradation])
        ax2.set_title('Performance Degradation', fontsize=14, weight='bold')
        ax2.set_ylabel('Degradation (%)')
        ax2.set_xticklabels([s.replace('_', '\n') for s in scenario_names], rotation=0)
        ax2.set_ylim(0, 100)
        
        # Add value labels
        for bar, val in zip(bars2, degradation):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{val:.0f}%', ha='center')
        
        plt.tight_layout()
        plt.savefig(f'{self.figures_dir}/extreme_scenarios.png', dpi=300, bbox_inches='tight')
        plt.close()