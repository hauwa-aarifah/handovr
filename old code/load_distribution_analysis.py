import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def analyze_load_distribution(results_df):
    """Analyze hospital load balancing improvements"""
    
    figures_dir = 'results/figures'
    os.makedirs(figures_dir, exist_ok=True)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Hospital utilization distribution
    proximity_loads = results_df.groupby('proximity_hospital').size()
    handovr_loads = results_df.groupby('handovr_hospital').size()
    
    # Get all hospitals and ensure consistent ordering
    all_hospitals = sorted(list(set(proximity_loads.index) | set(handovr_loads.index)))
    
    # Create comparison dataframe
    load_comparison = pd.DataFrame({
        'Proximity': [proximity_loads.get(h, 0) for h in all_hospitals],
        'Handovr': [handovr_loads.get(h, 0) for h in all_hospitals]
    }, index=all_hospitals)
    
    # Sort by proximity load for better visualization
    load_comparison = load_comparison.sort_values('Proximity', ascending=False)
    
    # Plot comparison
    x = np.arange(len(load_comparison))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, load_comparison['Proximity'], width, 
                    label='Proximity', color='lightcoral')
    bars2 = ax1.bar(x + width/2, load_comparison['Handovr'], width, 
                    label='Handovr', color='lightgreen')
    
    ax1.set_title('Incident Distribution by Hospital', fontsize=14, weight='bold')
    ax1.set_xlabel('Hospital')
    ax1.set_ylabel('Number of Incidents')
    ax1.set_xticks(x)
    ax1.set_xticklabels([h.split()[0] for h in load_comparison.index], rotation=45, ha='right')
    ax1.legend()
    
    # Add mean lines
    ax1.axhline(y=proximity_loads.mean(), color='red', linestyle='--', alpha=0.5)
    ax1.axhline(y=handovr_loads.mean(), color='green', linestyle='--', alpha=0.5)
    
    # 2. Variance comparison
    metrics = pd.DataFrame({
        'Metric': ['Variance', 'Std Dev', 'Range', 'CV'],
        'Proximity': [
            proximity_loads.var(),
            proximity_loads.std(),
            proximity_loads.max() - proximity_loads.min(),
            proximity_loads.std() / proximity_loads.mean()
        ],
        'Handovr': [
            handovr_loads.var(),
            handovr_loads.std(),
            handovr_loads.max() - handovr_loads.min(),
            handovr_loads.std() / handovr_loads.mean()
        ]
    })
    
    # Normalize for visualization
    metrics_norm = metrics.copy()
    for col in ['Proximity', 'Handovr']:
        metrics_norm[col] = metrics[col] / metrics['Proximity']
    
    metrics_norm.set_index('Metric')[['Proximity', 'Handovr']].plot(
        kind='bar', ax=ax2, color=['lightcoral', 'lightgreen']
    )
    ax2.set_title('Load Distribution Metrics (Normalized)', fontsize=14, weight='bold')
    ax2.set_ylabel('Relative Value')
    ax2.set_xticklabels(metrics['Metric'], rotation=0)
    ax2.legend(['Proximity', 'Handovr'])
    
    # 3. Lorenz curve for load distribution
    def lorenz_curve(loads):
        sorted_loads = np.sort(loads)
        cumsum = np.cumsum(sorted_loads)
        return np.insert(cumsum / cumsum[-1], 0, 0)
    
    proximity_lorenz = lorenz_curve(proximity_loads.values)
    handovr_lorenz = lorenz_curve(handovr_loads.values)
    x_lorenz_prox = np.linspace(0, 1, len(proximity_lorenz))
    x_lorenz_handovr = np.linspace(0, 1, len(handovr_lorenz))

    
    ax3.plot(x_lorenz_prox, proximity_lorenz, 'r-', linewidth=2, label='Proximity')
    ax3.plot(x_lorenz_handovr, handovr_lorenz, 'g-', linewidth=2, label='Handovr')
    ax3.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect equality')
    ax3.fill_between(x_lorenz_prox, x_lorenz_prox, proximity_lorenz, alpha=0.3, color='red')
    ax3.fill_between(x_lorenz_handovr, x_lorenz_handovr, handovr_lorenz, alpha=0.3, color='green')

    
    ax3.set_title('Lorenz Curve - Load Distribution Equality', fontsize=14, weight='bold')
    ax3.set_xlabel('Cumulative % of Hospitals')
    ax3.set_ylabel('Cumulative % of Incidents')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Improvement visualization
    variance_reduction = ((proximity_loads.var() - handovr_loads.var()) / 
                         proximity_loads.var() * 100)
    
    improvements = {
        'Variance\nReduction': variance_reduction,
        'Max Load\nReduction': ((proximity_loads.max() - handovr_loads.max()) / 
                               proximity_loads.max() * 100),
        'Equity\nImprovement': ((calculate_gini(handovr_loads) - calculate_gini(proximity_loads)) / 
                               calculate_gini(proximity_loads) * -100)
    }
    
    bars = ax4.bar(improvements.keys(), improvements.values(), 
                   color=['green' if v > 0 else 'red' for v in improvements.values()])
    ax4.set_title('Load Balancing Improvements', fontsize=14, weight='bold')
    ax4.set_ylabel('Improvement (%)')
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # Add value labels
    for bar, (key, val) in zip(bars, improvements.items()):
        ax4.text(bar.get_x() + bar.get_width()/2, 
                bar.get_height() + (2 if val > 0 else -5), 
                f'{val:.1f}%', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{figures_dir}/load_balancing_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'variance_reduction': variance_reduction,
        'gini_coefficient_proximity': calculate_gini(proximity_loads),
        'gini_coefficient_handovr': calculate_gini(handovr_loads),
        'max_load_proximity': int(proximity_loads.max()),
        'max_load_handovr': int(handovr_loads.max()),
        'cv_proximity': float(proximity_loads.std() / proximity_loads.mean()),
        'cv_handovr': float(handovr_loads.std() / handovr_loads.mean())
    }

def calculate_gini(loads):
    """Calculate Gini coefficient for load distribution"""
    sorted_loads = np.sort(loads)
    n = len(sorted_loads)
    cumsum = np.cumsum(sorted_loads)
    return (2 * np.sum((np.arange(1, n+1) * sorted_loads))) / (n * cumsum[-1]) - (n + 1) / n