# analyse skewness with charts and descrptions
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
def analyze_skewness(series, plot_kde=True):
    """
    Analyze the skewness of a pandas Series and provide interpretation with visualization.
    
    Parameters:
    - series: pandas Series to analyze
    - plot_kde: whether to display the KDE plot (default True)
    
    Returns:
    - Dictionary containing skewness value and interpretation
    - Displays KDE plot if plot_kde=True
    """
    # Calculate skewness
    skew_val = series.skew()
    
    # Interpret skewness value
    if pd.isna(skew_val):
        interpretation = "Unable to calculate skewness (possibly constant series)"
    elif abs(skew_val) < 0.5:
        interpretation = "Approximately symmetric distribution"
    elif 0.5 <= abs(skew_val) < 1:
        interpretation = "Moderately skewed distribution"
    else:
        interpretation = "Highly skewed distribution"
    
    interpretation += f" (skewness = {skew_val:.3f})"
    
    if skew_val > 0:
        interpretation += "\nPositive skew: Right-tailed distribution (mean > median)"
    elif skew_val < 0:
        interpretation += "\nNegative skew: Left-tailed distribution (mean < median)"
    else:
        interpretation += "\nPerfectly symmetric (mean ≈ median)"
    
    # Create results dictionary
    result = {
        'skewness': skew_val,
        'interpretation': interpretation,
        'mean': series.mean(),
        'median': series.median()
    }
    
    # Plot KDE if requested
    if plot_kde:
        plt.figure(figsize=(10, 6))
        
        # Plot actual data KDE
        sns.kdeplot(series, color='blue', label='Actual Data', linewidth=2)
        
        # Plot reference normal distribution
        if len(series) > 1:
            xmin, xmax = series.min(), series.max()
            x = np.linspace(xmin, xmax, 1000)
            normal_pdf = stats.norm.pdf(x, loc=series.mean(), scale=series.std())
            plt.plot(x, normal_pdf, 'r--', label='Normal Dist', linewidth=1.5)
        
        # Add plot decorations
        plt.title(f'Distribution of Values\nSkewness = {skew_val:.3f}', fontsize=14)
        plt.xlabel('Value', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Show vertical lines for mean and median
        plt.axvline(series.mean(), color='green', linestyle='--', linewidth=1, label=f'Mean ({series.mean():.2f})')
        plt.axvline(series.median(), color='purple', linestyle=':', linewidth=1, label=f'Median ({series.median():.2f})')
        plt.legend()
        
        plt.show()
    
    return result

# ------------------------------------------------------------------------------------------------


