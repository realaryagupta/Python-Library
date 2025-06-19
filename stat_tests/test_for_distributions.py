# test for uniform distribution
import pandas as pd
from scipy import stats
def ks_uniform_test(data_series, significance_level=0.05):
    """
    Perform Kolmogorov-Smirnov test for uniform distribution on a pandas Series.
    
    Parameters:
    - data_series: pandas Series to test
    - significance_level: threshold for rejecting null hypothesis (default 0.05)
    
    Returns:
    - Dictionary containing:
        'test_statistic': KS test statistic
        'p_value': p-value from the test
        'is_uniform': boolean indicating if data follows uniform distribution
        'conclusion': string with test interpretation
    """
    try:
        # Normalize data to [0,1] range if not already
        normalized_data = (data_series - data_series.min()) / (data_series.max() - data_series.min())
        
        # Perform KS test against uniform distribution
        test_statistic, p_value = stats.kstest(normalized_data, 'uniform')
        
        # Determine if uniform based on p-value
        is_uniform = p_value > significance_level
        
        conclusion = ("Data appears to be uniformly distributed (fail to reject H0)" 
                     if is_uniform 
                     else "Data does not appear to be uniformly distributed (reject H0)")
        
        return {
            'test_statistic': test_statistic,
            'p_value': p_value,
            'is_uniform': is_uniform,
            'conclusion': conclusion
        }
        
    except Exception as e:
        return {
            'error': str(e),
            'test_statistic': None,
            'p_value': None,
            'is_uniform': None,
            'conclusion': 'Test failed due to error'
        }
    
# final output like this 
# {'test_statistic': np.float64(0.23869343349637373),
#  'p_value': np.float64(1.2828958934919755e-50),
#  'is_uniform': np.False_,
#  'conclusion': 'Data does not appear to be uniformly distributed (reject H0)'}

# --------------------------------------------------------------------------------------------------

# test for normal distribution
import pandas as pd
from scipy import stats
def shapiro_normal_test(data_series, significance_level=0.05):
    """
    Perform Shapiro-Wilk test for normal distribution on a pandas Series.
    
    Parameters:
    - data_series: pandas Series to test
    - significance_level: threshold for rejecting null hypothesis (default 0.05)
    
    Returns:
    - Dictionary containing:
        'test_statistic': Shapiro-Wilk test statistic
        'p_value': p-value from the test
        'is_normal': boolean indicating if data follows normal distribution
        'conclusion': string with test interpretation
    """
    try:
        # Perform Shapiro-Wilk test
        test_statistic, p_value = stats.shapiro(data_series)
        
        # Determine if normal based on p-value
        is_normal = p_value > significance_level
        
        conclusion = ("Data appears to be normally distributed (fail to reject H0)" 
                     if is_normal 
                     else "Data does not appear to be normally distributed (reject H0)")
        
        return {
            'test_statistic': test_statistic,
            'p_value': p_value,
            'is_normal': is_normal,
            'conclusion': conclusion
        }
        
    except Exception as e:
        return {
            'error': str(e),
            'test_statistic': None,
            'p_value': None,
            'is_normal': None,
            'conclusion': 'Test failed due to error'
        }

# output will be like below
# {'test_statistic': np.float64(0.9493315658670367),
#  'p_value': np.float64(4.776418056805091e-18),
#  'is_normal': np.False_,
#  'conclusion': 'Data does not appear to be normally distributed (reject H0)'}

# --------------------------------------------------------------------------------------------------
# test for poission distribution
import pandas as pd
import numpy as np
from scipy import stats
def chi2_poisson_test(data_series, significance_level=0.05):
    """
    Perform chi-square goodness-of-fit test for Poisson distribution with robust handling.
    
    Parameters:
    - data_series: pandas Series of non-negative integer values
    - significance_level: threshold for rejecting null hypothesis (default 0.05)
    
    Returns:
    - Dictionary with test results and diagnostics
    """
    try:
        # Data preparation
        data = data_series.dropna().values
        if len(data) == 0:
            raise ValueError("No valid data after removing NAs")
        if np.any(data < 0):
            raise ValueError("Data contains negative values")
            
        # Estimate lambda
        estimated_lambda = np.mean(data)
        
        # Get observed counts
        unique, observed = np.unique(data, return_counts=True)
        max_observed = max(unique)
        
        # Create bins - we'll combine the tail to ensure expected counts >= 5
        bins = np.arange(0, max_observed + 2)  # +2 to include max_observed in last bin
        observed_binned, _ = np.histogram(data, bins=bins)
        
        # Calculate expected probabilities
        expected_probs = stats.poisson.pmf(bins[:-1], mu=estimated_lambda)
        # Last bin gets the remaining probability
        expected_probs[-1] = 1 - stats.poisson.cdf(bins[-2] - 1, mu=estimated_lambda)
        
        # Combine bins from the end until all expected counts >= 5
        n = len(data)
        i = len(expected_probs) - 1
        while i > 0:
            if n * expected_probs[i] < 5:
                expected_probs[i-1] += expected_probs[i]
                observed_binned[i-1] += observed_binned[i]
                expected_probs = np.delete(expected_probs, i)
                observed_binned = np.delete(observed_binned, i)
                bins = np.delete(bins, i)
            i -= 1
        
        # Ensure we have at least 2 bins
        if len(observed_binned) < 2:
            raise ValueError("Not enough bins after combining (need ≥2)")
        
        # Calculate expected counts
        expected_counts = n * expected_probs
        
        # Normalize to ensure sums match exactly
        observed_sum = np.sum(observed_binned)
        expected_sum = np.sum(expected_counts)
        scaling_factor = observed_sum / expected_sum
        expected_counts = expected_counts * scaling_factor
        
        # Perform chi-square test
        chi2_stat, p_value = 0, 1
        for obs, exp in zip(observed_binned, expected_counts):
            if exp > 0:  # Avoid division by zero
                chi2_stat += (obs - exp)**2 / exp
        
        dof = len(observed_binned) - 1 - 1  # bins - 1 - parameters estimated
        p_value = 1 - stats.chi2.cdf(chi2_stat, dof)
        
        is_poisson = p_value > significance_level
        
        return {
            'test_statistic': chi2_stat,
            'p_value': p_value,
            'is_poisson': is_poisson,
            'conclusion': "Poisson distribution" if is_poisson else "Not Poisson distribution",
            'estimated_lambda': estimated_lambda,
            'degrees_of_freedom': dof,
            'observed_counts': observed_binned,
            'expected_counts': expected_counts,
            'bins_used': bins[:-1]  # Return the bin boundaries
        }
        
    except Exception as e:
        return {
            'error': str(e),
            'test_statistic': None,
            'p_value': None,
            'is_poisson': None,
            'conclusion': 'Test failed',
            'estimated_lambda': None,
            'degrees_of_freedom': None
        }
    

# output 
# {'test_statistic': np.float64(10.72697783425192),
#  'p_value': np.float64(0.09718999826360342),
#  'is_poisson': np.True_,
#  'conclusion': 'Poisson distribution',
#  'estimated_lambda': np.float64(2.412),
#  'degrees_of_freedom': 6,
#  'observed_counts': array([ 92, 203, 288, 192, 119,  64,  34,   8]),
#  'expected_counts': array([ 89.63584349, 216.20165451, 260.73919534, 209.63431305,
#         126.40949077,  60.97993835,  24.51393522,  11.88562928]),
#  'bins_used': array([0, 1, 2, 3, 4, 5, 6, 7])}


# --------------------------------------------------------------------------------------------------
# chi square test for any distribution
import numpy as np
import pandas as pd
from scipy import stats

def chi_square_gof_test(data_series, dist='poisson', significance_level=0.05, min_bin_size=5):
    """
    Perform chi-square goodness-of-fit test for specified distribution.
    
    Parameters:
    - data_series: pandas Series of numerical values
    - dist: distribution to test ('poisson', 'normal', 'exponential', 'uniform')
    - significance_level: threshold for rejecting null hypothesis (default 0.05)
    - min_bin_size: minimum expected count per bin (default 5)
    
    Returns:
    - Dictionary containing:
        * test statistics
        * p-value
        * distribution parameters
        * fit conclusion
        * diagnostic information
    """
    result = {
        'test_statistic': None,
        'p_value': None,
        'is_fit': None,
        'conclusion': None,
        'dist_params': {},
        'observed': None,
        'expected': None,
        'bins': None,
        'degrees_of_freedom': None,
        'warnings': []
    }
    
    try:
        # Data preparation
        data = data_series.dropna().values
        if len(data) == 0:
            raise ValueError("No valid data after removing missing values")
        
        n = len(data)
        result['n_observations'] = n
        
        # Estimate distribution parameters
        if dist == 'poisson':
            if np.any(data < 0) or not np.all(np.equal(np.mod(data, 1), 0)):
                raise ValueError("Poisson requires non-negative integer values")
            lambda_ = np.mean(data)
            result['dist_params'] = {'lambda': lambda_}
            
            # Create bins for discrete data
            unique = np.unique(data)
            bins = np.arange(unique.min(), unique.max() + 2)  # +2 to include max value
            observed, _ = np.histogram(data, bins=bins)
            
            # Calculate expected probabilities
            expected_probs = stats.poisson.pmf(bins[:-1], mu=lambda_)
            # Last bin gets remaining probability
            expected_probs[-1] = 1 - stats.poisson.cdf(bins[-2] - 1, mu=lambda_)
            
        elif dist == 'normal':
            mu, sigma = np.mean(data), np.std(data, ddof=1)
            result['dist_params'] = {'mu': mu, 'sigma': sigma}
            
            # Create bins for continuous data (using Sturges' rule)
            k = int(np.ceil(np.log2(n)) + 1)
            bins = np.linspace(np.min(data), np.max(data), k + 1)
            observed, _ = np.histogram(data, bins=bins)
            
            # Calculate expected probabilities
            expected_probs = np.diff(stats.norm.cdf(bins, loc=mu, scale=sigma))
            
        elif dist == 'exponential':
            if np.any(data < 0):
                raise ValueError("Exponential requires non-negative values")
            beta = np.mean(data)
            result['dist_params'] = {'beta': beta}
            
            # Create bins
            k = int(np.ceil(np.log2(n)) + 1)
            bins = np.linspace(0, np.max(data), k + 1)
            observed, _ = np.histogram(data, bins=bins)
            
            # Calculate expected probabilities
            expected_probs = np.diff(stats.expon.cdf(bins, scale=beta))
            
        elif dist == 'uniform':
            a, b = np.min(data), np.max(data)
            result['dist_params'] = {'a': a, 'b': b}
            
            # Create bins
            k = int(np.ceil(np.log2(n)) + 1)
            bins = np.linspace(a, b, k + 1)
            observed, _ = np.histogram(data, bins=bins)
            
            # Calculate expected probabilities
            expected_probs = np.diff(stats.uniform.cdf(bins, loc=a, scale=b-a))
            
        else:
            raise ValueError(f"Unsupported distribution: {dist}")
        
        # Combine bins to meet minimum expected count requirement
        i = len(expected_probs) - 1
        while i > 0:
            if n * expected_probs[i] < min_bin_size:
                expected_probs[i-1] += expected_probs[i]
                observed[i-1] += observed[i]
                expected_probs = np.delete(expected_probs, i)
                observed = np.delete(observed, i)
                bins = np.delete(bins, i)
            i -= 1
        
        # Check if we have enough bins
        if len(observed) < 2:
            raise ValueError(f"Not enough bins after combining (need ≥2), try increasing sample size")
        
        # Calculate expected counts
        expected = n * expected_probs
        
        # Normalize to ensure sums match exactly
        scaling_factor = np.sum(observed) / np.sum(expected)
        expected = expected * scaling_factor
        
        # Calculate chi-square statistic manually
        chi2 = np.sum((observed - expected)**2 / expected)
        dof = len(observed) - 1 - len(result['dist_params'])
        p_value = 1 - stats.chi2.cdf(chi2, dof)
        
        # Prepare results
        result.update({
            'test_statistic': chi2,
            'p_value': p_value,
            'is_fit': p_value > significance_level,
            'conclusion': f"Data fits {dist} distribution" if p_value > significance_level 
                         else f"Data does not fit {dist} distribution",
            'observed': observed,
            'expected': expected,
            'bins': bins[:-1] if dist == 'poisson' else bins,
            'degrees_of_freedom': dof
        })
        
        # Add warnings if any
        if np.any(expected < min_bin_size):
            result['warnings'].append(f"Some bins have expected counts < {min_bin_size}")
        if dof < 1:
            result['warnings'].append("Low degrees of freedom - results may be unreliable")
        
    except Exception as e:
        result['error'] = str(e)
        result['conclusion'] = f"Test failed: {str(e)}"
    
    return result

# result is like this 
# {'test_statistic': np.float64(370.5843220132851),
#  'p_value': np.float64(0.0),
#  'is_fit': np.False_,
#  'conclusion': 'Data does not fit normal distribution',
#  'dist_params': {'mu': np.float64(2.412),
#   'sigma': np.float64(1.5500511793675262)},
#  'observed': array([ 92, 203, 288,   0, 192, 119,  64,   0,  42]),
#  'expected': array([ 83.72243733, 138.13341405, 183.60141314, 196.60046433,
#         169.60042279, 117.86843871,  65.99066684,  29.76186988,
#          14.72087293]),
#  'bins': array([0.        , 0.72727273, 1.45454545, 2.18181818, 2.90909091,
#         3.63636364, 4.36363636, 5.09090909, 5.81818182, 8.        ]),
#  'degrees_of_freedom': 6,
#  'warnings': [],
#  'n_observations': 1000}



# --------------------------------------------------------------------------------------------------
# test for geometric distribution
import numpy as np
import pandas as pd
from scipy import stats
def watson_geometric_test(series, alpha=0.05):
    """
    Discrete adaptation of Watson's test for geometric distribution.
    
    Parameters:
    - series: pandas Series of non-negative integers
    - alpha: significance level (default 0.05)
    
    Returns:
    - Dictionary containing:
        * test_statistic: U² statistic
        * p_value: approximate p-value
        * is_geometric: boolean test result
        * conclusion: textual interpretation
        * estimated_p: estimated success probability
    """
    # Input validation
    if not isinstance(series, pd.Series):
        raise ValueError("Input must be a pandas Series")
        
    data = series.dropna().values
    if len(data) == 0:
        return {'error': 'No valid data after removing NAs'}
    
    # Corrected condition - check for non-negative integers
    if not (np.all(np.mod(data, 1) == 0) and np.all(data >= 0)):
        return {'error': 'Data must contain non-negative integers'}
    
    # Estimate p parameter for geometric distribution (MLE)
    p_hat = 1 / (1 + np.mean(data))
    
    # Calculate empirical CDF
    unique_vals, counts = np.unique(data, return_counts=True)
    ecdf = np.cumsum(counts) / len(data)
    
    # Calculate theoretical CDF (geometric)
    theoretical_cdf = 1 - (1 - p_hat)**(unique_vals + 1)  # P(X ≤ x)
    
    # Discrete adaptation of Watson's U² statistic
    n = len(data)
    d = ecdf - theoretical_cdf
    mean_d = np.mean(d)
    U_squared = np.sum((d - mean_d)**2) + 1/(12*n)  # Discrete correction
    
    # Approximate p-value (using modified critical values)
    critical_value = 0.224  # Approximation for α=0.05
    
    is_geometric = U_squared < critical_value
    p_value = np.exp(-2 * np.pi**2 * U_squared)  # Approximate p-value
    
    conclusion = ("Data appears to follow geometric distribution (fail to reject H0)"
                if is_geometric 
                else "Data does not appear to follow geometric distribution (reject H0)")
    
    return {
        'test_statistic': U_squared,
        'p_value': p_value,
        'is_geometric': is_geometric,
        'conclusion': conclusion,
        'estimated_p': p_hat,
        'critical_value': critical_value,
        'alpha': alpha
    }


# -----------------------------------

