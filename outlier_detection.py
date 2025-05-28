

""" OUTLIER DETECTION"""
# ------------------------------------------------------------------------------------------------------------------------
# Outlier Detection Functions
# Find outliers using IQR method
def detect_outliers_iqr(series):
    """Detect outliers using IQR method"""
    q25 = series.quantile(0.25)
    q75 = series.quantile(0.75)
    iqr = q75 - q25
    lower_bound = q25 - 1.5 * iqr
    upper_bound = q75 + 1.5 * iqr
    
    has_outliers = ((series < lower_bound) | (series > upper_bound)).any()
    bounds = [lower_bound, upper_bound] if has_outliers else None
    
    return bounds, has_outliers



# ------------------------------------------------------------------------------------------------------------------------
# Find outliers using standard deviation method
def detect_outliers_std(series, std_threshold=3):
    """Detect outliers using standard deviation method"""
    mean = series.mean()
    std = series.std()
    margin = std_threshold * std
    lower_bound = mean - margin
    upper_bound = mean + margin
    
    has_outliers = ((series < lower_bound) | (series > upper_bound)).any()
    bounds = [lower_bound, upper_bound] if has_outliers else None
    
    return bounds, has_outliers