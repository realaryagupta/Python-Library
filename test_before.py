import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm
from scipy.stats import boxcox
from statsmodels.stats.outliers_influence import variance_inflation_factor
from IPython.display import display, HTML



# ------------------------------------------------------------------------------------------------------------------------
def plot_corr_heatmap(df, catcols=None, figsize=(12,8), cmap='coolwarm', annot=False, mask_upper=True):
    """
    Plots a correlation heatmap for the numerical columns of a DataFrame.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame.
        catcols (list, optional): List of categorical columns to exclude. If None, excludes non-numeric columns.
        figsize (tuple): Figure size for the plot.
        cmap (str): Colormap for the heatmap.
        annot (bool): Whether to annotate the correlation values.
        mask_upper (bool): Whether to mask the upper triangle of the heatmap.
    """
    # If catcols not provided, use only numeric columns
    if catcols is not None:
        numcols = [col for col in df.columns if col not in catcols and pd.api.types.is_numeric_dtype(df[col])]
    else:
        numcols = df.select_dtypes(include=np.number).columns.tolist()
    
    corr_matrix = df[numcols].corr()
    
    mask = None
    if mask_upper:
        mask = np.zeros_like(corr_matrix, dtype=bool)
        mask[np.triu_indices_from(mask)] = True
    
    plt.figure(figsize=figsize)
    sns.heatmap(
        corr_matrix,
        mask=mask,
        cmap=cmap,
        annot=annot,
        vmin=-1, vmax=1,
        square=True,
        linewidths=.5,
        cbar_kws={"shrink": .8}
    )
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
# For any DataFrame df
# plot_corr_heatmap(df)

# If you want to specify categorical columns to exclude:
# catcols = ['col1', 'col2', 'col3']
# plot_corr_heatmap(df, catcols=catcols)

# ------------------------------------------------------------------------------------------------------------------------
def plot_missing_values(df, figsize=(12, 8), color='blue'):
    """
    Plots a horizontal bar chart of missing values per column in a DataFrame.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame.
        figsize (tuple): Size of the plot.
        color (str): Color of the bars.
    """
    missing_df = df.isnull().sum().reset_index()
    missing_df.columns = ['column_name', 'missing_count']
    missing_df = missing_df[missing_df['missing_count'] > 0]
    missing_df = missing_df.sort_values(by='missing_count')
    
    if missing_df.empty:
        print("No missing values found in any column.")
        return
    
    ind = np.arange(missing_df.shape[0])
    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(ind, missing_df.missing_count.values, color=color)
    ax.set_yticks(ind)
    ax.set_yticklabels(missing_df.column_name.values, rotation='horizontal')
    ax.set_xlabel("Count of missing values")
    ax.set_title("Number of missing values in each column")
    plt.tight_layout()
    plt.show()
# plot_missing_values(df_train)


# ------------------------------------------------------------------------------------------------------------------------
# Data Type Detection Functions
# Automatically classify columns as numeric/categorical/binary
def detect_column_type(series, category_threshold=10, forced_types=None):
    """Automatically detect if a column is numeric, categorical, or binary"""
    forced_types = forced_types or {}
    column_name = series.name
    
    if column_name in forced_types:
        return forced_types[column_name], True
    
    if series.dtype == 'bool':
        return 'binary', False
    elif series.dtype == 'object':
        return 'category', False
    elif series.dtype in ['int64', 'float64']:
        unique_count = series.nunique()
        if unique_count == 2 and 0 in series.unique():
            return 'binary', False
        elif unique_count <= category_threshold:
            return 'category', False
        else:
            return 'number', False
    else:
        return 'unknown', False



# ------------------------------------------------------------------------------------------------------------------------
# Generate basic statistics for any column
def create_column_descriptor(df, column, max_unique_display=15):
    """Generate comprehensive descriptive statistics for a single column"""
    series = df[column]
    desc = {}
    
    # Basic descriptors
    desc['dtype'] = str(series.dtype)
    desc['count'] = series.count()
    desc['nulls'] = series.isna().sum()
    desc['nulls_perc'] = round(desc['nulls'] / len(series), 2)
    desc['unique'] = series.nunique()
    
    # Unique values (limited display)
    if desc['unique'] <= max_unique_display:
        desc['unique_values'] = list(series.unique())
    else:
        desc['unique_values'] = []
    
    return desc



# ------------------------------------------------------------------------------------------------------------------------
# Add mean, std, quartiles, skewness for numbers
def add_numeric_descriptors(desc_dict, series):
    """Add numeric-specific descriptors to column description"""
    desc_dict['mean'] = series.mean()
    desc_dict['std'] = series.std()
    desc_dict['min'] = series.min()
    desc_dict['25%'] = series.quantile(0.25)
    desc_dict['50%'] = series.quantile(0.50)
    desc_dict['75%'] = series.quantile(0.75)
    desc_dict['max'] = series.max()
    desc_dict['skew'] = series.skew()
    return desc_dict


# ------------------------------------------------------------------------------------------------------------------------
# Add mode, frequency stats for categories
def add_categorical_descriptors(desc_dict, series):
    """Add categorical-specific descriptors to column description"""
    if series.dtype in ['int64', 'float64']:
        desc_dict['min'] = series.min()
        desc_dict['max'] = series.max()
    
    # Mode and frequency
    value_counts = series.value_counts().sort_values(ascending=False)
    if len(value_counts) > 0 and len(value_counts) < len(series):
        desc_dict['mode'] = value_counts.index[0]
        desc_dict['freq'] = value_counts.iloc[0]
        desc_dict['freq_perc'] = round(value_counts.iloc[0] / desc_dict['count'], 2)
    
    return desc_dict


# ------------------------------------------------------------------------------------------------------------------------
# Comprehensive Dataset Description
# Complete dataset overview 
def create_dataset_description(df, target_col=None, category_threshold=10, forced_types=None):
    """Create comprehensive description of entire dataset"""
    desc_df = pd.DataFrame(index=df.columns)
    
    for column in df.columns:
        # Basic descriptors
        desc = create_column_descriptor(df, column)
        
        # Detect column type
        col_type, is_forced = detect_column_type(df[column], category_threshold, forced_types)
        desc['type'] = col_type
        desc['type_isforced'] = 1 if is_forced else 0
        
        # Add type-specific descriptors
        if col_type == 'number':
            desc = add_numeric_descriptors(desc, df[column])
            # Add outlier detection
            bounds, has_outliers = detect_outliers_iqr(df[column])
            desc['outliers'] = bounds
            desc['has_outliers'] = has_outliers
            
        elif col_type in ['category', 'binary']:
            desc = add_categorical_descriptors(desc, df[column])
        
        # Add to dataframe
        for key, value in desc.items():
            desc_df.loc[column, key] = value
    
    return desc_df


# ------------------------------------------------------------------------------------------------------------------------
# Visualization Functions
#  Distribution + boxplot for numbers
def plot_numeric_distribution(series, title_prefix=""):
    """Plot distribution and boxplot for numeric variables"""
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    fig.tight_layout(pad=3)
    
    # Distribution plot
    axes[0].set_title(f"{title_prefix} Distribution (skew: {series.skew():.2f})")
    axes[0].grid(True)
    try:
        sns.histplot(series.dropna(), kde=True, ax=axes[0])
    except:
        pass
    
    # Box plot
    axes[1].set_title(f"{title_prefix} Boxplot")
    axes[1].grid(True)
    sns.boxplot(x=series, ax=axes[1])
    
    plt.show()



# ------------------------------------------------------------------------------------------------------------------------
# Bar charts with percentages for categories
def plot_categorical_distribution(series, title_prefix="", max_categories=50, fillna_value="Missing"):
    """Plot bar chart for categorical variables"""
    if series.nunique() > max_categories:
        print(f"WARNING: Too many categories ({series.nunique()}) to plot effectively")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Fill missing values and get value counts
    series_filled = series.fillna(fillna_value)
    value_counts = series_filled.value_counts().sort_values(ascending=False)
    
    # Create bar plot
    ax.set_title(f"{title_prefix} Distribution")
    ax.grid(True, axis='y')
    bars = sns.barplot(x=value_counts.index, y=value_counts.values, ax=ax)
    
    # Add percentage labels
    total = value_counts.sum()
    for bar, count in zip(bars.patches, value_counts.values):
        percentage = f"{100*count/total:.1f}%"
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                percentage, ha='center', va='bottom')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------------------------------------------------------------------
# Smart plotting based on variable types
def plot_bivariate_analysis(x_series, y_series, x_type, y_type):
    """Plot bivariate relationship between two variables"""
    if x_type == 'number' and y_type == 'number':
        # Scatter plot with regression line
        plt.figure(figsize=(10, 6))
        sns.regplot(x=x_series, y=y_series)
        plt.title(f"{x_series.name} vs {y_series.name}")
        plt.grid(True)
        plt.show()
        
    elif x_type in ['category', 'binary'] and y_type == 'number':
        # Box plot
        plt.figure(figsize=(12, 6))
        sns.boxplot(x=x_series, y=y_series)
        plt.title(f"{x_series.name} vs {y_series.name}")
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.show()
        
    elif x_type in ['category', 'binary'] and y_type in ['category', 'binary']:
        # Cross-tabulation
        crosstab = pd.crosstab(x_series, y_series, margins=True, margins_name='Total')
        print(f"Cross-tabulation: {x_series.name} vs {y_series.name}")
        display(HTML(crosstab.to_html()))



# ------------------------------------------------------------------------------------------------------------------------
# Transformation Functions
# Visual transformation suggestions
def suggest_transformations(series):
    """Suggest and visualize potential transformations for numeric data"""
    series_clean = series.dropna()
    
    if len(series_clean) == 0:
        print("No data available for transformation analysis")
        return
    
    has_negative = series_clean.min() < 0
    has_zeros = (series_clean == 0).any()
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.tight_layout(pad=3)
    
    transformations = []
    
    # Square root transformation
    if not has_negative:
        sqrt_data = np.sqrt(series_clean)
        transformations.append(('Square Root', sqrt_data))
        sns.histplot(sqrt_data, kde=True, ax=axes[0,0])
        axes[0,0].set_title(f'Square Root (skew: {sqrt_data.skew():.2f})')
        sns.boxplot(x=sqrt_data, ax=axes[1,0])
    
    # Reciprocal transformation
    if not has_zeros:
        recip_data = 1 / series_clean
        transformations.append(('Reciprocal', recip_data))
        sns.histplot(recip_data, kde=True, ax=axes[0,1])
        axes[0,1].set_title(f'Reciprocal (skew: {recip_data.skew():.2f})')
        sns.boxplot(x=recip_data, ax=axes[1,1])
    
    # Log transformation
    if not has_negative:
        if has_zeros:
            log_data = np.log(series_clean + 1)
            log_title = 'Log(x+1)'
        else:
            log_data = np.log(series_clean)
            log_title = 'Log(x)'
        
        transformations.append((log_title, log_data))
        sns.histplot(log_data, kde=True, ax=axes[0,2])
        axes[0,2].set_title(f'{log_title} (skew: {log_data.skew():.2f})')
        sns.boxplot(x=log_data, ax=axes[1,2])
    
    plt.show()
    return transformations


# ------------------------------------------------------------------------------------------------------------------------
# Correlation and Multicollinearity Functions
# Correlation matrix visualization
def create_correlation_heatmap(df, figsize=(12, 10)):
    """Create correlation heatmap for numeric variables"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr()
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    plt.figure(figsize=figsize)
    sns.heatmap(corr_matrix, annot=True, mask=mask, cmap='coolwarm', 
                center=0, square=True, linewidths=0.5)
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.show()
    
    return corr_matrix



# ------------------------------------------------------------------------------------------------------------------------
# Variance Inflation Factor calculation (replaces your multivar method)
def calculate_vif(df, target_col=None):
    """Calculate Variance Inflation Factor for numeric variables"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove target column if specified
    if target_col and target_col in numeric_cols:
        numeric_cols.remove(target_col)
    
    if len(numeric_cols) < 2:
        print("Need at least 2 numeric variables for VIF calculation")
        return pd.DataFrame()
    
    # Remove rows with any missing values
    df_clean = df[numeric_cols].dropna()
    
    if len(df_clean) == 0:
        print("No complete cases available for VIF calculation")
        return pd.DataFrame()
    
    vif_data = pd.DataFrame()
    vif_data["Variable"] = numeric_cols
    vif_data["VIF"] = [variance_inflation_factor(df_clean.values, i) 
                       for i in range(len(numeric_cols))]
    
    return vif_data.sort_values('VIF', ascending=False)


# ------------------------------------------------------------------------------------------------------------------------
# Flag issues automatically
def identify_problematic_columns(desc_df, null_threshold=0.5, vif_threshold=5.0):
    """Identify columns that might need attention"""
    issues = {}
    
    # High null percentage
    high_nulls = desc_df[desc_df['nulls_perc'] > null_threshold].index.tolist()
    if high_nulls:
        issues['high_nulls'] = high_nulls
    
    # Columns with outliers
    has_outliers = desc_df[desc_df['has_outliers'] == True].index.tolist()
    if has_outliers:
        issues['outliers'] = has_outliers
    
    # Single value columns
    single_value = desc_df[desc_df['unique'] <= 1].index.tolist()
    if single_value:
        issues['single_value'] = single_value
    
    return issues



# ------------------------------------------------------------------------------------------------------------------------
# Main EDA Pipeline Function (calls other fxn)
# Main pipeline function for full analysis
def comprehensive_eda(df, target_col=None, category_threshold=10, forced_types=None):
    """Run comprehensive EDA pipeline on dataset"""
    print("Starting Comprehensive EDA...")
    print(f"Dataset shape: {df.shape}")
    
    # Create dataset description
    desc_df = create_dataset_description(df, target_col, category_threshold, forced_types)
    
    # Display summary
    print("\nDataset Summary:")
    type_summary = desc_df['type'].value_counts()
    display(HTML(type_summary.to_frame().T.to_html()))
    
    # Identify issues
    issues = identify_problematic_columns(desc_df)
    if issues:
        print("\nPotential Issues Identified:")
        for issue_type, columns in issues.items():
            print(f"- {issue_type}: {columns}")
    
    # Correlation analysis for numeric variables
    numeric_cols = desc_df[desc_df['type'] == 'number'].index.tolist()
    if len(numeric_cols) > 1:
        print(f"\nCorrelation Analysis ({len(numeric_cols)} numeric variables):")
        corr_matrix = create_correlation_heatmap(df[numeric_cols])
        
        # VIF analysis
        vif_df = calculate_vif(df, target_col)
        if not vif_df.empty:
            print("\nVariance Inflation Factors:")
            display(HTML(vif_df.to_html(index=False)))
    
    return desc_df



# ------------------------------------------------------------------------------------------------------------------------
# Detailed single-column analysis (replaces your univar method)
# Example usage function
# CALLS OTHER FXN INSIDE THIS
def analyze_single_column(df, column_name, target_col=None):
    """Perform detailed analysis of a single column"""
    if column_name not in df.columns:
        print(f"Column '{column_name}' not found in dataset")
        return
    
    series = df[column_name]
    col_type, _ = detect_column_type(series)
    
    print(f"Analyzing column: {column_name}")
    print(f"Type: {col_type}")
    
    # Basic statistics
    desc = create_column_descriptor(df, column_name)
    desc_df = pd.DataFrame([desc]).T
    desc_df.columns = ['Value']
    display(HTML(desc_df.to_html()))
    
    # Visualizations
    if col_type == 'number':
        plot_numeric_distribution(series, column_name)
        
        # Suggest transformations
        print("Transformation Analysis:")
        suggest_transformations(series)
        
    elif col_type in ['category', 'binary']:
        plot_categorical_distribution(series, column_name)
    
    # Bivariate analysis with target
    if target_col and target_col in df.columns and column_name != target_col:
        target_type, _ = detect_column_type(df[target_col])
        print(f"\nBivariate Analysis with {target_col}:")
        plot_bivariate_analysis(series, df[target_col], col_type, target_type)




# ------------------------------------------------------------------------------------------------------------------------
def fit_bayesian_target_encoder(df, group_col, target_col):
    """
    Computes group statistics and prior mean for Bayesian target encoding.
    
    Args:
        df (pd.DataFrame): The input dataframe.
        group_col (str): Name of the categorical column.
        target_col (str): Name of the target column.
        
    Returns:
        dict: Contains 'prior_mean' and 'stats' DataFrame.
    """
    prior_mean = np.mean(df[target_col])
    stats = (
        df[[target_col, group_col]]
        .groupby(group_col)
        .agg(['sum', 'count'])[target_col]
        .rename(columns={'sum': 'n', 'count': 'N'})
        .reset_index()
    )
    return {'prior_mean': prior_mean, 'stats': stats, 'group_col': group_col}
# fit_bayesian_target_encoder(df_train, group_col='category', target_col='target')




# ------------------------------------------------------------------------------------------------------------------------
def transform_bayesian_target_encoder(df, encoder, stat_type='mean', N_min=1):
    """
    Applies Bayesian target encoding to a DataFrame using precomputed statistics.
    
    Args:
        df (pd.DataFrame): The input dataframe.
        encoder (dict): Output from fit_bayesian_target_encoder.
        stat_type (str): Statistic to compute ('mean', 'mode', 'median', 'var', 'skewness', 'kurtosis').
        N_min (int): Minimum prior sample size.
        
    Returns:
        np.ndarray: Encoded values.
    """
    group_col = encoder['group_col']
    stats = encoder['stats']
    prior_mean = encoder['prior_mean']
    
    df_stats = pd.merge(df[[group_col]], stats, how='left', on=group_col)
    n = df_stats['n'].copy()
    N = df_stats['N'].copy()
    
    # Fill missing
    nan_indices = n.isna()
    n[nan_indices] = prior_mean
    N[nan_indices] = 1.0

    N_prior = np.maximum(N_min - N, 0)
    alpha_prior = prior_mean * N_prior
    beta_prior = (1 - prior_mean) * N_prior
    
    alpha = alpha_prior + n
    beta = beta_prior + N - n
    
    # Compute requested statistic
    if stat_type == 'mean':
        num = alpha
        dem = alpha + beta
    elif stat_type == 'mode':
        num = alpha - 1
        dem = alpha + beta - 2
    elif stat_type == 'median':
        num = alpha - 1/3
        dem = alpha + beta - 2/3
    elif stat_type == 'var':
        num = alpha * beta
        dem = (alpha + beta) ** 2 * (alpha + beta + 1)
    elif stat_type == 'skewness':
        num = 2 * (beta - alpha) * np.sqrt(alpha + beta + 1)
        dem = (alpha + beta + 2) * np.sqrt(alpha * beta)
    elif stat_type == 'kurtosis':
        num = 6 * (alpha - beta) ** 2 * (alpha + beta + 1) - alpha * beta * (alpha + beta + 2)
        dem = alpha * beta * (alpha + beta + 2) * (alpha + beta + 3)
    else:
        num = prior_mean
        dem = np.ones_like(N_prior)
    
    value = num / dem
    value[np.isnan(value)] = np.nanmedian(value)
    return value
# transform_bayesian_target_encoder(df_train, encoder, stat_type='mean')



# ------------------------------------------------------------------------------------------------------------------------
# =============================================================================
# COLLINEARITY FUNCTIONS
# =============================================================================

def drop_collinear_features(df, collinear_df, vif_thresh=5):
    """Drop highly collinear features based on VIF threshold."""
    if len(collinear_df) > 0:
        columns_to_drop = collinear_df[collinear_df["VIF"] >= vif_thresh]["variables"].to_list()
        df_cleaned = df.drop(columns_to_drop, axis=1)
        print(f"Dropped {len(columns_to_drop)} collinear columns: {columns_to_drop}")
        print(f"Shape changed from {df.shape} to {df_cleaned.shape}")
        return df_cleaned, columns_to_drop
    return df, []


# ------------------------------------------------------------------------------------------------------------------------
# =============================================================================
# CATEGORICAL ENCODING FUNCTIONS
# =============================================================================
def create_others_category(series, num_categories=0, min_elem_categories=0, 
                          categories=None, others_name="Others", fill_nulls_value=None):
    """Categorize series by grouping less frequent values into 'Others' category."""
    # Handle nulls first
    if fill_nulls_value is not None:
        series = series.fillna(fill_nulls_value)
    
    # Determine top categories
    if categories is not None:
        top_categories = list(categories)
    elif num_categories > 0:
        top_categories = series.value_counts().head(num_categories).index.tolist()
    elif min_elem_categories > 0:
        value_counts = series.value_counts()
        top_categories = value_counts[value_counts >= min_elem_categories].index.tolist()
    else:
        top_categories = series.unique().tolist()
    
    # Add null value and others to categories
    if fill_nulls_value is not None:
        top_categories.append(fill_nulls_value)
    top_categories.append(others_name)
    
    # Create categorical with others
    result = pd.Categorical(series, categories=top_categories).fillna(others_name)
    return result



# ------------------------------------------------------------------------------------------------------------------------
def apply_one_hot_encoding(df, column, drop_first=True, prefix=None):
    """Apply one-hot encoding to categorical column(s)."""
    if isinstance(column, list):
        # Multiple columns case
        if prefix is None:
            raise ValueError("prefix must be provided for multiple columns")
        
        dummy_df = pd.DataFrame()
        for col in column:
            col_dummies = pd.get_dummies(df[col], prefix=prefix, prefix_sep="_")
            for dummy_col in col_dummies.columns:
                if dummy_col in dummy_df.columns:
                    dummy_df[dummy_col] = dummy_df[dummy_col] + col_dummies[dummy_col]
                else:
                    dummy_df[dummy_col] = col_dummies[dummy_col]
        
        result_df = pd.concat([df.drop(column, axis=1), dummy_df], axis=1)
        print(f"Multiple OHE: Generated {len(dummy_df.columns)} columns, dropped {len(column)} original columns")
        
    else:
        # Single column case
        col_prefix = prefix if prefix else column
        dummies = pd.get_dummies(df[column], prefix=col_prefix, prefix_sep="_")
        
        if drop_first:
            dummies = dummies.drop(dummies.columns[-1], axis=1)
        
        result_df = pd.concat([df.drop(column, axis=1), dummies], axis=1)
        print(f"OHE: Generated {len(dummies.columns)} columns for {column}")
    
    return result_df




# ------------------------------------------------------------------------------------------------------------------------
def apply_ordinal_encoding(df, column, mapping):
    """Apply ordinal encoding using provided value mapping."""
    # Convert mapping to dictionary if it's list of tuples
    if isinstance(mapping, list):
        mapping_dict = dict(mapping)
    else:
        mapping_dict = mapping
    
    encoded_series = df[column].map(mapping_dict)
    
    if encoded_series.isna().any():
        print(f"WARNING: Ordinal encoding for {column} resulted in null values")
        return df, False
    
    df_result = df.copy()
    df_result[column] = encoded_series
    print(f"Ordinal encoding applied to {column}")
    return df_result, True




# ------------------------------------------------------------------------------------------------------------------------
def apply_binary_encoding(df, column, mapping):
    """Apply binary encoding (0/1) using provided value mapping."""
    return apply_ordinal_encoding(df, column, mapping)



# ------------------------------------------------------------------------------------------------------------------------
def create_pivot_features(df, index_col, pivot_col, value_col, prefix):
    """Create pivot table as feature encoding with custom prefix."""
    # Reset index to include it in pivot
    df_temp = df.reset_index()
    index_name = df_temp.columns[0]  # First column after reset_index
    
    # Create pivot table
    pivot_df = df_temp.pivot_table(
        index=index_name, 
        columns=pivot_col, 
        values=value_col, 
        fill_value=0
    )
    
    # Add prefix to column names
    pivot_df.columns = [f"{prefix}_{col}" for col in pivot_df.columns]
    
    # Merge back with original dataframe
    df_original_index = df.drop([pivot_col, value_col], axis=1)
    result_df = pd.concat([df_original_index, pivot_df], axis=1)
    
    print(f"Pivot: Generated {len(pivot_df.columns)} features from {pivot_col} and {value_col}")
    return result_df


# ------------------------------------------------------------------------------------------------------------------------
# =============================================================================
# NUMERICAL TRANSFORMATION FUNCTIONS
# =============================================================================
def apply_sqrt_transform(df, column):
    """Apply square root transformation to reduce right skewness."""
    if df[column].min() < 0:
        print(f"WARNING: {column} has negative values, cannot apply sqrt transform")
        return df, False
    
    df_result = df.copy()
    original_stats = f"mean: {df[column].mean():.2f}, skew: {df[column].skew():.2f}"
    
    df_result[column] = df[column] ** 0.5
    
    new_stats = f"mean: {df_result[column].mean():.2f}, skew: {df_result[column].skew():.2f}"
    print(f"Sqrt transform on {column}: {original_stats} -> {new_stats}")
    return df_result, True



# ------------------------------------------------------------------------------------------------------------------------
def apply_inverse_transform(df, column):
    """Apply inverse transformation (1/x) to reduce left skewness."""
    if (df[column] == 0).any():
        print(f"WARNING: {column} has zero values, cannot apply inverse transform")
        return df, False
    
    df_result = df.copy()
    original_stats = f"mean: {df[column].mean():.2f}, skew: {df[column].skew():.2f}"
    
    df_result[column] = 1 / df[column]
    
    new_stats = f"mean: {df_result[column].mean():.2f}, skew: {df_result[column].skew():.2f}"
    print(f"Inverse transform on {column}: {original_stats} -> {new_stats}")
    return df_result, True



# ------------------------------------------------------------------------------------------------------------------------
def apply_log_transform(df, column):
    """Apply logarithmic transformation to reduce right skewness."""
    if df[column].min() < 0:
        print(f"WARNING: {column} has negative values, cannot apply log transform")
        return df, False
    
    df_result = df.copy()
    original_stats = f"mean: {df[column].mean():.2f}, skew: {df[column].skew():.2f}"
    
    if (df[column] == 0).any():
        df_result[column] = np.log(df[column] + 1)
        print(f"Applied log(x+1) to {column} due to zero values")
    else:
        df_result[column] = np.log(df[column])
    
    new_stats = f"mean: {df_result[column].mean():.2f}, skew: {df_result[column].skew():.2f}"
    print(f"Log transform on {column}: {original_stats} -> {new_stats}")
    return df_result, True



# ------------------------------------------------------------------------------------------------------------------------
def create_ratio_feature(df, numerator_col, denominator_col, new_col_name=None, decimals=2):
    """Create ratio feature by dividing one column by another."""
    if new_col_name is None:
        new_col_name = f"{numerator_col}_to_{denominator_col}_ratio"
    
    df_result = df.copy()
    df_result[new_col_name] = round(df[numerator_col] / df[denominator_col], decimals)
    
    print(f"Created ratio feature {new_col_name} = {numerator_col} / {denominator_col}")
    return df_result


# ------------------------------------------------------------------------------------------------------------------------
# =============================================================================
# DATA CLEANING FUNCTIONS
# =============================================================================
def remove_outliers(df, column, condition, exclude_indices=None):
    """Remove outliers based on condition string, excluding specified indices."""
    if exclude_indices is None:
        exclude_indices = []
    
    original_shape = df.shape
    
    # Evaluate condition to find outlier indices
    outlier_mask = eval(f"df[column]{condition}")
    outlier_indices = df[outlier_mask].index.tolist()
    
    # Exclude protected indices (e.g., test set)
    indices_to_drop = [idx for idx in outlier_indices if idx not in exclude_indices]
    
    df_cleaned = df.drop(indices_to_drop, axis=0)
    
    print(f"Removed {len(indices_to_drop)} outliers from {column}")
    print(f"Shape changed from {original_shape} to {df_cleaned.shape}")
    return df_cleaned



# ------------------------------------------------------------------------------------------------------------------------
def drop_columns(df, columns):
    """Drop specified column(s) from dataframe."""
    if isinstance(columns, str):
        columns = [columns]
    
    df_result = df.drop(columns, axis=1)
    print(f"Dropped columns: {columns}")
    print(f"Shape changed from {df.shape} to {df_result.shape}")
    return df_result



# ------------------------------------------------------------------------------------------------------------------------
def drop_rows(df, indices):
    """Drop specified row(s) by index from dataframe."""
    if not isinstance(indices, list):
        indices = [indices]
    
    df_result = df.drop(indices, axis=0)
    print(f"Dropped {len(indices)} rows")
    print(f"Shape changed from {df.shape} to {df_result.shape}")
    return df_result



# ------------------------------------------------------------------------------------------------------------------------
def replace_values(df, column, mapping):
    """Replace values in column using mapping dictionary."""
    df_result = df.copy()
    df_result[column] = df_result[column].replace(mapping)
    
    print(f"Replaced values in {column}: {mapping}")
    return df_result



# ------------------------------------------------------------------------------------------------------------------------
def fill_missing_values(df, column, value):
    """Fill missing values in column with specified value."""
    df_result = df.copy()
    original_nulls = df[column].isna().sum()
    
    df_result[column] = df_result[column].fillna(value)
    
    final_nulls = df_result[column].isna().sum()
    print(f"Filled {original_nulls - final_nulls} null values in {column} with {value}")
    return df_result


# ------------------------------------------------------------------------------------------------------------------------
# =============================================================================
# DATASET SPLITTING FUNCTIONS
# =============================================================================
def split_train_test_target(df, target_column, test_indices=None):
    """Split dataframe into train/test sets and separate features from target."""
    if test_indices is not None and len(test_indices) > 0:
        # Split into train and test
        train_df = df.loc[~df.index.isin(test_indices)]
        test_df = df.loc[df.index.isin(test_indices)]
        
        # Separate features and target
        X_train = train_df.drop(target_column, axis=1)
        y_train = train_df[target_column]
        X_test = test_df.drop(target_column, axis=1) if target_column in test_df.columns else test_df
        
        print(f"Split data: Train {X_train.shape}, Test {X_test.shape}")
        return X_train, y_train, X_test
    else:
        # No test set, just separate features and target
        X = df.drop(target_column, axis=1)
        y = df[target_column]
        
        print(f"Separated features {X.shape} and target {y.shape}")
        return X, y, None


# ------------------------------------------------------------------------------------------------------------------------
# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def get_column_info(df, column):
    """Get comprehensive information about a specific column."""
    info = {
        'dtype': df[column].dtype,
        'null_count': df[column].isna().sum(),
        'null_percentage': (df[column].isna().sum() / len(df)) * 100,
        'unique_count': df[column].nunique(),
        'memory_usage': df[column].memory_usage(deep=True)
    }
    
    if pd.api.types.is_numeric_dtype(df[column]):
        info.update({
            'mean': df[column].mean(),
            'std': df[column].std(),
            'min': df[column].min(),
            'max': df[column].max(),
            'skewness': df[column].skew(),
            'kurtosis': df[column].kurtosis()
        })
    else:
        info.update({
            'top_values': df[column].value_counts().head().to_dict()
        })
    
    return info


# ------------------------------------------------------------------------------------------------------------------------
def generate_processing_summary(operations_log):
    """Generate HTML summary of all processing operations performed."""
    html_content = "<h2>Data Processing Summary</h2>"
    
    for operation in operations_log:
        html_content += f"<div style='margin: 10px; padding: 10px; border-left: 3px solid #007acc;'>"
        html_content += f"<strong>{operation['type']}</strong> on <em>{operation['column']}</em><br>"
        html_content += f"Status: {'✓ Success' if operation['success'] else '✗ Failed'}<br>"
        if 'details' in operation:
            html_content += f"Details: {operation['details']}<br>"
        html_content += "</div>"
    
    return html_content



# ------------------------------------------------------------------------------------------------------------------------
def validate_dataframe_integrity(df, expected_dtypes=None, required_columns=None):
    """Validate dataframe integrity after processing operations."""
    issues = []
    
    # Check for infinite values
    if np.isinf(df.select_dtypes(include=[np.number])).any().any():
        issues.append("Dataframe contains infinite values")
    
    # Check for unexpected data types
    if expected_dtypes:
        for col, expected_dtype in expected_dtypes.items():
            if col in df.columns and df[col].dtype != expected_dtype:
                issues.append(f"Column {col} has unexpected dtype: {df[col].dtype} (expected {expected_dtype})")
    
    # Check for required columns
    if required_columns:
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            issues.append(f"Missing required columns: {list(missing_cols)}")
    
    # Check for duplicate indices
    if df.index.duplicated().any():
        issues.append("Dataframe has duplicate indices")
    
    return issues


