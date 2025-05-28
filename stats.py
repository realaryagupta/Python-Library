# Data Manipulation and Analysis
import numpy as np
import pandas as pd

# Data Visualization
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from wordcloud import WordCloud, STOPWORDS

# Statistical Analysis
from scipy import stats
import statsmodels.api as sm

# Machine Learning and Preprocessing
from sklearn.preprocessing import (
    PowerTransformer,
    OneHotEncoder,
    StandardScaler
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Display and Output Formatting
from IPython.display import display, HTML


# -----------------------------------------------------------------------------------------------------------------------
# Identifies and returns outlier rows in a DataFrame column using the Interquartile Range (IQR) method with a customizable threshold.
def get_iqr_outliers(data, var, band = 1.5):
    """
    Identify outliers in a numerical column using the IQR (Interquartile Range) method.

    Parameters:
        data (pd.DataFrame): The DataFrame containing the data.
        var (str): The column name to check for outliers.
        band (float, optional): The multiplier for the IQR to set the outlier thresholds (default is 1.5).

    Returns:
        pd.DataFrame: DataFrame containing the outlier rows, sorted by the variable.
    """
    from IPython.display import display_html

    # Calculate the first (Q1) and third (Q3) quartiles
    q1, q3 = data[var].quantile([0.25, 0.75]).values

    # Compute the Interquartile Range (IQR)
    iqr = q3 - q1

    # Define lower and upper limits for outliers
    lower_limit = q1 - (band * iqr)
    upper_limit = q3 + (band * iqr)

    # Display IQR limits for reference
    display_html(3, f"{var} - IQR Limits:")
    print(f"{'Lower Limit':12}: {lower_limit}")
    print(f"{'Upper Limit':12}: {upper_limit}")
    print('  ' * 65)
    print('*-' * 65)

    # Filter and return rows where the variable is an outlier
    outliers = data[(data[var] < lower_limit) | (data[var] > upper_limit)].sort_values(var)
    return outliers

#  get_iqr_outliers(data, 'age', band = 2.0)



# -----------------------------------------------------------------------------------------------------------------------
# Display a string as HTML header of specified size in Jupyter/IPython environments.
def display_html(size=3, content="content"):
    """
    Display a string as HTML header of specified size in Jupyter/IPython environments.

    Parameters:
        size (int, optional): Header size (1-6, default is 3).
        content (str, optional): The content to display inside the header.

    Returns:
        None. Renders HTML in the notebook cell output.
    """
    display(HTML(f"<h{size}>{content}</h{size}>"))


# -----------------------------------------------------------------------------------------------------------------------
# Rotate the x-axis tick labels for better readability.
def rotate_xlabels(ax, angle=35):
    """
    Rotate the x-axis tick labels for better readability.

    Parameters:
        ax (matplotlib.axes.Axes): The axis object to modify.
        angle (int or float, optional): The rotation angle for the labels (default is 35 degrees).

    Returns:
        None. Modifies the axis in-place.
    """
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=angle,
        ha="right"
    )


# -----------------------------------------------------------------------------------------------------------------------
# Rotate the y-axis tick labels for better readability.
def rotate_ylabels(ax, angle=0):
    """
    Rotate the y-axis tick labels for better readability.

    Parameters:
        ax (matplotlib.axes.Axes): The axis object to modify.
        angle (int or float, optional): The rotation angle for the labels (default is 0 degrees).

    Returns:
        None. Modifies the axis in-place.
    """
    # Recommended approach: set rotation for each label
    for tick in ax.get_yticklabels():
        tick.set_rotation(angle)
# Rotate y-axis labels by 45 degrees
# rotate_ylabels(ax, angle=45)



# -----------------------------------------------------------------------------------------------------------------------
# Generate a summary DataFrame showing the count and percentage of missing values for each variable.
def missing_info(data):
    """
    Generate a summary DataFrame showing the count and percentage of missing values for each variable.

    Parameters:
        data (pd.DataFrame): The DataFrame to analyze.

    Returns:
        pd.DataFrame: DataFrame indexed by variable name, with columns for count and percentage of missing values,
                      sorted by count descending.
    """
    # Identify columns with any missing values
    na_cols = [col for col in data.columns if data[col].isna().any()]

    # Count missing values for each such column
    na_counts = [data[col].isna().sum() for col in na_cols]

    # Calculate percentage of missing values for each such column
    na_pct = [data[col].isna().mean() * 100 for col in na_cols]

    # Construct and return a summary DataFrame
    return (
        pd.DataFrame({
            "variable": na_cols,
            "count": na_counts,
            "percentage": na_pct
        })
        .sort_values(by="count", ascending=False)
        .set_index("variable")
    )
# summary = missing_info(tips)



# -----------------------------------------------------------------------------------------------------------------------
# Perform a Chi-Square test for association between two categorical variables.
def hyp_cat_cat(data, var1, var2, alpha=0.05):
    """
    Perform a Chi-Square test for association between two categorical variables.

    Parameters:
        data (pd.DataFrame): The DataFrame containing the variables.
        var1 (str): The name of the first categorical variable.
        var2 (str): The name of the second categorical variable.
        alpha (float, optional): Significance level for the test (default is 0.05).

    Returns:
        None. Prints and displays test results and interpretation.
    """
    # Display HTML header for context
    display_html(2, f"Hypothesis Test for Association between {var1} and {var2}")

    # Create a contingency table for the two categorical variables
    ct = pd.crosstab(data[var1], data[var2])

    # Display HTML subheader for the Chi-square test section
    display_html(3, "Chi-square Test")

    # Perform the Chi-Square test for independence
    chi2 = stats.chi2_contingency(ct)
    statistic = chi2.statistic
    pvalue = chi2.pvalue

    # Print Cramér's V (association strength), significance level, and hypotheses
    print(f"- {'Cramers V':21}: {cramers_v(data, var1, var2)}")
    print(f"- {'Significance Level':21}: {alpha * 100}%")
    print(f"- {'Null Hypothesis':21}: The samples are uncorrelated")
    print(f"- {'Alternate Hypothesis':21}: The samples are correlated")
    print(f"- {'Test Statistic':21}: {statistic}")
    print(f"- {'p-value':21}: {pvalue}")

    # Interpret the result based on p-value and significance level
    if pvalue < alpha:
        print(f"- Since p-value is less than {alpha}, we Reject the Null Hypothesis at {alpha * 100}% significance level")
        print(f"- CONCLUSION: The variables {var1} and {var2} are correlated")
    else:
        print(f"- Since p-value is greater than {alpha}, we Fail to Reject the Null Hypothesis at {alpha * 100}% significance level")
        print(f"- CONCLUSION: The variables {var1} and {var2} are uncorrelated")
# Optionally, specify a different significance level
# num_cat_hyp_testing(df, "Income", "Country", alpha=0.01)



# -----------------------------------------------------------------------------------------------------------------------
# Perform hypothesis tests (ANOVA and Kruskal-Wallis) to assess association between a numeric and a categorical variable.
def num_cat_hyp_testing(data, num_var, cat_var, alpha=0.05):
    """
    Perform hypothesis tests (ANOVA and Kruskal-Wallis) to assess association between a numeric and a categorical variable.

    Parameters:
        data (pd.DataFrame): The DataFrame containing the data.
        num_var (str): The numeric variable.
        cat_var (str): The categorical variable.
        alpha (float, optional): Significance level for the tests (default is 0.05).

    Returns:
        None. Prints and displays test results and interpretations.
    """
    # Display analysis header
    display_html(2, f"Hypothesis Test for Association between {num_var} and {cat_var}")

    # Group data by categorical variable, dropping missing values in numeric variable
    groups_df = data.dropna(subset=[num_var]).groupby(cat_var)
    groups = [group[num_var].values for _, group in groups_df]

    # --- ANOVA Test ---
    anova = stats.f_oneway(*groups)
    statistic = anova.statistic if hasattr(anova, "statistic") else anova[0]
    pvalue = anova.pvalue if hasattr(anova, "pvalue") else anova[1]
    display_html(3, "ANOVA Test")
    print(f"- {'Significance Level':21}: {alpha * 100}%")
    print(f"- {'Null Hypothesis':21}: The groups have similar population mean")
    print(f"- {'Alternate Hypothesis':21}: The groups don't have similar population mean")
    print(f"- {'Test Statistic':21}: {statistic}")
    print(f"- {'p-value':21}: {pvalue}")
    if pvalue < alpha:
        print(f"- Since p-value is less than {alpha}, we Reject the Null Hypothesis at {alpha * 100}% significance level")
        print(f"- CONCLUSION: The variables {num_var} and {cat_var} are associated with each other")
    else:
        print(f"- Since p-value is greater than {alpha}, we Fail to Reject the Null Hypothesis at {alpha * 100}% significance level")
        print(f"- CONCLUSION: The variables {num_var} and {cat_var} are not associated with each other")

    # --- Kruskal-Wallis Test ---
    kruskal = stats.kruskal(*groups)
    statistic = kruskal.statistic if hasattr(kruskal, "statistic") else kruskal[0]
    pvalue = kruskal.pvalue if hasattr(kruskal, "pvalue") else kruskal[1]
    display_html(3, "Kruskal-Wallis Test")
    print(f"- {'Significance Level':21}: {alpha * 100}%")
    print(f"- {'Null Hypothesis':21}: The groups have similar population median")
    print(f"- {'Alternate Hypothesis':21}: The groups don't have similar population median")
    print(f"- {'Test Statistic':21}: {statistic}")
    print(f"- {'p-value':21}: {pvalue}")
    if pvalue < alpha:
        print(f"- Since p-value is less than {alpha}, we Reject the Null Hypothesis at {alpha * 100}% significance level")
        print(f"- CONCLUSION: The variables {num_var} and {cat_var} are associated with each other")
    else:
        print(f"- Since p-value is greater than {alpha}, we Fail to Reject the Null Hypothesis at {alpha * 100}% significance level")
        print(f"- CONCLUSION: The variables {num_var} and {cat_var} are not associated with each other")


# -----------------------------------------------------------------------------------------------------------------------
# Display a detailed summary of a categorical variable, including meta-data, quick glance, descriptive statistics, and category distribution.
def cat_summary(data, var):
    """
    Display a detailed summary of a categorical variable, including meta-data, quick glance,
    descriptive statistics, and category distribution.

    Parameters:
        data (pd.DataFrame): The DataFrame containing the data.
        var (str): The categorical variable to summarize.

    Returns:
        None. Displays summary outputs.
    """
    import warnings
    warnings.filterwarnings("ignore")

    # Extract the column for analysis
    col = data.loc[:, var].copy()
    
    # Title
    display_html(2, var)

    # Quick glance at the raw column data
    display_html(3, "Quick Glance:")
    display(col)

    # Meta-data: data type, cardinality, missing data, available data
    display_html(3, "Meta-data:")
    print(f"{'Data Type':15}: {col.dtype}")
    print(f"{'Cardinality':15}: {col.nunique(dropna=True)} categories")
    print(f"{'Missing Data':15}: {col.isna().sum():,} rows ({col.isna().mean() * 100:.2f} %)")
    print(f"{'Available Data':15}: {col.count():,} / {len(col):,} rows")

    # Descriptive summary using pandas describe (includes count, unique, top, freq)
    display_html(3, "Summary:")
    display(
        col
        .describe()
        .rename("")
        .to_frame()
    )

    # Category distribution: count and percentage for each level
    display_html(3, "Categories Distribution:")
    with pd.option_context("display.max_rows", None):
        display(
            col
            .value_counts()
            .pipe(lambda ser: pd.concat(
                [
                    ser,
                    col.value_counts(normalize=True)
                ],
                axis=1
            ))
            .set_axis(["count", "percentage"], axis=1)
            .rename_axis(index="category")
        )
# Optionally, specify a different significance level
# num_num_hyp_testing(df, "Height", "Weight", alpha=0.01)        


# -----------------------------------------------------------------------------------------------------------------------
# Perform hypothesis testing for association between two numeric variables using Pearson and Spearman correlation tests.
def num_num_hyp_testing(data, var1, var2, alpha=0.05):
    """
    Perform hypothesis testing for association between two numeric variables using Pearson and Spearman correlation tests.

    Parameters:
        data (pd.DataFrame): The DataFrame containing the data.
        var1 (str): The first numeric variable.
        var2 (str): The second numeric variable.
        alpha (float, optional): Significance level for the tests (default is 0.05).

    Returns:
        None. Prints and displays test results and interpretations.
    """
    display_html(2, f"Hypothesis Test for Association between {var1} and {var2}")

    # Drop rows with missing values in either variable
    temp = data.dropna(subset=[var1, var2], how="any").copy()

    # --- Pearson Correlation Test (measures linear relationship) ---
    pearson = stats.pearsonr(temp[var1].values, temp[var2].values)
    statistic = pearson.statistic if hasattr(pearson, "statistic") else pearson[0]
    pvalue = pearson.pvalue if hasattr(pearson, "pvalue") else pearson[1]
    display_html(3, "Pearson Test")
    print(f"- {'Significance Level':21}: {alpha * 100}%")
    print(f"- {'Null Hypothesis':21}: The samples are uncorrelated")
    print(f"- {'Alternate Hypothesis':21}: The samples are correlated")
    print(f"- {'Test Statistic':21}: {statistic}")
    print(f"- {'p-value':21}: {pvalue}")
    if pvalue < alpha:
        print(f"- Since p-value is less than {alpha}, we Reject the Null Hypothesis at {alpha * 100}% significance level")
        print(f"- CONCLUSION: The variables {var1} and {var2} are correlated")
    else:
        print(f"- Since p-value is greater than {alpha}, we Fail to Reject the Null Hypothesis at {alpha * 100}% significance level")
        print(f"- CONCLUSION: The variables {var1} and {var2} are uncorrelated")

    # --- Spearman Correlation Test (measures monotonic relationship) ---
    spearman = stats.spearmanr(temp[var1].values, temp[var2].values)
    statistic = spearman.statistic if hasattr(spearman, "statistic") else spearman[0]
    pvalue = spearman.pvalue if hasattr(spearman, "pvalue") else spearman[1]
    display_html(3, "Spearman Test")
    print(f"- {'Significance Level':21}: {alpha * 100}%")
    print(f"- {'Null Hypothesis':21}: The samples are uncorrelated")
    print(f"- {'Alternate Hypothesis':21}: The samples are correlated")
    print(f"- {'Test Statistic':21}: {statistic}")
    print(f"- {'p-value':21}: {pvalue}")
    if pvalue < alpha:
        print(f"- Since p-value is less than {alpha}, we Reject the Null Hypothesis at {alpha * 100}% significance level")
        print(f"- CONCLUSION: The variables {var1} and {var2} are correlated")
    else:
        print(f"- Since p-value is greater than {alpha}, we Fail to Reject the Null Hypothesis at {alpha * 100}% significance level")
        print(f"- CONCLUSION: The variables {var1} and {var2} are uncorrelated")


# -----------------------------------------------------------------------------------------------------------------------
# Display a detailed summary of a numeric variable, including meta-data, percentiles, central tendency, spread, skewness, kurtosis, and normality tests.
def num_summary(data, var):
    """
    Display a detailed summary of a numeric variable, including meta-data, percentiles, central tendency,
    spread, skewness, kurtosis, and normality tests.

    Parameters:
        data (pd.DataFrame): The DataFrame containing the data.
        var (str): The numeric variable to summarize.

    Returns:
        None. Displays summary outputs.
    """
    import warnings
    warnings.filterwarnings("ignore")

    # Extract the column for analysis
    col = data.loc[:, var].copy()

    # Title
    display_html(size=2, content=var)

    # Quick glance at the raw column data
    display_html(3, "Quick Glance:")
    display(col)

    # Meta-data: data type, missing data, available data
    display_html(3, "Meta-data:")
    print(f"{'Data Type':15}: {col.dtype}")
    print(f"{'Missing Data':15}: {col.isna().sum():,} rows ({col.isna().mean() * 100:.2f} %)")
    print(f"{'Available Data':15}: {col.count():,} / {len(col):,} rows")

    # Percentiles (quantiles)
    display_html(3, "Percentiles:")
    display(
        col
        .quantile([0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0])
        .rename(index=lambda val: f"{val * 100:.0f}")
        .rename("value")
        .rename_axis(index="percentile")
        .to_frame()
    )

    # Central tendency (mean, trimmed mean, median)
    display_html(3, "Central Tendancy:")
    display(
        pd.Series({
            "mean": col.mean(),
            "trimmed mean (5%)": stats.trim_mean(col.values, 0.05),
            "trimmed mean (10%)": stats.trim_mean(col.values, 0.1),
            "median": col.median()
        })
        .rename("value")
        .to_frame()
    )

    # Measure of spread (variance, std, IQR, MAD, coefficient of variation)
    display_html(3, "Measure of Spread:")
    std = col.std()
    iqr = col.quantile(0.75) - col.quantile(0.25)
    display(
        pd.Series({
            "var": col.var(),
            "std": std,
            "IQR": iqr,
            "mad": stats.median_abs_deviation(col.dropna()),
            "coef_variance": std / col.mean() if col.mean() != 0 else float('nan')
        })
        .rename("value")
        .to_frame()
    )

    # Skewness and kurtosis
    display_html(3, "Skewness and Kurtosis:")
    display(
        pd.Series({
            "skewness": col.skew(),
            "kurtosis": col.kurtosis()
        })
        .rename("value")
        .to_frame()
    )

    alpha = 0.05
    # Test for normality
    display_html(3, "Hypothesis Testing for Normality:")

    # Shapiro-Wilk test
    display_html(4, "Shapiro-Wilk Test:")
    sw_test = stats.shapiro(col.dropna().values)
    sw_statistic = sw_test.statistic
    sw_pvalue = sw_test.pvalue
    print(f"{'Significance Level':21}: {alpha}")
    print(f"{'Null Hypothesis':21}: The data is normally distributed")
    print(f"{'Alternate Hypothesis':21}: The data is not normally distributed")
    print(f"{'p-value':21}: {sw_pvalue}")
    print(f"{'Test Statistic':21}: {sw_statistic}")
    if sw_pvalue < alpha:
        print(f"- Since p-value is less than alpha ({alpha}), we Reject the Null Hypothesis at {alpha * 100}% significance level")
        print("- CONCLUSION: We conclude that the data sample is not normally distributed")
    else:
        print(f"- Since p-value is greater than alpha ({alpha}), we Fail to Reject the Null Hypothesis at {alpha * 100}% significance level")
        print("- CONCLUSION: We conclude that the data sample is normally distributed")

    # Anderson-Darling test
    display_html(4, "Anderson-Darling Test:")
    ad_test = stats.anderson(col.dropna().values, dist="norm")
    ad_statistic = ad_test.statistic
    ad_critical = ad_test.critical_values[2]
    print(f"{'Significance Level':21}: {alpha}")
    print(f"{'Null Hypothesis':21}: The data is normally distributed")
    print(f"{'Alternate Hypothesis':21}: The data is not normally distributed")
    print(f"{'Critical Value':21}: {ad_critical}")
    print(f"{'Test Statistic':21}: {ad_statistic}")
    if ad_statistic >= ad_critical:
        print(f"- Since the Test-statistic is greater than Critical Value, we Reject the Null Hypothesis at {alpha * 100}% significance level")
        print("- CONCLUSION: We conclude that the data sample is not normally distributed")
    else:
        print(f"- Since the Test-statistic is less than Critical Value, we Fail to Reject the Null Hypothesis at {alpha * 100}% significance level")
        print("- CONCLUSION: We conclude that the data sample is normally distributed")
# Display detailed summary for the 'Income' column
# num_summary(df, "Income")



# -----------------------------------------------------------------------------------------------------------------------
# Performs the Jarque-Bera test for normality on a specified column of a DataFrame.
def test_for_normality(dataframe, column_name, alpha=0.05):
    """
    Performs the Jarque-Bera test for normality on a specified column of a DataFrame.

    Parameters:
        dataframe (pd.DataFrame): The input DataFrame.
        column_name (str): The column to test for normality.
        alpha (float): Significance level for the hypothesis test (default=0.05).

    Prints:
        Test statistic, p-value, and interpretation of the result.
    """
    # Drop missing values to ensure test validity
    data = dataframe[column_name].dropna()

    print("Jarque-Bera Test for Normality")

    # Perform the Jarque-Bera test using scipy.stats
    from scipy.stats import jarque_bera
    stat, p_val = jarque_bera(data)

    print(f"Test Statistic: {stat:.4f}")
    print(f"P-value: {p_val:.4g}")
    print(f"Significance Level (alpha): {alpha}")
    print("-*" * 32)

    # Interpret the result based on the p-value
    if p_val <= alpha:
        print("Reject the null hypothesis. The data is NOT normally distributed.")
    else:
        print("Fail to reject the null hypothesis. The data is likely normally distributed.")
# Optionally, specify a different significance level
# test_for_normality(df, "Income", alpha=0.01)


# -----------------------------------------------------------------------------------------------------------------------
# Performs a one-way ANOVA test to determine if there are significant differences between the means of groups defined by a categorical variable.
def anova_test(dataframe, num_col, cat_col, alpha=0.05):
    """
    Performs a one-way ANOVA test to determine if there are significant differences
    between the means of groups defined by a categorical variable.

    Parameters:
        dataframe (pd.DataFrame): Input DataFrame.
        num_col (str): Name of the numeric column (dependent variable).
        cat_col (str): Name of the categorical column (grouping variable).
        alpha (float): Significance level for the test (default=0.05).

    Prints:
        F-statistic, p-value, and interpretation of the result.
    """
    # Drop missing values to ensure valid input for ANOVA
    data = dataframe[[num_col, cat_col]].dropna()

    # Group the data by the categorical variable and extract the numeric values for each group
    groups = [group[num_col].values for _, group in data.groupby(cat_col)]

    # Perform one-way ANOVA using scipy.stats.f_oneway
    from scipy.stats import f_oneway
    f_stat, p_val = f_oneway(*groups)

    # Print the summary of ANOVA results
    print(f"ANOVA Results for '{num_col}' by '{cat_col}':")
    print(f"F-statistic: {f_stat:.4f}")
    print(f"P-value: {p_val:.4g}")
    print("-*" * 32)

    # Interpretation of the results
    if p_val <= alpha:
        print(f"Reject the null hypothesis (p ≤ {alpha}). There is a significant relationship between '{num_col}' and '{cat_col}'.")
    else:
        print(f"Fail to reject the null hypothesis (p > {alpha}). There is no significant relationship between '{num_col}' and '{cat_col}'.")
# Call the function to test if 'Score' means differ by 'Group'
# anova_test(df, "Score", "Group")


# -----------------------------------------------------------------------------------------------------------------------
# Performs the Chi-Square test of independence between two categorical columns.
def chi_2_test(dataframe, col1, col2, alpha=0.05):
    """
    Performs the Chi-Square test of independence between two categorical columns.

    Parameters:
        dataframe (pd.DataFrame): Input DataFrame.
        col1 (str): Name of the first categorical column.
        col2 (str): Name of the second categorical column.
        alpha (float): Significance level for the test (default=0.05).

    Prints:
        Contingency table, chi-squared statistic, degrees of freedom, p-value, and interpretation.
    """
    # Drop missing values to ensure a valid contingency table
    data = dataframe[[col1, col2]].dropna()

    # Create the contingency table
    contingency_table = pd.crosstab(data[col1], data[col2])
    print(f"\nContingency Table between '{col1}' and '{col2}':")
    print(contingency_table)
    print("-*" * 32)

    # Perform the chi-squared test of independence
    from scipy.stats import chi2_contingency
    chi2, p_val, dof, expected = chi2_contingency(contingency_table)

    # Print test results
    print(f"Chi-squared Statistic: {chi2:.4f}")
    print(f"Degrees of Freedom: {dof}")
    print(f"P-value: {p_val:.4g}")
    print("-*" * 32)

    # Interpret the result
    if p_val <= alpha:
        print(f"Reject the null hypothesis (p ≤ {alpha}). There is a significant association between '{col1}' and '{col2}'.")
    else:
        print(f"Fail to reject the null hypothesis (p > {alpha}). There is no significant association between '{col1}' and '{col2}'.")
# Call the function to test independence between 'Gender' and 'Purchased'
# chi_2_test(df, "Gender", "Purchased")



# -----------------------------------------------------------------------------------------------------------------------
# Visualizes the distribution of a numerical column using KDE, boxplot, and histogram. Optionally, groups by a categorical column.
def numerical_analysis(dataframe, column_name, cat_col=None, bins="auto"):
    """
    Visualizes the distribution of a numerical column using KDE, boxplot, and histogram.
    Optionally, groups by a categorical column.

    Parameters:
        dataframe (pd.DataFrame): Input data.
        column_name (str): Numerical column to analyze.
        cat_col (str, optional): Categorical column for grouping (default: None).
        bins (int, str, or sequence, optional): Histogram bins (default: "auto").

    Displays:
        - Kernel Density Estimate (KDE) plot
        - Boxplot
        - Histogram with KDE overlay
    """
    # Set up figure with white background and grid layout
    fig = plt.figure(figsize=(15, 10), facecolor='white')
    grid = GridSpec(nrows=2, ncols=2, figure=fig)
    ax1 = fig.add_subplot(grid[0, 0])
    ax2 = fig.add_subplot(grid[0, 1])
    ax3 = fig.add_subplot(grid[1, :])

    # Define color palette and style
    palette = 'viridis'
    single_color = '#2c7be5'
    grid_alpha = 0.4

    # --- KDE Plot ---
    if cat_col:
        # KDE by category
        sns.kdeplot(
            data=dataframe, x=column_name, hue=cat_col, ax=ax1,
            fill=True, palette=palette, alpha=0.6, linewidth=1.5
        )
    else:
        # KDE for entire column
        sns.kdeplot(
            data=dataframe, x=column_name, ax=ax1,
            fill=True, color=single_color, alpha=0.6, linewidth=1.5
        )
    ax1.set_title('Kernel Density Estimate', fontsize=12, pad=10)
    ax1.grid(True, linestyle=':', alpha=grid_alpha)

    # --- Boxplot ---
    if cat_col:
        # Boxplot by category (horizontal for clarity)
        sns.boxplot(
            data=dataframe, x=column_name, y=cat_col, ax=ax2,
            palette=palette, width=0.6, linewidth=1
        )
    else:
        # Boxplot for entire column
        sns.boxplot(
            data=dataframe, x=column_name, ax=ax2,
            color=single_color, width=0.4, linewidth=1
        )
    ax2.set_title('Boxplot', fontsize=12, pad=10)
    ax2.grid(True, linestyle=':', alpha=grid_alpha)

    # --- Histogram with KDE ---
    if cat_col:
        # Histogram by category
        sns.histplot(
            data=dataframe, x=column_name, bins=bins, hue=cat_col,
            kde=True, ax=ax3, palette=palette, alpha=0.7,
            edgecolor='white', linewidth=0.5
        )
    else:
        # Histogram for entire column
        sns.histplot(
            data=dataframe, x=column_name, bins=bins, kde=True,
            ax=ax3, color=single_color, alpha=0.7,
            edgecolor='white', linewidth=0.5
        )
    ax3.set_title('Histogram with KDE', fontsize=12, pad=10)
    ax3.grid(True, linestyle=':', alpha=grid_alpha)

    # --- Final layout and style tweaks ---
    for ax in [ax1, ax2, ax3]:
        sns.despine(ax=ax, left=True)

    plt.tight_layout()
    plt.show()
# Analyze 'Age' distribution grouped by 'Gender'
# numerical_analysis(df, "Age", cat_col="Gender")


# -----------------------------------------------------------------------------------------------------------------------
# Visualizes the relationship between one numerical and two categorical variables using barplot, boxplot, violin plot, and stripplot.
def multivariate_analysis(dataframe, num_column, cat_column_1, cat_column_2):
    """
    Visualizes the relationship between one numerical and two categorical variables
    using barplot, boxplot, violin plot, and stripplot.

    Parameters:
        dataframe (pd.DataFrame): Input data.
        num_column (str): Numerical column to analyze.
        cat_column_1 (str): First categorical column (x-axis).
        cat_column_2 (str): Second categorical column (hue/grouping).

    Displays:
        - Barplot (mean values)
        - Boxplot (distribution)
        - Violin plot (distribution + density)
        - Strip plot (individual observations)
    """
    # Set up a 2x2 grid of subplots with white background
    fig, axes = plt.subplots(2, 2, figsize=(16, 9), facecolor='white')
    ax1, ax2 = axes[0]
    ax3, ax4 = axes[1]

    palette = "Set2"

    # --- Barplot: Mean value by category ---
    sns.barplot(
        data=dataframe, x=cat_column_1, y=num_column,
        hue=cat_column_2, ax=ax1, palette=palette, edgecolor="black"
    )
    ax1.set_title("Mean Value by Category (Barplot)", fontsize=14)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(title=cat_column_2, loc="best")

    # --- Boxplot: Distribution by category ---
    sns.boxplot(
        data=dataframe, x=cat_column_1, y=num_column,
        hue=cat_column_2, ax=ax2, palette="pastel"
    )
    ax2.set_title("Distribution by Category (Boxplot)", fontsize=14)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(title=cat_column_2, loc="best")

    # --- Violin plot: Distribution and density ---
    sns.violinplot(
        data=dataframe, x=cat_column_1, y=num_column,
        hue=cat_column_2, ax=ax3, palette=palette, split=True, inner="quartile"
    )
    ax3.set_title("Distribution by Category (Violin)", fontsize=14)
    ax3.grid(True, linestyle="--", alpha=0.5)
    ax3.legend(title=cat_column_2, loc="best")

    # --- Strip plot: Individual observations ---
    sns.stripplot(
        data=dataframe, x=cat_column_1, y=num_column,
        hue=cat_column_2, dodge=True, ax=ax4, palette=palette, size=4, alpha=0.6
    )
    ax4.set_title("Individual Observations (Stripplot)", fontsize=14)
    ax4.grid(True, linestyle="--", alpha=0.5)
    ax4.legend(title=cat_column_2, loc="best")

    # --- Remove duplicate legends for clarity ---
    for ax in [ax1, ax2, ax3, ax4]:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(title=cat_column_2, loc="best")
        else:
            if hasattr(ax, "legend_") and ax.legend_:
                ax.legend_.remove()

    plt.tight_layout()
    plt.show()
# Call the function to analyze 'Age' by 'Gender' and 'Region'
# multivariate_analysis(df, "Age", "Gender", "Region")


# -----------------------------------------------------------------------------------------------------------------------
# Provides a detailed analysis of a categorical column, including value counts, percentages, unique categories, and a countplot visualization.
def categorical_analysis(dataframe, column_name):
    """
    Provides a detailed analysis of a categorical column, including value counts,
    percentages, unique categories, and a countplot visualization.

    Parameters:
        dataframe (pd.DataFrame): Input DataFrame.
        column_name (str): Name of the categorical column to analyze.

    Displays:
        - Table of value counts and percentages
        - List and count of unique categories
        - Countplot with annotated bars
    """
    # Calculate value counts and percentages
    counts = dataframe[column_name].value_counts()
    percentages = (
        dataframe[column_name]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
        .astype(str)
        .add("%")
    )

    # Display counts and percentages as a DataFrame
    summary_df = pd.DataFrame({"Count": counts, "Percentage": percentages})
    print(f"\nValue Counts and Percentages for '{column_name}':")
    print(summary_df)
    print("-*" * 32)

    # Get unique categories and their count
    unique_categories = dataframe[column_name].unique().tolist()
    number_of_categories = dataframe[column_name].nunique()
    print(f"The unique categories in '{column_name}' column are: {unique_categories}")
    print(f"The number of categories in '{column_name}' column is: {number_of_categories}")
    print("-*" * 32)

    # Enhanced countplot
    plt.figure(figsize=(10, 6), facecolor='white')
    ax = sns.countplot(
        data=dataframe,
        x=column_name,
        palette="Set2",
        edgecolor='black'
    )
    plt.xticks(rotation=45)
    plt.title(f'Count of Categories in {column_name}', fontsize=16)
    plt.xlabel(column_name, fontsize=13)
    plt.ylabel('Count', fontsize=13)
    ax.grid(axis='y', linestyle='--', alpha=0.6)

    # Annotate bars with counts
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{height}', (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom', fontsize=11, color='black', fontweight='bold')

    plt.tight_layout()
    plt.show()
# categorical_analysis(df, "Country")


# -----------------------------------------------------------------------------------------------------------------------
# Visualizes the relationship between a categorical and a numerical variable using barplot, boxplot, violin plot, and strip plot.
def numerical_categorical_analysis(dataframe, cat_column_1, num_column):
    """
    Visualizes the relationship between a categorical and a numerical variable
    using barplot, boxplot, violin plot, and strip plot.

    Parameters:
        dataframe (pd.DataFrame): Input DataFrame.
        cat_column_1 (str): Categorical column for grouping (x-axis).
        num_column (str): Numerical column to analyze (y-axis).

    Displays:
        - Barplot (mean values)
        - Boxplot (distribution)
        - Violin plot (distribution + density)
        - Strip plot (individual observations)
    """
    # Set up a 2x2 grid of subplots with white background
    fig, axes = plt.subplots(2, 2, figsize=(16, 9), facecolor='white')
    ax1, ax2 = axes[0]
    ax3, ax4 = axes[1]

    # Define color palettes for consistency
    palette = "Set2"
    single_color = "#3498db"

    # --- Barplot: Mean value by category ---
    sns.barplot(
        data=dataframe, x=cat_column_1, y=num_column, ax=ax1,
        palette=palette
    )
    ax1.set_title("Mean Value by Category", fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # --- Boxplot: Distribution by category ---
    sns.boxplot(
        data=dataframe, x=cat_column_1, y=num_column, ax=ax2,
        palette="pastel"
    )
    ax2.set_title("Distribution by Category (Boxplot)", fontsize=14)
    ax2.grid(True, linestyle='--', alpha=0.5)

    # --- Violin plot: Distribution and density ---
    sns.violinplot(
        data=dataframe, x=cat_column_1, y=num_column, ax=ax3,
        palette=palette, inner="quartile"  # Shows quartiles inside the violin[1][2]
    )
    ax3.set_title("Distribution by Category (Violin)", fontsize=14)
    ax3.grid(True, linestyle='--', alpha=0.5)

    # --- Strip plot: Individual observations ---
    sns.stripplot(
        data=dataframe, x=cat_column_1, y=num_column, ax=ax4,
        color=single_color, size=4, jitter=True, alpha=0.6
    )
    ax4.set_title("Individual Observations (Stripplot)", fontsize=14)
    ax4.grid(True, linestyle='--', alpha=0.5)

    # Improve overall layout
    plt.tight_layout()
    plt.show()
# numerical_categorical_analysis(df, "gender", "salary")

