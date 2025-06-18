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
# Generate univariate plots for date-time variables.
def dt_univar_plots(data, var, target = None, bins="auto"):
    """
    Generate univariate plots for date-time variables.
    
    Parameters:
        data (pd.DataFrame): Input DataFrame containing the data.
        var (str): The name of the date-time variable to plot.
        target (str, optional): The target variable for the line plot. If None, the line plot is skipped.
        bins (int, str, or sequence, optional): Bin specification for the histogram.
    """
    # Display a custom HTML header for the plots
    display_html(3, f"Univariate plots of {var}")
    
    # Extract the variable column for plotting
    col = data.loc[:, var].copy()
    
    # Create a figure with two subplots: one for histogram, one for line plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 4))
    
    # --- Histogram: visualize the distribution of the date-time variable ---
    sns.histplot(
        data=data,
        x=var,
        bins=bins,
        color="#1973bd",
        ax=ax1
    )
    # Add rugplot for additional distribution detail
    sns.rugplot(
        data=data,
        x=var,
        color="darkblue",
        height=0.035,
        ax=ax1
    )
    ax1.set(title="Histogram")
    rotate_xlabels(ax1)
    
    # --- Line plot: show trend of target variable over the date-time variable ---
    if target and target in data.columns:
        sns.lineplot(
            data=data,
            x=var,
            y=target,
            color="#d92b2b",
            ax=ax2
        )
        ax2.set(title="Line Plot")
        rotate_xlabels(ax2)
    else:
        # Hide the second axis if target is not provided
        ax2.set_visible(False)
    
    plt.tight_layout()
# dt_univar_plots(df, "your_datetime_column", target="your_target_column")


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
# Visualize the count of missing values for each variable in a DataFrame using a bar chart.
import matplotlib.pyplot as plt
import pandas as pd

def plot_missing_info(df, bar_label_params=None, figsize=(10, 4), rotation=45):
    """
    Plot the count of missing values per column in a DataFrame.

    Parameters:
        df (pd.DataFrame): Input DataFrame to analyze.
        bar_label_params (dict, optional): Parameters passed to ax.bar_label for customizing labels.
        figsize (tuple, optional): Size of the plot figure. Default is (10, 4).
        rotation (int, optional): Rotation angle for x-axis labels. Default is 45 degrees.

    Returns:
        tuple: (fig, ax) - Matplotlib Figure and Axes objects.
    """
    if bar_label_params is None:
        bar_label_params = {}

    # Calculate missing value counts per column
    na_data = df.isnull().sum()
    na_data = na_data[na_data > 0].sort_values(ascending=False)

    if na_data.empty:
        print("No missing data to plot.")
        return None, None

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(
        na_data.index,
        na_data.values,
        color="#1eba47",
        edgecolor="black",
        alpha=0.7
    )

    # Add value labels
    ax.bar_label(bars, **bar_label_params)

    # Set labels and title
    ax.set(
        xlabel="Variable",
        ylabel="Count",
        title="Missing Data Counts per Variable"
    )

    # Rotate x-axis labels
    ax.set_xticklabels(na_data.index, rotation=rotation, ha='right')

    # Final layout
    plt.tight_layout()
    plt.show()

    return fig, ax

# plot_missing_info(df)




# -----------------------------------------------------------------------------------------------------------------------
# Visualize bivariate relationships between two categorical variables using multiple plot types.
def cat_bivar_plots(
    data,
    var1,
    var2,
    k1=None,
    k2=None,
    order1=None,
    order2=None,
    figsize=(12, 8.5)
):
    """
    Visualize bivariate relationships between two categorical variables using multiple plot types.

    Parameters:
        data (pd.DataFrame): The DataFrame containing the variables.
        var1 (str): The first categorical variable.
        var2 (str): The second categorical variable.
        k1 (int, optional): If specified, restrict var1 to its top k1 categories.
        k2 (int, optional): If specified, restrict var2 to its top k2 categories.
        order1 (list, optional): Custom order for var1 categories.
        order2 (list, optional): Custom order for var2 categories.
        figsize (tuple, optional): Figure size for the plots.

    Returns:
        None. Displays a grid of bivariate plots.
    """
    import warnings
    warnings.filterwarnings("ignore")

    # Display analysis header
    display_html(2, f"Bi-variate Analysis between {var1} and {var2}")
    display_html(content="")

    # Optionally restrict to top-k categories for each variable
    if k1 is not None:
        data = get_top_k(data, var1, k=k1)
    if k2 is not None:
        data = get_top_k(data, var2, k=k2)

    # Set up a 2x2 grid of subplots
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.ravel()

    # --- Cross-tabulation heatmap ---
    ct = pd.crosstab(
        index=data[var1],
        columns=data[var2]
    ).reindex(index=order1, columns=order2)
    cat_heat_map(
        ct,
        mask=False,
        vmin=ct.values.min(),
        vmax=ct.values.max(),
        fmt="d",
        cmap="Blues",
        cbar_kws=dict(location="top", label="Counts"),
        ax=axes[0]
    )
    rotate_ylabels(axes[0])
    rotate_xlabels(axes[0])

    # --- Normalized cross-tabulation heatmap ---
    norm_ct = pd.crosstab(
        index=data[var1],
        columns=data[var2],
        normalize="index"
    ).reindex(index=order1, columns=order2)
    cat_heat_map(
        norm_ct,
        mask=False,
        vmin=0,
        vmax=1,
        fmt=".2f",
        cmap="Greens",
        cbar_kws=dict(location="top", label="Normalized"),
        ax=axes[1]
    )
    axes[1].set(ylabel="")
    rotate_ylabels(axes[1])
    rotate_xlabels(axes[1])

    # --- Grouped bar plot (absolute counts) ---
    ct.plot.bar(
        ax=axes[2],
        title="Bar Plot",
        legend=False
    )
    rotate_xlabels(axes[2])

    # --- Stacked bar plot (normalized counts) ---
    norm_ct.plot.bar(
        ax=axes[3],
        title="Stacked Bar Plot",
        stacked=True
    )
    rotate_xlabels(axes[3])
    axes[3].legend(
        loc="upper left",
        bbox_to_anchor=(1, 1),
        title=var2
    )

    plt.tight_layout()
    plt.show()
# cat_bivar_plots(df, "categorical_var1", "categorical_var2")



# -----------------------------------------------------------------------------------------------------------------------
# Plot a heatmap for categorical data using Seaborn, with optional masking of the upper triangle.
def cat_heat_map(data, mask=True, **kwargs):
    """
    Plot a heatmap for categorical data using Seaborn, with optional masking of the upper triangle.

    Parameters:
        data (pd.DataFrame): The matrix (e.g., cross-tabulation) to visualize.
        mask (bool, optional): If True, mask the upper triangle (useful for symmetric matrices).
        **kwargs: Additional keyword arguments passed to sns.heatmap.

    Returns:
        matplotlib.axes.Axes: The heatmap axes object.
    """
    # Create a mask for the upper triangle if requested (useful for symmetric matrices)
    if mask:
        mask_arr = np.zeros_like(data, dtype=bool)
        mask_arr[np.triu_indices_from(mask_arr)] = True
    else:
        mask_arr = None

    # Plot the heatmap with annotations and grid lines for clarity
    return sns.heatmap(
        data=data,
        mask=mask_arr,
        annot=True,            # Show values in each cell
        linewidths=1.5,        # Add lines between cells for readability
        linecolor="white",     # Set grid line color
        square=True,           # Make cells square-shaped
        **kwargs
    )
# cat_heat_map(crosstab_df)



# -----------------------------------------------------------------------------------------------------------------------
# This will plot bar chart, Box plot and voilin plot.
from IPython.display import HTML, display
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

def num_cat_bivar_plots(
    data,
    num_var,
    cat_var,
    k=None,
    estimator="mean",
    orient="v",
    order=None,
    figsize=(15, 4)
):
    """
    Generate bivariate plots to visualize the relationship between a numeric and a categorical variable.

    Parameters:
        data (pd.DataFrame): The DataFrame containing the data.
        num_var (str): The numeric variable.
        cat_var (str): The categorical variable.
        k (int, optional): If specified, restricts to top-k categories of cat_var.
        estimator (str or function, optional): Aggregation function for the bar plot (default is "mean").
        orient (str, optional): "v" for vertical, "h" for horizontal plots.
        order (list, optional): Custom order for categories.
        figsize (tuple, optional): Figure size.

    Returns:
        None. Displays bar, box, and violin plots.
    """

    def get_values(data, num_var, cat_var, estimator, order=None):
        """
        Helper function to aggregate numeric variable by categorical variable.
        """
        return (
            data
            .groupby(cat_var)
            .agg(estimator, numeric_only=True)
            .loc[:, num_var]
            .dropna()
            .sort_values()
            .reindex(index=order)
        )

    import warnings
    warnings.filterwarnings("ignore")

    display_html(2, f"Bi-variate Analysis between {cat_var} and {num_var}")
    display_html(content="")

    # Restrict to top-k categories if specified
    if k is None:
        temp = get_values(data, num_var, cat_var, estimator, order=order)
    else:
        data = get_top_k(data, cat_var, k=k)
        temp = get_values(data, num_var, cat_var, estimator)

    # Vertical orientation: bar, box, violin plots side by side
    if orient == "v":
        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # Bar plot: aggregated value (e.g., mean) for each category
        sns.barplot(
            x=temp.index,
            y=temp.values,
            color="#d92b2b",
            ax=axes[0],
            edgecolor="black",
            alpha=0.5
        )
        axes[0].set(
            title="Bar Plot",
            xlabel=cat_var,
            ylabel=f"{estimator} of {num_var}"
        )
        rotate_xlabels(axes[0])

        # Box plot: distribution of numeric variable by category
        sns.boxplot(
            data=data,
            x=cat_var,
            y=num_var,
            color="lightgreen",
            order=temp.index,
            ax=axes[1]
        )
        axes[1].set(
            title="Box Plot",
            xlabel=cat_var,
            ylabel=num_var
        )
        rotate_xlabels(axes[1])

        # Violin plot: distribution and density by category
        sns.violinplot(
            data=data,
            x=cat_var,
            y=num_var,
            color="#0630c9",
            order=temp.index,
            ax=axes[2],
            alpha=0.5
        )
        axes[2].set(
            title="Violin Plot",
            xlabel=cat_var,
            ylabel=num_var
        )
        rotate_xlabels(axes[2])

    # Horizontal orientation: bar, box, violin plots stacked
    else:
        fig, axes = plt.subplots(3, 1, figsize=figsize)

        # Bar plot: aggregated value for each category
        sns.barplot(
            y=temp.index,
            x=temp.values,
            color="#d92b2b",
            ax=axes[0],
            edgecolor="black",
            alpha=0.5
        )
        axes[0].set(
            title="Bar Plot",
            xlabel=f"{estimator} of {num_var}",
            ylabel=cat_var
        )

        # Box plot: distribution by category
        sns.boxplot(
            data=data,
            y=cat_var,
            x=num_var,
            color="lightgreen",
            order=temp.index,
            ax=axes[1]
        )
        axes[1].set(
            title="Box Plot",
            xlabel=num_var,
            ylabel=cat_var
        )

        # Violin plot: distribution and density by category
        sns.violinplot(
            data=data,
            y=cat_var,
            x=num_var,
            color="#0630c9",
            order=temp.index,
            ax=axes[2],
            alpha=0.5
        )
        axes[2].set(
            title="Violin Plot",
            xlabel=num_var,
            ylabel=cat_var
        )

    plt.tight_layout()
    plt.show()
# num_cat_bivar_plots(data = df, num_var = 'sweetness', cat_var = 'quality')


# -----------------------------------------------------------------------------------------------------------------------
# Generate univariate visualizations for a categorical variable, including bar chart, pie chart, and word cloud.
def cat_univar_plots(
    data,
    var,
    k=None,
    order=None,
    show_wordcloud=True,
    figsize=(12, 8.5)
):
    """
    Generate univariate visualizations for a categorical variable, including bar chart, pie chart, and word cloud.

    Parameters:
        data (pd.DataFrame): The DataFrame containing the data.
        var (str): The categorical variable to visualize.
        k (int, optional): If specified, restricts to top-k categories.
        order (list, optional): Custom order for categories.
        show_wordcloud (bool, optional): Whether to display the word cloud.
        figsize (tuple, optional): Figure size for the plots.

    Returns:
        None. Displays the plots.
    """

    # Display analysis header
    display_html(2, f"Univariate Analysis of {var}")
    display_html(content="")

    # Set up a grid for bar, pie, and word cloud plots
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 2, figure=fig)
    ax1 = fig.add_subplot(gs[0, 0])  # bar-chart
    ax2 = fig.add_subplot(gs[0, 1])  # pie-chart
    ax3 = fig.add_subplot(gs[1, :])  # word-cloud

    # Get category counts, optionally restricting to top-k
    if k is None:
        counts = (
            data[var]
            .value_counts()
            .reindex(index=order)
        )
    else:
        temp = get_top_k(data, var, k=k)
        counts = temp[var].value_counts()

    # Generate random colors for categories
    colors = [tuple(np.random.choice(256, size=3) / 255) for _ in range(len(counts))]

    # Bar chart: shows frequency of each category
    bar_chart(counts, colors, ax1)

    # Pie chart: shows proportion of each category
    pie_chart(counts, colors, ax2)

    # Word cloud: visual representation of category frequency
    if show_wordcloud:
        var_string = " ".join(
            data[var]
            .dropna()
            .astype(str)
            .str.replace(" ", "_")
            .to_list()
        )
        word_cloud = WordCloud(
            width=2000,
            height=700,
            random_state=42,
            background_color="black",
            colormap="Set2",
            stopwords=STOPWORDS
        ).generate(var_string)
        ax3.imshow(word_cloud)
        ax3.axis("off")
        ax3.set_title("Word Cloud")
    else:
        ax3.remove()

    plt.tight_layout()
    plt.show()
# fxn calling
# cat_univar_plots(df, "your_categorical_column")




# -----------------------------------------------------------------------------------------------------------------------
# Create a bar chart for categorical counts with custom colors and labels.
def bar_chart(counts, colors, ax):
    """
    Create a bar chart for categorical counts with custom colors and labels.

    Parameters:
        counts (pd.Series): Category counts (index as categories, values as counts).
        colors (list): List of colors for each bar.
        ax (matplotlib.axes.Axes): The axis to plot on.

    Returns:
        None. Draws the bar chart on the given axis.
    """
    # Plot the bar chart with custom colors and edge color
    barplot = ax.bar(
        x=range(len(counts)),
        height=counts.values,
        tick_label=counts.index,
        color=colors,
        edgecolor="black",
        alpha=0.7
    )

    # Add value labels above each bar for clarity
    ax.bar_label(
        barplot,
        padding=5,
        color="black"
    )

    # Set chart title and axis labels
    ax.set(
        title="Bar Chart",
        xlabel="Categories",
        ylabel="Count"
    )

    # Rotate x-tick labels for better readability
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        ha="right"
    )

# calling fxn
# fig, ax = plt.subplots()
# bar_chart(counts, colors, ax)
# plt.show()



# -----------------------------------------------------------------------------------------------------------------------
# Create a pie chart for categorical counts with custom colors and labels.
def pie_chart(counts, colors, ax):
    """
    Create a pie chart for categorical counts with custom colors and labels.

    Parameters:
        counts (pd.Series): Category counts (index as categories, values as counts).
        colors (list): List of colors for each slice.
        ax (matplotlib.axes.Axes): The axis to plot on.

    Returns:
        None. Draws the pie chart on the given axis.
    """
    # Plot the pie chart with percentage labels and custom styling
    wedges, texts, autotexts = ax.pie(
        counts.values,
        labels=counts.index,
        autopct="%.2f%%",
        colors=colors,
        wedgeprops=dict(alpha=0.7, edgecolor="black"),
    )

    # Set chart title
    ax.set_title("Pie Chart")

    # Add legend for categories, positioned outside the plot
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        title="Categories",
        title_fontproperties=dict(weight="bold", size=10)
    )

    # Style the percentage labels for better visibility
    plt.setp(
        autotexts,
        weight="bold",
        color="white"
    )

# calling fxn
# fig, ax = plt.subplots()
# pie_chart(counts, colors, ax)
# plt.show()




# -----------------------------------------------------------------------------------------------------------------------
# Replace all but the top-k most frequent categories in a column with 'Other'.
def get_top_k(data, var, k):
    """
    Replace all but the top-k most frequent categories in a column with 'Other'.

    Parameters:
        data (pd.DataFrame): The DataFrame containing the data.
        var (str): The categorical variable to process.
        k (int): Number of top categories to keep.

    Returns:
        pd.DataFrame: DataFrame with rare categories replaced by 'Other'.

    Raises:
        ValueError: If k is not less than the cardinality of the variable.
    """
    col = data.loc[:, var].copy()
    cardinality = col.nunique(dropna=True)
    if k >= cardinality:
        raise ValueError(f"Cardinality of {var} is {cardinality}. K must be less than {cardinality}.")
    else:
        # Get the top-k categories by frequency
        top_categories = col.value_counts(dropna=True).index[:k]
        # Replace all other categories with 'Other'
        data = data.assign(**{
            var: np.where(col.isin(top_categories), col, "Other")
        })
        return data
# get_top_k(df, "your_categorical_column", k)



# -----------------------------------------------------------------------------------------------------------------------
# Visualize the bivariate relationship between two numeric variables using scatter and hexbin plots.
def num_bivar_plots(
    data,
    var_x,
    var_y,
    figsize=(12, 4.5),
    scatter_kwargs=dict(),
    hexbin_kwargs=dict()
):
    """
    Visualize the bivariate relationship between two numeric variables using scatter and hexbin plots.

    Parameters:
        data (pd.DataFrame): The DataFrame containing the data.
        var_x (str): Name of the x-axis numeric variable.
        var_y (str): Name of the y-axis numeric variable.
        figsize (tuple, optional): Figure size.
        scatter_kwargs (dict, optional): Additional keyword arguments for seaborn.scatterplot.
        hexbin_kwargs (dict, optional): Additional keyword arguments for matplotlib hexbin.

    Returns:
        None. Displays the plots.
    """
    # Display analysis header
    display_html(2, f"Bi-variate Analysis between {var_x} and {var_y}")
    display_html(content="")

    # Set up figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # --- Scatter Plot: shows individual data points ---
    sns.scatterplot(
        data=data,
        x=var_x,
        y=var_y,
        ax=axes[0],
        edgecolors="black",
        **scatter_kwargs
    )
    axes[0].set(title="Scatter Plot")

    # --- Hexbin Plot: shows density of points in hexagonal bins ---
    col_x = data[var_x]
    col_y = data[var_y]
    hexbin = axes[1].hexbin(
        x=col_x,
        y=col_y,
        **hexbin_kwargs
    )
    axes[1].set(
        title="Hexbin Plot",
        xlabel=var_x,
        xlim=(col_x.min(), col_x.max()),
        ylim=(col_y.min(), col_y.max())
    )
    # Add colorbar to show counts per hexagon
    plt.colorbar(hexbin, ax=axes[1], label="Count")

    plt.tight_layout()
    plt.show()
# num_bivar_plots(df, "numeric_column_x", "numeric_column_y")



# -----------------------------------------------------------------------------------------------------------------------
# Plot a Cramer's V heatmap showing the association strength between all pairs of categorical variables.
def cramersV_heatmap(data, figsize=(12, 6), cmap="Blues"):
    """
    Plot a Cramer's V heatmap showing the association strength between all pairs of categorical variables.

    Parameters:
        data (pd.DataFrame): The DataFrame containing the data.
        figsize (tuple, optional): Figure size for the heatmap.
        cmap (str, optional): Colormap for the heatmap.

    Returns:
        None. Displays the heatmap.
    """
    # Select categorical columns (object dtype)
    cols = data.select_dtypes(include="O").columns.to_list()

    # Initialize a square DataFrame for the Cramer's V values
    matrix = (
        pd.DataFrame(data=np.ones((len(cols), len(cols))))
        .set_axis(cols, axis=0)
        .set_axis(cols, axis=1)
    )

    # Compute Cramer's V for each pair of categorical variables
    for col1 in cols:
        for col2 in cols:
            if col1 != col2:
                matrix.loc[col1, col2] = cramers_v(data, col1, col2)

    # Mask the upper triangle for cleaner visualization
    mask = np.zeros_like(matrix, dtype=bool)
    mask[np.triu_indices_from(mask)] = True

    # Plot the heatmap
    fig, ax = plt.subplots(figsize=figsize)
    hm = sns.heatmap(
        matrix,
        vmin=0,
        vmax=1,
        cmap=cmap,
        annot=True,
        fmt=".2f",
        square=True,
        linewidths=1.5,
        mask=mask,
        ax=ax
    )
    ax.set(title="Cramer's V Correlation Matrix Heatmap")
    rotate_xlabels(ax)
    rotate_ylabels(ax)
# cramersV_heatmap(df)



# -----------------------------------------------------------------------------------------------------------------------
# Generate a suite of univariate plots for a numeric variable, including histogram, CDF, power transform, box plot, violin plot, and QQ plot.
def num_univar_plots(data, var, bins=10, figsize=(15, 7)):
    """
    Generate a suite of univariate plots for a numeric variable, including histogram, CDF, power transform, box plot,
    violin plot, and QQ plot.

    Parameters:
        data (pd.DataFrame): The DataFrame containing the data.
        var (str): The numeric variable to visualize.
        bins (int, optional): Number of bins for the histogram.
        figsize (tuple, optional): Figure size for the plots.

    Returns:
        None. Displays the plots.
    """
    display_html(2, f"Univariate Analysis of {var}")
    display_html(content="")
    col = data.loc[:, var].copy()

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.ravel()

    # Histogram with KDE and rugplot
    sns.histplot(
        data,
        x=var,
        bins=bins,
        kde=True,
        color="#1973bd",
        ax=axes[0],
    )
    sns.rugplot(
        data,
        x=var,
        color="black",
        height=0.035,
        ax=axes[0]
    )
    axes[0].set(title="Histogram")

    # CDF (Empirical Cumulative Distribution Function)
    sns.ecdfplot(
        data,
        x=var,
        ax=axes[1],
        color="red"
    )
    axes[1].set(title="CDF")

    # Power transformed KDE and rugplot
    data = data.assign(**{
        f"{var}_pwt": (
            PowerTransformer()
            .fit_transform(data.loc[:, [var]])
            .ravel()
        )
    })
    sns.kdeplot(
        data,
        x=f"{var}_pwt",
        fill=True,
        color="#f2b02c",
        ax=axes[2]
    )
    sns.rugplot(
        data,
        x=f"{var}_pwt",
        color="black",
        height=0.035,
        ax=axes[2]
    )
    axes[2].set(title="Power Transformed")

    # Box plot
    sns.boxplot(
        data,
        x=var,
        color="#4cd138",
        ax=axes[3]
    )
    axes[3].set(title="Box Plot")

    # Violin plot
    sns.violinplot(
        data,
        x=var,
        color="#ed68b4",
        ax=axes[4]
    )
    axes[4].set(title="Violin Plot")

    # QQ plot for normality check
    sm.qqplot(
        col.dropna(),
        line="45",
        fit=True,
        ax=axes[5]
    )
    axes[5].set(title="QQ Plot")

    plt.tight_layout()
    plt.show()
# num_univar_plots(df, "your_numeric_column")



# -----------------------------------------------------------------------------------------------------------------------
# Create a grid of pairwise scatterplots for the variables in a DataFrame using seaborn's PairGrid.
def pair_plots(
    data,
    height=3,
    aspect=1.5,
    hue=None,
    legend=False
):
    """
    Create a grid of pairwise scatterplots for the variables in a DataFrame using seaborn's PairGrid.

    Parameters:
        data (pd.DataFrame): The DataFrame containing the data.
        height (float, optional): Height (in inches) of each facet (default is 3).
        aspect (float, optional): Aspect ratio of each facet (default is 1.5).
        hue (str, optional): Variable in data to map plot aspects to different colors.
        legend (bool, optional): Whether to display the legend (default is False).

    Returns:
        None. Displays the pair plot grid.
    """
    display_html(2, "Pair Plots")

    pair_grid = sns.PairGrid(
        data=data,
        aspect=aspect,
        height=height,
        hue=hue,
        corner=True  # Only plot lower triangle and diagonal for clarity
    )
    pair_grid.map_lower(sns.scatterplot)

    if legend:
        pair_grid.add_legend()
# pair_plots(df)



# -----------------------------------------------------------------------------------------------------------------------
# Plots a heatmap of the correlation matrix for the given DataFrame.
def correlation_heatmap(
    data,
    figsize=(12, 6),
    method="spearman",
    cmap="RdBu"
):
    """
    Plots a heatmap of the correlation matrix for the given DataFrame.

    Parameters:
        data (pd.DataFrame): Input data for which to compute the correlation matrix.
        figsize (tuple): Size of the matplotlib figure.
        method (str): Correlation method ('pearson', 'spearman', or 'kendall').
        cmap (str): Colormap for the heatmap.
    """
    # Compute the correlation matrix for numeric columns only
    cm = data.corr(method=method, numeric_only=True)

    # Create a mask for the upper triangle (to avoid duplicate information)
    mask = np.triu(np.ones_like(cm, dtype=bool))

    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=figsize)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(
        cm,
        mask=mask,
        vmin=-1, vmax=1,
        cmap=cmap,
        center=0,
        annot=True,         # Show correlation coefficients
        fmt=".2f",          # Format coefficients to two decimals
        linewidths=1.5,     # Line width between cells
        square=True,        # Make cells square-shaped
        cbar_kws={"shrink": 0.75, "ticks": [-1, -0.5, 0, 0.5, 1]},  # Colorbar customization
        ax=ax
    )

    # Rotate axis labels for better readability
    rotate_xlabels(ax)
    rotate_ylabels(ax)

    # Set the title
    ax.set_title(f"{method.title()} Correlation Matrix Heatmap")
# correlation_heatmap(df)



# -----------------------------------------------------------------------------------------------------------------------
def process_and_plot_employment_status(
    df, 
    grouping_col='title', 
    question_no_col='question_no', 
    question_no_value='Q23', 
    id_col='id'
):
    # Remove rows with missing group
    base_data = df.dropna(subset=[grouping_col])

    # Total distinct respondents per group
    total_counts = base_data.groupby(grouping_col)[id_col].nunique().reset_index(name='n')

    # Respondents who answered Q23
    answered_counts = base_data[base_data[question_no_col] == question_no_value] \
        .groupby(grouping_col)[id_col].nunique().reset_index(name='answered')

    # Merge and fill missing
    merged = pd.merge(total_counts, answered_counts, on=grouping_col, how='left')
    merged['answered'] = merged['answered'].fillna(0)
    merged['not_answered'] = merged['n'] - merged['answered']

    # Long format for answered/not answered
    long_df = pd.melt(
        merged, 
        id_vars=[grouping_col, 'n'], 
        value_vars=['answered', 'not_answered'], 
        var_name='Q23', 
        value_name='count'
    )
    long_df['Q23'] = long_df['Q23'].map({'answered': 'Answered', 'not_answered': 'Not answered'})
    long_df['Q23'] = pd.Categorical(long_df['Q23'], categories=['Answered', 'Not answered'], ordered=True)

    # Employment status mapping
    employment_map = {'Student': 'Student', 'Currently not employed': 'Currently not employed'}
    merged['employment'] = merged[grouping_col].map(employment_map).fillna('Employed')
    merged['employment'] = pd.Categorical(
        merged['employment'], 
        categories=['Employed', 'Student', 'Currently not employed'], 
        ordered=True
    )

    # Summarize and percent
    employment_summary = merged.groupby('employment')['n'].sum().reset_index()
    employment_summary['n_pct'] = employment_summary['n'] / employment_summary['n'].sum()

    # Plot
    plt.figure(figsize=(15, 3))
    sns.set_theme(style='whitegrid')
    barplot = sns.barplot(
        x='n_pct', y='employment', 
        data=employment_summary, 
        palette='Set2'
    )
    for index, row in employment_summary.iterrows():
        barplot.text(
            row.n_pct / 2, index, 
            f'{row.n_pct:.0%}', 
            color='black', ha='center', va='center', 
            fontweight='bold', fontsize=14
        )
    plt.title('Employment status')
    plt.xlabel('Percentage')
    plt.ylabel('Employment')
    plt.xlim(0, 1)
    plt.show()

    return long_df, employment_summary
# long_df, employment_summary = process_and_plot_employment_status(your_dataframe)



# -----------------------------------------------------------------------------------------------------------------------
