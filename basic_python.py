import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import schedule
import time
import logging
import joblib
from flask import Flask, request, jsonify
from fpdf import FPDF
import matplotlib.pyplot as plt
import os
from sklearn.feature_selection import SelectKBest, f_classif
from pandas.api.types import is_numeric_dtype
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import datetime



# ---------------------------------------------------------------------------------------------------
# tell shape, missing value in each colm and dtype of each col
def quick_data_exploration(df):
    """
    Provides a fast and informative overview of a pandas DataFrame.

    Parameters:
        df (pandas.DataFrame): The DataFrame to be explored.

    Returns:
        df (pandas.DataFrame): The same DataFrame for further analysis.

    Functionality:
        1. Prints the dataset's shape (number of rows and columns).
        2. Displays data types of each column.
        3. Shows the count of missing values in each column.
        4. Prints basic statistical summaries for numerical columns.
        5. Displays the first five rows of the dataset for a quick preview.

    Example Usage:
        import pandas as pd
        df = pd.read_csv("data/my_dataset.csv")
        quick_data_exploration(df)
    """
    # Basic information
    print("The Shape of Dataset is:", df.shape)
    print('*-' * 65)
    print("\nData Types:\n", pd.DataFrame(df.dtypes))
    print('*-' * 65)
    print("\nMissing Values:\n", df.isnull().sum())
    print('*-' * 65)
    return df
# quick_data_exploration(df)




# ---------------------------------------------------------------------------------------------------
# lists down numerical and categorical features, imputes numerical feature with median, categorical feature with mode, 
# remove duplicated and standardise the numerical features
def data_preprocessing(df):
    """
    Preprocesses a pandas DataFrame by handling missing values, removing duplicates, and normalizing numeric features.

    Parameters:
        df (pandas.DataFrame): The input DataFrame to preprocess.

    Returns:
        df (pandas.DataFrame): The preprocessed DataFrame.

    Functionality:
        1. Identifies numeric and categorical columns.
        2. Fills missing values in numeric columns with the median.
        3. Fills missing values in categorical columns with the mode.
        4. Removes duplicate rows.
        5. Normalizes numeric columns using StandardScaler.

    Example Usage:
        df_clean = data_preprocessing(df)
    """
    from sklearn.preprocessing import StandardScaler
    # Handle missing values
    numeric_columns = df.select_dtypes(include=[np.number]).columns.to_list()
    categorical_columns = df.select_dtypes(exclude=[np.number]).columns.to_list()
    print(f"Numerical Features are {numeric_columns}")
    print(f"Categorical Features are {categorical_columns}")

    print('*-' * 65)

    # Fill numeric missing values with median
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())

    # Fill categorical missing values with mode
    if len(categorical_columns) > 0:
        df[categorical_columns] = df[categorical_columns].fillna(df[categorical_columns].mode().iloc[0])

    # Remove duplicates
    df = df.drop_duplicates()

    # Normalize numeric features
    scaler = StandardScaler()
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

    return df
# data_preprocessing(df)
# data_preprocessing(df)


# ---------------------------------------------------------------------------------------------------
# draw histogram and box plot for all numerical features
def automated_eda(df):
    """
    Automates univariate exploratory data analysis for numeric columns in a DataFrame.

    Parameters:
        df (pandas.DataFrame): The input DataFrame to analyze.

    Functionality:
        1. Identifies all numeric columns in the DataFrame.
        2. For each numeric column:
            - Plots the distribution (histogram with KDE).
            - Plots the box plot for outlier detection.
        3. Displays the plots for visual inspection.

    Example Usage:
        automated_eda(df)
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        plt.figure(figsize=(10, 4))

        # Distribution plot
        plt.subplot(1, 2, 1)
        sns.histplot(df[col], kde=True)
        plt.title(f'Distribution of {col}')

        # Box plot
        plt.subplot(1, 2, 2)
        sns.boxplot(y=df[col])
        plt.title(f'Box Plot of {col}')

        plt.tight_layout()
        plt.show()
# automated_eda(df) 



# ---------------------------------------------------------------------------------------------------
# from datetime feature this will make indivisal features of all the things like day, time etc and
# for all numerical features this will make a feature interaction using multiplication between 2 numerical feature
def feature_engineering(df):
    """
    Performs feature engineering by extracting components from datetime columns and creating interaction features between numeric columns.

    Parameters:
        df (pandas.DataFrame): The input DataFrame.

    Returns:
        df (pandas.DataFrame): The DataFrame with new engineered features.

    Functionality:
        1. Identifies datetime columns and extracts year, month, day, and day of week as new features.
        2. Creates pairwise interaction features (multiplication) between all numeric columns.

    Example Usage:
        df_fe = feature_engineering(df)
    """
    # Date features
    date_columns = df.select_dtypes(include=['datetime64']).columns

    for col in date_columns:
        df[f'{col}_year'] = df[col].dt.year
        df[f'{col}_month'] = df[col].dt.month
        df[f'{col}_day'] = df[col].dt.day
        df[f'{col}_dayofweek'] = df[col].dt.dayofweek

    # Interaction features
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for i in range(len(numeric_cols)):
        for j in range(i+1, len(numeric_cols)):
            col1, col2 = numeric_cols[i], numeric_cols[j]
            df[f'{col1}_{col2}_interaction'] = df[col1] * df[col2]

    return df
# feature_engineering(df)



# ---------------------------------------------------------------------------------------------------
# A utility function to train and evaluate multiple classification models, returning key metrics for each.
# def train_evaluate_models(X, y):
#     """
#     Trains and evaluates multiple classification models on the provided features and target.

#     Parameters:
#         X (pd.DataFrame or np.ndarray): Feature matrix.
#         y (pd.Series or np.ndarray): Target vector.

#     Returns:
#         results (dict): Dictionary containing accuracy, precision, recall, and F1-score for each model.

#     Functionality:
#         1. Splits the data into training and test sets (80/20 split).
#         2. Initializes Random Forest and Logistic Regression classifiers.
#         3. Trains each model and predicts on the test set.
#         4. Computes accuracy, precision, recall, and F1-score for each model.
#         5. Returns a dictionary with metrics for each model.

#     Example Usage:
#         results = train_evaluate_models(X, y)
#     """
#     # Split the data
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     # Initialize models
#     models = {
#         'Random Forest': RandomForestClassifier(),
#         'Logistic Regression': LogisticRegression()
#     }

#     results = {}

#     # Train and evaluate each model
#     for name, model in models.items():
#         model.fit(X_train, y_train)
#         y_pred = model.predict(X_test)

#         results[name] = {
#             'accuracy': accuracy_score(y_test, y_pred),
#             'precision': precision_score(y_test, y_pred, average='weighted'),
#             'recall': recall_score(y_test, y_pred, average='weighted'),
#             'f1': f1_score(y_test, y_pred, average='weighted')
#         }

#     return results
# # train_evaluate_models(X, y)


# ---------------------------------------------------------------------------------------------------
# no of missing values in indivisual colm, no of unique value in all colms, dtype of all the colms, no of outliers in all the colms
def data_quality_check(df):
    """
    Generates a comprehensive data quality report for the given DataFrame.

    Parameters:
        df (pandas.DataFrame): The DataFrame to check.

    Returns:
        quality_report (dict): A dictionary summarizing data quality metrics:
            - total_rows: Total number of rows in the DataFrame.
            - duplicate_rows: Number of duplicate rows.
            - missing_values: Dictionary of missing value counts per column.
            - unique_values: Dictionary of unique value counts per column.
            - data_types: Dictionary of data types per column.
            - outliers: Dictionary of outlier counts per numeric column (using the IQR method).

    Functionality:
        1. Counts total and duplicate rows.
        2. Reports missing values per column.
        3. Reports unique value counts per column.
        4. Reports data types per column.
        5. Detects outliers in numeric columns using the IQR method.

    Example Usage:
        report = data_quality_check(df)
    """
    quality_report = {
        'total_rows': len(df),
        'duplicate_rows': len(df) - len(df.drop_duplicates()),
        'missing_values': df.isnull().sum().to_dict(),
        'unique_values': {col: df[col].nunique() for col in df.columns},
        'data_types': df.dtypes.to_dict()
    }

    # Check for outliers in numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outliers = {}

    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers[col] = int(((df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)).sum())

    quality_report['outliers'] = outliers

    return quality_report
# data_quality_check(tips)



# ---------------------------------------------------------------------------------------------------
# A utility function to deploy a trained model as a Flask API for real-time predictions.
# def deploy_model(model, model_name='model.pkl'):
#     """
#     Deploys a trained model as a Flask API for real-time predictions.

#     Parameters:
#         model: The trained model object to be deployed.
#         model_name (str): Filename for saving the model (default: 'model.pkl').

#     Returns:
#         app (Flask): A Flask app instance with a /predict endpoint.

#     Functionality:
#         1. Saves the provided model to disk using joblib.
#         2. Creates a Flask app with a /predict endpoint.
#         3. The /predict endpoint accepts POST requests with a JSON payload containing 'features'.
#         4. Loads the model, makes predictions, and returns the result as JSON.
#         5. Handles and returns errors as JSON if prediction fails.

#     Example Usage:
#         app = deploy_model(trained_model)
#         app.run(port=5000)
#     """
#     # Save the model
#     joblib.dump(model, model_name)

#     # Create Flask app
#     app = Flask(__name__)

#     @app.route('/predict', methods=['POST'])
#     def predict():
#         try:
#             # Get data from POST request
#             data = request.json

#             # Load model and make prediction
#             loaded_model = joblib.load(model_name)
#             prediction = loaded_model.predict([data['features']])

#             return jsonify({'prediction': prediction.tolist()})

#         except Exception as e:
#             return jsonify({'error': str(e)})

#     return app
# # deploy_model(model, model_name='iris_model.pkl')



# ---------------------------------------------------------------------------------------------------
# A utility function to generate a PDF report with basic statistics and visualizations from a DataFrame.
# def generate_report(df, title="Data Analysis Report"):
#     """
#     Generates a PDF report containing the dataset's shape and up to three numeric column distributions.

#     Parameters:
#         df (pandas.DataFrame): The input DataFrame.
#         title (str): Title of the report (default: "Data Analysis Report").

#     Functionality:
#         1. Adds a title and dataset shape to the PDF.
#         2. For up to three numeric columns, creates and embeds a histogram plot of their distribution.
#         3. Saves the PDF as 'report.pdf'.

#     Example Usage:
#         generate_report(df, title="My Data Report")
#     """
#     pdf = FPDF()
#     pdf.add_page()

#     # Add title
#     pdf.set_font("Arial", "B", 16)
#     pdf.cell(0, 10, title, ln=True, align='C')

#     # Add basic statistics
#     pdf.set_font("Arial", "", 12)
#     pdf.cell(0, 10, f"Dataset Shape: {df.shape}", ln=True)

#     # Add visualizations for up to 3 numeric columns
#     for col in df.select_dtypes(include=[np.number]).columns[:3]:
#         plt.figure(figsize=(10, 6))
#         df[col].hist()
#         plt.title(f"Distribution of {col}")
#         img_path = f"{col}_dist.png"
#         plt.savefig(img_path)
#         plt.close()
#         pdf.image(img_path, x=10, w=190)
#         os.remove(img_path)  # Clean up the image file after adding to PDF

#     # Save report
#     pdf.output("report.pdf")
# generate_report(df, title="My Data Report")



# ---------------------------------------------------------------------------------------------------
# Selects the top features from a dataset using either statistical tests (ANOVA F-value) or tree-based 
# (Random Forest) feature importances,
def feature_selection(X, y, method='statistical', n_features=10):
    """
    Selects the most important features from the dataset using either statistical or tree-based methods.

    Parameters:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series or np.ndarray): Target vector.
        method (str): Feature selection method ('statistical' or 'tree_based'). Default is 'statistical'.
        n_features (int): Number of top features to select. Default is 10.

    Returns:
        selected_features (list): List of selected feature names.

    Functionality:
        1. If 'statistical', uses SelectKBest with ANOVA F-value to select top features.
        2. If 'tree_based', uses RandomForestClassifier feature importances to select top features.

    Example Usage:
        selected = feature_selection(X, y, method='tree_based', n_features=5)
    """
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.ensemble import RandomForestClassifier
    if method == 'statistical':
        # Statistical feature selection
        selector = SelectKBest(score_func=f_classif, k=n_features)
        selector.fit(X, y)
        selected_features = X.columns[selector.get_support()].tolist()

    elif method == 'tree_based':
        # Tree-based feature selection
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        importance = pd.DataFrame({
            'feature': X.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        selected_features = importance.head(n_features)['feature'].tolist()

    else:
        raise ValueError("method must be 'statistical' or 'tree_based'")

    return selected_features

# feature_selection(X, y, method='tree_based', n_features=2)



# ---------------------------------------------------------------------------------------------------
# A class for building a simple ETL (Extract, Transform, Load) pipeline with error handling and logging.
class ETLPipeline:
    """
    A class for building and managing a simple ETL (Extract, Transform, Load) pipeline.

    Attributes:
        connection_params (dict): Contains connection strings for source and destination databases.

    Methods:
        extract(query): Extracts data from the source database using the provided SQL query.
        transform(data): Applies basic data cleaning transformations (drop NA and duplicates).
        load(data, table_name): Loads the transformed data into the destination database table.

    Example Usage:
        etl = ETLPipeline()
        data = etl.extract("SELECT * FROM my_table")
        data_clean = etl.transform(data)
        etl.load(data_clean, "clean_table")
    """

    def __init__(self):
        self.connection_params = {
            'source': 'source_connection_string',
            'destination': 'destination_connection_string'
        }

    def extract(self, query):
        """
        Extracts data from the source database using the provided SQL query.

        Parameters:
            query (str): SQL query to fetch data.

        Returns:
            data (pd.DataFrame or None): Extracted data as a DataFrame, or None if extraction fails.
        """
        try:
            # Add your extraction logic here
            data = pd.read_sql(query, self.connection_params['source'])
            return data
        except Exception as e:
            logging.error(f"Extraction failed: {str(e)}")
            return None

    def transform(self, data):
        """
        Applies basic data cleaning transformations: drops missing values and duplicates.

        Parameters:
            data (pd.DataFrame): Input data to transform.

        Returns:
            transformed_data (pd.DataFrame or None): Transformed data, or None if transformation fails.
        """
        try:
            transformed_data = data.copy()
            transformed_data = transformed_data.dropna()
            transformed_data = transformed_data.drop_duplicates()
            return transformed_data
        except Exception as e:
            logging.error(f"Transformation failed: {str(e)}")
            return None

    def load(self, data, table_name):
        """
        Loads the transformed data into the destination database table.

        Parameters:
            data (pd.DataFrame): Data to load.
            table_name (str): Name of the destination table.

        Returns:
            success (bool): True if loading is successful, False otherwise.
        """
        try:
            # Add your loading logic here
            data.to_sql(table_name, self.connection_params['destination'])
            return True
        except Exception as e:
            logging.error(f"Loading failed: {str(e)}")
            return False
"""
# Usage
etl = DemoETLPipeline()
data = etl.extract("SELECT * FROM my_table")
print("Extracted Data:\n", data)

data_clean = etl.transform(data)
print("\nTransformed Data:\n", data_clean)

success = etl.load(data_clean, "clean_table")
print("\nLoad Success:", success)

# Check loaded data in destination
loaded_data = pd.read_sql("SELECT * FROM clean_table", dest_conn)
print("\nLoaded Data in Destination:\n", loaded_data)
"""


# ---------------------------------------------------------------------------------------------------
# A utility function to validate DataFrame columns against type, range, and pattern rules.
# def validate_data(df, rules):
#     """
#     Validates DataFrame columns against a set of rules for data type, value range, and regex pattern.

#     Parameters:
#         df (pd.DataFrame): The DataFrame to validate.
#         rules (dict): Dictionary of validation rules. 
#             Example:
#                 {
#                     'age': {'type': 'numeric', 'range': (0, 120)},
#                     'email': {'pattern': r'^[\w\.-]+@[\w\.-]+\.\w+$'}
#                 }

#     Returns:
#         validation_results (dict): Dictionary with validation results for each rule and column.

#     Functionality:
#         1. Checks if each specified column exists in the DataFrame.
#         2. Validates column data type if 'type' is specified.
#         3. Validates value range if 'range' is specified.
#         4. Validates string pattern if 'pattern' is specified.

#     Example Usage:
#         rules = {
#             'age': {'type': 'numeric', 'range': (0, 120)},
#             'email': {'pattern': r'^[\w\.-]+@[\w\.-]+\.\w+$'}
#         }
#         results = validate_data(df, rules)
#     """
#     validation_results = {}

#     for column, rule in rules.items():
#         if column not in df.columns:
#             validation_results[column] = "Column not found"
#             continue

#         # Check data type
#         if rule.get('type') == 'numeric':
#             validation_results[f"{column}_type"] = is_numeric_dtype(df[column])

#         # Check range
#         if 'range' in rule:
#             min_val, max_val = rule['range']
#             validation_results[f"{column}_range"] = bool(df[column].between(min_val, max_val).all())

#         # Check regex pattern
#         if 'pattern' in rule:
#             pattern = rule['pattern']
#             validation_results[f"{column}_pattern"] = bool(df[column].astype(str).str.match(pattern).all())

#     return validation_results



# ---------------------------------------------------------------------------------------------------
# A utility function to create a multi-plot interactive dashboard using Plotly.
def create_dashboard(df):
    """
    Creates an interactive dashboard with four subplots: distribution histogram, scatter plot, box plot, and time series plot.

    Parameters:
        df (pandas.DataFrame): Input DataFrame with numeric columns and a datetime or index for time series.

    Functionality:
        1. Creates a 2x2 subplot layout with titles.
        2. Plots a histogram of the first numeric column.
        3. Plots a scatter plot of the first two numeric columns.
        4. Plots a box plot of the first numeric column.
        5. Plots a time series line plot of the first numeric column against the DataFrame index.
        6. Displays the interactive dashboard.

    Example Usage:
        create_dashboard(df)
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) < 2:
        raise ValueError("DataFrame must have at least two numeric columns for this dashboard.")

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Distribution Plot', 'Scatter Plot', 'Box Plot', 'Time Series')
    )

    # Distribution plot
    fig.add_trace(
        go.Histogram(x=df[numeric_cols[0]], name='Distribution'),
        row=1, col=1
    )

    # Scatter plot
    fig.add_trace(
        go.Scatter(x=df[numeric_cols[0]], y=df[numeric_cols[1]], mode='markers', name='Scatter'),
        row=1, col=2
    )

    # Box plot
    fig.add_trace(
        go.Box(y=df[numeric_cols[0]], name='Box Plot'),
        row=2, col=1
    )

    # Time series plot
    fig.add_trace(
        go.Scatter(x=df.index, y=df[numeric_cols[0]], mode='lines', name='Time Series'),
        row=2, col=2
    )

    fig.update_layout(height=800, showlegend=False)
    fig.show()
# create_dashboard(df)



# ---------------------------------------------------------------------------------------------------
# A utility function to monitor model performance, alert on threshold breach, and log metrics to a file.
def monitor_model_performance(model, X_test, y_test, threshold=0.8):
    """
    Monitors the performance of a trained model on test data, alerts if accuracy drops below a threshold,
    and logs performance metrics with a timestamp.

    Parameters:
        model: Trained model with a .predict() method.
        X_test: Test features.
        y_test: Test labels.
        threshold (float): Accuracy threshold for alerting (default: 0.8).

    Returns:
        metrics (dict): Dictionary containing timestamp, accuracy, precision, recall, and F1-score.

    Functionality:
        1. Computes accuracy, precision, recall, and F1-score.
        2. Prints an alert if accuracy is below the specified threshold.
        3. Appends metrics with a timestamp to 'model_performance_log.json'.

    Example Usage:
        metrics = monitor_model_performance(model, X_test, y_test, threshold=0.85)
    """
    # Get predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    metrics = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1': f1_score(y_test, y_pred, average='weighted')
    }

    # Alert if performance drops below threshold
    if metrics['accuracy'] < threshold:
        print(f"Alert: Model performance below threshold! Current accuracy: {metrics['accuracy']}")

    # Save metrics to file
    with open('model_performance_log.json', 'a') as f:
        json.dump(metrics, f)
        f.write('\n')

    return metrics
# monitor_model_performance(model, X_test, y_test, threshold=0.85)