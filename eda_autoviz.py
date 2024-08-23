import pandas as pd
from autoviz.AutoViz_Class import AutoViz_Class
import os

# Get the absolute path to the project directory
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Construct the absolute path to the CSV file
csv_path = os.path.join(project_dir, 'output', 'image_data.csv')

# Load the DataFrame
df = pd.read_csv(csv_path)

# Create an instance of AutoViz_Class
AV = AutoViz_Class()

# Use AutoViz to visualize the dataset
df_auto_viz = AV.AutoViz('', dfte=df, depVar='', verbose=1, chart_format='png', max_rows_analyzed=150000, max_cols_analyzed=30)
