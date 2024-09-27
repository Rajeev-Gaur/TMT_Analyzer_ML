import pandas as pd
from config import DATASET_PATH

def load_and_preprocess_data():
    # Load the dataset
    data = pd.read_csv(DATASET_PATH)

    # Strip whitespace from column names
    data.columns = data.columns.str.strip()

    # Print the columns for debugging
    print("DataFrame columns:", data.columns.tolist())

    # Check for missing values and drop them
    data.dropna(inplace=True)

    # Convert categorical variables to numeric
    if 'Sex' in data.columns:
        data['Sex'] = data['Sex'].map({1: 1, 0: 0})  # Assuming 1 = Male, 0 = Female
    if 'exang' in data.columns:
        data['exang'] = data['exang'].map({1: 1, 0: 0})  # Adjust mapping as per your dataset

    # One-hot encode categorical features
    data = pd.get_dummies(data, drop_first=True)

    # Define features and target variable
    columns_to_drop = ['ID', 'num', 'Place']
    existing_columns = [col for col in columns_to_drop if col in data.columns]
    
    # Drop columns that exist
    X = data.drop(columns=existing_columns)
    y = data['num']  # Assuming 'num' is your target variable

    return X, y

