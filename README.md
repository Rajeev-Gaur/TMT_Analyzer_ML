# TMT Analyzer ML

## Overview

The **TMT Analyzer ML** project aims to analyze Treadmill Test (TMT) results for heart patients using machine learning techniques. The project includes data preprocessing, exploratory data analysis (EDA), model training, evaluation, and saving the model for future predictions.

## Table of Contents

- [Installation]
- [Usage]
- [Data]
- [Exploratory Data Analysis]
- [Model Training]
- [Libraries used]
- [License]

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Rajeev-Gaur/TMT_Analyzer_ML.git
   cd TMT_Analyzer_ML
   '
2. python -m venv venv

3. Activate the virtual environment:
   venv\Scripts\activate

4. Install required libraries:
   pip install -r requirements.txt

Usage
1.Place your TMT dataset in the data directory and ensure it is in CSV format.

2.Modify the target_variable in the tmt_analyzer.py script to match your dataset's target column.

3.Run the analysis script:
python tmt_analyzer.py

4.The script will perform data preprocessing, EDA, train a machine learning model, and output evaluation metrics.

Data
This project uses a dataset containing Treadmill Test results. 

Dataset Headers and Their Relevance
ID: Unique identifier for each patient. Useful for tracking and managing records, but not typically used in analysis.

Age: Important demographic factor that can influence cardiovascular health and exercise capacity.

Sex: Gender can affect heart disease risk and physiological responses during exercise.

cp (chest pain type): Indicates the type of chest pain experienced (usually classified from 0 to 3). This is crucial for assessing heart risk during exercise.

trestbps (resting blood pressure): Baseline blood pressure measurement, important for evaluating cardiovascular response during TMT.

chol (cholesterol): Total cholesterol levels can indicate risk for cardiovascular disease.

fbs (fasting blood sugar): Indicates if blood sugar is above a certain level (1 = true, 0 = false), relevant for assessing metabolic health.

restecg (resting electrocardiographic results): Important for understanding the electrical activity of the heart at rest.

thalach (maximum heart rate): Maximum heart rate achieved during exercise; a key measure in evaluating cardiac function.

exang (exercise induced angina): Indicates whether exercise causes angina (1 = yes, 0 = no), very relevant for TMT analysis.


Exploratory Data Analysis
The EDA section of the script includes:

Histograms to visualize distributions (e.g., heart rate).
A correlation matrix to understand relationships between features.


Model Training
The project uses a Random Forest Classifier to predict the outcome of Treadmill Tests. The model is trained and evaluated with metrics such as accuracy, precision, recall, and F1-score. The confusion matrix is visualized to assess prediction performance.

Libraries Used:
pandas: For data manipulation and analysis.
numpy: For numerical operations and handling arrays.
scikit-learn: For implementing machine learning algorithms and model evaluation.
matplotlib: For creating static, animated, and interactive visualizations in Python.
seaborn: For statistical data visualization based on Matplotlib, offering a high-level interface for drawing attractive graphs.
joblib: For saving and loading Python objects, particularly models.


License
This project is licensed under the MIT License. See the LICENSE file for details.

