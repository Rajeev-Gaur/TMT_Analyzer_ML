import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from config import DATASET_PATH, TEST_SIZE, RANDOM_STATE
from train_model import train_and_evaluate_model
from preprocessing_data import load_and_preprocess_data
import os

def main():
    # Load and preprocess data
    X, y = load_and_preprocess_data()
    
    # Train the model and evaluate
    accuracy, conf_matrix, class_report, y_test, y_pred = train_and_evaluate_model(X, y, TEST_SIZE, RANDOM_STATE)
    
    print(f"Accuracy: {accuracy}")
    print("Confusion Matrix:\n", conf_matrix)
    print("Classification Report:\n", class_report)

    # Create output directory if it doesn't exist
    os.makedirs('output', exist_ok=True)

    # Save classification report to a text file
    with open('output/classification_report.txt', 'w') as f:
        f.write(class_report)

    # Plot and save confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.savefig('output/confusion_matrix.png')  # Save the figure
    plt.close()  # Close the plot

if __name__ == '__main__':
    main()
