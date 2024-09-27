from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import numpy as np

def train_and_evaluate_model(X, y, test_size, random_state):
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Initialize and train the model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
        # After calculating the metrics
    print(f"Returning values: {accuracy}, {conf_matrix}, {class_report}, {y_test}, {y_pred}")


    # Save the model
    joblib.dump(model, 'tmt_analyzer_model.pkl')

    return accuracy, conf_matrix, class_report, y_test, y_pred  

