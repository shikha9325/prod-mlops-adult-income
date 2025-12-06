import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set MLflow experiment
mlflow.set_experiment("Iris_Logistic_Regression")

# Start MLflow run
with mlflow.start_run():
    # Parameters
    lr_model = LogisticRegression(max_iter=200)
    mlflow.log_param("max_iter", 200)
    
    # Train model
    lr_model.fit(X_train, y_train)
    
    # Predict
    y_pred = lr_model.predict(X_test)
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    mlflow.log_metric("accuracy", acc)
    
    # Log model
    mlflow.sklearn.log_model(lr_model, "logistic_regression_model")
    print(f"Logged model with accuracy: {acc}")
