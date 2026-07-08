import argparse
import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", default="titanic_preprocessed.csv")
args = parser.parse_args()

df = pd.read_csv(args.data_path)
X = df.drop('Survived', axis=1)
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("CI_Titanic")

with mlflow.start_run():
    mlflow.sklearn.autolog()
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
    mlflow.sklearn.log_model(model, "model")
    joblib.dump(model, "model.pkl")
