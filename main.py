import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from compare_models import compare_models
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('dataset_footyf_v2.csv', on_bad_lines='skip')

@ignore_warnings(category=ConvergenceWarning)
def generate_model(compare=True):
    X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :-1], df.iloc[:, -1], test_size=0.2)
    X_train = X_train.dropna()
    y_train = y_train.dropna()
    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, y_train)
    if compare:
        loaded_model = joblib.load('FootyForecast_NEW.joblib')
        if compare_models(model, loaded_model, X_test, y_test):
            joblib.dump(model, 'FootyForecast_NEW.joblib')
            
while True:
    generate_model()