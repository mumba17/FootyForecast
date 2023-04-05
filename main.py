import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from compare_models import compare_models
from load_dataset import load_dataset
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

df = load_dataset('england-premier-league-matches-2018-to-2019-stats.csv')

@ignore_warnings(category=ConvergenceWarning)
def generate_model():
    X_train, X_test, y_train, y_test = train_test_split(df.drop('label', axis=1), df['label'], test_size=0.2)
    model = LogisticRegression(max_iter=2000)
    loaded_model = joblib.load('FootyForecast.joblib')
    cross_val_score(model, X_train, y_train, cv=10)
    model.fit(X_train, y_train)
    if compare_models(model, loaded_model, X_test, y_test):
        joblib.dump(model, 'FootyForecast.joblib')
        
for _ in range(500):
    generate_model()