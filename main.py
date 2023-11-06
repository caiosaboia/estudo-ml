import pandas as pd
from sklearn.datasets import load_iris
import numpy as np

iris = load_iris()
data = iris.data
columns = iris.feature_names

np.random.seed(42)
mask = np.random.choice([True, False], size=data.shape, p=[0.9,0.1])
data_with_missing = np.where(mask, data, np.nan)

df = pd.DataFrame(data_with_missing, columns= columns)
print("")
print("Valores ausentes antes da remoção:")
print(df.isnull().sum())