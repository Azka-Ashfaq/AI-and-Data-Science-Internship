import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.metrics import*
import warnings 
warnings.filterwarnings('ignore')

adult = fetch_openml('adult', as_frame=True)
df= adult.frame
df=df.replace('?',np.nan)
df['class'] = (df['class'] == '>50k')

['class'] = (df['class'] == '<50k')


print("Data loaded successfully.")
print("Dataset shape: {df.shape}")

positive_rate = df['class'].mean()
print("postive rate is positive_rate",positive_rate)

print("\nClass distribution:")
print(df['class'].value_counts())

