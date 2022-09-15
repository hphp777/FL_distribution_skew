# import tensorflow as tf
import torch
import numpy as np
from fedalign import client, server
import torch.nn.functional as F
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import precision_recall_curve, roc_curve, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.4, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

print(y_val)

rand_clf = RandomForestClassifier(n_estimators=200, random_state=42)
rand_clf.fit(X_train, y_train)

print(rand_clf.predict_proba(X_val))
print(rand_clf.predict_proba(X_val)[:,1])
