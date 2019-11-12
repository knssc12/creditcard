import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

data = pd.read_csv("creditcard_small.csv")
x = data[["V1","V2","V3","V4"]]
y = data["Class"]

svm_clf = Pipeline([
    ("Scaler", StandardScaler()),
    ("linear_svc", LinearSVC(C=1, loss="hinge")),
    ])

svm_clf.fit(x,y)

result = svm_clf.predict([[-2.3122265423262998,1.9519920106415802,-1.6098507322976898,3.9979055875468]])
print(result)
#print(pd.Series.to_numpy(y).nonzero())

