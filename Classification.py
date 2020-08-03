import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# pandas processor
data = pd.read_csv("iris_flowers.csv")
pd.set_option('display.max_rows', None)
values = data.drop(columns="class").to_numpy()
species = data["class"].to_frame().to_numpy()

# randomly splits into four different train/test objects
X_train, X_test, y_train, y_test = train_test_split(values, species, test_size=0.3, random_state=1)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train.ravel())

print("\nTESTING.....\n")
print("Parameters \t\t\t\tspecies         | prediction\n")

for i in range(0, 45):
    X_new = np.array([X_test[i]])
    prediction = knn.predict(X_new)
    if str(y_test[i][0]) != str(prediction[0]):
        print(X_test[i],
              "      %-15s" % y_test[i][0] + " | %-10s" % prediction[0],
              "INCORRECT")
    else:
        print(X_test[i],
              "      %-15s" % y_test[i][0] + " | %-10s" % prediction[0])

score = knn.score(X_test, y_test) * 100
print("\nPrediction accuracy: %.2f%%" % score)
