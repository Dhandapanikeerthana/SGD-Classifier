# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required Python libraries and load the Iris dataset.

2. Split the dataset into input features and output labels.

3. Divide the data into training and testing sets.

4. Create and train the SGD Classifier using the training data.

5. Predict the Iris flower species and evaluate the model accuracy.

## Program:
```
Developed by: KEERTHANA D 
RegisterNumber: 212224040155 

```


```


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=iris.target_names))

```

## Output:
<img width="789" height="377" alt="image" src="https://github.com/user-attachments/assets/f5941ab3-e67e-44a1-8a26-cb583d21ba7a" />


## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
