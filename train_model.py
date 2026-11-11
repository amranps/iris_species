from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

iris = load_iris()

print(iris.data)
print(iris.target)
print(iris.feature_names)
print(iris.target_names)
x = iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42,stratify=y)

model = RandomForestClassifier(n_estimators=100, random_state=42    )
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
joblib.dump(model, "iris_model.pkl")

model = joblib.load("iris_model.pkl")
prediction = model.predict([[5.1, 3.5, 1.4, 0.2]])
flower_name = iris.target_names[prediction][0]
print("Predicted flower name:", flower_name)