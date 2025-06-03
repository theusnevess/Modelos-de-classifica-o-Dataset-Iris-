from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

rda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
rda.fit(X_train, y_train)

accuracy = rda.score(X_test, y_test)
print(f"Acurácia do RDA: {accuracy:.2f}")