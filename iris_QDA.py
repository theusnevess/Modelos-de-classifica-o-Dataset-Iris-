from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis # Importando a classe LDA da biblioteca sklearn
from sklearn.datasets import load_iris # # Importa o dataset iris da biblioteca 
from sklearn.model_selection import train_test_split # Importa a função que irá dividir o dataset em subconjuntos de treino e teste 

iris = load_iris()
X = iris.data
y = iris.target 

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.3, random_state=42)

qda = QuadraticDiscriminantAnalysis()
qda.fit(X_train, y_train)

accuracy = qda.score(X_test, y_test)
print(f"Acurácia do QDA: {accuracy:.2f}")

