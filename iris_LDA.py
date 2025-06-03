# Importação das bibliotecas necessárias 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis # Importa a classe LDA da biblioteca sciktlearn
from sklearn.datasets import load_iris # Importa o dataset iris da biblioteca 
from sklearn.model_selection import train_test_split # Importa a função que irá dividir o dataset em subconjuntos de treino e teste 

# No dataset iris contém informações sobre três tipos de flores (setosa, versicolor e virginica) com 4 características para cada amostra 
iris = load_iris() # Faz a instanciação do dataset iris
X = iris.data # É armazenado na variável a característica das amostras (largura e comprimento das sépalas e pétalas )
y = iris.target # Corresponde ao rótulo indicando a qual espécie cada amostra pertence 

# Divide o conjunto em dois subconjuntos 
# X_train e y_train = contem as amostras que serão usadas para treinar o modelo
# X_test e y_test = contem as amostras que serão usadas para testar o modelo, comprovando o seu desempenho
# 30% das amostras serão reservadas para o teste, enquanto 70% serão usadas para treinamento 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

lda = LinearDiscriminantAnalysis() # Cria o modelo LDA que será utilizado para a classificação das amostras 
lda.fit(X_train, y_train) # Treina o modelo com os dados especificados para treino

accuracy = lda.score(X_test, y_test) # Calcula a acurácia do modelo, utilizando os dados de teste definidos anteriormente 
print(f"Acurácia do LDA: {accuracy:.2f}")

