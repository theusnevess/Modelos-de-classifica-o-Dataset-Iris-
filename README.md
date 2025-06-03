# 🌸 Classificação do Dataset Iris com LDA, QDA e RDA

Este projeto tem como objetivo aplicar e comparar os métodos de classificação **LDA (Linear Discriminant Analysis)**, **QDA (Quadratic Discriminant Analysis)** e **RDA (Regularized Discriminant Analysis)** no clássico dataset **Iris** utilizando as bibliotecas **scikit-learn**, **pandas**, **seaborn** e **matplotlib**.

## 🛠️ Tecnologias Utilizadas

- [Python 3.x](https://www.python.org/)
- [scikit-learn](https://scikit-learn.org/)
- [pandas](https://pandas.pydata.org/)
- [seaborn](https://seaborn.pydata.org/)
- [matplotlib](https://matplotlib.org/)

## 📊 Sobre os Métodos de Classificação

- **LDA (Linear Discriminant Analysis)**: assume que todas as classes possuem a mesma matriz de covariância.
- **QDA (Quadratic Discriminant Analysis)**: permite que cada classe tenha sua própria matriz de covariância.
- **RDA (Regularized Discriminant Analysis)**: interpolação entre LDA e QDA através de um parâmetro de regularização.

**Obs:** O RDA não está implementado nativamente no scikit-learn, sendo necessário implementá-lo manualmente ou adaptar bibliotecas auxiliares.

## 📝 Passos Realizados

1. **Carregamento do dataset Iris** utilizando `load_iris` do `scikit-learn`.
2. **Análise exploratória de dados (EDA)** com `pandas` e `seaborn` para visualizar correlações e distribuições.
3. **Pré-processamento** dos dados.
4. **Implementação dos classificadores**:
   - LDA via `LinearDiscriminantAnalysis`.
   - QDA via `QuadraticDiscriminantAnalysis`.
   - RDA implementado manualmente com interpolação entre LDA e QDA.
5. **Avaliação de desempenho** utilizando métricas como acurácia, matriz de confusão e visualização das fronteiras de decisão.
6. **Visualização dos resultados** com `matplotlib` e `seaborn`.

## 📈 Exemplos de Visualizações

- Matriz de correlação das variáveis.
- Gráficos de dispersão com as fronteiras de decisão para cada modelo.
- Matriz de confusão para avaliação de desempenho.


