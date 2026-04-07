# Classificação de Flores Iris com Machine Learning
# Caio Oliveira - Março 2026
# Meu primeiro projeto usando scikit-learn pra treinar um modelo de classificação

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# carregando o dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["especie"] = iris.target

print(f"Total de amostras: {df.shape[0]}")
print(f"Features: {df.shape[1] - 1}")
print(f"Espécies: {list(iris.target_names)}")
print()
print(df.head())
print()
print(df["especie"].value_counts())

# plotando pra ver se dá pra separar as espécies visualmente
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

for especie in [0, 1, 2]:
    subset = df[df["especie"] == especie]
    ax1.scatter(
        subset["petal length (cm)"],
        subset["petal width (cm)"],
        label=iris.target_names[especie],
        alpha=0.7,
    )

ax1.set_xlabel("Comprimento da pétala (cm)")
ax1.set_ylabel("Largura da pétala (cm)")
ax1.set_title("Pétalas por espécie")
ax1.legend()

ax2.bar(iris.target_names, df["especie"].value_counts().sort_index(), color=["#2a9d8f", "#e76f51", "#457b9d"])
ax2.set_ylabel("Quantidade")
ax2.set_title("Amostras por espécie")

plt.tight_layout()
plt.savefig("exploracao.png", dpi=150)
plt.close()
print("Gráfico salvo: exploracao.png")

# separando treino e teste
X = df[iris.feature_names]
y = df["especie"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTreino: {X_train.shape[0]} amostras")
print(f"Teste: {X_test.shape[0]} amostras")

# treinando com decision tree
modelo = DecisionTreeClassifier(random_state=42)
modelo.fit(X_train, y_train)

y_pred = modelo.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\nAcurácia: {acc:.1%}")

# matriz de confusão pra ver onde errou
fig, ax = plt.subplots(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=iris.target_names)
disp.plot(ax=ax, cmap="Blues")
ax.set_title(f"Matriz de Confusão (Acurácia: {acc:.1%})")
plt.tight_layout()
plt.savefig("matriz_confusao.png", dpi=150)
plt.close()
print("Gráfico salvo: matriz_confusao.png")

# vendo quais features mais importaram pro modelo
importancias = pd.DataFrame({
    "feature": iris.feature_names,
    "importancia": modelo.feature_importances_,
}).sort_values("importancia", ascending=True)

fig, ax = plt.subplots(figsize=(8, 4))
ax.barh(importancias["feature"], importancias["importancia"], color="#2a9d8f")
ax.set_xlabel("Importância")
ax.set_title("Quais medidas mais influenciam a classificação?")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.close()
print("Gráfico salvo: feature_importance.png")

print(f"\nResumo: Decision Tree com {acc:.1%} de acurácia")
print(f"Feature mais importante: {importancias.iloc[-1]['feature']}")