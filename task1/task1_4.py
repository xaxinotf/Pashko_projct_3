import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
from scipy.linalg import eig
import pandas as pd

# Визначення кількості станів
num_states = 7
states = list(range(num_states))

# Створення матриці переходів P з випадковими ймовірностями
np.random.seed(42)  # Для відтворюваності результатів
P = np.random.rand(num_states, num_states)

# Нормалізація рядків матриці P, щоб сума дорівнювала 1
P = P / P.sum(axis=1, keepdims=True)

print("Матриця переходів P:")
for i, row in enumerate(P):
    print(f"Стан {i}: {row}")

# Вектор початкових ймовірностей (наприклад, початковий стан - стан 0)
pi = np.zeros(num_states)
pi[0] = 1.0  # Початковий стан - стан 0

print("\nВектор початкових станів pi:")
print(pi)


def is_regular(P):
    """
    Перевіряє, чи є матриця переходів P регулярною.

    Parameters:
        P (numpy.ndarray): Матриця переходів.

    Returns:
        bool: True, якщо P регулярна, інакше False.
    """
    # Піднесення матриці P до степеня k, поки всі елементи не стануть додатніми
    k = 1
    while k <= 100:  # Максимум 100 ітерацій
        P_k = np.linalg.matrix_power(P, k)
        if np.all(P_k > 0):
            print(f"\nМатриця переходів P є регулярною при k = {k}.")
            return True
        k += 1
    print("\nМатриця переходів P НЕ є регулярною.")
    return False


# Перевірка регулярності
regular = is_regular(P)


def compute_stationary_distribution(P):
    """
    Обчислює стаціонарну розподіл для матриці переходів P.

    Parameters:
        P (numpy.ndarray): Матриця переходів.

    Returns:
        pi_star (numpy.ndarray): Стаціонарна розподіл.
    """
    # Обчислення власних значень та власних векторів
    eigenvalues, eigenvectors = eig(P.T)

    # Знаходимо індекс власного значення, яке дорівнює 1
    index = np.argmin(np.abs(eigenvalues - 1))

    # Отримуємо відповідний власний вектор
    pi_star = np.real(eigenvectors[:, index])

    # Нормалізація вектора
    pi_star = pi_star / pi_star.sum()

    return pi_star


# Обчислення стаціонарної розподілу
pi_star = compute_stationary_distribution(P)

print("\nТеоретична стаціонарна розподіл pi*:")
print(pi_star)
print(f"Сума елементів pi*: {pi_star.sum():.4f}")


def simulate_markov_chain(P, initial_state, num_steps):
    """
    Симулює ланцюг Маркова на задану кількість кроків.

    Parameters:
        P (numpy.ndarray): Матриця переходів.
        initial_state (int): Початковий стан.
        num_steps (int): Кількість кроків для симуляції.

    Returns:
        path (list): Послідовність станів у симуляції.
    """
    current_state = initial_state
    path = [current_state]

    for _ in range(num_steps):
        next_state = np.random.choice(states, p=P[current_state])
        path.append(next_state)
        current_state = next_state

    return path


# Параметри симуляції
num_simulations = 10000  # Кількість симуляцій
num_steps = 50  # Кількість кроків у кожній симуляції

# Збір статистичних даних
state_counts = np.zeros(num_states)
state_visit_counts = np.zeros(num_states)
paths = []

for _ in range(num_simulations):
    path = simulate_markov_chain(P, initial_state=0, num_steps=num_steps)
    paths.append(path)
    # Підрахунок кількості відвідувань кожного стану
    unique, counts = np.unique(path, return_counts=True)
    state_visit_counts[unique] += counts
    # Підрахунок кінцевих станів
    state_counts[path[-1]] += 1

# Обчислення експериментальної стаціонарної розподілу
experimental_pi_star = state_visit_counts / (num_simulations * num_steps)

print("\nЕкспериментальна стаціонарна розподіл pi*:")
print(experimental_pi_star)
print(f"Сума елементів експериментальної pi*: {experimental_pi_star.sum():.4f}")

# Обчислення експериментальних ймовірностей кінцевих станів
experimental_final_probs = state_counts / num_simulations

print("\nЕкспериментальні ймовірності кінцевих станів:")
for i, prob in enumerate(experimental_final_probs):
    print(f"Стан {i}: {prob:.4f}")

# Обчислення експериментальної матриці переходів
transition_counts = np.zeros((num_states, num_states), dtype=int)

for path in paths:
    for i in range(len(path) - 1):
        transition_counts[path[i]][path[i + 1]] += 1

# Обчислення експериментальної матриці переходів
experimental_P = np.zeros((num_states, num_states))

for i in range(num_states):
    total = transition_counts[i].sum()
    if total > 0:
        experimental_P[i] = transition_counts[i] / total
    else:
        experimental_P[i][i] = 1.0  # Для випадків без переходів

print("\nЕкспериментальна матриця переходів P:")
for i, row in enumerate(experimental_P):
    print(f"Стан {i}: {row}")

# Обчислення часу перебування в кожному стані
# Середній час перебування в кожному стані
mean_time_in_states = state_visit_counts / num_simulations
print("\nСередній час перебування в кожному стані:")
for i, mean_time in enumerate(mean_time_in_states):
    print(f"Стан {i}: {mean_time:.4f} кроків")

# Візуалізація Матриці Переходів
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.heatmap(P, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=[f"S{state}" for state in states],
            yticklabels=[f"S{state}" for state in states])
plt.title("Теоретична Матриця Переходів")
plt.xlabel("Наступний стан")
plt.ylabel("Поточний стан")

plt.subplot(1, 2, 2)
sns.heatmap(experimental_P, annot=True, fmt=".2f", cmap="Greens",
            xticklabels=[f"S{state}" for state in states],
            yticklabels=[f"S{state}" for state in states])
plt.title("Експериментальна Матриця Переходів")
plt.xlabel("Наступний стан")
plt.ylabel("Поточний стан")

plt.tight_layout()
plt.show()

# Візуалізація Стаціонарної Розподілу
stationary_df = {
    'Стан': [f"S{i}" for i in states],
    'Теоретична pi*': pi_star,
    'Експериментальна pi*': experimental_pi_star
}

stationary_df = pd.DataFrame(stationary_df)

# Візуалізація стаціонарної розподілу
stationary_df_melted = stationary_df.melt(id_vars='Стан', var_name='Тип', value_name='Ймовірність')

plt.figure(figsize=(10, 6))
sns.barplot(data=stationary_df_melted, x='Стан', y='Ймовірність', hue='Тип')
plt.title("Теоретична vs Експериментальна Стаціонарна Розподіл")
plt.ylabel("Ймовірність")
plt.show()

# Візуалізація Ланцюга Маркова за Допомогою NetworkX
G = nx.DiGraph()

# Додавання вузлів
for state in states:
    G.add_node(state, label=f"S{state}")

# Додавання ребер з ймовірностями
for i in states:
    for j in states:
        if P[i][j] > 0:
            G.add_edge(i, j, weight=P[i][j])

# Отримання міток вузлів
labels_graph = {node: G.nodes[node]['label'] for node in G.nodes}

# Отримання міток ребер з ймовірностями
edge_labels = {(i, j): f"{P[i][j]:.2f}" for i, j in G.edges()}

# Отримання позицій вузлів
pos = nx.spring_layout(G, seed=42)  # Фіксована позиція для стабільності

# Візуалізація графа
plt.figure(figsize=(12, 8))
nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=800)
nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=20, edge_color='gray')
nx.draw_networkx_labels(G, pos, labels_graph, font_size=12)
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)

plt.title("Граф Регулярного Ланцюга Маркова з 7 Станами")
plt.axis('off')
plt.show()

# Візуалізація Стаціонарної Розподілу через Діаграму Пирога
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Теоретична pi*
axes[0].pie(pi_star, labels=[f"S{i}" for i in states], autopct='%1.1f%%', startangle=140,
            colors=plt.cm.Blues(np.linspace(0, 1, num_states)))
axes[0].set_title("Теоретична Стаціонарна Розподіл")

# Експериментальна pi*
axes[1].pie(experimental_pi_star, labels=[f"S{i}" for i in states], autopct='%1.1f%%', startangle=140,
            colors=plt.cm.Greens(np.linspace(0, 1, num_states)))
axes[1].set_title("Експериментальна Стаціонарна Розподіл")

plt.show()

# Візуалізація Конвергенції Розподілу Ймовірностей до Стаціонарної pi*
# Візьмемо одну симуляцію для демонстрації
sample_path = simulate_markov_chain(P, initial_state=0, num_steps=num_steps)
sample_distribution = np.zeros((num_steps + 1, num_states))
current_distribution = pi.copy()
sample_distribution[0] = current_distribution

for step in range(1, num_steps + 1):
    current_distribution = np.dot(current_distribution, P)
    sample_distribution[step] = current_distribution

plt.figure(figsize=(12, 8))
for state in states:
    plt.plot(range(num_steps + 1), sample_distribution[:, state], label=f"S{state}")

plt.hlines(pi_star, 0, num_steps, colors='k', linestyles='dashed', label='Стаціонарна pi*')
plt.title("Конвергенція Розподілу Ймовірностей до Стаціонарної pi*")
plt.xlabel("Кроки")
plt.ylabel("Ймовірність")
plt.legend()
plt.grid(True)
plt.show()
