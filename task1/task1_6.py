import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import pandas as pd

# Визначення кількості станів
num_states = 7
states = list(range(num_states))

# Створення матриці переходів P з випадковими ймовірностями
np.random.seed(42)  # Для відтворюваності результатів
P = np.random.rand(num_states, num_states)

# Нормалізація рядків матриці P, щоб сума дорівнювала 1
P = P / P.sum(axis=1, keepdims=True)

# Вивід матриці переходів
print("Матриця переходів P:")
for i, row in enumerate(P):
    print(f"Стан {i}: {row}")

# Вектор початкових ймовірностей (початковий стан – стан 0)
pi = np.zeros(num_states)
pi[0] = 1.0  # Початковий стан - стан 0

print("\nВектор початкових станів pi:")
print(pi)


def is_regular(P, max_power=100):
    """
    Перевіряє, чи є матриця переходів P регулярною.

    Parameters:
        P (numpy.ndarray): Матриця переходів.
        max_power (int): Максимальна кількість піднесень для перевірки.

    Returns:
        bool: True, якщо P регулярна, інакше False.
    """
    for k in range(1, max_power + 1):
        P_k = np.linalg.matrix_power(P, k)
        if np.all(P_k > 0):
            print(f"\nМатриця переходів P є регулярною при k = {k}.")
            return True
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
    eigenvalues, eigenvectors = np.linalg.eig(P.T)

    # Знаходимо індекс власного значення, яке найліпше наближається до 1
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
num_simulations = 1000  # Кількість симуляцій
num_steps = 10  # Кількість кроків у кожній симуляції

# Збір статистичних даних
state_visit_counts = np.zeros(num_states)  # Кількість відвідувань кожного стану
transition_counts = np.zeros((num_states, num_states), dtype=int)  # Кількість переходів між станами
final_state_counts = np.zeros(num_states)  # Кількість завершень у кожному стані

paths = []  # Зберігаємо всі шляхи для подальшого аналізу (необов'язково)

for sim in range(num_simulations):
    path = simulate_markov_chain(P, initial_state=0, num_steps=num_steps)
    paths.append(path)

    # Підрахунок відвідувань станів
    for state in path:
        state_visit_counts[state] += 1

    # Підрахунок переходів між станами
    for i in range(len(path) - 1):
        transition_counts[path[i]][path[i + 1]] += 1

    # Підрахунок завершень у станах
    final_state = path[-1]
    final_state_counts[final_state] += 1

# Обчислення експериментальної стаціонарної розподілу
experimental_pi_star = state_visit_counts / (num_simulations * num_steps)

# Обчислення експериментальних ймовірностей кінцевих станів
experimental_final_probs = final_state_counts / num_simulations

print("\nЕкспериментальна стаціонарна розподіл pi*:")
print(experimental_pi_star)
print(f"Сума елементів експериментальної pi*: {experimental_pi_star.sum():.4f}")

print("\nЕкспериментальні ймовірності кінцевих станів:")
for i, prob in enumerate(experimental_final_probs):
    print(f"Стан {i}: {prob:.4f}")

# Обчислення експериментальної матриці переходів
experimental_P = np.zeros((num_states, num_states))

for i in range(num_states):
    total_transitions = transition_counts[i].sum()
    if total_transitions > 0:
        experimental_P[i] = transition_counts[i] / total_transitions
    else:
        experimental_P[i][i] = 1.0  # Для станів без вихідних переходів

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
transition_df_theoretical = pd.DataFrame(P, index=[f"S{i}" for i in states], columns=[f"S{j}" for j in states])
transition_df_experimental = pd.DataFrame(experimental_P, index=[f"S{i}" for i in states],
                                          columns=[f"S{j}" for j in states])

plt.figure(figsize=(16, 6))

plt.subplot(1, 2, 1)
sns.heatmap(transition_df_theoretical, annot=True, fmt=".2f", cmap="Blues")
plt.title("Теоретична Матриця Переходів")
plt.xlabel("Наступний стан")
plt.ylabel("Поточний стан")

plt.subplot(1, 2, 2)
sns.heatmap(transition_df_experimental, annot=True, fmt=".2f", cmap="Greens")
plt.title("Експериментальна Матриця Переходів")
plt.xlabel("Наступний стан")
plt.ylabel("Поточний стан")

plt.tight_layout()
plt.show()

# Візуалізація Стаціонарної Розподілу
stationary_df = pd.DataFrame({
    'Стан': [f"S{i}" for i in states],
    'Теоретична pi*': pi_star,
    'Експериментальна pi*': experimental_pi_star
})

# Перетворення даних для Seaborn
stationary_df_melted = stationary_df.melt(id_vars='Стан', var_name='Тип', value_name='Ймовірність')

# Візуалізація стаціонарної розподілу
plt.figure(figsize=(12, 6))
sns.barplot(data=stationary_df_melted, x='Стан', y='Ймовірність', hue='Тип', palette='pastel')
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

# Діаграма пирога для стаціонарної розподілу
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Теоретична pi*
axes[0].pie(pi_star, labels=[f"S{i}" for i in states], autopct='%1.1f%%',
            startangle=140, colors=plt.cm.Blues(np.linspace(0, 1, num_states)))
axes[0].set_title("Теоретична Стаціонарна Розподіл")

# Експериментальна pi*
axes[1].pie(experimental_pi_star, labels=[f"S{i}" for i in states], autopct='%1.1f%%',
            startangle=140, colors=plt.cm.Greens(np.linspace(0, 1, num_states)))
axes[1].set_title("Експериментальна Стаціонарна Розподіл")

plt.show()

# Візуалізація Гістограми Частот Відвідувань Станів
df_states = pd.DataFrame({
    'Стан': [f"S{i}" for i in states],
    'Середній час перебування': mean_time_in_states
})

plt.figure(figsize=(10, 6))
sns.barplot(data=df_states, x='Стан', y='Середній час перебування', palette='Blues_d')
plt.title('Середній час перебування в кожному стані')
plt.xlabel('Стан')
plt.ylabel('Середній час перебування (кроків)')
plt.show()

# Візуалізація Теплової Карти Частот Переходів
transition_df_counts = pd.DataFrame(transition_counts, index=[f"S{i}" for i in states],
                                    columns=[f"S{j}" for j in states])

plt.figure(figsize=(10, 8))
sns.heatmap(transition_df_counts, annot=True, fmt='d', cmap='YlGnBu')
plt.title('Частоти переходів між станами')
plt.xlabel('Наступний стан')
plt.ylabel('Поточний стан')
plt.show()
