import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict
import seaborn as sns

# Визначення кількості станів
num_states = 5

# Індексування станів від 0 до 4
states = list(range(num_states))

# Визначення поглинаючих станів (стани 3 та 4)
absorbing_states = [3, 4]

# Створення матриці переходів P з нулями
P = np.zeros((num_states, num_states))

# Заповнення матриці переходів
# Задаємо конкретні ймовірності переходів

# Стан 0
P[0] = [0.1, 0.6, 0.3, 0.0, 0.0]

# Стан 1
P[1] = [0.2, 0.2, 0.5, 0.1, 0.0]

# Стан 2
P[2] = [0.0, 0.3, 0.2, 0.4, 0.1]

# Поглинаючі стани залишаються самі у собі
for absorb_state in absorbing_states:
    P[absorb_state][absorb_state] = 1.0

# Вектор початкових ймовірностей (початковий стан – стан 0)
initial_state = 0

print("Матриця переходів P:")
for i, row in enumerate(P):
    print(f"Стан {i}: {row}")


def simulate_markov_chain(P, initial_state, absorbing_states):
    """
    Симулює одну реалізацію поглинаючого ланцюга Маркова до моменту поглинання.

    Parameters:
        P (numpy.ndarray): Матриця переходів.
        initial_state (int): Початковий стан.
        absorbing_states (list): Список поглинаючих станів.

    Returns:
        steps (int): Кількість кроків до поглинання.
        final_state (int): Поглинаючий стан, у який потрапив ланцюг.
        path (list): Послідовність станів у цій реалізації.
    """
    current_state = initial_state
    steps = 0
    path = [current_state]

    while current_state not in absorbing_states:
        # Вибір наступного стану на основі ймовірностей переходів
        next_state = np.random.choice(states, p=P[current_state])
        path.append(next_state)
        steps += 1
        current_state = next_state

    return steps, current_state, path


# Кількість симуляцій
num_simulations = 1000

# Збір статистичних даних
steps_to_absorption = []
absorption_counts = defaultdict(int)
paths = []

for _ in range(num_simulations):
    steps, final_state, path = simulate_markov_chain(P, initial_state, absorbing_states)
    steps_to_absorption.append(steps)
    absorption_counts[final_state] += 1
    paths.append(path)

# Обчислення ймовірності поглинання в кожен поглинаючий стан
absorption_probabilities = {state: count / num_simulations for state, count in absorption_counts.items()}

print(f"\nПроведено {num_simulations} симуляцій.")
print("Ймовірності поглинання:")
for state in absorbing_states:
    print(f"Поглинання у стан {state}: {absorption_probabilities.get(state, 0):.4f}")

# Основні статистичні показники для кроків до поглинання
mean_steps = np.mean(steps_to_absorption)
median_steps = np.median(steps_to_absorption)
std_steps = np.std(steps_to_absorption)

print(f"\nСередня кількість кроків до поглинання: {mean_steps:.2f}")
print(f"Медіана кількості кроків до поглинання: {median_steps}")
print(f"Стандартне відхилення кількості кроків до поглинання: {std_steps:.2f}")

# Візуалізація Результатів

# a. Гістограма Кроків до Поглинання
plt.figure(figsize=(10, 6))
sns.histplot(steps_to_absorption, bins=range(0, max(steps_to_absorption) + 2), kde=False, color='skyblue')
plt.title('Розподіл кількості кроків до поглинання')
plt.xlabel('Кроки до поглинання')
plt.ylabel('Частота')
plt.grid(axis='y', alpha=0.75)
plt.show()

# b. Розподіл Поглинальних Станів
labels = [f"Стан {state}" for state in absorbing_states]
sizes = [absorption_probabilities[state] * 100 for state in absorbing_states]
colors_pie = ['lightcoral', 'lightskyblue']

plt.figure(figsize=(8, 8))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors_pie, shadow=True)
plt.title('Розподіл поглинальних станів після симуляцій')
plt.axis('equal')  # Рівні пропорції
plt.show()

# c. Візуалізація Ланцюга Маркова
G = nx.DiGraph()

# Додавання вузлів
for state in states:
    if state in absorbing_states:
        G.add_node(state, color='red', label=f"S{state} (Поглинаючий)")
    else:
        G.add_node(state, color='lightblue', label=f"S{state}")

# Додавання ребер з ймовірностями
for i in states:
    for j in states:
        if P[i][j] > 0:
            G.add_edge(i, j, weight=P[i][j])

# Отримання кольорів вузлів
node_colors = [G.nodes[node]['color'] for node in G.nodes]

# Отримання міток вузлів
labels = {node: G.nodes[node]['label'] for node in G.nodes}

# Отримання міток ребер з ймовірностями
edge_labels = {(i, j): f"{P[i][j]:.2f}" for i, j in G.edges()}

# Відображення графа
plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G, seed=42)  # Розміщення вузлів

# Малювання вузлів
nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=800)

# Малювання ребер
nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=20, edge_color='gray')

# Малювання міток вузлів
nx.draw_networkx_labels(G, pos, labels, font_size=12)

# Малювання міток ребер
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)

plt.title("Поглинаючий ланцюг Маркова з 5 станами")
plt.axis('off')
plt.show()
