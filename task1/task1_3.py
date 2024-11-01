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

# Визначення непоглинаючих станів
transient_states = [s for s in states if s not in absorbing_states]

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

# Вивід матриці переходів
print("Теоретична матриця переходів P:")
for i, row in enumerate(P):
    print(f"Стан {i}: {row}")

# Теоретичні обчислення
# Визначення розмірів
t = len(transient_states)
r = len(absorbing_states)

# Створення матриці Q
Q = P[np.ix_(transient_states, transient_states)]

# Створення матриці R
R = P[np.ix_(transient_states, absorbing_states)]

print("\nМатриця Q (переходи між непоглинаючими станами):")
print(Q)

print("\nМатриця R (переходи до поглинаючих станів):")
print(R)

# Одинична матриця розміру Q
I = np.eye(t)

# Обчислення фундаментальної матриці N
N = np.linalg.inv(I - Q)

print("\nФундаментальна матриця N:")
print(N)

# Обчислення матриці поглинання B
B = np.dot(N, R)

print("\nМатриця поглинання B:")
print(B)

# Теоретичні ймовірності поглинання
print("\nТеоретичні ймовірності поглинання з початкового стану S0:")
for idx, state in enumerate(absorbing_states):
    print(f"Поглинання у стан {state}: {B[0][idx]:.4f}")

# Обчислення очікуваного часу до поглинання
expected_time_to_absorption = np.dot(N, np.ones((t, 1))).flatten()

print("\nТеоретичний очікуваний час до поглинання для кожного непоглинаючого стану:")
for idx, state in enumerate(transient_states):
    print(f"Стан {state}: {expected_time_to_absorption[idx]:.4f} кроків")

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
num_simulations = 10000

# Збір статистичних даних
steps_to_absorption = []
absorption_counts = defaultdict(int)
paths = []
state_visit_counts = defaultdict(int)  # Для часу перебування в заданому стані

# Визначення заданого стану для часу перебування (наприклад, стан 1)
target_state = 1

for _ in range(num_simulations):
    steps, final_state, path = simulate_markov_chain(P, initial_state, absorbing_states)
    steps_to_absorption.append(steps)
    absorption_counts[final_state] += 1
    paths.append(path)
    # Підрахунок кількості відвідувань заданого стану
    state_visit_counts[path.count(target_state)] += 1

# Обчислення ймовірності поглинання в кожен поглинаючий стан
experimental_absorption_probabilities = {state: count / num_simulations for state, count in absorption_counts.items()}

# Обчислення середнього часу до поглинання
mean_steps = np.mean(steps_to_absorption)
median_steps = np.median(steps_to_absorption)
std_steps = np.std(steps_to_absorption)

print(f"\nПроведено {num_simulations} симуляцій.")
print("\nЕкспериментальні ймовірності поглинання:")
for state in absorbing_states:
    print(f"Поглинання у стан {state}: {experimental_absorption_probabilities.get(state, 0):.4f}")

print(f"\nЕкспериментальні характеристики часу до поглинання:")
print(f"Середня кількість кроків до поглинання: {mean_steps:.4f}")
print(f"Медіана кількості кроків до поглинання: {median_steps}")
print(f"Стандартне відхилення кількості кроків до поглинання: {std_steps:.4f}")

# Експериментальна матриця переходів
transition_counts = np.zeros((num_states, num_states), dtype=int)

for path in paths:
    for i in range(len(path) - 1):
        transition_counts[path[i]][path[i+1]] += 1

# Обчислення експериментальної матриці переходів
experimental_P = np.zeros((num_states, num_states))

for i in range(num_states):
    total = transition_counts[i].sum()
    if total > 0:
        experimental_P[i] = transition_counts[i] / total
    else:
        experimental_P[i][i] = 1.0  # Поглинаючі стани залишаються самі у собі

print("\nЕкспериментальна матриця переходів P:")
for i, row in enumerate(experimental_P):
    print(f"Стан {i}: {row}")

# Час перебування в заданому стані
time_in_target_state = []

for path in paths:
    time_in_target_state.append(path.count(target_state))

# Середній час перебування в заданому стані
mean_time_in_target = np.mean(time_in_target_state)
median_time_in_target = np.median(time_in_target_state)
std_time_in_target = np.std(time_in_target_state)

print(f"\nЧас перебування в стані {target_state}:")
print(f"Середній час: {mean_time_in_target:.4f} кроків")
print(f"Медіана часу: {median_time_in_target}")
print(f"Стандартне відхилення часу: {std_time_in_target:.4f}")

# Порівняння експериментальних та теоретичних характеристик
print("\nПорівняння ймовірностей поглинання:")
for idx, state in enumerate(absorbing_states):
    theoretical = B[0][idx]
    experimental = experimental_absorption_probabilities.get(state, 0)
    print(f"Стан {state}: Теоретична = {theoretical:.4f}, Експериментальна = {experimental:.4f}")

print("\nПорівняння очікуваного часу до поглинання:")
for idx, state in enumerate(transient_states):
    theoretical = expected_time_to_absorption[idx]
    # Обчислення експериментального очікуваного часу для кожного стану
    # Тут ми обчислюємо лише для початкового стану
    if state == initial_state:
        experimental = mean_steps
        print(f"Стан {state}: Теоретичний = {theoretical:.4f}, Експериментальний = {experimental:.4f}")

# Візуалізація Результатів

# a. Експериментальна Матриця Переходів
plt.figure(figsize=(8, 6))
sns.heatmap(experimental_P, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=[f"S{state}" for state in states],
            yticklabels=[f"S{state}" for state in states])
plt.title("Експериментальна Матриця Переходів")
plt.xlabel("Наступний стан")
plt.ylabel("Поточний стан")
plt.show()

# b. Гістограма Кроків до Поглинання
plt.figure(figsize=(10, 6))
sns.histplot(steps_to_absorption, bins=range(0, max(steps_to_absorption)+2), kde=False, color='skyblue')
plt.title('Розподіл кількості кроків до поглинання')
plt.xlabel('Кроки до поглинання')
plt.ylabel('Частота')
plt.grid(axis='y', alpha=0.75)
plt.show()

# c. Діаграма Пирога для Ймовірностей Поглинання
labels = [f"Стан {state}" for state in absorbing_states]
sizes = [experimental_absorption_probabilities[state] * 100 for state in absorbing_states]
colors_pie = ['lightcoral', 'lightskyblue']

plt.figure(figsize=(8, 8))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors_pie, shadow=True)
plt.title('Розподіл поглинальних станів після симуляцій')
plt.axis('equal')  # Рівні пропорції
plt.show()

# d. Граф Ланцюга Маркова
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
labels_graph = {node: G.nodes[node]['label'] for node in G.nodes}

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
nx.draw_networkx_labels(G, pos, labels_graph, font_size=12)

# Малювання міток ребер
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)

plt.title("Поглинаючий ланцюг Маркова з 5 станами")
plt.axis('off')
plt.show()

# e. Гістограма Часу Перебування в Заданому Стані
plt.figure(figsize=(10, 6))
sns.histplot(time_in_target_state, bins=range(0, max(time_in_target_state)+2), kde=False, color='lightgreen')
plt.title(f'Розподіл часу перебування в стані {target_state}')
plt.xlabel('Кількість відвідувань стану')
plt.ylabel('Частота')
plt.grid(axis='y', alpha=0.75)
plt.show()
