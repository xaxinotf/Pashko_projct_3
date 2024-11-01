import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# Визначення кількості станів
num_states = 7

# Індексування станів від 0 до 6
states = list(range(num_states))

# Визначення поглинаючих станів (наприклад, стан 5 і 6)
absorbing_states = [5, 6]

# Створення матриці переходів P з нулями
P = np.zeros((num_states, num_states))

# Заповнення матриці переходів
for i in range(num_states):
    if i in absorbing_states:
        P[i][i] = 1.0  # Поглинаючий стан
    else:
        # Визначаємо кількість можливих переходів (усі стани крім поточного)
        # Тут задаємо конкретні ймовірності переходів
        P[i] = np.array([0.1 if j != i else 0 for j in range(num_states)])
        # Збільшуємо ймовірність переходу до поглинаючих станів
        P[i][5] = 0.3
        P[i][6] = 0.2
        # Нормалізуємо рядок, щоб сума дорівнювала 1
        P[i] = P[i] / P[i].sum()

print("Матриця переходів P:")
print(P)

# Визначення непоглинаючих станів
transient_states = [s for s in states if s not in absorbing_states]

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

# Створення графа
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
colors = [G.nodes[node]['color'] for node in G.nodes]

# Отримання міток вузлів
labels = {node: G.nodes[node]['label'] for node in G.nodes}

# Отримання міток ребер з ймовірностями
edge_labels = {(i, j): f"{P[i][j]:.2f}" for i, j in G.edges()}

# Відображення графа
plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G, seed=42)  # Розміщення вузлів

# Малювання вузлів
nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=800)

# Малювання ребер
nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=20, edge_color='gray')

# Малювання міток вузлів
nx.draw_networkx_labels(G, pos, labels, font_size=12)

# Малювання міток ребер
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)

plt.title("Поглинаючий ланцюг Маркова з 7 станами")
plt.axis('off')
plt.show()

# Початковий стан S0
initial_state_index = transient_states.index(0)  # Індекс S0 у transient_states

# Ймовірності поглинання з S0
absorption_probabilities = B[initial_state_index]

print("\nЙмовірності поглинання зі стану S0:")
for idx, prob in enumerate(absorption_probabilities):
    print(f"Поглинання у стан {absorbing_states[idx]}: {prob:.4f}")
