import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm

# Налаштування стилю Seaborn
sns.set(style="whitegrid")

# Кількість ув'язнених та коробок
NUM_PRISONERS = 100
NUM_BOXES = 100

# Кількість відкриттів, дозволених ув'язненим
OPEN_LIMITS = [50, 60, 75]

# Кількість симуляцій для експериментів
NUM_SIMULATIONS = 10000

def simulate_prisoners_strategy(boxes, open_limit):
    """
    Симулює стратегію ув'язнених.

    Parameters:
        boxes (list or array): Розташування табличок у коробках.
        open_limit (int): Максимальна кількість відкриттів для кожного ув'язненого.

    Returns:
        bool: True, якщо всі ув'язнені знайшли свої таблички, інакше False.
    """
    for prisoner in range(1, NUM_PRISONERS + 1):
        current_box = prisoner
        found = False
        for _ in range(open_limit):
            card = boxes[current_box - 1]
            if card == prisoner:
                found = True
                break
            current_box = card
        if not found:
            return False
    return True

def simulate_random_strategy(boxes, open_limit):
    """
    Симулює випадкову стратегію ув'язнених.

    Parameters:
        boxes (list or array): Розташування табличок у коробках.
        open_limit (int): Максимальна кількість відкриттів для кожного ув'язненого.

    Returns:
        bool: True, якщо всі ув'язнені знайшли свої таблички, інакше False.
    """
    for prisoner in range(1, NUM_PRISONERS + 1):
        opened_boxes = np.random.choice(range(1, NUM_BOXES + 1), size=open_limit, replace=False)
        if prisoner not in boxes[opened_boxes - 1]:
            return False
    return True

def run_random_strategy_simulation(num_simulations, open_limit):
    """
    Виконує симуляцію випадкового вибору коробок.

    Parameters:
        num_simulations (int): Кількість симуляцій.
        open_limit (int): Максимальна кількість відкриттів для кожного ув'язненого.

    Returns:
        float: Емпірична ймовірність успіху.
    """
    successes = 0
    for _ in tqdm(range(num_simulations), desc=f"Random Strategy with {open_limit} opens"):
        # Генеруємо випадкову перестановку табличок у коробках
        boxes = np.random.permutation(NUM_BOXES) + 1  # Таблички від 1 до 100
        if simulate_random_strategy(boxes, open_limit):
            successes += 1
    probability = successes / num_simulations
    return probability

def run_prisoners_strategy_simulation(num_simulations, open_limit):
    """
    Виконує симуляцію стратегії ув'язнених.

    Parameters:
        num_simulations (int): Кількість симуляцій.
        open_limit (int): Максимальна кількість відкриттів для кожного ув'язненого.

    Returns:
        float: Емпірична ймовірність успіху.
    """
    successes = 0
    for _ in tqdm(range(num_simulations), desc=f"Prisoners Strategy with {open_limit} opens"):
        # Генеруємо випадкову перестановку табличок у коробках
        boxes = np.random.permutation(NUM_BOXES) + 1  # Таблички від 1 до 100
        if simulate_prisoners_strategy(boxes, open_limit):
            successes += 1
    probability = successes / num_simulations
    return probability

def analyze_open_limits(open_limits, num_simulations):
    """
    Аналізує вплив різних лімітів відкриттів на ймовірність успіху.

    Parameters:
        open_limits (list): Список лімітів відкриттів.
        num_simulations (int): Кількість симуляцій для кожного ліміту.

    Returns:
        dict: Словник з лімітами відкриттів та відповідними ймовірностями.
    """
    results = {}
    for limit in open_limits:
        prob = run_prisoners_strategy_simulation(num_simulations, limit)
        results[limit] = prob
    return results

def visualize_results(random_prob, strategy_probs, open_limits):
    """
    Візуалізує результати симуляцій.

    Parameters:
        random_prob (float): Ймовірність успіху при випадковому виборі.
        strategy_probs (dict): Ймовірності успіху при використанні алгоритму.
        open_limits (list): Список лімітів відкриттів.
    """
    # Підготовка даних
    data = {
        'Strategy': ['Random Strategy'] + [f'Strategy {limit} opens' for limit in open_limits],
        'Probability of Success': [random_prob] + [strategy_probs[limit] for limit in open_limits],
        'Type': ['Random'] + ['Prisoners'] * len(open_limits)
    }
    df = pd.DataFrame(data)

    # Візуалізація порівняльного барплоту з використанням hue
    plt.figure(figsize=(12, 7))
    sns.barplot(data=df, x='Strategy', y='Probability of Success', hue='Type', palette='viridis')
    plt.ylabel('Probability of Success')
    plt.title('Probability of Survival: Random vs Prisoners\' Strategy')
    plt.ylim(0, 1)
    plt.legend(title='Type')

    # Додавання текстових міток
    for index, row in df.iterrows():
        plt.text(index, row['Probability of Success'] + 0.01, f"{row['Probability of Success']:.4f}", ha='center')

    plt.show()

def visualize_open_limit_change(strategy_probs, open_limits):
    """
    Візуалізує зміну ймовірності успіху з збільшенням ліміту відкриттів.

    Parameters:
        strategy_probs (dict): Ймовірності успіху при використанні алгоритму.
        open_limits (list): Список лімітів відкриттів.
    """
    limits = open_limits
    probabilities = [strategy_probs[limit] for limit in limits]

    plt.figure(figsize=(10, 6))
    sns.lineplot(x=limits, y=probabilities, marker='o', linewidth=2.5, color='blue')
    plt.xlabel('Number of Box Opens Allowed')
    plt.ylabel('Probability of Success')
    plt.title('Effect of Increasing Open Limit on Success Probability')
    plt.ylim(0, 1)

    # Додавання текстових міток
    for i, prob in enumerate(probabilities):
        plt.text(limits[i], probabilities[i] + 0.01, f"{prob:.4f}", ha='center')

    plt.show()

# 1. Симуляція Випадкового Вибору Коробок з Лімітом 50
print("Simulating Random Strategy...")
random_success_prob = run_random_strategy_simulation(NUM_SIMULATIONS, 50)
print(f"\nProbability of Success (Random Strategy, 50 opens): {random_success_prob:.10f}")

# 2. Симуляція Алгоритму Ув’язнених для Різних Лімітів Відкриттів
print("\nSimulating Prisoners' Strategy...")
strategy_success_probs = analyze_open_limits(OPEN_LIMITS, NUM_SIMULATIONS)
for limit, prob in strategy_success_probs.items():
    print(f"Probability of Success (Prisoners' Strategy, {limit} opens): {prob:.4f}")

# 3. Візуалізація Результатів
print("\nVisualizing Results...")
visualize_results(random_success_prob, strategy_success_probs, OPEN_LIMITS)
visualize_open_limit_change(strategy_success_probs, OPEN_LIMITS)
