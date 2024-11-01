import random
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from scipy import stats
from collections import Counter


class CoinGame:
    def __init__(self, bet_amount: float = 1.0, win_amount: float = 5.0):
        self.bet_amount = bet_amount
        self.win_amount = win_amount

    def flip_coin(self) -> str:
        return random.choice(['О', 'Р'])

    def play_game_head_tail(self, num_flips: int) -> Tuple[List[str], float, List[float]]:
        balance = 0
        flips = []
        balances = [0]
        last_flip = None
        wins = 0

        for _ in range(num_flips):
            current_flip = self.flip_coin()
            flips.append(current_flip)
            balance -= self.bet_amount

            if last_flip == 'О' and current_flip == 'Р':
                balance += self.win_amount
                wins += 1

            balances.append(balance)
            last_flip = current_flip

        return flips, balance, balances, wins

    def play_game_head_head(self, num_flips: int) -> Tuple[List[str], float, List[float]]:
        balance = 0
        flips = []
        balances = [0]
        last_flip = None
        wins = 0

        for _ in range(num_flips):
            current_flip = self.flip_coin()
            flips.append(current_flip)
            balance -= self.bet_amount

            if last_flip == 'О' and current_flip == 'О':
                balance += self.win_amount
                wins += 1

            balances.append(balance)
            last_flip = current_flip

        return flips, balance, balances, wins


def analyze_sequences(flips: List[str]) -> dict:
    sequences = {'О': 0, 'Р': 0, 'ОР': 0, 'РО': 0, 'ОО': 0, 'РР': 0}

    sequences['О'] = flips.count('О')
    sequences['Р'] = flips.count('Р')

    for i in range(len(flips) - 1):
        pair = flips[i] + flips[i + 1]
        if pair in sequences:
            sequences[pair] += 1

    return sequences


def calculate_advanced_statistics(results: List[float]) -> dict:
    stats_dict = {
        'Середній виграш': np.mean(results),
        'Медіанний виграш': np.median(results),
        'Стандартне відхилення': np.std(results),
        'Коефіцієнт варіації': np.std(results) / np.mean(results) * 100,
        'Асиметрія': stats.skew(results),
        'Ексцес': stats.kurtosis(results),
        'Мінімальний виграш': min(results),
        'Максимальний виграш': max(results),
        '25-й перцентиль': np.percentile(results, 25),
        '75-й перцентиль': np.percentile(results, 75),
        'Відсоток прибуткових ігор': (sum(1 for x in results if x > 0) / len(results)) * 100
    }
    return stats_dict


def simulate_multiple_players(game_type: str, num_players: int, flips_per_player: int) -> tuple:
    game = CoinGame()
    results = []
    all_flips = []
    all_balances = []
    total_wins = []

    for _ in range(num_players):
        if game_type == 'head_tail':
            flips, balance, balances, wins = game.play_game_head_tail(flips_per_player)
        else:
            flips, balance, balances, wins = game.play_game_head_head(flips_per_player)

        results.append(balance)
        all_flips.append(flips)
        all_balances.append(balances)
        total_wins.append(wins)

    return results, all_flips, all_balances, total_wins


def plot_advanced_results(results_ht, results_hh, balances_ht, balances_hh, wins_ht, wins_hh):
    plt.figure(figsize=(20, 15))

    # 1. Гістограма виграшів
    plt.subplot(3, 2, 1)
    plt.hist(results_ht, bins=30, alpha=0.5, label='Орел-Решка', color='blue')
    plt.hist(results_hh, bins=30, alpha=0.5, label='Орел-Орел', color='green')
    plt.title('Розподіл виграшів')
    plt.xlabel('Виграш ($)')
    plt.ylabel('Кількість гравців')
    plt.legend()

    # 2. Boxplot виграшів
    plt.subplot(3, 2, 2)
    box_data = [results_ht, results_hh]
    plt.boxplot(box_data, tick_labels=['Орел-Решка', 'Орел-Орел'])  # Змінено labels на tick_labels
    plt.title('Порівняння розподілів виграшів')
    plt.ylabel('Виграш ($)')

    # 3. Графік зміни балансу для випадкового гравця
    plt.subplot(3, 2, 3)
    plt.plot(balances_ht[0], label='Орел-Решка', color='blue')
    plt.plot(balances_hh[0], label='Орел-Орел', color='green')
    plt.title('Зміна балансу протягом гри (випадковий гравець)')
    plt.xlabel('Кількість підкидань')
    plt.ylabel('Баланс ($)')
    plt.legend()

    # 4. Гістограма кількості виграшів
    plt.subplot(3, 2, 4)
    plt.hist(wins_ht, bins=20, alpha=0.5, label='Орел-Решка', color='blue')
    plt.hist(wins_hh, bins=20, alpha=0.5, label='Орел-Орел', color='green')
    plt.title('Розподіл кількості виграшних комбінацій')
    plt.xlabel('Кількість виграшів')
    plt.ylabel('Кількість гравців')
    plt.legend()

    # 5. Q-Q plots
    plt.subplot(3, 2, 5)
    stats.probplot(results_ht, dist="norm", plot=plt)
    plt.title('Q-Q plot (Орел-Решка)')

    plt.subplot(3, 2, 6)
    stats.probplot(results_hh, dist="norm", plot=plt)
    plt.title('Q-Q plot (Орел-Орел)')

    plt.tight_layout()
    plt.show()


# Параметри симуляції
NUM_PLAYERS = 1000
FLIPS_PER_PLAYER = 100

# Проведення симуляцій
results_ht, flips_ht, balances_ht, wins_ht = simulate_multiple_players('head_tail', NUM_PLAYERS, FLIPS_PER_PLAYER)
results_hh, flips_hh, balances_hh, wins_hh = simulate_multiple_players('head_head', NUM_PLAYERS, FLIPS_PER_PLAYER)

# Візуалізація результатів
plot_advanced_results(results_ht, results_hh, balances_ht, balances_hh, wins_ht, wins_hh)

# Виведення статистики
print("\nРозширена статистика для гри 'Орел-Решка':")
stats_ht = calculate_advanced_statistics(results_ht)
for key, value in stats_ht.items():
    print(f"{key}: {value:.2f}")

print("\nРозширена статистика для гри 'Орел-Орел':")
stats_hh = calculate_advanced_statistics(results_hh)
for key, value in stats_hh.items():
    print(f"{key}: {value:.2f}")

# Аналіз послідовностей
print("\nАналіз послідовностей для випадкового гравця (Орел-Решка):")
seq_analysis_ht = analyze_sequences(flips_ht[0])
for key, value in seq_analysis_ht.items():
    print(f"Комбінація '{key}': {value} разів")

print("\nАналіз послідовностей для випадкового гравця (Орел-Орел):")
seq_analysis_hh = analyze_sequences(flips_hh[0])
for key, value in seq_analysis_hh.items():
    print(f"Комбінація '{key}': {value} разів")

# Фінансові показники
print("\nФінансові показники:")
print(f"Середній ROI (Орел-Решка): {(np.mean(results_ht) / (FLIPS_PER_PLAYER * 1)) * 100:.2f}%")
print(f"Середній ROI (Орел-Орел): {(np.mean(results_hh) / (FLIPS_PER_PLAYER * 1)) * 100:.2f}%")
print(f"Максимальний прибуток (Орел-Решка): ${max(results_ht):.2f}")
print(f"Максимальний прибуток (Орел-Орел): ${max(results_hh):.2f}")
print(f"Максимальний збиток (Орел-Решка): ${min(results_ht):.2f}")
print(f"Максимальний збиток (Орел-Орел): ${min(results_hh):.2f}")