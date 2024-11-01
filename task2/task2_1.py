# Імпорт необхідних бібліотек
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
from scipy.special import beta as beta_function  # Виправлення імпорту бета-функції
import math

# Налаштування стилю Seaborn для більш привабливих графіків
sns.set(style="whitegrid")


def beta_distribution(x, alpha, beta):
    """
    Функція щільності бета-розподілу.

    Parameters:
        x (float або np.ndarray): Точка або масив точок для обчислення щільності.
        alpha (float): Параметр α бета-розподілу.
        beta (float): Параметр β бета-розподілу.

    Returns:
        float або np.ndarray: Значення щільності бета-розподілу в точці x.
    """
    # Використання numpy для обробки масивів
    x = np.array(x)
    density = np.zeros_like(x, dtype=float)
    # Умови для знаходження щільності в межах [0,1]
    valid = (x > 0) & (x < 1)  # Виключаємо крайні точки, де щільність дорівнює 0
    density[valid] = (x[valid] ** (alpha - 1) * (1 - x[valid]) ** (beta - 1)) / beta_function(alpha, beta)
    return density


def metropolis_hastings(target_density, proposal_std, initial, iterations, alpha, beta):
    """
    Реалізація алгоритму Метрополіса-Гастінгса для вибірки з заданого розподілу.

    Parameters:
        target_density (function): Функція щільності цільового розподілу.
        proposal_std (float): Стандартне відхилення нормального пропозиційного розподілу.
        initial (float): Початкове значення вибірки.
        iterations (int): Кількість ітерацій алгоритму.
        alpha (float): Параметр α бета-розподілу.
        beta (float): Параметр β бета-розподілу.

    Returns:
        np.ndarray: Масив вибірок.
    """
    samples = np.zeros(iterations)
    current = initial
    current_density = target_density(current, alpha, beta)

    for i in tqdm(range(iterations), desc="Метрополіс-Гастінгс вибірка"):
        # Генерація пропозиції з нормального розподілу
        proposal = np.random.normal(current, proposal_std)

        # Переконаємося, що пропозиція лежить в межах [0, 1]
        if proposal <= 0 or proposal >= 1:
            samples[i] = current  # Відхилення пропозиції
            continue

        # Обчислення щільності цільового розподілу для пропозиції
        proposal_density = target_density(proposal, alpha, beta)

        # Обчислення відношення прийнятності
        acceptance_ratio = proposal_density / current_density

        # Прийняття або відхилення пропозиції
        if acceptance_ratio >= 1:
            current = proposal
            current_density = proposal_density
        else:
            if np.random.rand() < acceptance_ratio:
                current = proposal
                current_density = proposal_density

        # Збереження поточного стану
        samples[i] = current

    return samples


def plot_results(samples, alpha, beta):
    """
    Візуалізація результатів вибірки.

    Parameters:
        samples (np.ndarray): Масив вибірок.
        alpha (float): Параметр α бета-розподілу.
        beta (float): Параметр β бета-розподілу.
    """
    # Створення DataFrame для зручності
    df = pd.DataFrame({'Sample': samples})

    # Графік реалізації (Trace Plot)
    plt.figure(figsize=(14, 6))
    sns.lineplot(data=df, x=df.index, y='Sample', color='blue')
    plt.xlabel('Ітерація', fontsize=14)
    plt.ylabel('Значення', fontsize=14)
    plt.title('Trace Plot Метрополіса-Гастінгса', fontsize=16)
    plt.show()

    # Гістограма вибірки з накладеною теоретичною щільністю
    plt.figure(figsize=(14, 6))
    sns.histplot(df['Sample'], bins=30, stat='density', color='skyblue', edgecolor='black', label='Вибірка', alpha=0.6)

    # Накладення теоретичної щільності бета-розподілу
    x = np.linspace(0, 1, 1000)
    y = beta_distribution(x, alpha, beta)
    plt.plot(x, y, color='red', lw=2, label='Теоретична щільність')

    plt.xlabel('Значення', fontsize=14)
    plt.ylabel('Щільність', fontsize=14)
    plt.title('Гістограма вибірки з теоретичною щільністю бета-розподілу', fontsize=16)
    plt.legend(fontsize=12)
    plt.show()

    # Графік щільності розподілу (KDE Plot)
    plt.figure(figsize=(14, 6))
    sns.kdeplot(df['Sample'], color='blue', label='KDE вибірки', lw=2)
    plt.plot(x, y, color='red', label='Теоретична щільність', lw=2)

    plt.xlabel('Значення', fontsize=14)
    plt.ylabel('Щільність', fontsize=14)
    plt.title('KDE Plot вибірки та теоретичної щільності бета-розподілу', fontsize=16)
    plt.legend(fontsize=12)
    plt.show()


def main():
    # Параметри бета-розподілу
    alpha = 2.0  # Параметр α
    beta_param = 5.0  # Параметр β

    # Початкове значення
    initial = 0.5

    # Стандартне відхилення пропозиційного розподілу
    proposal_std = 0.1

    # Кількість ітерацій
    iterations = 1000

    # Реалізація алгоритму Метрополіса-Гастінгса
    samples = metropolis_hastings(beta_distribution, proposal_std, initial, iterations, alpha, beta_param)

    # Візуалізація результатів
    plot_results(samples, alpha, beta_param)


if __name__ == "__main__":
    main()
