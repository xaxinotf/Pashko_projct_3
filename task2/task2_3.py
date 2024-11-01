# Імпорт необхідних бібліотек
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
from scipy.special import gamma as gamma_function  # Гамма-функція з scipy
import math

# Налаштування стилю Seaborn для більш привабливих графіків
sns.set(style="whitegrid")

def cauchy_distribution(x, x0, gamma):
    """
    Функція щільності Cauchy-розподілу.

    Parameters:
        x (float або np.ndarray): Точка або масив точок для обчислення щільності.
        x0 (float): Центр розподілу.
        gamma (float): Параметр масштабу.

    Returns:
        float або np.ndarray: Значення щільності Cauchy-розподілу в точці x.
    """
    x = np.array(x)
    density = (1 / (gamma * np.pi)) * (gamma**2 / ((x - x0)**2 + gamma**2))
    return density

def metropolis_hastings(target_density, proposal_std, initial, iterations, x0, gamma):
    """
    Реалізація алгоритму Метрополіса-Гастінгса для вибірки з заданого розподілу.

    Parameters:
        target_density (function): Функція щільності цільового розподілу.
        proposal_std (float): Стандартне відхилення нормального пропозиційного розподілу.
        initial (float): Початкове значення вибірки.
        iterations (int): Кількість ітерацій алгоритму.
        x0 (float): Центр розподілу Cauchy.
        gamma (float): Параметр масштабу Cauchy-розподілу.

    Returns:
        np.ndarray: Масив вибірок.
    """
    samples = np.zeros(iterations)
    current = initial
    current_density = target_density(current, x0, gamma)

    for i in tqdm(range(iterations), desc="Метрополіс-Гастінгс вибірка"):
        # Генерація пропозиції з нормального розподілу
        proposal = np.random.normal(current, proposal_std)

        # Обчислення щільності цільового розподілу для пропозиції
        proposal_density = target_density(proposal, x0, gamma)

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

def plot_results(samples, x0, gamma):
    """
    Візуалізація результатів вибірки.

    Parameters:
        samples (np.ndarray): Масив вибірок.
        x0 (float): Центр розподілу Cauchy.
        gamma (float): Параметр масштабу Cauchy-розподілу.
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

    # Накладення теоретичної щільності Cauchy-розподілу
    x = np.linspace(df['Sample'].min(), df['Sample'].max(), 1000)
    y = cauchy_distribution(x, x0, gamma)
    plt.plot(x, y, color='red', lw=2, label='Теоретична щільність')

    plt.xlabel('Значення', fontsize=14)
    plt.ylabel('Щільність', fontsize=14)
    plt.title('Гістограма вибірки з теоретичною щільністю Cauchy-розподілу', fontsize=16)
    plt.legend(fontsize=12)
    plt.show()

    # Графік щільності розподілу (KDE Plot)
    plt.figure(figsize=(14, 6))
    sns.kdeplot(df['Sample'], color='blue', label='KDE вибірки', lw=2)
    plt.plot(x, y, color='red', label='Теоретична щільність', lw=2)

    plt.xlabel('Значення', fontsize=14)
    plt.ylabel('Щільність', fontsize=14)
    plt.title('KDE Plot вибірки та теоретичної щільності Cauchy-розподілу', fontsize=16)
    plt.legend(fontsize=12)
    plt.show()

def main():
    # Параметри Cauchy-розподілу
    x0 = 0.0    # Центр розподілу
    gamma = 1.0  # Параметр масштабу

    # Початкове значення
    initial = 0.0

    # Стандартне відхилення пропозиційного розподілу
    proposal_std = 1.0

    # Кількість ітерацій
    iterations = 1000

    # Реалізація алгоритму Метрополіса-Гастінгса
    samples = metropolis_hastings(cauchy_distribution, proposal_std, initial, iterations, x0, gamma)

    # Візуалізація результатів
    plot_results(samples, x0, gamma)

if __name__ == "__main__":
    main()
