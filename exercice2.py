import env
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


N0 = 100        
NUM_EPISODES = 800_000  


DEALER_DIM = 10
PLAYER_DIM = 21
ACTION_DIM = 2

# Fonction de valeur action Q(s, a)
Q = np.zeros((DEALER_DIM, PLAYER_DIM, ACTION_DIM))

# Compteur de visites N(s) et N(s, a)
N_s  = np.zeros((DEALER_DIM, PLAYER_DIM), dtype=int)
N_sa = np.zeros((DEALER_DIM, PLAYER_DIM, ACTION_DIM), dtype=int)


def state_to_index(state) :
    """
    Convertit un état du jeu en indices de tableau.

    Parametres : 
    state : dict  {'dealer': int [1-10], 'player': int [1-21]}

    retourne (d_idx, p_idx) : indices dans les tableaux numpy
    """
    d_idx = np.clip(state['dealer'] - 1, 0, DEALER_DIM - 1)
    p_idx = np.clip(state['player'] - 1, 0, PLAYER_DIM - 1)
    return d_idx, p_idx


def epsilon_greedy(state) :
    """
    Choisit une action selon la politique ε-greedy.

    Parametres : 
    state : dict  état courant

    Retour : 
    action : int  0 = 'hit', 1 = 'stick'
    """
    d_idx, p_idx = state_to_index(state)
    epsilon = N0 / (N0 + N_s[d_idx, p_idx])

    if np.random.rand() < epsilon:
        # Exploration : action aléatoire
        return np.random.randint(0, ACTION_DIM)
    else:
        # Exploitation : meilleure action connue
        return int(np.argmax(Q[d_idx, p_idx]))


def action_to_str(action) :
    """Convertit un entier d'action en chaîne pour l'environnement."""
    return 'hit' if action == 0 else 'stick'


def generate_episode() :
    """
    Joue un épisode complet en suivant la politique ε-greedy courante.

    Retour : 
    trajectory : list de (state, action, reward)
    Chaque triplet correspond à un pas de temps.
    """
    state = env.init_game()
    trajectory = []
    terminate = 0

    while not terminate:
        action = epsilon_greedy(state)
        next_state, reward, terminate = env.step(state, action_to_str(action))
        trajectory.append((state, action, reward))
        state = next_state

    return trajectory


def monte_carlo_control(num_episodes = NUM_EPISODES) :
    """
    Entraîne la politique par la méthode de contrôle Monte-Carlo.

    Algorithme :
    1. Générer un épisode complet avec la politique ε-greedy.
    2. Calculer le retour cumulé G (sans actualisation, y = 1).
    3. Pour chaque (s, a) visité, mettre à jour N(s), N(s,a) et Q(s,a)
       avec le pas d'apprentissage at = 1/N(st, at).

    Parametres : 
    num_episodes : int  nombre d'épisodes d'entraînement
    """
    for episode_idx in range(num_episodes):

        trajectory = generate_episode()

        # Calcul du retour cumulé depuis la fin (γ = 1, pas d'actualisation)
        G = 0.0
        for (state, action, reward) in reversed(trajectory):
            G += reward   # γ = 1 → simple somme

            d_idx, p_idx = state_to_index(state)

            # Mise à jour des compteurs
            N_s[d_idx, p_idx]        += 1
            N_sa[d_idx, p_idx, action] += 1

            # Pas d'apprentissage variable
            alpha = 1.0 / N_sa[d_idx, p_idx, action]

            # Mise à jour incrémentale de Q(s, a)
            Q[d_idx, p_idx, action] += alpha * (G - Q[d_idx, p_idx, action])

        if (episode_idx + 1) % 100_000 == 0:
            print(f"  Épisode {episode_idx + 1:>10} / {num_episodes}")



def plot_value_function() :
    """
    Trace la fonction de valeur optimale V*(s) = max_a Q*(s, a)
    sous forme de surface 3D.

    Axes :
        x : carte visible du croupier (1-10)
        y : somme du joueur (1-21)
        z : V*(s)
    """
    # V*(s) = max sur les actions de Q*(s, a)
    V_star = np.max(Q, axis=2)   # shape (10, 21)

    dealer_range = np.arange(1, DEALER_DIM + 1)   # 1..10
    player_range = np.arange(1, PLAYER_DIM + 1)   # 1..21

    D, P = np.meshgrid(dealer_range, player_range)  # shape (21, 10)
    Z    = V_star[D - 1, P - 1]                     # shape (21, 10)

    fig = plt.figure(figsize=(12, 7))
    ax  = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(D, P, Z, cmap='viridis', edgecolor='none', alpha=0.9)

    ax.set_xlabel("Carte du croupier (1-10)", labelpad=10)
    ax.set_ylabel("Somme du joueur (1-21)", labelpad=10)
    ax.set_zlabel("V*(s)", labelpad=10)
    ax.set_title(f"Fonction de valeur optimale V*(s) — Easy21\n({NUM_EPISODES:,} épisodes)", pad=15)

    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label="V*(s)")
    plt.tight_layout()
    plt.savefig("./value_function.png", dpi=150)
    plt.show()
    print("Figure sauvegardée : value_function.png")



if __name__ == '__main__':
    print(f"Démarrage de l'entraînement Monte Carlo ({NUM_EPISODES:,} épisodes)…")
    monte_carlo_control(NUM_EPISODES)
    print("Entraînement terminé.\n")

    print("Politique optimale (0=hit, 1=stick) :")
    optimal_policy = np.argmax(Q, axis=2)
    print(optimal_policy.T)   # transposé : lignes = player, colonnes = dealer

    plot_value_function()