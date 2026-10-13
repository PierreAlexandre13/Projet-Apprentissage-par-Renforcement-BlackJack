import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  

import env
import exercice2
import exercice3


N0 = 100 # Constante ε-greedy
NUM_EPISODES = 10_000 # Épisodes Q-Learning (et comparaisons)
NUM_RUNS = 10      # Nombre de runs indépendants pour l'écart-type

DEALER_DIM = 10
PLAYER_DIM = 21
ACTION_DIM = 2   # 0 = hit, 1 = stick


def action_to_str(action) :
    """
    Convertit un entier d'action en chaîne pour l'environnement.

    Paramètres:
    action : int
        0 pour 'hit', 1 pour 'stick'.

    Retour : 
    str
        'hit' ou 'stick'.
    """
    return 'hit' if action == 0 else 'stick'


def state_to_index(state) :
    """
    Convertit un état du jeu en indices de tableau numpy.

    Paramètres:
    state : dict
        Dictionnaire {'dealer': int [1-10], 'player': int [1-21]}.

    Retour:
    (d_idx, p_idx) : tuple[int, int]
        Indices (dealer - 1, player - 1) bornés dans [0, DEALER_DIM-1]
        et [0, PLAYER_DIM-1].
    """
    d = np.clip(state['dealer'] - 1, 0, DEALER_DIM - 1)
    p = np.clip(state['player'] - 1, 0, PLAYER_DIM - 1)
    return int(d), int(p)


def epsilon_greedy(state, Q, N_s) :
    """
    Choisit une action selon la politique ε-greedy avec ε variable.

    εt = N0 / (N0 + N(s))

    Paramètres:
    state : dict
        État courant du jeu.
    Q     : np.ndarray de forme (10, 21, 2)
        Fonction de valeur action courante.
    N_s   : np.ndarray de forme (10, 21)
        Compteur de visites d'état.

    Retour:
    action : int
        0 = hit, 1 = stick.
    """
    d, p    = state_to_index(state)
    epsilon = N0 / (N0 + N_s[d, p])
    if np.random.rand() < epsilon:
        return np.random.randint(0, ACTION_DIM)
    return int(np.argmax(Q[d, p]))


def compute_mse(Q, Q_star) :
    """
    Calcule la MSE entre l'estimation courante et Q*.

    MSE = mean_{s,a} (Q(s,a) - Q*(s,a))²

    Paramètres
    ----------
    Q      : np.ndarray de forme (10, 21, 2)  Estimé courant.
    Q_star : np.ndarray de forme (10, 21, 2)  Référence Monte-Carlo.

    Retour
    ------
    float
        Valeur de la MSE.
    """
    return float(np.mean(np.square(Q - Q_star)))


def load_q_star(num_episodes = 800_000) :
    """
    Calcule Q* en appelant directement les fonctions de exercice2.py.

    Paramètres:
    num_episodes : int
        Nombre d'épisodes passés à monte_carlo_control.

    Retour:
    Q_star : np.ndarray de forme (10, 21, 2)
        Copie de la fonction de valeur optimale calculée par Monte-Carlo.
    """
    exercice2.Q[:]    = 0.0
    exercice2.N_s[:]  = 0
    exercice2.N_sa[:] = 0

    print(f"Calcul de Q* via Monte-Carlo ({num_episodes:,} épisodes)…")
    exercice2.monte_carlo_control(num_episodes)
    print("Q* calculé.\n")

    return exercice2.Q.copy()


def q_learning(Q_star, num_episodes = NUM_EPISODES, record_curve = False) :
    """
    Exécute l'algorithme Q-Learning (Watkins, 1989) sur Easy21.

    Algorithme off-policy : la mise à jour cible la politique greedy.
    Règle de mise à jour :
        Q(s, a) ← Q(s, a) + alpha [r + gamma · max_a' Q(s', a') - Q(s, a)]
    avec :
      - gamma = 1 (pas d'actualisation),
      - alpha_t = 1 / N(s, a)  (pas d'apprentissage variable),
      - εt = N0 / (N0 + N(s))  (exploration décroissante).

    Paramètres:
    Q_star       : np.ndarray de forme (10, 21, 2)
        Valeurs de référence pour le calcul de la MSE.
    num_episodes : int
        Nombre d'épisodes d'entraînement.
    record_curve : bool
        Si True, enregistre la MSE à chaque épisode.

    Retour:
    Q         : np.ndarray de forme (10, 21, 2)
        Fonction de valeur action apprise.
    mse_curve : list[float]
        Courbe de MSE épisode par épisode (vide si record_curve=False).
    """
    Q    = np.zeros((DEALER_DIM, PLAYER_DIM, ACTION_DIM))
    N_s  = np.zeros((DEALER_DIM, PLAYER_DIM), dtype=int)
    N_sa = np.zeros((DEALER_DIM, PLAYER_DIM, ACTION_DIM), dtype=int)

    mse_curve = []

    for _ in range(num_episodes):

        state    = env.init_game()
        terminal = 0

        while not terminal:
            d, p   = state_to_index(state)

            # Choix de l'action avec la politique ε-greedy (comportement)
            action = epsilon_greedy(state, Q, N_s)

            # Mise à jour des compteurs
            N_s[d, p] += 1
            N_sa[d, p, action] += 1

            # Pas d'apprentissage variable
            alpha = 1.0 / N_sa[d, p, action]
            
            # Transition dans l'environnement
            next_state, reward, terminal = env.step(state, action_to_str(action))

            # Cible Q-Learning : max sur les actions de l'état suivant
            if terminal:
                # État terminal : max_a' Q(s', a') = 0
                target = reward
            else:
                nd, np_ = state_to_index(next_state)
                target  = reward + np.max(Q[nd, np_])

            # Mise à jour de Q(s, a) — règle off-policy
            Q[d, p, action] += alpha * (target - Q[d, p, action])
            state  = next_state

        if record_curve:
            mse_curve.append(compute_mse(Q, Q_star))

    return Q, mse_curve


def run_multiple_ql(Q_star, num_episodes, num_runs = NUM_RUNS, record_curve = False) :
    """
    Exécute q_learning plusieurs fois indépendamment et agrège les résultats.

    L'écart-type est calculé entre les runs indépendants, ce qui reflète la
    variabilité de l'algorithme due à l'aléatoire des épisodes.

    Paramètres:
    Q_star       : np.ndarray  Référence Monte-Carlo, forme (10, 21, 2).
    num_episodes : int         Nombre d'épisodes par run.
    num_runs     : int         Nombre de runs indépendants.
    record_curve : bool        Si True, agrège les courbes de MSE épisode par épisode.

    Retour:
    mse_mean   : float            MSE moyenne finale sur les runs.
    mse_std    : float            Écart-type de la MSE finale entre runs.
    curve_mean : np.ndarray|None  Courbe de MSE moyenne par épisode.
    curve_std  : np.ndarray|None  Courbe d'écart-type par épisode.
    Q_last     : np.ndarray       Q appris lors du dernier run (pour la surface 3D).
    """
    mse_finals = []
    all_curves = []
    Q_last     = None

    for _ in range(num_runs):
        Q_est, curve = q_learning(Q_star, num_episodes=num_episodes,
                                  record_curve=record_curve)
        mse_finals.append(compute_mse(Q_est, Q_star))
        Q_last = Q_est
        if record_curve:
            all_curves.append(curve)

    mse_mean = float(np.mean(mse_finals))
    mse_std  = float(np.std(mse_finals))

    if record_curve:
        curves_array = np.array(all_curves)          # shape (num_runs, num_episodes)
        curve_mean   = np.mean(curves_array, axis=0)
        curve_std    = np.std(curves_array,  axis=0)
    else:
        curve_mean = curve_std = None

    return mse_mean, mse_std, curve_mean, curve_std, Q_last


def plot_comparison_curves(curve_mean_ql, curve_std_ql, curve_mean_s0, curve_std_s0, curve_mean_s1, curve_std_s1, num_episodes = NUM_EPISODES) :
    """
    Trace sur un même graphique les courbes d'apprentissage (moyenne ± écart-type)
    pour Q-Learning, Sarsa(λ=0) et Sarsa(λ=1).

    Paramètres:
    curve_mean_ql : np.ndarray  MSE moyenne par épisode — Q-Learning.
    curve_std_ql  : np.ndarray  Écart-type par épisode  — Q-Learning.
    curve_mean_s0 : np.ndarray  MSE moyenne par épisode — Sarsa(λ=0).
    curve_std_s0  : np.ndarray  Écart-type par épisode  — Sarsa(λ=0).
    curve_mean_s1 : np.ndarray  MSE moyenne par épisode — Sarsa(λ=1).
    curve_std_s1  : np.ndarray  Écart-type par épisode  — Sarsa(λ=1).
    num_episodes  : int         Nombre d'épisodes (pour le titre).
    """
    episodes = np.arange(1, len(curve_mean_ql) + 1)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(episodes, curve_mean_ql, label="Q-Learning",   linewidth=2, color='seagreen')
    ax.fill_between(episodes, curve_mean_ql - curve_std_ql, curve_mean_ql + curve_std_ql, alpha=0.2, color='seagreen')

    ax.plot(episodes, curve_mean_s0, label="Sarsa(λ = 0)", linewidth=2, color='steelblue')
    ax.fill_between(episodes, curve_mean_s0 - curve_std_s0, curve_mean_s0 + curve_std_s0, alpha=0.2, color='steelblue')

    ax.plot(episodes, curve_mean_s1, label="Sarsa(λ = 1)", linewidth=2, color='coral')
    ax.fill_between(episodes, curve_mean_s1 - curve_std_s1, curve_mean_s1 + curve_std_s1, alpha=0.2, color='coral')

    ax.set_xlabel("Numéro d'épisode", fontsize=13)
    ax.set_ylabel("MSE  (Q - Q*)²",   fontsize=13)
    ax.set_title(f"Comparaison Q-Learning vs Sarsa(λ) — Courbes d'apprentissage\n({num_episodes:,} épisodes, {NUM_RUNS} runs)", fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig("./qlearning_comparison_curves.png", dpi=150)
    plt.show()
    print("Figure sauvegardée : qlearning_comparison_curves.png")


def plot_value_function(Q, num_episodes = NUM_EPISODES) :
    """
    Trace la fonction de valeur optimale V*(s) = max_a Q(s, a)
    sous forme de surface 3D.

    Paramètres:
    Q            : np.ndarray de forme (10, 21, 2)
        Fonction de valeur action apprise par Q-Learning.
    num_episodes : int
        Nombre d'épisodes (pour le titre).
    """
    V_star = np.max(Q, axis=2) # shape (10, 21)

    dealer_range = np.arange(1, DEALER_DIM + 1) # 1..10
    player_range = np.arange(1, PLAYER_DIM + 1) # 1..21

    D, P = np.meshgrid(dealer_range, player_range) # shape (21, 10)
    Z = V_star[D - 1, P - 1] # shape (21, 10)

    fig = plt.figure(figsize=(12, 7))
    ax  = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(D, P, Z, cmap='viridis', edgecolor='none', alpha=0.9)

    ax.set_xlabel("Carte du croupier (1-10)", labelpad=10)
    ax.set_ylabel("Somme du joueur (1-21)",   labelpad=10)
    ax.set_zlabel("V*(s)",                    labelpad=10)
    ax.set_title(f"Fonction de valeur optimale V*(s) — Q-Learning Easy21\n({num_episodes:,} épisodes)", pad=15)

    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label="V*(s)")
    plt.tight_layout()
    plt.savefig("./qlearning_value_function.png", dpi=150)
    plt.show()
    print("Figure sauvegardée : qlearning_value_function.png")


if __name__ == '__main__':

    # Calcul de Q* via Monte-Carlo (référence)
    Q_star = load_q_star(num_episodes=800_000)

    # Q-Learning : courbe d'apprentissage moyenne + écart-type sur NUM_RUNS runs
    print(f"Exécution de Q-Learning ({NUM_EPISODES:,} épisodes, {NUM_RUNS} runs)…")
    mse_ql, std_ql, curve_mean_ql, curve_std_ql, Q_ql = run_multiple_ql(Q_star, num_episodes=NUM_EPISODES, num_runs=NUM_RUNS, record_curve=True)
    print(f"  MSE finale Q-Learning : {mse_ql:.4f} ± {std_ql:.4f}\n")

    # Sarsa(λ=0) et Sarsa(λ=1) : courbes d'apprentissage via run_multiple d'exercice3
    print(f"Exécution de Sarsa(λ=0) ({NUM_EPISODES:,} épisodes, {NUM_RUNS} runs)…")
    mse_s0, std_s0, curve_mean_s0, curve_std_s0 = exercice3.run_multiple(0.0, Q_star, num_episodes=NUM_EPISODES, num_runs=NUM_RUNS, record_curve=True)
    print(f"  MSE finale Sarsa(λ=0) : {mse_s0:.4f} ± {std_s0:.4f}\n")

    print(f"Exécution de Sarsa(λ=1) ({NUM_EPISODES:,} épisodes, {NUM_RUNS} runs)…")
    mse_s1, std_s1, curve_mean_s1, curve_std_s1 = exercice3.run_multiple(1.0, Q_star, num_episodes=NUM_EPISODES, num_runs=NUM_RUNS, record_curve=True)
    print(f"  MSE finale Sarsa(λ=1) : {mse_s1:.4f} ± {std_s1:.4f}\n")

    # Graphe comparatif avec bandes d'écart-type
    plot_comparison_curves(curve_mean_ql, curve_std_ql, curve_mean_s0, curve_std_s0, curve_mean_s1, curve_std_s1, NUM_EPISODES)

    # Surface 3D de la politique apprise (sur un run long pour la qualité visuelle)
    print("Calcul d'un run long pour la surface 3D (300 000 épisodes)…")
    Q_ql_long, _ = q_learning(Q_star, num_episodes=800_000, record_curve=False)
    print("Politique optimale Q-Learning (0=hit, 1=stick) :")
    print(np.argmax(Q_ql_long, axis=2).T)
    
    plot_value_function(Q_ql_long, num_episodes=800_000)

    print("\nTerminé.")