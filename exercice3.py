import env
import numpy as np
import matplotlib.pyplot as plt
import exercice2


def load_q_star(num_episodes = 800_000) :
    """
    Calcule Q* en appelant directement les fonctions de exercice2.py.

    Parametres : 
    num_episodes : nombre d'épisodes passés à monte_carlo_control

    Retour : 
    Q_star : np.ndarray  shape (10, 21, 2)
    """
    # Réinitialise les structures globales d'exercice2
    exercice2.Q[:]    = 0.0
    exercice2.N_s[:]  = 0
    exercice2.N_sa[:] = 0

    print(f"Calcul de Q* via exercice2.monte_carlo_control ({num_episodes:,} épisodes)…")
    exercice2.monte_carlo_control(num_episodes)
    print("Q* calculé.\n")

    # Retourne une copie pour isoler exercice3 des globales d'exercice2
    return exercice2.Q.copy()


N0  = 100 # Constante ε-greedy 100
DEALER_DIM   = 10
PLAYER_DIM  = 21
ACTION_DIM = 2
NUM_EPISODES_SARSA = 1_000
NUM_RUNS  = 10   # Nombre de runs indépendants pour l'écart-type


def state_to_index(state) :
    """Convertit un état du jeu en indices numpy (dealer-1, player-1)."""
    d = np.clip(state['dealer'] - 1, 0, DEALER_DIM - 1)
    p = np.clip(state['player'] - 1, 0, PLAYER_DIM - 1)
    return d, p


def epsilon_greedy(state, Q, N_s) :
    """
    Choisit une action selon la politique ε-greedy.

    Parametres : 
    state : dict         état courant
    Q     : np.ndarray   fonction de valeur action courante (10, 21, 2)
    N_s   : np.ndarray   compteur de visites d'état (10, 21)

    Retour : 
    action : int  0 = 'hit', 1 = 'stick'
    """
    d, p    = state_to_index(state)
    epsilon = N0 / (N0 + N_s[d, p])
    if np.random.rand() < epsilon:
        return np.random.randint(0, ACTION_DIM)
    return int(np.argmax(Q[d, p]))


def action_to_str(action) :
    """Convertit 0/1 en 'hit'/'stick'."""
    return 'hit' if action == 0 else 'stick'


def compute_mse(Q, Q_star) :
    """
    Calcule l'erreur quadratique moyenne entre Q et Q*.

    Parametres : 
    Q      : np.ndarray  shape (10, 21, 2)  — estimé courant
    Q_star : np.ndarray  shape (10, 21, 2)  — référence MC

    Retour : 
    mse : float
    """
    return float(np.mean(np.square(Q - Q_star)))


def sarsa_lambda(lam, Q_star, num_episodes = NUM_EPISODES_SARSA, record_curve = False) :
    """
    Exécute Sarsa(lam) sur Easy21 pendant `num_episodes` épisodes.

    Paramètres d'exploration/apprentissage identiques à Monte Carlo :
      at = 1/N(st, at)
      εt = N0/(N0 + N(st))

    Parametres : 
    lam          : float    paramètre lam ∈ [0, 1]
    Q_star       : ndarray  fonction de valeur optimale de référence
    num_episodes : int      nombre d'épisodes
    record_curve : bool     si True, enregistre la MSE à chaque épisode

    Retour : 
    Q         : np.ndarray   fonction de valeur estimée finale (10, 21, 2)
    mse_curve : list[float]  courbe de MSE (vide si record_curve=False)
    """
    Q    = np.zeros((DEALER_DIM, PLAYER_DIM, ACTION_DIM))
    N_s  = np.zeros((DEALER_DIM, PLAYER_DIM), dtype=int)
    N_sa = np.zeros((DEALER_DIM, PLAYER_DIM, ACTION_DIM), dtype=int)

    mse_curve = []

    for _ in range(num_episodes):

        # Initialisation des traces d'éligibilité pour l'épisode
        E = np.zeros((DEALER_DIM, PLAYER_DIM, ACTION_DIM))

        # Premier état et première action 
        state = env.init_game()
        action = epsilon_greedy(state, Q, N_s)
        terminate = 0

        while not terminate:
            d, p = state_to_index(state)

            # Mise à jour des compteurs
            N_s[d, p] += 1
            N_sa[d, p, action] += 1

            # Pas d'apprentissage
            alpha = 1.0 / N_sa[d, p, action]

            # Transition
            next_state, reward, terminate = env.step(state, action_to_str(action))

            if terminate:
                # État terminal : Q(s', a') = 0
                delta = reward - Q[d, p, action]
                next_action = None
            else:
                next_action = epsilon_greedy(next_state, Q, N_s)
                nd, np_ = state_to_index(next_state)
                delta = reward + Q[nd, np_, next_action] - Q[d, p, action]

            # Accumulation de la trace
            E[d, p, action] += 1

            # Mise à jour de Q et décroissance des traces
            Q  += alpha * delta * E
            E  *= lam

            # Transition vers le prochain état/action
            state  = next_state
            action = next_action if next_action is not None else action

        if record_curve:
            mse_curve.append(compute_mse(Q, Q_star))

    return Q, mse_curve


def run_multiple(lam, Q_star, num_episodes, num_runs = NUM_RUNS, record_curve = False) :
    """
    Exécute sarsa_lambda plusieurs fois indépendamment et agrège les résultats.

    L'écart-type est calculé entre les runs indépendants, ce qui reflète la
    variabilité de l'algorithme due à l'aléatoire des épisodes.

    Parametres :
    lam          : float   paramètre lam ∈ [0, 1]
    Q_star       : ndarray référence Monte-Carlo
    num_episodes : int     nombre d'épisodes par run
    num_runs     : int     nombre de runs indépendants
    record_curve : bool    si True, agrège les courbes de MSE épisode par épisode

    Retour :
    mse_mean   : float            MSE moyenne finale sur les runs
    mse_std    : float            écart-type de la MSE finale entre runs
    curve_mean : np.ndarray|None  courbe de MSE moyenne par épisode
    curve_std  : np.ndarray|None  courbe d'écart-type par épisode
    """
    mse_finals = []
    all_curves = []

    for run in range(num_runs):
        Q_est, curve = sarsa_lambda(lam, Q_star, num_episodes=num_episodes,
                                    record_curve=record_curve)
        mse_finals.append(compute_mse(Q_est, Q_star))
        if record_curve:
            all_curves.append(curve)

    mse_mean = float(np.mean(mse_finals))
    mse_std  = float(np.std(mse_finals))

    if record_curve:
        curves_array = np.array(all_curves)           # shape (num_runs, num_episodes)
        curve_mean   = np.mean(curves_array, axis=0)
        curve_std    = np.std(curves_array,  axis=0)
    else:
        curve_mean = curve_std = None

    return mse_mean, mse_std, curve_mean, curve_std


def plot_mse_vs_lambda(lambdas, mse_means, mse_stds) :
    """
    Trace la MSE finale en fonction de lam, avec bande d'écart-type
    calculée sur plusieurs runs indépendants.

    Parametres : 
    lambdas   : list[float]  valeurs de lam testées
    mse_means : list[float]  MSE moyenne par lam
    mse_stds  : list[float]  écart-type inter-runs par lam
    """
    lambdas   = np.array(lambdas)
    mse_means = np.array(mse_means)
    mse_stds  = np.array(mse_stds)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(lambdas, mse_means, marker='o', linewidth=2, markersize=6, color='steelblue', label='MSE moyenne')
    ax.fill_between(lambdas, mse_means - mse_stds, mse_means + mse_stds, alpha=0.2, color='steelblue', label=f'± 1 écart-type ({NUM_RUNS} runs)')
    ax.set_xlabel("lam", fontsize=13)
    ax.set_ylabel("MSE  (Q - Q*)²", fontsize=13)
    ax.set_title(f"Sarsa(lam) — MSE finale vs lam  ({NUM_EPISODES_SARSA} épisodes, {NUM_RUNS} runs)", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig("./sarsa_mse_vs_lambda.png", dpi=150)
    plt.show()
    print("Figure sauvegardée : sarsa_mse_vs_lambda.png")


def plot_learning_curves(curve_mean_0, curve_mean_1, curve_std_0, curve_std_1) :
    """
    Trace les courbes d'apprentissage (MSE vs épisode) pour lam=0 et lam=1,
    avec bande d'écart-type calculée sur plusieurs runs indépendants.

    Parametres : 
    curve_mean_0 : np.ndarray  MSE moyenne par épisode pour lam = 0
    curve_mean_1 : np.ndarray  MSE moyenne par épisode pour lam = 1
    curve_std_0  : np.ndarray  écart-type inter-runs par épisode pour lam = 0
    curve_std_1  : np.ndarray  écart-type inter-runs par épisode pour lam = 1
    """
    episodes = np.arange(1, len(curve_mean_0) + 1)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(episodes, curve_mean_0, label="lam = 0", linewidth=2, color='steelblue')
    ax.fill_between(episodes, curve_mean_0 - curve_std_0, curve_mean_0 + curve_std_0, alpha=0.2, color='steelblue')
    ax.plot(episodes, curve_mean_1, label="lam = 1", linewidth=2, color='coral')
    ax.fill_between(episodes, curve_mean_1 - curve_std_1, curve_mean_1 + curve_std_1, alpha=0.2, color='coral')
    ax.set_xlabel("Numéro d'épisode", fontsize=13)
    ax.set_ylabel("MSE  (Q - Q*)²", fontsize=13)
    ax.set_title(f"Sarsa(lam) — Courbe d'apprentissage (lam=0 et lam=1, {NUM_RUNS} runs)", fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig("./sarsa_learning_curves.png", dpi=150)
    plt.show()
    print("Figure sauvegardée : sarsa_learning_curves.png")


if __name__ == '__main__':

    # Calcul de Q* via Monte Carlo (référence)
    Q_star = load_q_star(num_episodes=100_000)

    # Sarsa(lam) pour lam dans {0, 0.1, …, 1} 
    lambdas   = [round(l, 1) for l in np.arange(0.0, 1.1, 0.1)]
    mse_means = []
    mse_stds  = []

    print(f"Exécution de Sarsa(lam) pour chaque valeur de lam ({NUM_RUNS} runs)…")
    for lam in lambdas:
        mean, std, _, _ = run_multiple(lam, Q_star, num_episodes=NUM_EPISODES_SARSA, num_runs=NUM_RUNS, record_curve=False)
        mse_means.append(mean)
        mse_stds.append(std)
        print(f"  lam = {lam:.1f}  —  MSE = {mean:.4f} ± {std:.4f}")

    # Graphe MSE vs lam
    plot_mse_vs_lambda(lambdas, mse_means, mse_stds)

    # Courbes d'apprentissage pour lam=0 et lam=1 
    print(f"\nCalcul des courbes d'apprentissage (lam=0 et lam=1, {NUM_RUNS} runs)…")

    _, _, curve_mean_0, curve_std_0 = run_multiple(0.0, Q_star, num_episodes=100_000, num_runs=NUM_RUNS, record_curve=True)
    _, _, curve_mean_1, curve_std_1 = run_multiple(1.0, Q_star, num_episodes=100_000, num_runs=NUM_RUNS, record_curve=True)

    plot_learning_curves(curve_mean_0, curve_mean_1, curve_std_0, curve_std_1)

    print("\nTerminé.")