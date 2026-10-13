import numpy as np
import matplotlib.pyplot as plt

import env
import exercice2


EPSILON            = 0.05  # Exploration constante
ALPHA              = 0.01  # Pas d'apprentissage constant
NUM_EPISODES       = 1_000 # Épisodes pour la MSE finale vs lambda
NUM_EPISODES_CURVE = 1_000 # Épisodes pour les courbes d'apprentissage
NUM_RUNS           = 10      # Nombre de runs indépendants pour l'écart-type

DEALER_DIM = 10
PLAYER_DIM = 21
ACTION_DIM = 2  # 0 = hit, 1 = stick

# Intervalles de codage grossier (bornes incluses)
DEALER_INTERVALS = [(1, 4), (4, 7), (7, 10)] # 3 intervalles
PLAYER_INTERVALS = [(1, 6), (4, 9), (7, 12), (10, 15), (13, 18), (16, 21)] # 6 intervalles
# Dimension du vecteur de caractéristiques : 3 * 6 * 2 = 36
FEATURE_DIM = len(DEALER_INTERVALS) * len(PLAYER_INTERVALS) * ACTION_DIM


def action_to_str(action) :
    """
    Convertit 0/1 en 'hit'/'stick'
    """
    return 'hit' if action == 0 else 'stick'


def feature_vector(state, action) :
    """
    Construit le vecteur de caractéristiques binaires phi(s, a).

    Le vecteur a dimension 3 * 6 * 2 = 36.
    Pour chaque combinaison (intervalle_dealer, intervalle_player, action),
    le bit correspondant vaut 1 si et seulement si :
      - la carte du croupier appartient à l'intervalle_dealer,
      - la somme du joueur appartient à l'intervalle_player,
      - l'action courante correspond à l'indice action.

    Paramètres:
    state : dict
        Dictionnaire {'dealer': int [1-10], 'player': int [1-21]}.
    action : int
        0 = hit, 1 = stick.

    Retour:
    phi : np.ndarray de forme (36,)
        Vecteur de caractéristiques binaires.
    """
    d   = state['dealer']
    p   = state['player']
    phi = np.zeros(FEATURE_DIM, dtype=float)

    idx = 0
    for di, (d_lo, d_hi) in enumerate(DEALER_INTERVALS):
        for pi, (p_lo, p_hi) in enumerate(PLAYER_INTERVALS):
            for a in range(ACTION_DIM):
                if d_lo <= d <= d_hi and p_lo <= p <= p_hi and a == action:
                    phi[idx] = 1.0
                idx += 1

    return phi


def q_approx(state, action, theta) :
    """
    Calcule la valeur Q(s, a) par approximation linéaire.

    Q(s, a) = phi(s, a)^T * theta

    Paramètres:
    state  : dict       État courant du jeu.
    action : int        Action (0=hit, 1=stick).
    theta  : np.ndarray Vecteur de paramètres de forme (36,).

    Retour:
    float
        Valeur estimée de Q(s, a).
    """
    return float(np.dot(feature_vector(state, action), theta))


def epsilon_greedy(state, theta, epsilon = EPSILON) :
    """
    Choisit une action selon la politique ε-greedy avec exploration constante.

    Paramètres:
    state   : dict       État courant.
    theta   : np.ndarray Vecteur de paramètres courant.
    epsilon : float      Probabilité d'exploration (défaut = 0.05).

    Retour:
    action : int   0 = hit, 1 = stick.
    """
    if np.random.rand() < epsilon:
        return np.random.randint(0, ACTION_DIM)
    q_values = [q_approx(state, a, theta) for a in range(ACTION_DIM)]
    return int(np.argmax(q_values))


def compute_mse(theta, Q_star) :
    """
    Calcule la MSE entre l'approximation linéaire et Q*.

    MSE = (1/N) * sum_{s,a} (Q(s,a) - Q*(s,a))^2
    avec N = DEALER_DIM * PLAYER_DIM * ACTION_DIM.

    Paramètres:
    theta  : np.ndarray  Vecteur de paramètres courant, forme (36,).
    Q_star : np.ndarray  Valeurs de référence Monte-Carlo, forme (10, 21, 2).

    Retour:
    float : Valeur de la MSE.
    """
    mse = 0.0
    for d_idx in range(DEALER_DIM):
        for p_idx in range(PLAYER_DIM):
            for a in range(ACTION_DIM):
                state = {'dealer': d_idx + 1, 'player': p_idx + 1}
                q_hat = q_approx(state, a, theta)
                mse  += (q_hat - Q_star[d_idx, p_idx, a]) ** 2
    return mse / (DEALER_DIM * PLAYER_DIM * ACTION_DIM)


def sarsa_lambda_linear(lam, Q_star, num_episodes = NUM_EPISODES, record_curve = False) :
    """
    Exécute Sarsa(lam) avec approximation linéaire sur Easy21.

    Règles d'entraînement :
      - Exploration constante ε = 0,05.
      - Pas d'apprentissage constant alpha = 0,01.
      - Traces d'éligibilité accumulées : e ← gamma*lam·e + phi(s, a).
      - Mise à jour des paramètres : θ ← θ + alpha · δ · e.
      - gamma = 1 (pas d'actualisation).

    Paramètres:
    lam          : float     Paramètre lam ∈ [0, 1].
    Q_star       : ndarray   Fonction de valeur optimale de référence (10,21,2).
    num_episodes : int       Nombre d'épisodes d'entraînement.
    record_curve : bool      Si True, enregistre la MSE à chaque épisode.

    Retour:
    theta     : np.ndarray   Vecteur de paramètres final, forme (36,).
    mse_curve : list[float]  Courbe de MSE épisode par épisode (vide si record_curve=False).
    """
    theta     = np.zeros(FEATURE_DIM)
    mse_curve = []

    for episode_id in range(num_episodes):

        # Initialisation des traces d'éligibilité pour l'épisode
        e = np.zeros(FEATURE_DIM)

        # Premier état et première action
        state = env.init_game()
        action = epsilon_greedy(state, theta)
        terminate = 0

        while not terminate:
            phi_sa = feature_vector(state, action)

            # Transition
            next_state, reward, terminate = env.step(state, action_to_str(action))

            if terminate:
                # État terminal : Q(s', a') = 0
                delta = reward - q_approx(state, action, theta)
                next_action = None
            else:
                next_action = epsilon_greedy(next_state, theta)
                delta = (reward + q_approx(next_state, next_action, theta) - q_approx(state, action, theta))

            # Accumulation de la trace d'éligibilité
            e = lam * e + phi_sa

            # Mise à jour des paramètres
            theta += ALPHA * delta * e
            
            # Transition vers le prochain état/action
            state  = next_state
            action = next_action if next_action is not None else action

        if (episode_id + 1) % 1_000 == 0:
            print(f"  Épisode {episode_id + 1:>10} / {num_episodes}")

        if record_curve:
            mse_curve.append(compute_mse(theta, Q_star))

    return theta, mse_curve


def run_multiple(lam, Q_star, num_episodes, num_runs = NUM_RUNS, record_curve = False) :
    """
    Exécute sarsa_lambda_linear plusieurs fois indépendamment et agrège les résultats.

    L'écart-type est calculé entre les runs indépendants, ce qui reflète la
    variabilité de l'algorithme due à l'aléatoire des épisodes.

    Paramètres:
    lam          : float   Paramètre lam ∈ [0, 1].
    Q_star       : ndarray Référence Monte-Carlo, forme (10, 21, 2).
    num_episodes : int     Nombre d'épisodes par run.
    num_runs     : int     Nombre de runs indépendants.
    record_curve : bool    Si True, agrège les courbes de MSE épisode par épisode.

    Retour:
    mse_mean   : float            MSE moyenne finale sur les runs.
    mse_std    : float            Écart-type de la MSE finale entre runs.
    curve_mean : np.ndarray|None  Courbe de MSE moyenne par épisode.
    curve_std  : np.ndarray|None  Courbe d'écart-type par épisode.
    """
    mse_finals = []
    all_curves = []

    for _ in range(num_runs):
        theta, curve = sarsa_lambda_linear(lam, Q_star, num_episodes=num_episodes,
                                           record_curve=record_curve)
        mse_finals.append(compute_mse(theta, Q_star))
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

    return mse_mean, mse_std, curve_mean, curve_std


def load_q_star(num_episodes = 800_000) :
    """
    Calcule Q* en appelant directement les fonctions de exercice2.py.

    Paramètres: 
    num_episodes : int
        Nombre d'épisodes passés à monte_carlo_control.

    Retour:
    Q_star : np.ndarray  forme (10, 21, 2)
        Copie de la fonction de valeur optimale calculée par Monte-Carlo.
    """
    exercice2.Q[:]    = 0.0
    exercice2.N_s[:]  = 0
    exercice2.N_sa[:] = 0

    print(f"Calcul de Q* via Monte-Carlo ({num_episodes:,} épisodes)…")
    exercice2.monte_carlo_control(num_episodes)
    print("Q* calculé.\n")

    return exercice2.Q.copy()


def plot_mse_vs_lambda(lambdas, mse_means, mse_stds) :
    """
    Trace la MSE finale en fonction de lam, avec bande d'écart-type
    calculée sur plusieurs runs indépendants.

    Paramètres:
    lambdas   : list[float]  Valeurs de lam testées.
    mse_means : list[float]  MSE moyenne par lam.
    mse_stds  : list[float]  Écart-type inter-runs par lam.
    """
    lambdas   = np.array(lambdas)
    mse_means = np.array(mse_means)
    mse_stds  = np.array(mse_stds)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(lambdas, mse_means, marker='o', linewidth=2, markersize=6, color='steelblue', label='MSE moyenne')
    ax.fill_between(lambdas, mse_means - mse_stds, mse_means + mse_stds, alpha=0.2, color='steelblue', label=f'± 1 écart-type ({NUM_RUNS} runs)')
    ax.set_xlabel("lam", fontsize=13)
    ax.set_ylabel("MSE  (Q - Q*)²", fontsize=13)
    ax.set_title(f"Sarsa(lam) linéaire — MSE finale vs lam  ({NUM_EPISODES} épisodes, {NUM_RUNS} runs)", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig("./linear_mse_vs_lambda.png", dpi=150)
    plt.show()
    print("Figure sauvegardée : linear_mse_vs_lambda.png")


def plot_learning_curves(curve_mean_0, curve_mean_1, curve_std_0, curve_std_1) :
    """
    Trace les courbes d'apprentissage (MSE vs épisode) pour lam=0 et lam=1,
    avec bande d'écart-type calculée sur plusieurs runs indépendants.

    Paramètres:
    curve_mean_0 : np.ndarray  MSE moyenne par épisode pour lam = 0.
    curve_mean_1 : np.ndarray  MSE moyenne par épisode pour lam = 1.
    curve_std_0  : np.ndarray  Écart-type inter-runs par épisode pour lam = 0.
    curve_std_1  : np.ndarray  Écart-type inter-runs par épisode pour lam = 1.
    """
    episodes = np.arange(1, len(curve_mean_0) + 1)

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(episodes, curve_mean_0, label="lam = 0", linewidth=2, color='steelblue')
    ax.fill_between(episodes, curve_mean_0 - curve_std_0, curve_mean_0 + curve_std_0,alpha=0.2, color='steelblue')

    ax.plot(episodes, curve_mean_1, label="lam = 1", linewidth=2, color='coral')
    ax.fill_between(episodes,  curve_mean_1 - curve_std_1,  curve_mean_1 + curve_std_1, alpha=0.2, color='coral')

    ax.set_xlabel("Numéro d'épisode", fontsize=13)
    ax.set_ylabel("MSE  (Q - Q*)²", fontsize=13)
    ax.set_title(f"Sarsa(lam) linéaire — Courbe d'apprentissage (lam=0 et lam=1, {NUM_RUNS} runs)", fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig("./linear_learning_curves.png", dpi=150)
    plt.show()
    print("Figure sauvegardée : linear_learning_curves.png")


if __name__ == '__main__':

    # Calcul de Q* via Monte-Carlo (référence)
    Q_star = load_q_star(num_episodes=800_000)

    # Sarsa(lam) linéaire pour lam ∈ {0, 0.1, …, 1} — MSE finale + écart-type
    lambdas   = [round(l, 1) for l in np.arange(0.0, 1.1, 0.1)]
    mse_means = []
    mse_stds  = []

    print(f"Exécution de Sarsa(lam) linéaire pour chaque valeur de lam ({NUM_RUNS} runs)…")
    for lam in lambdas:
        mean, std, _, _ = run_multiple(lam, Q_star,num_episodes=NUM_EPISODES, num_runs=NUM_RUNS, record_curve=False)
        mse_means.append(mean)
        mse_stds.append(std)
        print(f"  lam = {lam:.1f}  →  MSE = {mean:.4f} ± {std:.4f}")

    # Graphe MSE vs lam
    plot_mse_vs_lambda(lambdas, mse_means, mse_stds)

    # Courbes d'apprentissage pour lam = 0 et lam = 1
    print(f"\nCalcul des courbes d'apprentissage (lam=0 et lam=1, {NUM_RUNS} runs)…")

    _, _, curve_mean_0, curve_std_0 = run_multiple(0.0, Q_star, num_episodes=NUM_EPISODES_CURVE, num_runs=NUM_RUNS, record_curve=True)
    _, _, curve_mean_1, curve_std_1 = run_multiple(1.0, Q_star, num_episodes=NUM_EPISODES_CURVE, num_runs=NUM_RUNS, record_curve=True)

    plot_learning_curves(curve_mean_0, curve_mean_1, curve_std_0, curve_std_1)

    print("\nTerminé.")


    