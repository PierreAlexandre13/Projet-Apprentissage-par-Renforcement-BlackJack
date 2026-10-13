# Apprentissage par renforcement : Blackjack (Easy21)

Projet d'apprentissage par renforcement en Python. Une variante simplifiée du blackjack, appelée **Easy21**, sert d'environnement pour comparer plusieurs algorithmes : contrôle Monte-Carlo, Sarsa(λ), Sarsa(λ) avec approximation linéaire et Q-Learning.

Le détail de l'analyse et des résultats se trouve dans `Rapport Projet RL.pdf`.

## Règles du jeu (Easy21)

- Les cartes sont tirées dans un paquet infini : valeur uniforme entre 1 et 10, de couleur **noire** (probabilité 2/3) ou **rouge** (probabilité 1/3). Il n'y a ni as ni figures.
- Les cartes noires s'**ajoutent** à la somme, les cartes rouges se **soustraient**.
- Au départ, le joueur et le croupier tirent chacun une carte noire.
- À chaque tour, le joueur choisit `hit` (tirer) ou `stick` (rester).
- Le joueur perd (récompense -1) si sa somme dépasse 21 ou tombe sous 1.
- Si le joueur reste, le croupier joue (il tire tant que sa somme est insuffisante). La partie se termine par une victoire (+1), une défaite (-1) ou une égalité (0).
- Il n'y a pas d'actualisation (γ = 1).

L'état est le couple (carte visible du croupier, somme du joueur).

## Contenu du dépôt

| Fichier | Description |
|---|---|
| `env.py` | Environnement Easy21 (`init_game()`, `step(state, action)`) |
| `exercice2.py` | Contrôle Monte-Carlo (ε-greedy, ε = N₀/(N₀+N(s))). Calcule Q* et trace V*(s) |
| `exercice3.py` | Sarsa(λ) tabulaire : MSE par rapport à Q* en fonction de λ, courbes d'apprentissage (λ = 0 et λ = 1) |
| `exercice5.py` | Sarsa(λ) avec approximation linéaire (codage grossier, 36 caractéristiques binaires, ε et α constants) |
| `exercice6.py` | Q-Learning, comparé à Sarsa(λ = 0) et Sarsa(λ = 1) |
| `Rapport Projet RL.pdf` | Rapport du projet |

## Prérequis

- Python 3
- `numpy`
- `matplotlib`

```bash
pip install numpy matplotlib
```

## Utilisation

Chaque exercice se lance indépendamment depuis la racine du dépôt :

```bash
python env.py          # exemple d'utilisation de l'environnement
python exercice2.py    # Monte-Carlo
python exercice3.py    # Sarsa(λ)
python exercice5.py    # Sarsa(λ) linéaire
python exercice6.py    # Q-Learning
```

Les scripts affichent des graphiques et les enregistrent au format PNG dans le dossier courant :

| Script | Figures générées |
|---|---|
| `exercice2.py` | `value_function.png` |
| `exercice3.py` | `sarsa_mse_vs_lambda.png`, `sarsa_learning_curves.png` |
| `exercice5.py` | `linear_mse_vs_lambda.png`, `linear_learning_curves.png` |
| `exercice6.py` | `qlearning_comparison_curves.png`, `qlearning_value_function.png` |

> ⏱️ Les exercices 3, 5 et 6 calculent d'abord Q* par Monte-Carlo (`exercice2.py`), puis exécutent plusieurs runs indépendants. L'exécution peut prendre plusieurs minutes. Pour accélérer les tests, réduisez `NUM_EPISODES` et `NUM_RUNS` en tête des fichiers.

## Algorithmes

- **Monte-Carlo control** : mise à jour de Q(s, a) à partir du retour complet de chaque épisode, avec un pas d'apprentissage 1/N(s, a).
- **Sarsa(λ)** : méthode TD *on-policy* avec traces d'éligibilité.
- **Sarsa(λ) linéaire** : Q(s, a) = φ(s, a)ᵀθ, avec un codage grossier en intervalles qui se chevauchent (3 intervalles pour le croupier, 6 pour le joueur, 2 actions).
- **Q-Learning** : méthode TD *off-policy* dont la cible utilise max Q(s', a').

Les méthodes TD sont évaluées par l'erreur quadratique moyenne (MSE) par rapport à la fonction de valeur Q* obtenue par Monte-Carlo.
