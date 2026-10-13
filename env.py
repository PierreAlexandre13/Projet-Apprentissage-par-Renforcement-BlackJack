import random


def step(state, action) : 
    """
    Entrées : 
    - état s (première carte du croupier 1-10, et somme du joueur 1-21)
    - action a (hit ou stick)
    Retour : 
    - chantillon de l'état suivant s' (potentiellement terminal si la partie est terminée)
    - récompense r
    - bool representant l'état terminale
    
    Env utilise pour model-free : pas de matrice de transition
    Pas d'actualisation (no discounting)

    Regles : 
    - Jeu de cartes infini 
    - Chaque tirage produit une valeur entre 1 et 10 (distribution uniforme) avec une couleur : rouge (probabilité 1/3) ou noire (probabilité 2/3).
    — Il n'y a ni as ni figures dans ce jeu.
    — En début de partie, le joueur et le croupier tirent chacun une carte noire (entièrement observable).
    — À chaque tour, le joueur peut tirer (hit) ou rester (stick).
    — Les cartes noires sont ajoutées à la somme du joueur ; les cartes rouges sont soustraites.
    — Si la somme du joueur dépasse 21 ou passe en dessous de 1, le joueur est éliminé et perd (récompense -1).
    — Si le joueur reste, le croupier joue : il reste toujours sur une somme ≥ 17 et tire sinon. Si le croupier est éliminé, le joueur gagne ;
    sinon, la victoire (récompense +1), la défaite (-1) ou l'égalité (0) est déterminée par celui qui a la plus grande somme.

    """

    carteCroupier = state['dealer']
    sommeJoueur = state['player']

    if (action == 'stick') : 
        while (carteCroupier <= 17) : 
            randomNb = random.randint(1, 10)
            randomColor = random.randint(1, 3) #1:rouge, 2et3:noir
            #print("nb : ", randomNb, "/ color : ", randomColor)
            if (randomColor == 1) : 
                carteCroupier -= randomNb
            else : 
                carteCroupier += randomNb
            
            if (carteCroupier > 21 or carteCroupier < 1) : # or carteCroupier < 1
                return ({'dealer': carteCroupier, 'player': sommeJoueur}, 1, 1) #le joueur gagne
            
        if (carteCroupier == sommeJoueur) : 
            return ({'dealer': carteCroupier, 'player': sommeJoueur}, 0, 1) #egalite
        if (carteCroupier > sommeJoueur) : 
            return ({'dealer': carteCroupier, 'player': sommeJoueur}, -1, 1) #joueur perd
        if (carteCroupier < sommeJoueur) : 
            return ({'dealer': carteCroupier, 'player': sommeJoueur}, 1, 1) #joueur gagne
        

    if (action == 'hit') : 
        randomNb = random.randint(1, 10)
        randomColor = random.randint(1, 3) #1:rouge, 2et3:noir
        #print("nb : ", randomNb, "/ color : ", randomColor)
        if (randomColor == 1) : 
            sommeJoueur -= randomNb
        else : 
            sommeJoueur += randomNb
        
        if (sommeJoueur > 21 or sommeJoueur < 1) : 
            return ({'dealer': carteCroupier, 'player': sommeJoueur}, -1, 1) #le joueur est éliminé et perd (récompense -1)
        
        return ({'dealer': carteCroupier, 'player': sommeJoueur}, 0, 0)


    
def draw_card() : 
    """
    simule un tirage 
    valeur entre 1 et 10 avec couleur rouge ou noire
    """
    randomNb = random.randint(1, 10)
    randomColor = random.randint(1, 3) #1:rouge, 2et3:noir
    return (randomNb, randomColor)

def init_game():
    """
    initialise l'état de départ en faisant tirer une carte noire à chaque joueur
    """
    state = {'dealer': 0, 'player': 0}
    valeur, _ = draw_card()
    state['dealer'] = valeur
    valeur, _ = draw_card()
    state['player'] = valeur
    return state



if __name__ == '__main__' :
    # Exemple d’interface attendue
    #state = {'dealer': 5, 'player': 14}

    state = init_game()

    print(state)

    # Action ’hit’ : le joueur tire une carte
    next_state, reward, terminal = step(state, action='hit')

    print(next_state, " ", reward, " ", terminal)

    # Action ’stick’ : le croupier joue en entier, renvoie le resultat
    next_state, reward, terminal = step(state, action='stick')

    print(next_state, " ", reward, " ", terminal)