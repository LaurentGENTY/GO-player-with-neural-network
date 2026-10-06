# Relance du joueur de Go 9x9 — design

- **Date** : 2026-10-06
- **Statut** : validé en conversation, en attente de relecture de la spec
- **Repo** : `LaurentGENTY/GO-player-with-neural-network` (projet ENSEIRB-MATMECA 2020, Laurent Genty & Johan Chataigner)

## 1. Intention

Relancer le joueur de Go 9x9 de 2020 comme side project, **en gardant la stack d'origine (Python + Keras/TensorFlow) au cœur**. Pas de réécriture dans un autre langage.

Ce qui est livré :
1. le projet tourne sur un Python et un TensorFlow modernes ;
2. un nouveau joueur **MCTS** qui réutilise le CNN de 2020 comme fonction de valeur ;
3. des **médias de showcase** reproductibles (GIF/MP4 de parties et tableau de matchs) intégrés au README et réutilisables tels quels sur le portfolio (`~/perso/portfolio`).

L'histoire racontée : *« même réseau, meilleure recherche : Alpha-Beta 2020 vs MCTS 2026 »*.

### Contraintes (énoncées par Laurent)
- La stack d'origine reste le cœur : Python et ML Keras.
- Le jeu dans le navigateur est optionnel et hors périmètre ici. Le projet n'a pas besoin d'être jouable en ligne.
- Le livrable obligatoire, ce sont les GIF/vidéos pour le README et le portfolio.

### Hors périmètre
- Navigateur / Pyodide.
- Ré-entraînement du réseau, réseau de policy.
- Intégration à un « playground » commun avec le Gomoku.
- Modifications du dépôt portfolio.

## 2. État des lieux (exploration du 2026-10-06)

| Sujet | Constat |
|---|---|
| Moteur `GO/Goban.py` | Fonctionne en Python 3.14 avec numpy seul. ~1 574 coups/s, ~13 parties aléatoires/s. |
| Modèle `model.json` + `model.h5` | Sauvegardé avec Keras 2.3.0-tf. **`model_from_json` échoue** avec TF 2.21 / Keras 3 (`Could not locate class 'Sequential'`). Si l'on **reconstruit l'architecture en code** et charge les poids couche par couche depuis le `.h5` via `h5py`, **ça fonctionne** : plateau vide → `[0.600, 0.400]`. Débit : ~300 évaluations/s unitaires, ~9 000 plateaux/s par lots de 64. |
| Architecture du CNN | Entrée `(9,9,2)` → 3 × [Conv2D 64, 5×5, same, relu + BatchNorm] → Flatten → Dropout 0.5 → Dense 32 relu → Dense 2 softmax. Sortie : `[P(Noir gagne), P(Blanc gagne)]`. |
| Contrat joueur | `PlayerInterface` : `getPlayerName`, `getPlayerMove() -> "E5"/"PASS"`, `playOpponentMove(move)`, `newGame(color)`, `endGame(winner)`. |
| Runners | `localGame.py` (arbitre console) et `visualGame.ipynb` (rendu `Board.svg()`, avec export PNG via cairosvg déjà prévu en commentaire). |
| GnuGo | Bridge GTP dans `GnuGo.py`. Binaire absent en local. Formule Homebrew : `gnu-go`. |
| Score | Score chinois **sans komi**. Le réseau donne 60 % à Noir sur un plateau vide. |

### Bugs connus dans `GO/myPlayer.py`
1. `MaxMin` appelle `self._NNboard.push` au lieu de `pop` après l'exploration (code MinMax, non utilisé par défaut).
2. `IterativeDeepening` choisit le meilleur score **toutes profondeurs confondues** au lieu de garder le résultat de la dernière profondeur terminée.
3. Chemins relatifs au répertoire courant (`./model.json`, `games.json`).

## 3. Architecture

```
pyproject.toml          # uv, Python 3.12; deps: numpy, tensorflow, h5py, cairosvg, pillow, imageio(+ffmpeg)
go_player/
  goban.py              # Board from GO/Goban.py + configurable komi (default 0 = 2020 behavior)
  nn.py                 # ValueNet: rebuilds the 2020 CNN in Keras 3, loads model.h5 weights, batched predict
  players/
    base.py             # PlayerInterface (unchanged contract)
    alphabeta.py        # 2020 myPlayer cleaned up: bugs fixed, same algorithm
    mcts.py             # UCT + batched leaf evaluation; pluggable Evaluator
    gnugo.py            # GTP bridge (from GO/GnuGo.py + GO/gnugoPlayer.py)
    random_player.py
  arena.py              # N games A vs B, alternating colors -> results table
  record.py             # one game -> SVG frames -> PNG -> GIF + MP4, with overlay
  cli.py                # `go-player play|arena|record ...`
  assets/               # model.h5, games.json (copies of the 2020 files)
tests/
media/                  # generated GIF/MP4 + arena.md / arena.json, embedded in README
GO/, ML/                # 2020 archive, untouched
```

Principes :
- Les joueurs ne communiquent **que** via `PlayerInterface`. L'arène et l'enregistreur ne connaissent pas l'algorithme utilisé.
- Le MCTS prend un `Evaluator` interchangeable : `NNEvaluator` (CNN) ou `RolloutEvaluator` (partie aléatoire jusqu'à la fin).
- Les assets sont chargés via `importlib.resources`, jamais selon le répertoire courant.
- `GO/` et `ML/` ne sont pas modifiés. Le README les présente comme la « version école 2020 ».

### Unités

| Unité | Rôle | Interface | Dépend de |
|---|---|---|---|
| `goban.Board` | Règles, push/pop, score, SVG | API de `GO/Goban.py` + `komi` au constructeur | numpy |
| `nn.ValueNet` | Charger le CNN et prédire par lot | `ValueNet.load() -> ValueNet` ; `predict(boards: np.ndarray[N,9,9,2]) -> np.ndarray[N,2]` ; `encode(board) -> np.ndarray[9,9,2]` | tensorflow, h5py |
| `players.mcts.MCTSPlayer` | Choisir un coup sous budget de temps | `PlayerInterface` + `MCTSPlayer(evaluator, time_budget=5.0, batch_size=32, c=1.4, seed=None, opening_book=True)` | goban, Evaluator |
| `players.alphabeta.AlphaBetaPlayer` | Joueur 2020 corrigé | `PlayerInterface` + `AlphaBetaPlayer(value_net, time_budget=5.0, seed=None, opening_book=True)` | goban, nn |
| `players.gnugo.GnuGoPlayer` | Adversaire de référence | `PlayerInterface` + `GnuGoPlayer(level=1)` | binaire `gnugo` |
| `arena` | Matchs et statistiques | `run_match(factory_a, factory_b, games, seed) -> MatchResult` ; `write_report(results, dir)` | goban, players |
| `record` | Une partie → GIF/MP4 | `record_game(black, white, out_stem, value_net=None, seed=None)` | goban, cairosvg, pillow, imageio |
| `cli` | Point d'entrée `go-player` | sous-commandes `play`, `arena`, `record` | tout |

## 4. Flux de données

### Un coup MCTS (budget de 5 s par défaut, `--time`)
1. **Sélection** : descente par UCT `Q + c·√(ln N_parent / n)`, avec c = 1.4. Le réseau ne fournit pas de policy, donc tous les coups partent avec le même prior. Les coups non visités passent en premier.
2. **Expansion** : on copie le plateau et on joue le coup. Les coups refusés par `push` (super-ko) sont retirés des enfants.
3. **Évaluation par lots** : on accumule jusqu'à `batch_size` feuilles (16–32) en appliquant une virtual loss sur leur chemin, puis un seul appel `ValueNet.predict`. La valeur d'une feuille est la P(victoire) **du joueur qui vient de jouer**. Une position terminale est évaluée par `board.result()` (1 / 0 / 0.5).
4. **Remontée** : on retire la virtual loss, puis on propage la valeur vers la racine en l'inversant (`1 − v`) à chaque niveau.
5. **Choix** : l'enfant de la racine le plus visité. Règle reprise de 2020 : si l'adversaire vient de passer et que `compute_score()` (komi compris) nous donne gagnant, on passe.

`RolloutEvaluator` : partie aléatoire avec `weak_legal_moves` jusqu'à la fin ou au plafond de coups, résultat 1 / 0 / 0.5. Il sert de **joueur de référence** pour mesurer ce qu'apporte le réseau.

### Encodage réseau
`(9,9,2)` avec le plan 0 pour Noir et le plan 1 pour Blanc, selon la convention `[col][lin]` de 2020 (`Board.unflatten`). Il est recalculé à chaque feuille depuis le plateau. Le `NNboard` incrémental de 2020 ne sert plus au MCTS. L'Alpha-Beta corrigé peut le garder.

### Ouvertures
Livre `games.json` sur les 5 premiers coups, comme en 2020. Activé par défaut et désactivable (`--no-book`) pour les deux joueurs IA.

### Arène
- N parties (par défaut 20) en **alternant les couleurs**, avec une seed dérivée par partie.
- Sorties :
  - `media/arena.md` : victoires, défaites et nulles par couleur, temps moyen par coup, simulations moyennes par coup pour le MCTS ;
  - `media/arena.json` : données brutes.
- Matchs prévus :
  1. MCTS-NN vs AlphaBeta-2020 ;
  2. MCTS-NN vs MCTS-rollout ;
  3. chaque IA (MCTS-NN, AlphaBeta-2020) vs GnuGo niveau 1.

### Enregistrement
1. Partie jouée, puis un SVG par position (`Board.svg()`).
2. Surcouche :
   - bandeau avec les noms des joueurs et le numéro du coup ;
   - cercle sur le dernier coup ;
   - barre « P(Noir gagne) » calculée par `ValueNet` si disponible.
3. cairosvg → PNG → GIF (~0,6 s par image, 3 s sur la dernière) + MP4.
4. Fichiers produits : `media/<black>-vs-<white>.gif|mp4`.

### Komi
Défaut 0, fidèle au comportement 2020 et aux données d'entraînement du réseau. L'arène alterne les couleurs pour compenser. `--komi 7` est disponible.

## 5. Gestion des erreurs

| Situation | Comportement |
|---|---|
| Modèle introuvable ou TF absent | Erreur explicite au démarrage d'un joueur NN. **Pas de repli silencieux** sur l'heuristique « nombre de pierres ». Il reste possible d'utiliser `--evaluator rollout`. |
| `gnugo` absent du PATH | Vérifié avant `arena`/`record` s'ils l'utilisent. Message : `brew install gnu-go`. Les tests concernés sont sautés (`skipif`). |
| Coup illégal d'un joueur | Il perd la partie (règle de `localGame.py`). L'incident est noté dans `MatchResult`. |
| Budget de temps | Le MCTS vérifie l'horloge entre deux lots et s'arrête à 90 % du budget. Sans aucune simulation, il joue un coup légal au hasard. |
| Partie trop longue | Plafond de 200 coups dans l'arène, la partie est alors comptée nulle. |
| `cairo` absent | Message `brew install cairo`, uniquement lors de `record`. |

## 6. Tests (`pytest`, hors-ligne, en moins d'une minute hors tests GnuGo)

1. **Règles** : capture, interdiction du suicide, super-ko, score chinois avec et sans komi.
2. **Réseau** :
   - le chargement fonctionne ;
   - le plateau vide donne `≈ [0.60, 0.40]` (tolérance 0.01) ;
   - une position tournée de 90° donne une valeur proche (écart absolu ≤ 0.05) ;
   - `predict` accepte des lots.
3. **MCTS** :
   - trouve la capture évidente en un coup sur une position préparée ;
   - ne renvoie que des coups légaux ;
   - respecte le budget (temps ≤ 1,1 × budget) ;
   - donne un résultat reproductible avec une seed fixe (avec un nombre de simulations fixé plutôt qu'un temps fixé, pour être déterministe).
4. **Alpha-Beta** :
   - même test de capture ;
   - après `getPlayerMove`, l'encodage interne correspond exactement à `ValueNet.encode(board)`, ce qui couvre la dérive `push`/`pop`.
5. **Smoke tests** :
   - arène : 2 parties random vs random avec un budget de 0,1 s ;
   - `record` : produit un GIF non vide.

## 7. Critères de réussite

1. `uv sync && uv run go-player record --black mcts --white gnugo` produit `media/mcts-vs-gnugo.gif` et `.mp4`.
2. `media/arena.md` contient les matchs listés en §4. Objectif visé, mais pas bloquant : MCTS-NN gagne plus de 60 % des parties contre AlphaBeta-2020.
3. Le README est réécrit en anglais :
   - GIF en tête ;
   - histoire 2020 → 2026 ;
   - tableau de l'arène ;
   - Quick start ;
   - lien vers l'archive `GO/` + `ML/` et crédits (Laurent Genty, Johan Chataigner).
4. Les fichiers de `media/` peuvent être copiés tels quels dans `~/perso/portfolio`.
5. `uv run pytest` passe.
