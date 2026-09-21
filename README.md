# Desktop Band

Un petit groupe dessiné au-dessus du bureau et animé en fonction du son joué
par le PC. Ce premier jet cible Windows et utilise `desktop-overlay` uniquement
comme bibliothèque de fenêtre transparente.

## Ce que fait cette version

- groupe de 1 à 7 personnages ;
- chanteur, batterie, basse, clavier, guitare, percussions et danseur en
  prisyadka ;
- au moins 8 sprites par personnage, avec une vitesse propre à chaque rôle ;
- capture de la sortie Windows par WASAPI loopback ;
- suivi automatique de la sortie audio Windows par défaut ;
- séparation StemgenRT en temps réel : batterie, basse, voix et autres instruments ;
- détection du BPM et modulation dynamique des animations ;
- réactions contextuelles en bon langage gopnik sur les moments forts ;
- fenêtre transparente, toujours visible et click-through ;
- mode démo pour essayer les animations sans configuration audio.

La guitare et le clavier partagent la piste `other` de StemgenRT. Une analyse
spectrale légère les différencie visuellement, sans prétendre les séparer.

## Installation

Avec Python 3.10 ou plus récent :

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e .
```

## Essai immédiat

Le mode démo permet de valider l'affichage avant la capture audio :

```powershell
python -m desktop_band --demo --members 4
```

Puis avec le son réellement joué par Windows :

```powershell
python -m desktop_band --members 4
```

Le mode `compact` est utilisé par défaut. Il réduit les sprites, supprime les
gros halos et les libellés permanents, et rapproche les membres du groupe. Pour
retrouver l'ancienne présentation :

```powershell
python -m desktop_band --members 7 --layout wide
```

Pour afficher le BPM estimé en direct :

```powershell
python -m desktop_band --members 7 --debug
```

Le mode StemgenRT cherche son modèle dans `models/stemgen_rt.onnx`. Le modèle
n'est pas versionné dans ce dépôt. Sous PowerShell, il peut être récupéré
depuis le dépôt officiel StemgenRT :

```powershell
New-Item -ItemType Directory -Force models
Invoke-WebRequest `
  -Uri "https://media.githubusercontent.com/media/sweetspotsoundsystem/stemgen-rt/main/model/model.onnx" `
  -OutFile "models/stemgen_rt.onnx"
```

Il est aussi possible d'indiquer un autre emplacement avec `--model-path`.
Pour revenir à l'ancien analyseur sans modèle :

```powershell
python -m desktop_band --members 4 --analysis lightweight
```

Exemples :

```powershell
python -m desktop_band --members 2 --corner bottom-left
python -m desktop_band --members 6 --monitor 1
```

Le processus se ferme avec `Ctrl+C` dans le terminal. La fenêtre est
click-through : elle ne vole ni le focus ni les clics aux autres applications.

## Vérifications

```powershell
python -m unittest discover -s tests -v
python -m compileall -q desktop_band tests
```

## Limites connues du premier jet

- Windows est la plateforme de capture validée en priorité ;
- StemgenRT ne sépare pas individuellement guitare, clavier et piano ;
- le changement de composition nécessite actuellement de relancer la commande.

## Crédits

- Fenêtre transparente : [desktop-overlay](https://github.com/boubou666/desktop-overlay)
- Séparation audio : [StemgenRT](https://github.com/sweetspotsoundsystem/stemgen-rt)
- Sprites générés avec OpenAI ImageGen.

Les notices des dépendances tierces sont regroupées dans
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). Les changements sont suivis
selon [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) dans
[`CHANGELOG.md`](CHANGELOG.md).
