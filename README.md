# Gopnik Band

Un petit groupe de gopniks dessiné au-dessus du bureau et animé en fonction du
son joué par le PC. L'application utilise `desktop-overlay` uniquement comme
bibliothèque de fenêtre transparente.

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
- routeur de détection V2 avec confiance temporelle, compétition entre rôles
  voisins et états repos/groove/jeu ;
- détection de drops avec emballement du groupe, spotlights et réplique dédiée ;
- mise en avant automatique du soliste dominant ;
- animations de repos procédurales : fumée, bouteille, regard et batte ;
- packs de sprites interchangeables ;
- icône de tray pour masquer le groupe, changer sa taille, sa sortie audio, son
  pack et son profil de détection ;
- mode déplacement/redimensionnement de 30 secondes depuis le tray ;
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

Le mode debug affiche aussi les trois meilleures confiances (`I` = idle,
`G` = groove, `P` = playing). Les calibrations fournies se choisissent avec :

```powershell
python -m desktop_band --members 7 --profile taiko
python -m desktop_band --members 7 --profile hardbass
python -m desktop_band --members 7 --profile soft
```

`balanced` reste le profil par défaut. Les cas taiko, hardbass, guitare et
clavier font partie de la suite de tests afin que les réglages ne régressent
pas silencieusement.

## Contrôles du tray

Un clic droit sur l'icône Desktop Band permet de :

- afficher ou masquer l'overlay ;
- activer le déplacement/redimensionnement pendant 30 secondes ;
- passer de 1 à 7 membres sans redémarrer ;
- suivre la sortie Windows par défaut ou choisir une sortie précise ;
- changer de pack de sprites et de profil de détection ;
- quitter proprement l'application.

En mode déplacement, toute la fenêtre devient saisissable et le triangle en
bas à droite sert de poignée de redimensionnement. Le click-through est remis
automatiquement après 30 secondes. Utiliser `--no-tray` pour désactiver cette
fonctionnalité.

## Packs de sprites

Le pack intégré s'appelle `gopnik`. Les packs placés dans
`desktop_band/assets/packs/<nom>/` apparaissent automatiquement dans le tray.
Un dossier externe se charge aussi avec
`--sprite-pack C:\chemin\vers\mon-pack`.

Le format complet du manifeste et des feuilles est documenté dans
[`docs/SPRITE_PACKS.md`](docs/SPRITE_PACKS.md). Le fonctionnement interne est
résumé dans [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

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

## Annoter des morceaux pour améliorer la détection

Gopnik Lab permet de déposer directement un MP3 ou de coller un lien YouTube,
de laisser StemgenRT proposer les instruments et leurs passages, puis de
corriger cette timeline :

```powershell
gopnik-band-annotator
```

L'outil fonctionne entièrement en local dans le navigateur. L'import YouTube
est effectué par `yt-dlp` dans un cache local ; l'audio n'est envoyé à aucun
service autre que YouTube. Gopnik Lab convertit ensuite le morceau en WAV
d'entraînement dans `training/audio/` et maintient
`training/annotations.json`. Ces emplacements sont ignorés par Git. Le workflow
complet est décrit dans [`training/README.md`](training/README.md).
Le tableau du dataset indique précisément les morceaux utilisés et leur
couverture. L'entraînement automatisé réserve un morceau entier à la
validation, compare le candidat au modèle actif et refuse toute régression.
Le modèle validé s'exporte ensuite en archive ONNX autonome sans inclure les
fichiers audio.

## Limites connues

- Windows est la plateforme de capture validée en priorité ;
- StemgenRT ne sépare pas individuellement guitare, clavier et piano ;
- guitare et clavier restent des inférences à l'intérieur de la même piste
  `other`, même si le routeur V2 réduit fortement les doubles détections.

## Crédits

- Fenêtre transparente : [desktop-overlay](https://github.com/boubou666/desktop-overlay)
- Séparation audio : [StemgenRT](https://github.com/sweetspotsoundsystem/stemgen-rt)
- Sprites générés avec OpenAI ImageGen.

Les notices des dépendances tierces sont regroupées dans
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). Les changements sont suivis
selon [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) dans
[`CHANGELOG.md`](CHANGELOG.md). Les règles de contribution sont dans
[`CONTRIBUTING.md`](CONTRIBUTING.md).
