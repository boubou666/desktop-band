# Entraînement du classifieur

Les fichiers audio d'entraînement restent locaux et sont ignorés par Git. Le
repo ne contient que le modèle ONNX final et les outils reproductibles.

## 1. Annoter les morceaux dans Gopnik Lab

L'annotateur accepte directement un MP3, WAV, FLAC ou M4A, ainsi qu'un lien
YouTube :

```powershell
gopnik-band-annotator
```

On peut aussi le lancer sans réinstaller le point d'entrée :

```powershell
python -m gopnik_band.annotator
```

Le navigateur décode le morceau localement, l'envoie uniquement au serveur
`127.0.0.1`, puis StemgenRT et le routeur V2 proposent une première timeline.
Il reste à corriger les instruments présents, les débuts et les fins de leurs
passages, puis à cliquer sur **Enregistrer les corrections**.

Le mode punch-in évite de saisir les temps à la main : choisir un instrument,
cliquer le départ sur la timeline, relire avec `R`, utiliser `Espace` pour
lecture/pause, puis `E` pour terminer et ajouter le passage. Les touches `1` à
`6` choisissent rapidement l'instrument.

Lorsqu'un fichier ne contient réellement qu'un seul type d'instrument, cocher
**Morceau mono-instrument** et choisir ce rôle. À l'entraînement, les
prédictions automatiques et les annotations de timeline des autres rôles sont
alors ignorées : les fenêtres audibles deviennent des exemples positifs du rôle
choisi et les silences restent des exemples négatifs. Ces solos propres sont
très utiles, mais le dataset doit aussi contenir des morceaux mixés et corrigés
pour apprendre à séparer les rôles lorsqu'ils jouent ensemble.

Pour YouTube, `yt-dlp` récupère une seule piste audio dans
`training/youtube/`. Elle est ensuite traitée comme un fichier droppé. Le cache
YouTube est lui aussi ignoré par Git. Seuls des contenus que l'utilisateur est
autorisé à télécharger doivent être importés.

Les WAV normalisés sont déposés dans `training/audio/` et le manifeste est
mis à jour dans `training/annotations.json` :

```json
{
  "clips": [
    {
      "audio": "audio/taiko.wav",
      "global_labels": ["drummer", "percussion"],
      "segments": [
        {"start": 0.0, "end": 18.4, "labels": ["drummer"]},
        {"start": 18.4, "end": 31.2, "labels": ["drummer", "percussion"]}
      ]
    }
  ]
}
```

Les suggestions automatiques ne deviennent jamais silencieusement la vérité :
le manifeste n'est écrit qu'après validation humaine. Une sélection globale
sans aucun segment reste utilisable comme label faible sur tout le morceau.

Le petit modèle apprend maintenant les six labels `singer`, `drummer`,
`bassist`, `keyboard`, `guitarist` et `percussion`. StemgenRT fournit les
stems ; le classifieur apprend à accepter ou rejeter leurs fuites à partir des
corrections humaines.

## Amorcer le dataset avec BabySlakh

Le corpus officiel BabySlakh contient 20 compositions avec des stems isolés
et des métadonnées d'instruments. L'importeur télécharge les 882,8 Mo depuis
Zenodo, vérifie le checksum, extrait localement le corpus puis fabrique un lot
équilibré de passages actifs :

```powershell
python -m pip install -e ".[training]"
python tools/import_slakh.py --download-baby --minutes-per-role 30
```

Les fichiers bruts restent dans `training/datasets/`, les extraits retenus
dans `training/slakh/audio/` et le manifeste dans
`training/slakh_manifest.json`. Tout reste local et ignoré par Git. Basse,
guitare, clavier, batterie et percussions sont importés ; Slakh ne contient pas
de chant, qui doit donc toujours venir des morceaux corrigés dans Gopnik Lab.
BabySlakh/Slakh2100 est distribué sous licence CC BY 4.0 ; l'attribution et la
citation demandée sont conservées dans `THIRD_PARTY_NOTICES.md`.

L'importeur choisit les fenêtres les plus audibles en tournant entre les 20
compositions pour éviter qu'un seul morceau domine. Tous les extraits issus du
même morceau portent aussi le même groupe de validation : ils restent ensemble
pendant l'examen du candidat et ne peuvent pas artificiellement gonfler son
score. Un corpus Slakh déjà extrait peut être utilisé sans téléchargement :

```powershell
python tools/import_slakh.py D:\datasets\babyslakh_16k
```

## Entraînement automatisé

Le tableau **Dataset du commissaire** liste chaque morceau, sa durée et la
couverture corrigée par instrument. Le bouton **Entraîner un candidat** devient
disponible quand au moins trois morceaux existent et que chaque instrument est
présent dans deux morceaux distincts.

Le travail s'effectue en arrière-plan :

1. extraction des features StemgenRT ;
2. réservation d'un morceau entier pour la validation ;
3. entraînement d'un candidat ONNX à six sorties ;
4. comparaison macro-F1 avec le modèle actif ;
5. promotion uniquement si le candidat progresse d'au moins un point.

L'extraction est conservée dans `training/cache/` et invalidée seulement si
le WAV, StemgenRT ou le schéma de caractéristiques change. Une correction de
timeline peut donc être réentraînée sans repasser tout l'audio. Le Lab affiche
le morceau courant, le temps écoulé, l'ETA, les cache hits, les F1 par rôle,
la matrice de confusion et les raisons exactes d'une promotion ou d'un rejet.

**Fabriquer 12 mixes** combine les pistes mono-instrument avec variations de
gain, EQ, compression, bruit et vitesse. Les mixes sont locaux dans
`training/synthetic/` et leur manifeste est fusionné uniquement pour
l'entraînement. Une augmentation légère des caractéristiques est également
appliquée au train, jamais au morceau de validation.

Le raccourci global `Ctrl+Alt+B` crée un signalement des douze dernières
secondes de caractéristiques, sans audio. Gopnik Lab permet ensuite de lui
attribuer le bon instrument ; il rejoint automatiquement l'entraînement
suivant. `Ctrl+Alt+R` enregistre une session complète de caractéristiques dans
`training/replays/`, reproductible avec `gopnik-band --replay <fichier>`.

Le modèle précédent est archivé dans `training/models/`. **Exporter le
modèle** télécharge un ZIP contenant le modèle actif, l'ordre des features et
des labels, ainsi que le rapport de validation. Aucun audio n'est exporté.
Le registre du Lab permet aussi une comparaison A/B et une restauration en un
clic. Les annotations disposent de leurs propres révisions dans
`training/history/`.

## Extraction manuelle des features

```powershell
python tools/collect_features.py training/annotations.json
```

Le modèle StemgenRT doit être disponible comme pour l'application. Les
features sont écrites dans `training/features.jsonl`.

## Entraînement manuel et export

```powershell
python -m pip install -e ".[training]"
python tools/train_classifier.py training/features.jsonl
```

Toujours réserver des morceaux entiers à la validation au lieu de mélanger des
fenêtres du même morceau entre entraînement et validation. Sinon le score sera
beau comme propagande, mais faux comme promesse de vodka gratuite.
