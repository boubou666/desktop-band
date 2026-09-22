# Packs de sprites

Un pack est un dossier contenant un `manifest.json` et une ou plusieurs
feuilles PNG. Les packs livrés avec l'application résident dans
`desktop_band/assets/packs/`. Un dossier externe peut aussi être passé à
`--sprite-pack`.

## Format d'une feuille

- quatre colonnes exactement ;
- une pose par colonne ;
- une ou plusieurs lignes ;
- fond transparent ;
- le personnage doit conserver une échelle et un point d'appui cohérents.

Le renderer normalise ensuite toutes les poses d'un même rôle sur une toile
commune afin d'éviter les changements de taille pendant l'animation.

## Manifeste

```json
{
  "schema_version": 1,
  "name": "Mon pack",
  "sheets": [
    {
      "file": "bass.png",
      "rows": 1,
      "roles": {"bassist": 0}
    }
  ]
}
```

`rows` indique le nombre de lignes de la feuille. Pour chaque rôle, la valeur
est l'index de sa ligne en partant de zéro.

Rôles acceptés :

- `singer`
- `drummer`
- `bassist`
- `keyboard`
- `guitarist`
- `percussion`
- `vibing`

Plusieurs feuilles peuvent alimenter le même rôle. Deux feuilles de quatre
colonnes donnent ainsi huit poses.

## Installation

Pack externe :

```powershell
python -m desktop_band --sprite-pack C:\chemin\vers\mon-pack
```

Pack intégré : placer son dossier dans `desktop_band/assets/packs/<nom>/`. Il
apparaîtra automatiquement dans le menu du tray.
