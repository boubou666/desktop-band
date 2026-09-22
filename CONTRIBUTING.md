# Contributing

Merci de garder les changements petits, testables et centrés sur une seule
fonctionnalité.

## Environnement

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e .
```

Le modèle StemgenRT n'est pas nécessaire pour exécuter les tests unitaires.
Il reste volontairement hors du dépôt.

## Vérifications

Avant une proposition de changement :

```powershell
python -m unittest discover -s tests -v
python -m compileall -q desktop_band tests
```

Toute modification du routeur doit inclure un cas reproductible dans
`tests/test_detection.py`. Toute modification du format de pack doit inclure
un fixture dans `tests/fixtures/` et mettre à jour `docs/SPRITE_PACKS.md`.

## Organisation

- `desktop_band/analysis.py` et `stemgen.py` produisent les caractéristiques
  audio ;
- `desktop_band/detection.py` décide qui joue ;
- `desktop_band/renderer.py` ne doit pas réimplémenter l'analyse musicale ;
- `desktop_band/app.py` relie la fenêtre, le tray et la source audio ;
- les sprites distribués vivent dans `desktop_band/assets/packs/`.

Les changements visibles doivent être ajoutés à la section `Unreleased` du
`CHANGELOG.md` selon Keep a Changelog.
