# Architecture

Desktop Band sépare la capture, l'analyse et l'affichage pour que chaque bloc
puisse être testé sans ouvrir de fenêtre Windows.

```text
WASAPI loopback
      │
      ▼
capture.py ──► analysis.py ──► stemgen.py
                                  │
                                  ▼
                            detection.py
                                  │
                    ┌─────────────┴─────────────┐
                    ▼                           ▼
               moments.py                 reactions.py
                    └─────────────┬─────────────┘
                                  ▼
                             renderer.py
                                  │
                                  ▼
                    desktop-overlay / Tk Canvas
```

## Responsabilités

- `capture.py` capture la sortie sélectionnée et reconnecte WASAPI lorsque le
  périphérique change.
- `analysis.py` extrait niveau, bandes fréquentielles, transitoires, tempo,
  stabilité de hauteur et densité de notes.
- `stemgen.py` exécute StemgenRT et enrichit ses quatre stems.
- `detection.py` accumule les preuves, met les rôles concurrents en compétition
  et produit les états `idle`, `groove` et `playing`.
- `moments.py` détecte les transitions globales comme les drops.
- `reactions.py` choisit les répliques ponctuelles.
- `renderer.py` anime les sprites sans capturer ni analyser le son.
- `app.py` possède le cycle de vie de l'overlay et relaie les actions du tray
  vers le thread Tk.

## Assets

Chaque style visuel est un pack autonome sous
`desktop_band/assets/packs/<nom>/`. Le chargeur et le format du manifeste sont
décrits dans [SPRITE_PACKS.md](SPRITE_PACKS.md).

Le modèle `models/stemgen_rt.onnx` est une dépendance locale ignorée par Git.
