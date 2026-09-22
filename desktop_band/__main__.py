"""Command-line entry point for Gopnik Band."""

from __future__ import annotations

import argparse

from .app import AppOptions, run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gopnik-band",
        description="Affiche un groupe de gopniks animé par le son du système.",
    )
    parser.add_argument(
        "--members",
        type=int,
        choices=range(1, 8),
        default=4,
        metavar="1..7",
        help="nombre de musiciens (défaut : 4)",
    )
    parser.add_argument(
        "--corner",
        choices=("top-left", "top-right", "bottom-left", "bottom-right"),
        default="bottom-right",
        help="coin de l'écran (défaut : bottom-right)",
    )
    parser.add_argument(
        "--monitor",
        default="primary",
        help="primary, index d'écran ou partie de son nom",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="anime le groupe sans capturer de son",
    )
    parser.add_argument(
        "--analysis",
        choices=("stemgen", "lightweight"),
        default="stemgen",
        help="moteur d'analyse (défaut : stemgen)",
    )
    parser.add_argument(
        "--model-path",
        help="chemin du modèle ONNX StemgenRT",
    )
    parser.add_argument(
        "--layout",
        choices=("compact", "wide"),
        default="compact",
        help="mise en page compacte ou historique (défaut : compact)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="affiche le BPM détecté en direct",
    )
    parser.add_argument(
        "--sprite-pack",
        default="gopnik",
        help="nom ou dossier d'un pack de sprites (défaut : gopnik)",
    )
    parser.add_argument(
        "--profile",
        choices=("balanced", "taiko", "hardbass", "soft"),
        default="balanced",
        help="calibration de détection (défaut : balanced)",
    )
    parser.add_argument(
        "--no-tray",
        action="store_true",
        help="désactive l'icône et les commandes de zone de notification",
    )
    parser.add_argument(
        "--no-hotkeys",
        action="store_true",
        help="désactive les raccourcis clavier globaux",
    )
    parser.add_argument(
        "--fps",
        type=int,
        choices=range(5, 61),
        default=30,
        metavar="5..60",
        help="limite de rendu en images par seconde (défaut : 30)",
    )
    parser.add_argument(
        "--eco",
        action="store_true",
        help="réduit automatiquement le FPS, surtout pendant le silence",
    )
    parser.add_argument(
        "--decor",
        choices=("none", "garage", "panelki"),
        default="none",
        help="décor optionnel derrière le groupe",
    )
    parser.add_argument(
        "--no-lighting",
        action="store_true",
        help="désactive l'éclairage musical",
    )
    parser.add_argument(
        "--variant-seed",
        type=int,
        help="graine reproductible pour les variantes visuelles",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return run(
        AppOptions(
            members=args.members,
            corner=args.corner,
            monitor=args.monitor,
            demo=args.demo,
            analysis=args.analysis,
            model_path=args.model_path,
            layout=args.layout,
            debug=args.debug,
            sprite_pack=args.sprite_pack,
            detection_profile=args.profile,
            tray=not args.no_tray,
            hotkeys=not args.no_hotkeys,
            fps=args.fps,
            eco=args.eco,
            decor=args.decor,
            lighting=not args.no_lighting,
            variant_seed=args.variant_seed,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
