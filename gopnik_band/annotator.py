"""Compatibility entry point for ``python -m gopnik_band.annotator``."""

from desktop_band.annotator import main


if __name__ == "__main__":
    raise SystemExit(main())
