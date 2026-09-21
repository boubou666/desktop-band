"""Application bootstrap and transparent overlay lifecycle."""

from __future__ import annotations

from dataclasses import dataclass
import sys

from .band import create_lineup
from .capture import DemoAudioSource, SystemAudioSource
from .renderer import BandRenderer


@dataclass(frozen=True, slots=True)
class AppOptions:
    members: int = 4
    corner: str = "bottom-right"
    monitor: str = "primary"
    demo: bool = False
    analysis: str = "stemgen"
    model_path: str | None = None
    layout: str = "compact"
    debug: bool = False


def _window_size(member_count: int, layout: str) -> tuple[int, int]:
    if layout == "compact":
        return max(280, 112 * member_count + 40), 235
    return max(320, 190 * member_count + 60), 330


def run(options: AppOptions) -> int:
    try:
        from desktop_overlay import enumerate_monitors
        from desktop_overlay import window
    except ImportError:
        print(
            "desktop-overlay est absent. Installe le projet avec "
            "`python -m pip install -e .`.",
            file=sys.stderr,
        )
        return 2

    tk, _tkfont = window.import_tk()
    lineup = create_lineup(options.members)
    width, height = _window_size(len(lineup), options.layout)

    root = tk.Tk()
    root.withdraw()
    overlay = tk.Toplevel(root)
    background = window.prepare_window(overlay)
    canvas = tk.Canvas(
        overlay,
        width=width,
        height=height,
        bg=background,
        highlightthickness=0,
        borderwidth=0,
    )
    canvas.pack()

    monitor = _select_monitor(enumerate_monitors(), options.monitor)
    x, y = monitor.corner_position(width, height, options.corner, margin=20)
    window.move_window(overlay, width, height, x, y)
    window.show_window(overlay)

    def ensure_overlay_visible() -> None:
        """Finish the hidden-to-visible transition after Tk created the HWND."""

        try:
            overlay.update_idletasks()
            window.set_opacity(overlay, 1.0)
            overlay.wm_attributes("-topmost", True)
            overlay.lift()
        except tk.TclError:
            pass

    ensure_overlay_visible()

    source = (
        DemoAudioSource()
        if options.demo
        else SystemAudioSource(analysis=options.analysis, model_path=options.model_path)
    )
    renderer = BandRenderer(
        canvas,
        lineup,
        width,
        height,
        compact=options.layout == "compact",
        debug=options.debug,
    )
    source.start()

    closing = False

    def shutdown() -> None:
        nonlocal closing
        if closing:
            return
        closing = True
        source.stop()
        try:
            root.destroy()
        except tk.TclError:
            pass

    def tick() -> None:
        if closing:
            return
        renderer.draw(
            source.features,
            source.status,
            source.error or renderer.sprite_error,
        )
        root.after(33, tick)

    root.protocol("WM_DELETE_WINDOW", shutdown)
    overlay.protocol("WM_DELETE_WINDOW", shutdown)
    root.after(0, tick)
    # On Windows, Tk can create the native layered window only while handling
    # its first event-loop turn. Reassert alpha/topmost once that has happened.
    root.after(120, ensure_overlay_visible)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        shutdown()
    finally:
        source.stop()
    return 0


def _select_monitor(monitors, preference: str):
    if not monitors:
        raise RuntimeError("desktop-overlay n'a retourné aucun écran")
    if preference == "primary":
        return next((monitor for monitor in monitors if monitor.primary), monitors[0])
    try:
        index = int(preference)
        return monitors[index]
    except (ValueError, IndexError):
        wanted = preference.casefold()
        return next(
            (monitor for monitor in monitors if wanted in monitor.name.casefold()),
            monitors[0],
        )
