"""Application bootstrap and transparent overlay lifecycle."""

from __future__ import annotations

from dataclasses import dataclass
import queue
import sys

from .band import create_lineup
from .capture import DemoAudioSource, SystemAudioSource
from .detection import PROFILES
from .renderer import BandRenderer
from .sprites import available_sprite_packs


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
    sprite_pack: str = "gopnik"
    detection_profile: str = "balanced"
    tray: bool = True


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
        sprite_pack=options.sprite_pack,
        detection_profile=options.detection_profile,
    )
    source.start()

    closing = False
    visible = True
    editing = False
    edit_generation = 0
    member_count = options.members
    selected_pack = options.sprite_pack
    selected_profile = options.detection_profile
    drag_state = {"mode": None, "x": 0, "y": 0, "wx": 0, "wy": 0, "w": width, "h": height}
    tray = None
    ui_actions: queue.SimpleQueue[tuple[object, tuple]] = queue.SimpleQueue()

    def schedule(callback, *args) -> None:
        ui_actions.put((callback, args))

    def set_click_through(enabled: bool) -> bool:
        if sys.platform != "win32":
            return False
        try:
            import ctypes

            overlay.update_idletasks()
            hwnd = ctypes.windll.user32.GetParent(overlay.winfo_id()) or overlay.winfo_id()
            user32 = ctypes.windll.user32
            style = user32.GetWindowLongW(hwnd, -20)
            transparent = 0x00000020
            no_activate = 0x08000000
            if enabled:
                style |= transparent | no_activate
            else:
                style &= ~(transparent | no_activate)
            user32.SetWindowLongW(hwnd, -20, style)
            user32.SetWindowPos(hwnd, 0, 0, 0, 0, 0, 0x0027)
            return True
        except Exception:
            return False

    def end_edit_mode(generation: int | None = None) -> None:
        nonlocal editing
        if generation is not None and generation != edit_generation:
            return
        editing = False
        canvas.delete("edit-ui")
        set_click_through(True)

    def enable_edit_mode() -> None:
        nonlocal editing, edit_generation
        editing = True
        edit_generation += 1
        set_click_through(False)
        overlay.lift()
        root.after(30_000, lambda value=edit_generation: end_edit_mode(value))

    def on_pointer_down(event) -> None:
        if not editing:
            return
        drag_state.update(
            x=event.x_root, y=event.y_root,
            wx=overlay.winfo_x(), wy=overlay.winfo_y(),
            w=renderer.width, h=renderer.height,
        )
        drag_state["mode"] = (
            "resize"
            if event.x >= renderer.width - 28 and event.y >= renderer.height - 28
            else "move"
        )

    def on_pointer_move(event) -> None:
        if not editing or drag_state["mode"] is None:
            return
        dx = event.x_root - drag_state["x"]
        dy = event.y_root - drag_state["y"]
        if drag_state["mode"] == "move":
            window.move_window(
                overlay, renderer.width, renderer.height,
                drag_state["wx"] + dx, drag_state["wy"] + dy,
            )
            return
        new_width = max(280, drag_state["w"] + dx)
        new_height = max(190, drag_state["h"] + dy)
        renderer.resize(new_width, new_height)
        canvas.configure(width=new_width, height=new_height)
        window.move_window(
            overlay, new_width, new_height,
            drag_state["wx"], drag_state["wy"],
        )

    def on_pointer_up(_event) -> None:
        drag_state["mode"] = None

    canvas.bind("<ButtonPress-1>", on_pointer_down)
    canvas.bind("<B1-Motion>", on_pointer_move)
    canvas.bind("<ButtonRelease-1>", on_pointer_up)

    def shutdown() -> None:
        nonlocal closing
        if closing:
            return
        closing = True
        if tray is not None:
            tray.stop()
        source.stop()
        try:
            root.destroy()
        except tk.TclError:
            pass

    def tick() -> None:
        if closing:
            return
        while True:
            try:
                callback, args = ui_actions.get_nowait()
            except queue.Empty:
                break
            callback(*args)
            if closing:
                return
        renderer.draw(
            source.features,
            source.status,
            source.error or renderer.sprite_error or (
                tray.error if tray is not None else None
            ),
        )
        if editing:
            canvas.delete("edit-ui")
            canvas.create_rectangle(
                2, 2, renderer.width - 3, renderer.height - 3,
                outline="#22d3ee", width=3, dash=(8, 4), tags="edit-ui",
            )
            canvas.create_polygon(
                renderer.width - 26, renderer.height - 4,
                renderer.width - 4, renderer.height - 26,
                renderer.width - 4, renderer.height - 4,
                fill="#22d3ee", outline="#f8fafc", tags="edit-ui",
            )
            canvas.create_text(
                9, 39, text="DÉPLACEMENT — tire le coin pour redimensionner",
                anchor="nw", fill="#22d3ee", font=("Segoe UI", 9, "bold"),
                tags="edit-ui",
            )
        root.after(33, tick)

    def toggle_visible() -> None:
        nonlocal visible
        visible = not visible
        if visible:
            overlay.deiconify()
            overlay.lift()
        else:
            overlay.withdraw()

    def set_members(count: int) -> None:
        nonlocal member_count
        old_width = renderer.width
        new_width, new_height = _window_size(count, options.layout)
        old_right = overlay.winfo_x() + old_width
        old_bottom = overlay.winfo_y() + renderer.height
        member_count = count
        renderer.set_lineup(create_lineup(count))
        renderer.resize(new_width, new_height)
        canvas.configure(width=new_width, height=new_height)
        window.move_window(
            overlay, new_width, new_height,
            old_right - new_width, old_bottom - new_height,
        )

    def set_pack(name: str) -> None:
        nonlocal selected_pack
        renderer.set_sprite_pack(name)
        if renderer.sprite_pack == name:
            selected_pack = name

    def set_profile(name: str) -> None:
        nonlocal selected_profile
        renderer.set_detection_profile(name)
        selected_profile = name

    if options.tray:
        try:
            from .tray import TrayCallbacks, TrayController

            output_list = getattr(source, "available_outputs", lambda: ())
            output_get = lambda: getattr(source, "selected_output", None)
            output_set = getattr(source, "select_output", lambda _value: None)
            packs = tuple(dict.fromkeys((*available_sprite_packs(), options.sprite_pack)))
            callbacks = TrayCallbacks(
                toggle_visible=lambda: schedule(toggle_visible),
                enable_edit=lambda: schedule(enable_edit_mode),
                set_members=lambda value: schedule(set_members, value),
                set_output=output_set,
                set_pack=lambda value: schedule(set_pack, value),
                set_profile=lambda value: schedule(set_profile, value),
                quit=lambda: schedule(shutdown),
                get_members=lambda: member_count,
                get_output=output_get,
                get_pack=lambda: selected_pack,
                get_profile=lambda: selected_profile,
                list_outputs=output_list,
            )
            tray = TrayController(callbacks, packs, PROFILES)
            tray.start()
        except Exception as exc:
            renderer.sprite_error = "tray indisponible : {}".format(exc)

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
