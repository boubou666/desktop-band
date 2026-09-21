"""Optional system-tray controls for the desktop overlay."""

from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import Callable


@dataclass(slots=True)
class TrayCallbacks:
    toggle_visible: Callable[[], None]
    enable_edit: Callable[[], None]
    set_members: Callable[[int], None]
    set_output: Callable[[str | None], None]
    set_pack: Callable[[str], None]
    set_profile: Callable[[str], None]
    quit: Callable[[], None]
    get_members: Callable[[], int]
    get_output: Callable[[], str | None]
    get_pack: Callable[[], str]
    get_profile: Callable[[], str]
    list_outputs: Callable[[], tuple[tuple[str, str], ...]]


class TrayController:
    def __init__(self, callbacks: TrayCallbacks, packs: tuple[str, ...], profiles) -> None:
        self.callbacks = callbacks
        self.packs = packs
        self.profiles = tuple(profiles)
        self._icon = None
        self._thread: threading.Thread | None = None
        self.error: str | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        icon = self._icon
        if icon is not None:
            try:
                icon.stop()
            except Exception:
                pass

    def _run(self) -> None:
        try:
            import pystray
            from PIL import Image, ImageDraw

            image = Image.new("RGBA", (64, 64), (17, 24, 39, 255))
            draw = ImageDraw.Draw(image)
            draw.ellipse((8, 8, 56, 56), fill=(239, 71, 111, 255))
            draw.rectangle((18, 17, 25, 47), fill=(248, 250, 252, 255))
            draw.rectangle((31, 11, 38, 47), fill=(248, 250, 252, 255))
            draw.rectangle((44, 24, 51, 47), fill=(248, 250, 252, 255))

            def choose(callback, value):
                def action(_icon, _item):
                    callback(value)
                return action

            def is_selected(getter, value):
                def checked(_item):
                    return getter() == value
                return checked

            members = pystray.Menu(*(
                pystray.MenuItem(
                    "{} gopnik{}".format(count, "s" if count > 1 else ""),
                    choose(self.callbacks.set_members, count),
                    checked=is_selected(self.callbacks.get_members, count),
                    radio=True,
                )
                for count in range(1, 8)
            ))
            outputs = [
                pystray.MenuItem(
                    "Sortie Windows par défaut",
                    lambda _icon, _item: self.callbacks.set_output(None),
                    checked=lambda _item: self.callbacks.get_output() is None,
                    radio=True,
                )
            ]
            outputs.extend(
                pystray.MenuItem(
                    name,
                    choose(self.callbacks.set_output, key),
                    checked=is_selected(self.callbacks.get_output, key),
                    radio=True,
                )
                for key, name in self.callbacks.list_outputs()
            )
            packs = pystray.Menu(*(
                pystray.MenuItem(
                    name,
                    choose(self.callbacks.set_pack, name),
                    checked=is_selected(self.callbacks.get_pack, name),
                    radio=True,
                )
                for name in self.packs
            ))
            profiles = pystray.Menu(*(
                pystray.MenuItem(
                    name,
                    choose(self.callbacks.set_profile, name),
                    checked=is_selected(self.callbacks.get_profile, name),
                    radio=True,
                )
                for name in self.profiles
            ))
            menu = pystray.Menu(
                pystray.MenuItem("Afficher / masquer", lambda _i, _m: self.callbacks.toggle_visible()),
                pystray.MenuItem("Déplacer / redimensionner (30 s)", lambda _i, _m: self.callbacks.enable_edit()),
                pystray.MenuItem("Taille du groupe", members),
                pystray.MenuItem("Sortie audio", pystray.Menu(*outputs)),
                pystray.MenuItem("Pack de sprites", packs),
                pystray.MenuItem("Profil de détection", profiles),
                pystray.Menu.SEPARATOR,
                pystray.MenuItem("Quitter", lambda _i, _m: self.callbacks.quit()),
            )
            self._icon = pystray.Icon("desktop-band", image, "Desktop Band", menu)
            self._icon.run()
        except Exception as exc:
            self.error = str(exc)
