"""Cross-platform global shortcuts with a graceful no-backend fallback."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True, slots=True)
class HotkeyBindings:
    toggle_visible: str = "<ctrl>+<alt>+g"
    edit_mode: str = "<ctrl>+<alt>+m"
    next_lineup: str = "<ctrl>+<alt>+<right>"
    mark_issue: str = "<ctrl>+<alt>+b"
    toggle_recording: str = "<ctrl>+<alt>+r"


class GlobalHotkeys:
    def __init__(
        self,
        toggle_visible: Callable[[], None],
        edit_mode: Callable[[], None],
        next_lineup: Callable[[], None],
        mark_issue: Callable[[], None] | None = None,
        toggle_recording: Callable[[], None] | None = None,
        bindings: HotkeyBindings | None = None,
    ) -> None:
        self.bindings = bindings or HotkeyBindings()
        self._callbacks = {
            self.bindings.toggle_visible: toggle_visible,
            self.bindings.edit_mode: edit_mode,
            self.bindings.next_lineup: next_lineup,
        }
        if mark_issue is not None:
            self._callbacks[self.bindings.mark_issue] = mark_issue
        if toggle_recording is not None:
            self._callbacks[self.bindings.toggle_recording] = toggle_recording
        self._listener = None
        self.error: str | None = None

    def start(self) -> bool:
        try:
            from pynput import keyboard

            self._listener = keyboard.GlobalHotKeys(self._callbacks)
            self._listener.start()
            return True
        except Exception as exc:
            self.error = str(exc)
            return False

    def stop(self) -> None:
        if self._listener is not None:
            try:
                self._listener.stop()
            except Exception:
                pass
