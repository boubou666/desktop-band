"""Small native helpers that complement desktop-overlay on each desktop OS."""

from __future__ import annotations

import sys


def set_click_through(window, enabled: bool, width: int, height: int) -> bool:
    """Toggle mouse passthrough on Windows, macOS or X11/XWayland."""

    if sys.platform == "win32":
        return _windows_click_through(window, enabled)
    if sys.platform == "darwin":
        return _macos_click_through(window, enabled)
    if sys.platform.startswith("linux"):
        return _x11_click_through(window, enabled, width, height)
    return False


def _windows_click_through(window, enabled: bool) -> bool:
    try:
        import ctypes

        window.update_idletasks()
        hwnd = ctypes.windll.user32.GetParent(window.winfo_id()) or window.winfo_id()
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


def _macos_click_through(window, enabled: bool) -> bool:
    try:
        import ctypes
        import ctypes.util

        objc = ctypes.cdll.LoadLibrary(ctypes.util.find_library("objc"))
        objc.sel_registerName.restype = ctypes.c_void_p
        objc.sel_registerName.argtypes = [ctypes.c_char_p]
        objc.objc_msgSend.restype = ctypes.c_void_p
        view = ctypes.c_void_p(int(window.winfo_id()))
        ns_window = objc.objc_msgSend(
            view, ctypes.c_void_p(objc.sel_registerName(b"window"))
        )
        if not ns_window:
            return False
        objc.objc_msgSend.restype = None
        objc.objc_msgSend(
            ctypes.c_void_p(ns_window),
            ctypes.c_void_p(objc.sel_registerName(b"setIgnoresMouseEvents:")),
            ctypes.c_bool(bool(enabled)),
        )
        return True
    except Exception:
        return False


def _x11_click_through(window, enabled: bool, width: int, height: int) -> bool:
    try:
        import ctypes
        import ctypes.util

        class XRectangle(ctypes.Structure):
            _fields_ = (
                ("x", ctypes.c_short),
                ("y", ctypes.c_short),
                ("width", ctypes.c_ushort),
                ("height", ctypes.c_ushort),
            )

        x11 = ctypes.cdll.LoadLibrary(ctypes.util.find_library("X11"))
        xext = ctypes.cdll.LoadLibrary(ctypes.util.find_library("Xext"))
        x11.XOpenDisplay.restype = ctypes.c_void_p
        display = x11.XOpenDisplay(None)
        if not display:
            return False
        try:
            rectangles = None
            count = 0
            if not enabled:
                rectangle = XRectangle(0, 0, max(1, width), max(1, height))
                rectangles = ctypes.pointer(rectangle)
                count = 1
            # ShapeInput=2, ShapeSet=0, Unsorted=0.
            xext.XShapeCombineRectangles(
                ctypes.c_void_p(display), ctypes.c_ulong(int(window.winfo_id())),
                2, 0, 0, rectangles, count, 0, 0,
            )
            x11.XFlush(ctypes.c_void_p(display))
            return True
        finally:
            x11.XCloseDisplay(ctypes.c_void_p(display))
    except Exception:
        return False
