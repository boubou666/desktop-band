"""Simple Canvas-rendered musicians with audio-driven poses."""

from __future__ import annotations

import math
import time
import tkinter as tk

import numpy as np

from .band import Musician, gated_activity
from .detection import RoleRouter
from .moments import MusicalMomentDetector
from .reactions import ShoutDetector, ShoutEvent
from .sprites import SPRITE_SHEETS as _SPRITE_SHEETS
from .sprites import load_sprite_pack


SKIN = "#f3c6a5"
INK = "#111827"
METAL = "#cbd5e1"

_ACTIVITY_ENVELOPES = {
    # Vocals need time to visually form a phrase; an instant attack makes the
    # singer look as if every syllable were a drum hit.
    "singer": (0.24, 0.34),
    # Keep struck instruments readable without making the rest of the band
    # snap between idle and playing poses.
    "drummer": (0.055, 0.16),
    "percussion": (0.055, 0.16),
    "vibing": (0.16, 0.32),
}

# Number of musical beats used for a full active animation cycle. A longer
# cycle looks calmer even though it contains more sprite frames.
_ROLE_BEATS_PER_CYCLE = {
    "keyboard": 6.0,
    "bassist": 4.0,
    "vibing": 4.0,
    "guitarist": 3.0,
    "drummer": 2.0,
    "percussion": 1.5,
}

# Minimum and maximum multipliers applied across the 0..1 activity range.
# Each instrument keeps its musical subdivision while gaining its own dynamics.
_ROLE_DYNAMIC_RANGE = {
    "keyboard": (0.82, 1.04),
    "bassist": (0.80, 1.12),
    "vibing": (0.72, 1.18),
    "guitarist": (0.78, 1.22),
    "drummer": (0.82, 1.18),
    "percussion": (0.82, 1.25),
}

_ROLE_MAX_RATE = {
    "keyboard": 8.0,
    "bassist": 10.0,
    "vibing": 11.0,
    "guitarist": 12.0,
    "drummer": 14.0,
    "percussion": 14.0,
}


def _role_frame_rate(
    role: str,
    bpm: float,
    activity: float,
    frame_count: int,
    onset: float = 0.0,
) -> float:
    """Return a role-specific sprite rate synchronized to musical tempo."""

    if role == "singer":
        # Mouth articulation follows the vocal envelope, not drum subdivisions.
        return 3.0 + max(0.0, min(1.0, activity)) * 2.5
    if frame_count <= 1:
        return 0.0

    # Keep the debug value musically honest while allowing a slow song's
    # half-time pulse to drive a readable animation subdivision.
    reported_bpm = max(50.0, min(180.0, float(bpm)))
    stable_bpm = reported_bpm
    while stable_bpm < 90.0:
        stable_bpm *= 2.0
    beats_per_second = stable_bpm / 60.0
    active_frames = frame_count if role == "vibing" else frame_count - 1
    beats_per_cycle = _ROLE_BEATS_PER_CYCLE.get(role, 4.0)
    rate = active_frames * beats_per_second / beats_per_cycle
    low_multiplier, high_multiplier = _ROLE_DYNAMIC_RANGE.get(
        role, (0.85, 1.10)
    )
    strength = max(0.0, min(1.0, float(activity)))
    rate *= low_multiplier + (high_multiplier - low_multiplier) * strength
    transient = max(0.0, min(1.0, float(onset)))
    if role == "drummer":
        rate *= 1.0 + transient * 0.22
    elif role == "percussion":
        rate *= 1.0 + transient * 0.30
    # A genuine fast tempo should feel substantially more frantic than a slow
    # tempo merely played in double-time for visual readability.
    fast_tempo = max(0.0, min(1.0, (reported_bpm - 110.0) / 50.0))
    rate *= 1.0 + fast_tempo * 0.55
    # Avoid a slideshow at very slow estimates and frantic motion at high BPM.
    maximum_rate = _ROLE_MAX_RATE.get(role, 10.0)
    return max(2.0, min(maximum_rate, rate))


def _smooth_activity(
    current: float,
    target: float,
    dt: float,
    attack: float,
    release: float,
) -> float:
    """Apply a frame-rate-independent visual attack/release envelope."""

    time_constant = attack if target > current else release
    blend = 1.0 - math.exp(-max(0.0, dt) / max(1e-6, time_constant))
    return current + (target - current) * blend


class BandRenderer:
    def __init__(
        self,
        canvas,
        lineup: tuple[Musician, ...],
        width: int,
        height: int,
        compact: bool = False,
        debug: bool = False,
        sprite_pack: str = "gopnik",
        detection_profile: str = "balanced",
    ):
        self.canvas = canvas
        self.lineup = lineup
        self.width = width
        self.height = height
        self.compact = compact
        self.debug = debug
        self.sprite_pack = sprite_pack
        self._frame_size = 166 if compact else 220
        self._sprite_size = 158 if compact else 210
        self.started = time.monotonic()
        self._last_draw = self.started
        self._activity = [0.0 for _ in lineup]
        self._playing = [False for _ in lineup]
        self._animation_phase = [0.0 for _ in lineup]
        self._shout_detector = ShoutDetector()
        self._moment_detector = MusicalMomentDetector()
        self._role_router = RoleRouter(detection_profile)
        self._last_decision = None
        self._shout_event = None
        self._shout_until = 0.0
        self.sprite_error = None
        self._sprite_frames = self._load_sprite_frames()

    def set_lineup(self, lineup: tuple[Musician, ...]) -> None:
        self.lineup = lineup
        self._role_router.reset()
        self._activity = [0.0 for _ in lineup]
        self._playing = [False for _ in lineup]
        self._animation_phase = [0.0 for _ in lineup]

    def resize(self, width: int, height: int) -> None:
        self.width = max(220, int(width))
        self.height = max(180, int(height))

    def set_detection_profile(self, profile: str) -> None:
        self._role_router.set_profile(profile)

    def set_sprite_pack(self, sprite_pack: str) -> None:
        previous = self.sprite_pack
        self.sprite_pack = sprite_pack
        frames = self._load_sprite_frames()
        if not frames:
            self.sprite_pack = previous
            self._sprite_frames = self._load_sprite_frames()
            return
        self._sprite_frames = frames

    def _load_sprite_frames(self):
        frames = {}
        self.sprite_error = None
        try:
            from PIL import Image, ImageTk

            content_by_role = {}
            for sheet in load_sprite_pack(self.sprite_pack):
                path = sheet.path
                row_count = sheet.rows
                role_rows = sheet.role_rows
                if not path.is_file():
                    continue
                with Image.open(path) as source:
                    atlas = source.convert("RGBA")
                atlas_width, atlas_height = atlas.size
                for role, row in role_rows.items():
                    cell_height = atlas_height / row_count
                    content_frames = []
                    for column in range(4):
                        left = round(column * atlas_width / 4)
                        right = round((column + 1) * atlas_width / 4)
                        top = round(row * atlas_height / row_count)
                        bottom = round((row + 1) * atlas_height / row_count)
                        if row_count == 1:
                            cell_width = right - left
                            horizontal_bleed = max(16, round(cell_width * 0.08))
                            horizontal_margin = max(24, round(cell_width * 0.22))
                            extended_left = max(0, left - horizontal_bleed)
                            extended_right = min(atlas_width, right + horizontal_bleed)
                            crop = atlas.crop(
                                (extended_left, top, extended_right, bottom)
                            )
                            crop = self._remove_neighbor_fragments(
                                crop,
                                0,
                                crop.height,
                                left - extended_left + horizontal_margin,
                                right - extended_left - horizontal_margin,
                            )
                        else:
                            bleed = max(16, round(cell_height * 0.08))
                            interior_margin = max(18, round(cell_height * 0.24))
                            extended_top = max(0, top - bleed)
                            extended_bottom = min(atlas_height, bottom + bleed)
                            crop = atlas.crop(
                                (left, extended_top, right, extended_bottom)
                            )
                            crop = self._remove_neighbor_fragments(
                                crop,
                                top - extended_top + interior_margin,
                                bottom - extended_top - interior_margin,
                                0,
                                crop.width,
                            )
                        content_bounds = crop.getbbox()
                        if content_bounds is not None:
                            crop = crop.crop(content_bounds)
                        content_frames.append(crop)
                    content_by_role.setdefault(role, []).extend(content_frames)

            for role, content_frames in content_by_role.items():
                # Every pose of one role, including multiple singer sheets,
                # shares a normalization canvas. Props and open-mouth frames
                # therefore never change the character's apparent scale.
                common_width = max(frame.width for frame in content_frames)
                common_height = max(frame.height for frame in content_frames)
                role_frames = []
                for crop in content_frames:
                    normalized = Image.new(
                        "RGBA", (common_width, common_height), (0, 0, 0, 0)
                    )
                    normalized.alpha_composite(
                        crop,
                        ((common_width - crop.width) // 2, common_height - crop.height),
                    )
                    normalized.thumbnail(
                        (self._sprite_size, self._sprite_size),
                        Image.Resampling.LANCZOS,
                    )
                    frame = Image.new(
                        "RGBA",
                        (self._frame_size, self._frame_size),
                        (0, 0, 0, 0),
                    )
                    frame.alpha_composite(
                        normalized,
                        (
                            (self._frame_size - normalized.width) // 2,
                            self._frame_size - normalized.height,
                        ),
                    )
                    # Windows/Tk implements transparency with a chroma key.
                    # Hard alpha avoids a visible magenta fringe.
                    alpha = frame.getchannel("A").point(
                        lambda value: 255 if value >= 80 else 0
                    )
                    frame.putalpha(alpha)
                    role_frames.append(ImageTk.PhotoImage(frame, master=self.canvas))
                frames[role] = tuple(role_frames)
            return frames
        except Exception as exc:
            self.sprite_error = "sprites indisponibles : {}".format(exc)
            return {}

    @staticmethod
    def _remove_neighbor_fragments(
        crop,
        core_top: int,
        core_bottom: int,
        core_left: int,
        core_right: int,
    ):
        """Keep only alpha components connected to the cell's safe interior.

        Image-generated atlases occasionally let a shoe or hat cross a grid
        boundary. We read a small bleed around the cell to recover the wanted
        character, then discard disconnected pieces belonging to its neighbor.
        """

        from PIL import Image

        alpha = np.asarray(crop.getchannel("A"))
        mask = alpha >= 32
        height, width = mask.shape
        core_top = max(0, min(height, core_top))
        core_bottom = max(core_top, min(height, core_bottom))
        core_left = max(0, min(width, core_left))
        core_right = max(core_left, min(width, core_right))
        core = np.zeros_like(mask)
        core[core_top:core_bottom, core_left:core_right] = mask[
            core_top:core_bottom, core_left:core_right
        ]
        visited = core.copy()
        ys, xs = np.nonzero(core)
        stack = list(zip(ys.tolist(), xs.tolist()))

        while stack:
            y, x = stack.pop()
            if y > 0 and mask[y - 1, x] and not visited[y - 1, x]:
                visited[y - 1, x] = True
                stack.append((y - 1, x))
            if y + 1 < height and mask[y + 1, x] and not visited[y + 1, x]:
                visited[y + 1, x] = True
                stack.append((y + 1, x))
            if x > 0 and mask[y, x - 1] and not visited[y, x - 1]:
                visited[y, x - 1] = True
                stack.append((y, x - 1))
            if x + 1 < width and mask[y, x + 1] and not visited[y, x + 1]:
                visited[y, x + 1] = True
                stack.append((y, x + 1))

        cleaned = crop.copy()
        cleaned_alpha = np.where(visited, alpha, 0).astype(np.uint8)
        cleaned.putalpha(Image.fromarray(cleaned_alpha, mode="L"))
        return cleaned

    def draw(self, features, status: str, error: str | None = None) -> None:
        self.canvas.delete("band")
        now = time.monotonic()
        elapsed = now - self.started
        # A debugger pause or a temporarily blocked UI must not skip the whole
        # envelope on the first frame back.
        dt = min(0.1, max(0.0, now - self._last_draw))
        self._last_draw = now
        margin = 20 if self.compact else 40
        usable = self.width - 2 * margin
        spacing = usable / max(1, len(self.lineup))
        ground = self.height - (24 if self.compact else 34)
        roles = tuple(musician.role for musician in self.lineup)
        decision = self._role_router.update(features, roles, dt)
        self._last_decision = decision
        role_scores = decision.scores
        moment = self._moment_detector.update(features, role_scores, now)
        drop_active = self._moment_detector.is_active(now)
        if moment is not None and now >= self._shout_until:
            self._shout_event = ShoutEvent(moment.text, "vibing", moment.duration)
            self._shout_until = now + moment.duration

        role_positions = {
            musician.role: margin + spacing * (index + 0.5)
            for index, musician in enumerate(self.lineup)
        }
        if drop_active:
            self._draw_drop_spotlight(elapsed)

        for index, musician in enumerate(self.lineup):
            raw_score = role_scores[musician.role]
            state = decision.states[musician.role]
            if state == "playing":
                target, self._playing[index] = gated_activity(
                    musician.role, raw_score, self._playing[index]
                )
            elif state == "groove":
                target = 0.08 + raw_score * 0.32
                self._playing[index] = False
            else:
                target = 0.0
                self._playing[index] = False
            if drop_active and musician.role != "singer":
                target = max(target, 0.58)
            attack, release = _ACTIVITY_ENVELOPES.get(
                musician.role, (0.14, 0.28)
            )
            self._activity[index] = _smooth_activity(
                self._activity[index], target, dt, attack, release
            )
            role_frames = self._sprite_frames.get(musician.role, ())
            if self._activity[index] >= 0.08 and role_frames:
                self._animation_phase[index] += dt * _role_frame_rate(
                    musician.role,
                    features.bpm,
                    self._activity[index],
                    len(role_frames),
                    features.onset,
                )
            else:
                self._animation_phase[index] = 0.0

        dominant = decision.dominant_role
        draw_order = list(range(len(self.lineup)))
        if dominant is not None:
            draw_order.sort(key=lambda i: self.lineup[i].role == dominant)
        for index in draw_order:
            musician = self.lineup[index]
            x = role_positions[musician.role]
            self._draw_musician(
                musician, x, ground, elapsed, features, index,
                self._activity[index],
                state=decision.states[musician.role],
                solo=musician.role == dominant,
                drop_active=drop_active,
            )

        if not drop_active:
            shout = self._shout_detector.update(features, role_scores, now)
            if shout is not None:
                self._shout_event = shout
                self._shout_until = now + shout.duration
        if self._shout_event is not None and now < self._shout_until:
            self._draw_shout(self._shout_event, role_positions)

        # The compact overlay stays visually silent during normal operation.
        # Status remains available in wide mode, while errors are always shown.
        if error or not self.compact:
            color = "#fb7185" if error else "#d1fae5"
            message = "{} — {:.0f} BPM — {}".format(
                status,
                features.bpm,
                "son actif" if features.active else "silence",
            )
            if error:
                message += " (utilise --demo pour tester)"
            self._rectangle(
                8, 8, min(self.width - 8, 520), 31,
                fill="#111827", outline=color, width=1,
            )
            self._text(
                12, 12, message, anchor="nw", fill=color,
                font=("Segoe UI", 9, "bold"),
            )
        elif self.debug:
            ranked = sorted(
                (
                    (score, role, decision.states[role])
                    for role, score in role_scores.items()
                    if role != "vibing"
                ),
                reverse=True,
            )[:3]
            details = "  ".join(
                "{}:{:.0f}% {}".format(role[:4].upper(), score * 100, state[0].upper())
                for score, role, state in ranked
            )
            badge_left = 8
            badge_right = min(self.width - 8, badge_left + 330)
            self._rectangle(
                badge_left, 7, badge_right, 33,
                fill="#111827", outline="#22d3ee", width=2,
            )
            self._text(
                badge_left + 8,
                11,
                "BPM {:.0f}  {}".format(features.bpm, details),
                anchor="nw",
                fill="#ecfeff",
                font=("Consolas", 8, "bold"),
            )

    def _draw_drop_spotlight(self, elapsed: float) -> None:
        pulse = 0.5 + 0.5 * math.sin(elapsed * 14.0)
        center = self.width / 2
        color = "#facc15" if pulse > 0.42 else "#fb7185"
        self._line(center - 150, 0, center - 45, self.height - 20,
                   fill=color, width=4)
        self._line(center + 150, 0, center + 45, self.height - 20,
                   fill=color, width=4)
        self._oval(center - 180, self.height - 34, center + 180, self.height - 15,
                   fill="", outline=color, width=3)

    def _draw_shout(self, event, role_positions: dict[str, float]) -> None:
        target_x = role_positions.get(event.role, self.width / 2)
        bubble_width = max(68, min(126, 24 + len(event.text) * 8))
        half = bubble_width / 2
        bubble_x = max(half + 4, min(self.width - half - 4, target_x))
        color = next(
            (
                musician.color
                for musician in self.lineup
                if musician.role == event.role
            ),
            "#f8fafc",
        )
        self._rectangle(
            bubble_x - half, 7, bubble_x + half, 34,
            fill="#111827", outline=color, width=2,
        )
        self._text(
            bubble_x, 12, event.text, anchor="n", fill="#f8fafc",
            font=("Segoe UI", 10, "bold"),
        )
        self._line(
            bubble_x, 34, target_x, 42,
            fill=color, width=3,
        )

    def _draw_musician(
        self, musician, x, ground, elapsed, features, index, activity,
        *, state="playing", solo=False, drop_active=False,
    ) -> None:
        if musician.role in self._sprite_frames:
            self._draw_sprite_musician(
                musician, x, ground, elapsed, features, index, activity,
                state=state, solo=solo, drop_active=drop_active,
            )
            return

        beat_hz = max(0.5, features.bpm / 60.0)
        sway = math.sin(elapsed * math.tau * beat_hz + index * 0.8)
        bob = (-2.0 - 3.0 * features.onset) * activity + 1.8 * sway * activity
        x += sway * activity * 1.6

        if activity > 0.08:
            halo = 2 + int(activity * 4)
            self._oval(
                x - 48 - halo, ground - 202 - halo,
                x + 48 + halo, ground + 8 + halo,
                fill="", outline=musician.color, width=halo,
            )
        self._oval(x - 38, ground - 7, x + 38, ground + 5, fill="#020617", outline="")
        hip_y = ground - 72 + bob
        shoulder_y = ground - 126 + bob
        head_y = ground - 163 + bob

        self._line(x - 10, hip_y, x - 14 - sway * 3, ground, fill=INK, width=7)
        self._line(x + 10, hip_y, x + 14 + sway * 3, ground, fill=INK, width=7)
        self._oval(x - 24, shoulder_y - 7, x + 24, hip_y + 10,
                   fill=musician.color, outline=INK, width=3)
        self._oval(x - 20, head_y - 24, x + 20, head_y + 18,
                   fill=SKIN, outline=INK, width=3)
        self._arc(x - 21, head_y - 27, x + 21, head_y + 5,
                  start=0, extent=180, fill=INK, outline=INK)
        self._oval(x - 9, head_y - 4, x - 5, head_y, fill=INK, outline="")
        self._oval(x + 5, head_y - 4, x + 9, head_y, fill=INK, outline="")

        mouth_open = 2 + 6 * features.vocal if musician.role == "singer" else 2
        self._oval(x - 5, head_y + 7, x + 5, head_y + 7 + mouth_open,
                   fill="#7f1d1d", outline=INK, width=1)

        draw_instrument = getattr(self, "_instrument_" + musician.role)
        draw_instrument(
            x, shoulder_y, hip_y, ground, activity, features, sway, elapsed
        )
        self._rectangle(
            x - 46, ground + 9, x + 46, ground + 27,
            fill="#111827", outline=musician.color, width=1,
        )
        self._text(x, ground + 12, musician.label, anchor="n", fill="#f8fafc",
                   font=("Segoe UI", 8, "bold"))
        meter_left = x - 28
        meter_top = ground + 28
        self._rectangle(
            meter_left, meter_top, meter_left + 56, meter_top + 5,
            fill="#111827", outline="#475569", width=1,
        )
        if activity > 0.01:
            self._rectangle(
                meter_left + 1, meter_top + 1,
                meter_left + 1 + 54 * activity, meter_top + 4,
                fill=musician.color, outline="",
            )

    def _draw_sprite_musician(
        self, musician, x, ground, elapsed, features, index, activity,
        *, state="playing", solo=False, drop_active=False,
    ) -> None:
        beat_hz = max(0.5, features.bpm / 60.0)
        pulse = math.sin(elapsed * math.tau * beat_hz + index * 0.7)
        idle_breath = math.sin(elapsed * 1.35 + index) * 0.65
        bob = (
            -activity * (3.0 + features.onset * 8.0)
            + pulse * activity * 2.0
            + idle_breath * (1.0 - min(1.0, activity * 3.0))
        )
        if solo:
            ground += 4

        role_frames = self._sprite_frames[musician.role]
        if state != "playing" and not drop_active:
            frame_index = 0
        elif musician.role == "singer" and len(role_frames) >= 8:
            # The smoothed visual envelope prevents raw vocal estimates from
            # snapping the mouth from closed to fully open in one UI frame.
            vocal_strength = activity
            if vocal_strength < 0.34:
                mouth_frames = (4, 7)
            elif vocal_strength < 0.68:
                mouth_frames = (4, 5, 7)
            else:
                mouth_frames = (5, 6, 7)
            frame_index = mouth_frames[
                int(self._animation_phase[index]) % len(mouth_frames)
            ]
        elif musician.role == "vibing" and len(role_frames) >= 8:
            # Eight consecutive poses make the alternating-leg prisyadka read
            # as a dance instead of a two-frame kick.
            frame_index = int(self._animation_phase[index]) % len(role_frames)
        elif len(role_frames) > 4:
            frame_index = 1 + int(self._animation_phase[index]) % (
                len(role_frames) - 1
            )
        elif features.onset > 0.42:
            frame_index = 2
        else:
            pose_speed = min(4.0, max(2.0, beat_hz * 1.5))
            frame_index = 1 if int(elapsed * pose_speed) % 2 == 0 else 3

        sprite = role_frames[frame_index % len(role_frames)]
        if solo:
            halo_width = 32 + 4 * math.sin(elapsed * 5.0)
            self._oval(
                x - halo_width, ground - 8 + bob,
                x + halo_width, ground + 7 + bob,
                fill=musician.color, outline="#f8fafc", width=2,
            )
        if self.compact and activity > 0.08:
            glow_width = 24 + activity * 22
            self._oval(
                x - glow_width, ground - 4 + bob,
                x + glow_width, ground + 5 + bob,
                fill=musician.color, outline="",
            )
        elif activity > 0.08:
            halo_width = 2 + int(activity * 5)
            self._oval(
                x - 82, ground - 218 + bob, x + 82, ground + 5 + bob,
                fill="", outline=musician.color, width=halo_width,
            )
        self.canvas.create_image(
            x, ground + 5 + bob, image=sprite, anchor="s", tags="band"
        )
        if state == "idle":
            self._draw_idle_details(musician.role, x, ground + bob, elapsed)

        if self.compact:
            meter_left = x - 24
            meter_top = ground + 10
            self._rectangle(
                meter_left, meter_top, meter_left + 48, meter_top + 4,
                fill="#111827", outline="", width=0,
            )
            if activity > 0.01:
                self._rectangle(
                    meter_left, meter_top,
                    meter_left + 48 * activity, meter_top + 4,
                    fill=musician.color, outline="",
                )
            return

        self._rectangle(
            x - 48, ground + 9, x + 48, ground + 28,
            fill="#111827", outline=musician.color, width=1,
        )
        self._text(
            x, ground + 12, musician.label, anchor="n", fill="#f8fafc",
            font=("Segoe UI", 9, "bold"),
        )
        meter_left = x - 38
        meter_top = ground + 29
        self._rectangle(
            meter_left, meter_top, meter_left + 76, meter_top + 6,
            fill="#111827", outline="#64748b", width=1,
        )
        if activity > 0.01:
            self._rectangle(
                meter_left + 1, meter_top + 1,
                meter_left + 1 + 74 * activity, meter_top + 5,
                fill=musician.color, outline="",
            )

    def _draw_idle_details(self, role: str, x: float, ground: float, elapsed: float) -> None:
        """Small procedural idle loops layered over the resting sprite."""

        phase = elapsed % 6.0
        if role == "bassist" and phase < 3.5:
            rise = phase * 7.0
            drift = math.sin(phase * 2.2) * 4.0
            radius = 2.0 + phase * 0.45
            self._oval(
                x + 29 + drift - radius, ground - 123 - rise - radius,
                x + 29 + drift + radius, ground - 123 - rise + radius,
                fill="", outline="#cbd5e1", width=2,
            )
        elif role == "vibing":
            tap = max(0.0, math.sin(elapsed * 1.8))
            if tap > 0.82:
                self._line(x + 30, ground - 2, x + 48, ground - 2,
                           fill="#f8fafc", width=2)
                self._line(x + 39, ground - 7, x + 39, ground + 2,
                           fill="#f8fafc", width=2)
            glint = 2.0 + 1.5 * math.sin(elapsed * 2.4)
            self._oval(x - 38 - glint, ground - 60 - glint,
                       x - 38 + glint, ground - 60 + glint,
                       fill="#e0f2fe", outline="")
        elif role == "singer" and phase > 4.5:
            self._line(x - 17, ground - 111, x - 7, ground - 109,
                       fill="#ef4444", width=2)
            self._line(x + 7, ground - 109, x + 17, ground - 111,
                       fill="#ef4444", width=2)

    def _instrument_singer(self, x, shoulder, hip, ground, activity, f, sway, elapsed):
        hand_y = shoulder + 22 - activity * 12
        self._line(x - 17, shoulder + 12, x - 34, hand_y, fill=SKIN, width=7)
        wave = math.sin(elapsed * 8.0) * activity * 10
        self._line(x + 17, shoulder + 12, x + 31 + wave, shoulder + 31,
                   fill=SKIN, width=7)
        self._line(x - 36, hand_y - 3, x - 36, ground - 3, fill=METAL, width=3)
        self._oval(x - 42, hand_y - 10, x - 32, hand_y, fill="#334155", outline=INK)

    def _instrument_drummer(self, x, shoulder, hip, ground, activity, f, sway, elapsed):
        stroke = math.sin(elapsed * math.tau * max(1.0, f.bpm / 30.0))
        left_lift = max(0.0, stroke) * 35 * activity
        right_lift = max(0.0, -stroke) * 35 * activity
        accent = f.onset * 24
        left_hand = hip - 17 - left_lift - accent
        right_hand = hip - 17 - right_lift - accent * 0.65
        self._line(x - 17, shoulder + 10, x - 34, left_hand, fill=SKIN, width=7)
        self._line(x + 17, shoulder + 10, x + 34, right_hand, fill=SKIN, width=7)
        self._line(x - 34, left_hand, x - 45, left_hand - 27, fill="#f8fafc", width=3)
        self._line(x + 34, right_hand, x + 45, right_hand - 27, fill="#f8fafc", width=3)
        self._oval(x - 43, hip - 20, x + 6, ground - 2, fill="#be123c", outline=INK, width=3)
        self._oval(x + 3, hip - 9, x + 39, ground - 8, fill="#e11d48", outline=INK, width=3)
        self._line(x - 50, hip - 42, x - 50, ground - 5, fill=METAL, width=2)
        cymbal_tilt = f.onset * 6
        self._line(x - 68, hip - 46 - cymbal_tilt, x - 32, hip - 40 + cymbal_tilt,
                   fill="#fde68a", width=6)
        if f.onset > 0.25:
            self._line(x - 76, hip - 52, x - 86, hip - 62, fill="#fde68a", width=2)
            self._line(x - 31, hip - 50, x - 21, hip - 61, fill="#fde68a", width=2)

    def _instrument_bassist(self, x, shoulder, hip, ground, activity, f, sway, elapsed):
        self._string_instrument(
            x, shoulder, hip, activity, f, "#0f766e", elapsed, bass=True
        )

    def _instrument_guitarist(self, x, shoulder, hip, ground, activity, f, sway, elapsed):
        self._string_instrument(
            x, shoulder, hip, activity, f, "#7c3aed", elapsed, bass=False
        )

    def _string_instrument(self, x, shoulder, hip, activity, f, color, elapsed, *, bass):
        speed = 12.0 if bass else 20.0
        strum = math.sin(elapsed * speed) * activity * (11 if bass else 18)
        fret = math.sin(elapsed * 5.0) * activity * 9
        self._line(x - 16, shoulder + 12, x - 36 + fret, hip - 31 - fret * 0.35,
                   fill=SKIN, width=7)
        self._line(x + 15, shoulder + 13, x + 12, hip - 11 + strum, fill=SKIN, width=7)
        self._oval(x - 30, hip - 35, x + 24, hip + 9, fill=color, outline=INK, width=3)
        neck_end = x + (66 if bass else 58)
        self._line(x + 10, hip - 25, neck_end, shoulder + 2, fill="#92400e", width=9)
        self._line(x + 7, hip - 25, neck_end, shoulder, fill="#fef3c7", width=2)
        self._oval(x - 6, hip - 19, x + 7, hip - 7, fill="#020617", outline="")
        if activity > 0.35:
            self._line(x + 28, hip - 23, x + 38, hip - 30, fill=color, width=2)
            self._line(x + 31, hip - 15, x + 43, hip - 17, fill=color, width=2)

    def _instrument_keyboard(self, x, shoulder, hip, ground, activity, f, sway, elapsed):
        travel = math.sin(elapsed * 10.0) * activity * 23
        left_hand_x = x - 22 + travel
        right_hand_x = x + 22 - travel
        self._line(x - 15, shoulder + 13, left_hand_x, hip - 7,
                   fill=SKIN, width=7)
        self._line(x + 15, shoulder + 13, right_hand_x, hip - 7,
                   fill=SKIN, width=7)
        self._rectangle(x - 50, hip - 10, x + 50, hip + 12, fill="#e2e8f0", outline=INK, width=3)
        for offset in range(-40, 41, 10):
            self._line(x + offset, hip - 8, x + offset, hip + 10, fill="#475569", width=1)
        if activity > 0.1:
            self._rectangle(left_hand_x - 5, hip - 8, left_hand_x + 4, hip + 9,
                            fill="#38bdf8", outline="")
            self._rectangle(right_hand_x - 4, hip - 8, right_hand_x + 5, hip + 9,
                            fill="#38bdf8", outline="")
        self._line(x - 38, hip + 12, x - 28, ground, fill=METAL, width=3)
        self._line(x + 38, hip + 12, x + 28, ground, fill=METAL, width=3)

    def _instrument_percussion(self, x, shoulder, hip, ground, activity, f, sway, elapsed):
        stroke = math.sin(elapsed * 17.0)
        left_hit = max(0.0, stroke) * activity * 34 + f.onset * 12
        right_hit = max(0.0, -stroke) * activity * 34 + f.onset * 12
        self._line(x - 16, shoulder + 12, x - 24, hip - 8 - left_hit,
                   fill=SKIN, width=7)
        self._line(x + 16, shoulder + 12, x + 24, hip - 8 - right_hit,
                   fill=SKIN, width=7)
        self._polygon(x - 38, hip - 12, x - 3, hip - 12, x - 8, ground,
                      x - 32, ground, fill="#f97316", outline=INK, width=3)
        self._polygon(x + 3, hip - 12, x + 38, hip - 12, x + 32, ground,
                      x + 8, ground, fill="#fb923c", outline=INK, width=3)
        self._oval(x - 39, hip - 17, x - 2, hip - 8, fill="#fed7aa", outline=INK)
        self._oval(x + 2, hip - 17, x + 39, hip - 8, fill="#fed7aa", outline=INK)

    def _line(self, *coords, **options):
        return self.canvas.create_line(*coords, tags="band", **options)

    def _oval(self, *coords, **options):
        return self.canvas.create_oval(*coords, tags="band", **options)

    def _rectangle(self, *coords, **options):
        return self.canvas.create_rectangle(*coords, tags="band", **options)

    def _polygon(self, *coords, **options):
        return self.canvas.create_polygon(*coords, tags="band", **options)

    def _arc(self, *coords, **options):
        return self.canvas.create_arc(*coords, tags="band", **options)

    def _text(self, x, y, text, **options):
        return self.canvas.create_text(
            x, y, text=text, tags="band", **options
        )
