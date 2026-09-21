# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Live debug badge showing the detected musical BPM.
- Automatic reconnection when the default Windows audio output changes.
- Contextual gopnik speech bubbles on drops, vocal entries, bass arrivals,
  strong drum hits, climaxes and playback restarts.
- V2 confidence router with temporal evidence, mutually competitive roles and
  idle/groove/playing states.
- Calibrated balanced, taiko, hardbass and soft detection profiles.
- Drop/chorus energy transitions with full-band boost, spotlights and dedicated
  reactions.
- Dominant-role solo foregrounding and procedural idle animations.
- Interchangeable manifest-based sprite packs.
- System-tray controls for visibility, output selection, lineup size, sprite
  pack, detection profile and clean shutdown.
- Temporary 30-second move/resize mode that safely disables click-through.

### Changed

- Animation speed now combines musical tempo, per-role subdivisions, stem
  activity and short percussive boosts.
- Reported BPM and internal animation double-time are now separate, keeping
  slow music measurements honest without making the band look static.
- Tempo detection now uses a continuous rhythmic-envelope autocorrelation
  instead of sparse isolated onsets.
- The debug badge now shows the top role confidences and their routing states.
- StemgenRT's drum and `other` outputs now receive extra transient, harmonic,
  timbral, pitch-stability and note-density analysis before visual routing.

## [0.1.0] - 2026-09-21

### Added

- Transparent, click-through Windows desktop overlay powered by
  `desktop-overlay`.
- One-to-seven-member gopnik band with singer, drummer, bassist, guitarist,
  keyboardist, percussionist and prisyadka dancer.
- Eight or more normalized animation frames for every role.
- Real-time Windows WASAPI loopback capture.
- StemgenRT separation into drums, bass, vocals and other instruments.
- Lightweight DSP fallback when StemgenRT is unavailable.
- Compact and wide layouts, configurable monitor and screen corner.
- Deterministic demo mode and automated unit tests.

[Unreleased]: https://github.com/boubou666/desktop-band/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/boubou666/desktop-band/releases/tag/v0.1.0
