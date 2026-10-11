# Example gallery

Precomputed results of the released lifter for twenty 1920x1080 ranges from
five public highlight videos, with a display copy of each range.

- `gallery.json` is the examples manifest for `tennis3d dashboard --gallery`.
- `<clip>/result.json` is the exported result: every source frame with its XYZ
  estimate or the reason it is a gap.
- `<clip>/timeline.json` gives each frame's integer PTS, time base, index in
  the complete downloaded video and decoded RGB SHA-256.
- `<clip>/playback.mp4` is the display copy the app plays beside the 3D path.
- `<clip>/playback.json` maps every frame of the display copy to its source
  PTS, and names the source and timeline by SHA-256.

The launch scripts in the repository folder open the app with these examples.
From a development environment:

```bash
uv sync --locked --extra app
uv run tennis3d dashboard --gallery examples/gallery-v2/gallery.json
```

Each example opens with its video, 3D playback and frame inspection. Nothing
else needs to be downloaded.

## Display copies

The reconstruction read lossless 1920x1080 clips, 4.4 GB in total. They are
not in this repository. The manifest's `source_sha256` and each timeline's
`decoded_rgb_sha256` values identify them.

The display copies are lossy re-encodings of those clips, 98 MB in total:
VP9 video in an MP4 file, 1920x1080, without audio. Their pixels differ from
the source pixels. The results were not computed from them, and they cannot
reproduce the results.

Each display copy has the same frames as its source clip, 10,749 in total.
Every frame's timestamp equals its source time elapsed since the clip's first
frame, as an exact rational in the source time base. To check the shipped
files against their receipts and timelines:

```bash
uv run python scripts/gallery_display_copies.py verify --gallery examples/gallery-v2/gallery.json
```

Chrome, Edge and Firefox play VP9 in MP4. Playback was tested in Chromium
only. A browser that cannot decode a clip reports that in the video panel and
still shows the 3D result.

## Sources

Each clip retains the frames whose source time is within its requested
start-inclusive, end-exclusive range. The timeline records those exact times,
and each manifest entry's `rights.source_url` names its source video.

| Clips | Source video | YouTube channel |
| --- | --- | --- |
| 01-01 to 01-04 | [Roger Federer vs Rafael Nadal - Indian Wells 2012 ● 4K Highlights](https://www.youtube.com/watch?v=Mo3COU35RNQ) | Raz Ols |
| 02-01 to 02-04 | [Flavio Cobolli vs Alexander Zverev, Men's Final Highlights by Emirates, Roland-Garros 2026](https://www.youtube.com/watch?v=WfxpIK_TvEQ) | Roland-Garros |
| 03-01 to 03-04 | [Highlights Sinner vs Alcaraz Final, Roland-Garros 2025](https://www.youtube.com/watch?v=aERKJuBtIGw) | Roland-Garros |
| 04-01 to 04-04 | [Aryna Sabalenka vs. Elena Rybakina Highlights, 2026 US Open Final](https://www.youtube.com/watch?v=YCAQnmJBJIk) | US Open Tennis Championships |
| 05-01 to 05-04 | [Novak Djokovic vs Alex de Minaur For The Title, Beijing 2026 Final Highlights](https://www.youtube.com/watch?v=8Dc6BvkUEZM) | Tennis TV |

## What the results are

The results are model estimates. No independent real 3D, ball-image or camera
ground truth exists for these clips, so they do not measure accuracy.

## Rights

The results, timelines and receipts are project output under GPL-3.0-only.
The footage in the display copies belongs to its owners. It is included so the
examples can be watched. Its owners have not given this project permission
to redistribute it. The project's licence does not cover it and grants no
rights to it. The manifest records its redistribution status as unknown.
