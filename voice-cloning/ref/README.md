# Reference voices

Drop a clean 3–10 s speech clip here as `<speaker_id>.<ext>`
(`.wav/.mp3/.m4a/.ogg/.flac`). On server start, each clip is encoded once and
cached to `voices/<speaker_id>.pt`, then available via:

    POST /tts/speaker/<speaker_id>

You can also register at runtime: `POST /speakers` (saves the clip here + caches it).
