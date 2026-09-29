# ServoSkull voice clips

This folder holds the short voice clips for the bench state console.
You type a state, the status LED shows its pattern, and the skull
speaks one word through the amp.

The `.wav` files are not in git (see `.gitignore`). Record your own.

## Files and words

| File            | Word ServoSkull uses | Required |
|-----------------|----------------------|----------|
| `idle.wav`      | "Awaiting"           | yes      |
| `listening.wav` | "Attending"          | yes      |
| `thinking.wav`  | "Cogitating"         | yes      |
| `speaking.wav`  | "Transmitting"       | yes      |
| `offline.wav`   | "Severed"            | yes      |
| `error.wav`     | "Malfunction"        | yes      |
| `test.wav`      | "Hello World!"       | optional |

The file names must match exactly. The words are up to you: say
whatever you like, in any language.

- The six state files are required; the console will not start without them.
- `test.wav` is optional: typing `test` plays it with the speaking LED pattern.
  If it is missing, the console warns once and carries on.

## Format

- WAV, plain PCM, 48000 Hz, 16-bit, mono
- One word per file, silence trimmed from both ends

Float WAVs (the default export in some apps) will not load.

## Record and convert

Record anywhere (phone, laptop), in any format, one word per file,
as `raw/<name>.<ext>`, e.g. `raw/idle.m4a`. Keep one file per name in
`raw/`. Then convert, trim and fade them all from the repo root:

```bash
mkdir -p software/skull_core/voice
for s in idle listening thinking speaking offline error test; do
  ffmpeg -hide_banner -y -i raw/$s.* \
    -af "silenceremove=start_periods=1:start_threshold=-45dB,areverse,silenceremove=start_periods=1:start_threshold=-45dB,afade=t=in:d=0.01,areverse,afade=t=in:d=0.01" \
    -ar 48000 -ac 1 -c:a pcm_s16le software/skull_core/voice/$s.wav
done
```

- If a soft first sound gets cut off, change `-45dB` to `-55dB`.
- If you skipped `test`, ffmpeg prints an error for `raw/test.*`. That is fine.
- Listen to each file once.

## Check a file

```bash
ffprobe -hide_banner software/skull_core/voice/idle.wav
```

The `Stream` line should show `pcm_s16le`, `48000 Hz` and `1 channels`:

```text
Stream #0:0: Audio: pcm_s16le ([1][0][0][0] / 0x0001), 48000 Hz, 1 channels, s16, 768 kb/s
```

Shorter, for all files at once:

```bash
for f in software/skull_core/voice/*.wav; do
  echo "$f"
  ffprobe -v error -show_entries stream=codec_name,sample_rate,channels \
    -show_entries format=duration -of default=nw=1 "$f"
done
```

Each file should print `codec_name=pcm_s16le`, `sample_rate=48000`,
`channels=1`, and a `duration` about as long as the word itself.

## Fixing WAVs you already recorded

If your `.wav` files are already in the voice folder but have the wrong
format (stereo, 44.1 kHz, float) or still have silence around the word,
fix them in place.

ffmpeg cannot overwrite the file it is reading. So each command writes
to a temporary `.part` file first and only replaces the original if
ffmpeg succeeded. If it fails, the temp file is removed and your
original is untouched.

### 1. Keep the originals

Copy them to `raw/` first. Files already in `raw/` are never
overwritten, so running this twice is safe:

```bash
mkdir -p raw
for f in software/skull_core/voice/*.wav; do
  [ -e "raw/${f##*/}" ] || cp "$f" raw/
done
```

With the originals in `raw/`, you can also just rerun the
[Record and convert](#record-and-convert) loop instead of the commands below.

### 2. Fix one file

Format only (48000 Hz, 16-bit, mono):

```bash
f=software/skull_core/voice/idle.wav
tmp="${f%.wav}.part"
ffmpeg -hide_banner -loglevel error -y -i "$f" \
  -ar 48000 -ac 1 -c:a pcm_s16le -f wav "$tmp" && mv "$tmp" "$f" || rm -f "$tmp"
```

Format, trim and fade:

```bash
f=software/skull_core/voice/idle.wav
tmp="${f%.wav}.part"
ffmpeg -hide_banner -loglevel error -y -i "$f" \
  -af "silenceremove=start_periods=1:start_threshold=-45dB,areverse,silenceremove=start_periods=1:start_threshold=-45dB,afade=t=in:d=0.01,areverse,afade=t=in:d=0.01" \
  -ar 48000 -ac 1 -c:a pcm_s16le -f wav "$tmp" && mv "$tmp" "$f" || rm -f "$tmp"
```

### 3. Fix the whole folder

Format only:

```bash
for f in software/skull_core/voice/*.wav; do
  tmp="${f%.wav}.part"
  ffmpeg -hide_banner -loglevel error -y -i "$f" \
    -ar 48000 -ac 1 -c:a pcm_s16le -f wav "$tmp" && mv "$tmp" "$f" || rm -f "$tmp"
done
```

Format, trim and fade:

```bash
for f in software/skull_core/voice/*.wav; do
  tmp="${f%.wav}.part"
  ffmpeg -hide_banner -loglevel error -y -i "$f" \
    -af "silenceremove=start_periods=1:start_threshold=-45dB,areverse,silenceremove=start_periods=1:start_threshold=-45dB,afade=t=in:d=0.01,areverse,afade=t=in:d=0.01" \
    -ar 48000 -ac 1 -c:a pcm_s16le -f wav "$tmp" && mv "$tmp" "$f" || rm -f "$tmp"
done
```

Trimming twice is harmless, but each run adds another short fade.
If a file sounds wrong afterwards, copy it back from `raw/` and try again.

Then [check the files](#check-a-file) and listen to each one.

### Symptoms and causes

| Symptom                               | Likely cause                              | Fix                                   |
|---------------------------------------|-------------------------------------------|---------------------------------------|
| File won't load                       | Float WAV (`pcm_f32le`) or not 48000 Hz   | Format only, or format + trim         |
| Pause before the word                 | Silence not trimmed                       | Format + trim                         |
| Click at the start or end             | No fade                                   | Format + trim (it adds a 10 ms fade)  |
| First soft sound is cut off           | Trim threshold too high                   | Use `-55dB` instead of `-45dB`        |
| ffprobe shows `2 channels` or `stereo` | Stereo file                               | Format only, or format + trim         |

## Run

From the repo root (the `-m` is required):

```bash
python -m software.skull_core.state_console
```

Type a state name, `test`, or `quit`.

These clips are a bench stand-in. In M3 the skull speaks with
text-to-speech instead, and this folder is no longer needed.
