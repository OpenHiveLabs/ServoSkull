import math
import sys
import types
import warnings
import wave

import numpy as np
import pytest

# record.py imports sounddevice at the top. The Surface venv doesn't have it,
# and these tests never touch audio hardware, so a stand-in module is enough.
try:
    import sounddevice  # noqa: F401
except (ImportError, OSError):
    sys.modules["sounddevice"] = types.ModuleType("sounddevice")

from software.skull_core import record as recorder
from software.skull_core.record import SAMPLE_RATE, rms_dbfs, save_wav


def test_sample_rate_matches_the_amp_path():
    assert SAMPLE_RATE == 48_000


def test_half_scale_is_minus_6_dbfs():
    samples = np.full(4800, 16384, dtype=np.int16)
    assert rms_dbfs(samples) == pytest.approx(-6.0206, abs=0.01)


def test_full_scale_sine_is_minus_3_dbfs():
    t = np.arange(SAMPLE_RATE) / SAMPLE_RATE
    samples = np.round(32767 * np.sin(2 * np.pi * 1000 * t)).astype(np.int16)
    assert rms_dbfs(samples) == pytest.approx(-3.0103, abs=0.01)


def test_loud_int16_does_not_overflow():
    # Squaring int16 before converting to float wraps around. -6 dBFS must stay -6.
    samples = np.array([30000, -30000] * 2400, dtype=np.int16)
    assert rms_dbfs(samples) == pytest.approx(20 * math.log10(30000 / 32768), abs=0.01)


def test_silence_is_finite_and_very_quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a log10(0) RuntimeWarning fails the test
        value = rms_dbfs(np.zeros(4800, dtype=np.int16))
    assert math.isfinite(value)
    assert value < -90.0  # one LSB of int16 is about -90.3 dBFS


def test_accepts_the_frames_by_1_shape_from_sd_rec():
    samples = np.full((4800, 1), 16384, dtype=np.int16)
    assert rms_dbfs(samples) == pytest.approx(-6.0206, abs=0.01)


@pytest.mark.parametrize("shape", [(4800,), (4800, 1)])
def test_save_wav_round_trip(tmp_path, shape):
    samples = (np.arange(4800) % 200 - 100).astype(np.int16).reshape(shape)
    path = tmp_path / "clip.wav"
    save_wav(path, samples)
    with wave.open(str(path), "rb") as w:
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
        assert w.getframerate() == SAMPLE_RATE
        assert w.getnframes() == 4800
        back = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    assert np.array_equal(back, samples.reshape(-1))


def test_save_wav_uses_the_rate_it_is_given(tmp_path):
    path = tmp_path / "clip16k.wav"
    save_wav(path, np.zeros(1600, dtype=np.int16), rate=16_000)
    with wave.open(str(path), "rb") as w:
        assert w.getframerate() == 16_000


@pytest.mark.parametrize("seconds", ["0", "-1", "nan", "inf"])
def test_main_rejects_nonpositive_or_nonfinite_duration(monkeypatch, seconds):
    monkeypatch.setattr(sys, "argv", ["record.py", "--seconds", seconds])
    with pytest.raises(SystemExit) as exc_info:
        recorder.main()
    assert exc_info.value.code == 2


def test_main_records_saves_prints_and_plays(tmp_path, monkeypatch, capsys):
    samples = np.full(4800, 16384, dtype=np.int16)
    output_dir = tmp_path / "nested" / "captures"
    recorded = {}
    played = {}

    def fake_record(seconds, device=None):
        recorded["seconds"] = seconds
        recorded["device"] = device
        return samples

    def fake_play(audio, **kwargs):
        played["audio"] = audio
        played.update(kwargs)

    monkeypatch.setattr(sys, "argv", [
        "record.py", "--seconds", "0.1", "--out", str(output_dir),
        "--device", "2", "--play", "--out-device", "3",
    ])
    monkeypatch.setattr(recorder, "record", fake_record)
    monkeypatch.setattr(recorder.sd, "play", fake_play, raising=False)

    assert recorder.main() == 0

    recordings = list(output_dir.glob("recording_*.wav"))
    assert len(recordings) == 1
    with wave.open(str(recordings[0]), "rb") as wav_file:
        assert wav_file.getnframes() == len(samples)
    assert recorded == {"seconds": 0.1, "device": 2}
    assert played == {
        "audio": samples,
        "samplerate": SAMPLE_RATE,
        "device": 3,
        "blocking": True,
    }
    output = capsys.readouterr().out
    assert str(recordings[0]) in output
    assert "-6.02 dBFS" in output
