import wave

import numpy as np
import pytest

from software.skull_core.state_console import (
    WAV_RATE,
    load_wav,
    missing_optional_wavs,
    missing_wavs,
    wav_path,
)
from software.skull_core.status_leds import STATES, TEST

SIX = {"idle", "listening", "thinking", "speaking", "offline", "error"}


def write_wav(path, rate=WAV_RATE, channels=1, sampwidth=2, samples=(0, 16384, -16384, 32767)):
    """A tiny WAV for the tests. 16-bit: the given samples. 8-bit: silence."""
    if sampwidth == 2:
        data = np.repeat(np.array(samples, dtype="<i2"), channels).tobytes()
    else:
        data = b"\x80" * (len(samples) * channels * sampwidth)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(sampwidth)
        w.setframerate(rate)
        w.writeframes(data)
    return path


def test_wav_path_is_folder_slash_state_dot_wav(tmp_path):
    assert wav_path("idle", tmp_path) == tmp_path / "idle.wav"
    assert {wav_path(s, tmp_path).name for s in STATES} == {f"{s}.wav" for s in SIX}


def test_wav_path_knows_the_test_command(tmp_path):
    assert wav_path(TEST, tmp_path) == tmp_path / "test.wav"


@pytest.mark.parametrize("bad", ["dancing", "", "../idle", "idle.wav", "IDLE", "test.wav"])
def test_wav_path_rejects_names_that_are_not_states(tmp_path, bad):
    with pytest.raises(ValueError):
        wav_path(bad, tmp_path)


def test_empty_folder_misses_all_six_in_states_order(tmp_path):
    assert missing_wavs(tmp_path) == [tmp_path / f"{s}.wav" for s in STATES]
    assert len(missing_wavs(tmp_path)) == 6


def test_folder_that_does_not_exist_misses_all_six(tmp_path):
    assert len(missing_wavs(tmp_path / "nope")) == 6


def test_all_six_present_means_nothing_missing(tmp_path):
    for s in SIX:
        (tmp_path / f"{s}.wav").touch()
    assert missing_wavs(tmp_path) == []


def test_one_missing_is_named(tmp_path):
    for s in SIX - {"thinking"}:
        (tmp_path / f"{s}.wav").touch()
    assert missing_wavs(tmp_path) == [tmp_path / "thinking.wav"]


def test_missing_test_wav_is_a_warning_not_a_failure(tmp_path):
    for s in SIX:
        (tmp_path / f"{s}.wav").touch()
    assert missing_wavs(tmp_path) == []                          # nothing required is missing
    assert missing_optional_wavs(tmp_path) == [tmp_path / "test.wav"]


def test_test_wav_present_means_no_warning(tmp_path):
    (tmp_path / "test.wav").touch()
    assert missing_optional_wavs(tmp_path) == []
    assert len(missing_wavs(tmp_path)) == 6                      # test.wav never stands in for a state


def test_load_mono_wav(tmp_path):
    assert WAV_RATE == 48_000
    x = load_wav(write_wav(tmp_path / "idle.wav"))
    assert x.dtype == np.float32
    assert x.shape == (4, 1)
    assert x[:, 0] == pytest.approx([0.0, 0.5, -0.5, 32767 / 32768])


def test_load_stereo_wav(tmp_path):
    assert load_wav(write_wav(tmp_path / "idle.wav", channels=2)).shape == (4, 2)


def test_wrong_rate_is_refused_with_a_clear_message(tmp_path):
    with pytest.raises(ValueError) as e:
        load_wav(write_wav(tmp_path / "idle.wav", rate=44_100))
    assert "44100" in str(e.value)
    assert "idle.wav" in str(e.value)


def test_8_bit_is_refused(tmp_path):
    with pytest.raises(ValueError):
        load_wav(write_wav(tmp_path / "idle.wav", sampwidth=1))


def test_three_channels_are_refused(tmp_path):
    with pytest.raises(ValueError):
        load_wav(write_wav(tmp_path / "idle.wav", channels=3))


def test_a_file_that_is_not_a_wav_is_a_value_error(tmp_path):
    bad = tmp_path / "error.wav"
    bad.write_text("not audio")
    with pytest.raises(ValueError) as e:
        load_wav(bad)
    assert "error.wav" in str(e.value)
