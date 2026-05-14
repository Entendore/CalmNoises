import os
import math
import numpy as np
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

# --- HEADLESS KIVY CONFIG ---
# MUST be set before importing Kivy or the App
os.environ['KIVY_NO_ARGS'] = '1'
os.environ['KIVY_NO_CONSOLELOG'] = '1'
os.environ['KIVY_WINDOW'] = 'sdl2' # Use sdl2, but headless logic applies

# Import app components
from app import (
    App, AudioEngine, SonicCalmApp,
    _gauss, get_psych_dimensions, get_perception_scores, 
    get_brain_wave_activation, aggregate_dimensions, aggregate_perception,
    SOLFEGGIO, SOLFEGGIO_PSYCH, BRAIN_WAVES, TONAL_CONSTRUCTIONS,
    FreqRow, PsychRadarWidget, BrainWaveWidget, PerceptionWidget,
    SAMPLE_RATE, BLOCK_SIZE
)


# ==========================================
# FIXTURES
# ==========================================

@pytest.fixture(scope='module')
def mock_sounddevice():
    """Mock sounddevice to prevent audio hardware errors in testing."""
    with patch('app.sd.OutputStream') as mock_sd:
        mock_instance = MagicMock()
        mock_sd.return_value = mock_instance
        yield mock_sd

@pytest.fixture
def audio_engine(mock_sounddevice):
    """Provides a fresh AudioEngine instance for each test."""
    return AudioEngine()

@pytest.fixture
def app_instance(mock_sounddevice):
    """Provides a fully built Kivy App instance without running the main loop."""
    app = SonicCalmApp()
    app.build() # Build the UI tree
    yield app
    app.stop()


# ==========================================
# TEST HELPERS & MATH LOGIC
# ==========================================

class TestHelpers:
    def test_gauss_peak(self):
        assert _gauss(5, 5, 1) == 1.0

    def test_gauss_off_peak(self):
        # Value should decrease significantly away from mean
        assert _gauss(0, 5, 1) < 0.01

    def test_get_psych_dimensions_exact_match(self):
        # If frequency is exactly a Solfeggio, return its exact array
        freq = 528
        dims = get_psych_dimensions(freq)
        assert dims == SOLFEGGIO_PSYCH[528]

    def test_get_psych_dimensions_interpolation(self):
        # Test a frequency between two adjacent anchors
        # Sorted anchors: [174, 285, 396, 417, 432, 528, ...]
        # 476 falls between 432 and 528, NOT 417 and 528
        freq = 476
        dims = get_psych_dimensions(freq)
        d1 = np.array(SOLFEGGIO_PSYCH[432])
        d2 = np.array(SOLFEGGIO_PSYCH[528])
        t = (476 - 432) / (528 - 432)
        expected = (d1 * (1 - t) + d2 * t).tolist()
        assert dims == pytest.approx(expected)

    def test_get_psych_dimensions_out_of_bounds_low(self):
        # Frequencies below the lowest anchor should clamp
        dims = get_psych_dimensions(50)
        assert dims == list(SOLFEGGIO_PSYCH[174])

    def test_get_psych_dimensions_out_of_bounds_high(self):
        # Frequencies above highest anchor should clamp
        dims = get_psych_dimensions(1500)
        assert dims == list(SOLFEGGIO_PSYCH[963])

    def test_get_perception_scores(self):
        scores = get_perception_scores(450)
        assert len(scores) == 6
        # Emotional peaks near log(450), so it should be highest
        assert scores[1] > scores[0] 

    def test_get_brain_wave_activation_no_binaural(self):
        acts = get_brain_wave_activation(0, [400])
        # Without binaural, activations rely only on weak carrier influence
        assert all(0 <= a <= 1 for a in acts)

    def test_get_brain_wave_activation_theta_binaural(self):
        acts = get_brain_wave_activation(6, [400]) # 6Hz is Theta
        # Theta should be highly activated
        theta_idx = [i for i, w in enumerate(BRAIN_WAVES) if w[0] == 'Theta'][0]
        assert acts[theta_idx] > 0.8

    def test_aggregate_dimensions_empty(self):
        assert aggregate_dimensions([]) == [0.0] * 8

    def test_aggregate_dimensions_max(self):
        # Mixing two frequencies should take the max of each dimension
        d1 = get_psych_dimensions(174) # High Healing
        d2 = get_psych_dimensions(741) # High Focus
        agg = aggregate_dimensions([174, 741])
        assert agg[1] == max(d1[1], d2[1]) # Focus
        assert agg[5] == max(d1[5], d2[5]) # Healing


# ==========================================
# TEST AUDIO ENGINE
# ==========================================

class TestAudioEngine:
    def test_initial_state(self, audio_engine):
        assert audio_engine.playing is False
        assert audio_engine.envelope == 0.0
        assert audio_engine.master_volume == 0.5

    def test_start_stream(self, audio_engine, mock_sounddevice):
        audio_engine.start()
        mock_sounddevice.assert_called_once()
        assert audio_engine.playing is True
        assert audio_engine._target_env == 1.0

    def test_stop_stream(self, audio_engine, mock_sounddevice):
        audio_engine.start()
        audio_engine.stop()
        assert audio_engine.playing is False
        assert audio_engine._target_env == 0.0

    def test_callback_ramp_up(self, audio_engine):
        audio_engine._target_env = 1.0
        outdata = np.zeros((BLOCK_SIZE, 2), dtype='float32')
        
        # Run callback multiple times to simulate time passing
        for _ in range(300):
            audio_engine._cb(outdata, BLOCK_SIZE, None, None)
        
        assert audio_engine.envelope == pytest.approx(1.0, abs=0.01)

    def test_callback_generates_audio(self, audio_engine):
        audio_engine.frequencies = [(432, 1.0)]
        audio_engine.master_volume = 1.0
        audio_engine.envelope = 1.0 # Force envelope open for test
        
        outdata = np.zeros((BLOCK_SIZE, 2), dtype='float32')
        audio_engine._cb(outdata, BLOCK_SIZE, None, None)
        
        # Outdata should not be all zeros
        assert np.sum(np.abs(outdata)) > 0

    def test_callback_binaural_offset(self, audio_engine):
        audio_engine.frequencies = [(432, 1.0)]
        audio_engine.binaural_offset = 10.0
        audio_engine.master_volume = 1.0
        audio_engine.envelope = 1.0
        
        outdata = np.zeros((BLOCK_SIZE, 2), dtype='float32')
        audio_engine._cb(outdata, BLOCK_SIZE, None, None)
        
        # Left and right channels should be different due to 10Hz offset
        assert not np.array_equal(outdata[:, 0], outdata[:, 1])

    def test_snapshot(self, audio_engine):
        audio_engine.frequencies = [(432, 1.0)]
        audio_engine.envelope = 1.0
        snap = audio_engine.snapshot(512)
        assert len(snap) == 512
        assert isinstance(snap, np.ndarray)


# ==========================================
# TEST UI WIDGETS & APP LOGIC
# ==========================================

class TestSonicCalmAppUI:
    def test_app_builds(self, app_instance):
        assert app_instance is not None
        assert len(app_instance.rows) == 2 # Default 432 + 528

    def test_add_tone(self, app_instance):
        initial_len = len(app_instance.rows)
        app_instance._add()
        assert len(app_instance.rows) == initial_len + 1
        
        # Test max limit (5)
        for _ in range(3):
            app_instance._add()
        assert len(app_instance.rows) == 5
        app_instance._add() # Should not add
        assert len(app_instance.rows) == 5

    def test_remove_tone(self, app_instance):
        app_instance._add() # Now 3 tones
        initial_len = len(app_instance.rows)
        row_to_del = app_instance.rows[-1]
        app_instance._del(row_to_del)
        assert len(app_instance.rows) == initial_len - 1

    def test_cannot_remove_last_tone(self, app_instance):
        # Remove until 1 left
        while len(app_instance.rows) > 1:
            app_instance._del(app_instance.rows[-1])
        
        assert len(app_instance.rows) == 1
        app_instance._del(app_instance.rows[0]) # Try to delete last
        assert len(app_instance.rows) == 1 # Should persist

    def test_apply_construction(self, app_instance):
        app_instance._apply_construction("Deep Sleep")
        
        # Check frequencies changed
        freqs = [f for f, _ in app_instance.audio.frequencies]
        assert 174 in freqs
        assert 285 in freqs
        
        # Check binaural changed
        assert app_instance.audio.binaural_offset == 2.0
        assert app_instance.b_check.active is True
        assert app_instance.b_slider.value == 2

        # Check play state
        assert app_instance.play_btn.state == 'down'

    def test_volume_slider(self, app_instance):
        app_instance._vol(None, 75)
        assert app_instance.audio.master_volume == 0.75
        assert app_instance.vol_v.text == "75 %"

    def test_binaural_toggle(self, app_instance):
        # Enable
        app_instance.b_check.active = True
        app_instance.b_slider.value = 18  # 18 Hz is in Beta range (13-30), not Alpha (8-13)
        app_instance._bin(None, None) # Simulate callback
        
        assert app_instance.audio.binaural_offset == 18.0
        assert "Beta" in app_instance.b_lbl.text
        
        # Disable
        app_instance.b_check.active = False
        app_instance._bin(None, None)
        assert app_instance.audio.binaural_offset == 0
        assert "Disabled" in app_instance.b_lbl.text

    def test_play_stop_toggle(self, app_instance):
        # Simulate play
        app_instance.play_btn.state = 'down'
        app_instance._toggle(app_instance.play_btn)
        assert app_instance.audio.playing is True
        assert "S T O P" in app_instance.play_btn.text

        # Simulate stop
        app_instance.play_btn.state = 'normal'
        app_instance._toggle(app_instance.play_btn)
        assert app_instance.audio.playing is False
        assert "P L A Y" in app_instance.play_btn.text


class TestAnalysisWidgets:
    def test_psych_radar_tick(self, app_instance):
        radar = app_instance.radar
        radar.set_dimensions([0.5]*8)
        # Tick interpolates
        radar.tick()
        for dim in radar.dims:
            assert dim > 0.0 # Should have moved from 0 towards 0.5

    def test_brain_wave_tick(self, app_instance):
        bw = app_instance.brain_wv
        bw.set_activations([1.0]*5)
        bw.tick()
        for act in bw.activations:
            assert act > 0.05 # Should have moved from 0.05 towards 1.0

    def test_perception_tick(self, app_instance):
        pw = app_instance.perc_wv
        pw.set_scores([1.0]*6)
        pw.tick()
        for sc in pw.scores:
            assert sc > 0.0 # Should have moved towards 1.0