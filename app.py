import sys
import math
import numpy as np
import threading
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.slider import Slider
from kivy.uix.checkbox import CheckBox
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.scrollview import ScrollView
from kivy.uix.togglebutton import ToggleButton
from kivy.graphics import Color, Line, Ellipse, Rectangle, ClearBuffers
from kivy.core.text import Label as CoreLabel
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.metrics import dp, sp

try:
    import sounddevice as sd
except ImportError:
    print("Error: 'sounddevice' library missing.\nInstall with: pip install sounddevice")
    sys.exit(1)

# --- Constants ---
SAMPLE_RATE = 44100
BLOCK_SIZE = 1024
GOLDEN_RATIO = (1 + math.sqrt(5)) / 2

SOLFEGGIO = {
    174: "Pain Relief", 285: "Tissue Heal", 396: "Liberation",
    417: "Change", 432: "Universal", 528: "Love/Heal",
    639: "Connection", 741: "Expression", 852: "Intuition", 963: "Crown"
}

RING_COLORS = [
    (0, 255, 200), (147, 112, 255), (255, 100, 200),
    (100, 200, 255), (255, 200, 100),
]

# Convert RGB 0-255 to 0-1 for Kivy
def c(rgb):
    return [rgb[0]/255, rgb[1]/255, rgb[2]/255, 1.0]

RING_COLORS_KIVY = [c(col) for col in RING_COLORS]

SOLFEGGIO_PSYCH = {
    174: [0.85, 0.15, 0.20, 0.50, 0.15, 0.95, 0.20, 0.20],
    285: [0.75, 0.15, 0.15, 0.35, 0.15, 0.90, 0.15, 0.25],
    396: [0.80, 0.30, 0.35, 0.92, 0.30, 0.35, 0.35, 0.50],
    417: [0.50, 0.60, 0.70, 0.80, 0.40, 0.40, 0.45, 0.60],
    432: [0.92, 0.50, 0.70, 0.70, 0.60, 0.55, 0.55, 0.70],
    528: [0.80, 0.40, 0.65, 0.92, 0.70, 0.88, 0.50, 0.60],
    639: [0.70, 0.40, 0.50, 0.92, 0.50, 0.40, 0.60, 0.50],
    741: [0.40, 0.85, 0.88, 0.50, 0.60, 0.30, 0.70, 0.90],
    852: [0.60, 0.50, 0.70, 0.50, 0.92, 0.30, 0.95, 0.70],
    963: [0.50, 0.30, 0.55, 0.40, 0.95, 0.20, 0.92, 0.55],
}

DIM_NAMES = ["Relaxation", "Focus", "Creativity", "Emotion", "Spiritual", "Healing", "Intuition", "Clarity"]
DIM_COLORS_KIVY = [c(col) for col in [(0, 210, 180), (255, 210, 50), (200, 100, 255), (255, 100, 150), (170, 120, 255), (100, 255, 150), (140, 175, 255), (255, 220, 100)]]

PERCEPTION_CHANNELS = ["Visceral", "Emotional", "Cognitive", "Spiritual", "Somatic", "Subconscious"]
PERCEPTION_COLORS_KIVY = [c(col) for col in [(255, 80, 80), (255, 120, 180), (100, 180, 255), (170, 120, 255), (100, 255, 180), (140, 140, 180)]]

BRAIN_WAVES = [
    ("Delta", "δ", 0.5, 4,  c((90, 70, 200)),  "Deep Sleep"),
    ("Theta", "θ", 4,   8,  c((60, 150, 230)), "Meditation"),
    ("Alpha", "α", 8,   13, c((60, 210, 150)), "Relaxation"),
    ("Beta",  "β", 13,  30, c((230, 190, 60)), "Focus"),
    ("Gamma", "γ", 30,  100, c((230, 80, 80)), "Insight"),
]

TONAL_CONSTRUCTIONS = {
    "Anxiety Relief": {
        "freqs": [396, 528], "binaural": 10,
        "desc": "CARRIER: 396 Hz dissolves fear → 528 Hz fills with love\nBINAURAL: Alpha (10 Hz) — Calm alertness",
    },
    "Deep Sleep": {
        "freqs": [174, 285], "binaural": 2,
        "desc": "CARRIER: 174 Hz tension release → 285 Hz tissue heal\nBINAURAL: Delta (2 Hz) — Deep restoration",
    },
    "Meditation": {
        "freqs": [432, 852], "binaural": 6,
        "desc": "CARRIER: 432 Hz harmony → 852 Hz intuition\nBINAURAL: Theta (6 Hz) — Inner awareness",
    },
    "Focus Flow": {
        "freqs": [741, 417], "binaural": 18,
        "desc": "CARRIER: 741 Hz expression → 417 Hz change\nBINAURAL: Beta (18 Hz) — Active concentration",
    },
    "Heart Opening": {
        "freqs": [528, 639], "binaural": 8,
        "desc": "CARRIER: 528 Hz love → 639 Hz connection\nBINAURAL: Alpha (8 Hz) — Heart-brain coherence",
    },
    "Spiritual Growth": {
        "freqs": [852, 963], "binaural": 7,
        "desc": "CARRIER: 852 Hz intuition → 963 Hz crown\nBINAURAL: Theta (7 Hz) — Transcendence",
    },
    "Creative Break": {
        "freqs": [417, 741], "binaural": 8,
        "desc": "CARRIER: 417 Hz unblocks → 741 Hz sparks\nBINAURAL: Alpha (8 Hz) — Flow state",
    },
    "Pain Release": {
        "freqs": [174, 528], "binaural": 4,
        "desc": "CARRIER: 174 Hz relief → 528 Hz repair\nBINAURAL: Theta (4 Hz) — Endorphin release",
    },
}

# --- Helpers ---
def _gauss(x, mu, sigma):
    return math.exp(-0.5 * ((x - mu) / sigma) ** 2)

def get_psych_dimensions(freq):
    anchors = sorted(SOLFEGGIO_PSYCH.keys())
    if freq <= anchors[0]: return list(SOLFEGGIO_PSYCH[anchors[0]])
    if freq >= anchors[-1]: return list(SOLFEGGIO_PSYCH[anchors[-1]])
    for i in range(len(anchors) - 1):
        if anchors[i] <= freq <= anchors[i + 1]:
            t = (freq - anchors[i]) / (anchors[i + 1] - anchors[i])
            d1 = np.array(SOLFEGGIO_PSYCH[anchors[i]])
            d2 = np.array(SOLFEGGIO_PSYCH[anchors[i + 1]])
            return (d1 * (1 - t) + d2 * t).tolist()
    return [0.5] * 8

def get_perception_scores(freq):
    lf = math.log(max(freq, 20))
    return [
        _gauss(lf, math.log(80),  0.90), _gauss(lf, math.log(450), 1.00),
        _gauss(lf, math.log(650), 0.85), _gauss(lf, math.log(880), 0.75),
        _gauss(lf, math.log(280), 1.10), _gauss(lf, math.log(130), 1.20),
    ]

def get_brain_wave_activation(binaural_offset, frequencies):
    bo = binaural_offset
    result = []
    for name, sym, lo, hi, color, desc in BRAIN_WAVES:
        activation = 0.05
        if bo > 0:
            center = (lo + hi) / 2
            width = (hi - lo) / 2 + 2
            activation = max(0.0, _gauss(bo, center, width))
        for f in frequencies:
            lf = math.log(max(f, 20))
            if name == "Delta": activation = max(activation, _gauss(lf, math.log(100), 1.2) * 0.25)
            elif name == "Theta": activation = max(activation, _gauss(lf, math.log(250), 1.0) * 0.25)
            elif name == "Alpha": activation = max(activation, _gauss(lf, math.log(450), 0.9) * 0.25)
            elif name == "Beta": activation = max(activation, _gauss(lf, math.log(700), 0.8) * 0.25)
            elif name == "Gamma": activation = max(activation, _gauss(lf, math.log(900), 0.7) * 0.25)
        result.append(min(activation, 1.0))
    return result

def aggregate_dimensions(frequencies):
    if not frequencies: return [0.0] * 8
    dims = [0.0] * 8
    for f in frequencies:
        scores = get_psych_dimensions(f)
        for i in range(8): dims[i] = max(dims[i], scores[i])
    return dims

def aggregate_perception(frequencies):
    if not frequencies: return [0.0] * 6
    perc = [0.0] * 6
    for f in frequencies:
        scores = get_perception_scores(f)
        for i in range(6): perc[i] = max(perc[i], scores[i])
    return perc

# --- Audio Engine ---
class AudioEngine:
    def __init__(self):
        self.frequencies = []
        self.binaural_offset = 0.0
        self.master_volume = 0.5
        self.playing = False
        self.phase_l = {}
        self.phase_r = {}
        self.envelope = 0.0
        self._target_env = 0.0
        self._lock = threading.Lock()
        self.stream = None
        self._t = np.arange(BLOCK_SIZE) / SAMPLE_RATE

    def start(self):
        if self.stream is None:
            try:
                self.stream = sd.OutputStream(samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE, channels=2, dtype='float32', callback=self._cb)
                self.stream.start()
            except Exception as e:
                print(f"Audio Error: {e}")
                return
        self._target_env = 1.0
        self.playing = True

    def stop(self):
        self._target_env = 0.0
        self.playing = False
        Clock.schedule_once(self._kill_stream, 0.6)

    def _kill_stream(self, dt):
        if self.stream and not self.playing:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception: pass
            self.stream = None
            self.phase_l.clear()
            self.phase_r.clear()
            self.envelope = 0.0

    def _cb(self, outdata, frames, _time, _status):
        ramp_speed = 0.004
        if self.envelope < self._target_env:
            self.envelope = min(self.envelope + ramp_speed, self._target_env)
        else:
            self.envelope = max(self.envelope - ramp_speed, self._target_env)

        outdata.fill(0)
        if self.envelope < 0.001: return

        with self._lock:
            freqs_snapshot = list(self.frequencies)
            offset = self.binaural_offset
            
        if not freqs_snapshot: return
        n_freq = len(freqs_snapshot)
        
        for freq, vol in freqs_snapshot:
            if freq <= 0: continue
            p_l = self.phase_l.get(freq, 0.0)
            wave_l = np.sin(2.0 * np.pi * freq * self._t + p_l)
            self.phase_l[freq] = (p_l + 2.0 * np.pi * freq * (frames / SAMPLE_RATE)) % (2.0 * np.pi)

            freq_r = freq + offset
            p_r = self.phase_r.get(freq, 0.0)
            wave_r = np.sin(2.0 * np.pi * freq_r * self._t + p_r)
            self.phase_r[freq] = (p_r + 2.0 * np.pi * freq_r * (frames / SAMPLE_RATE)) % (2.0 * np.pi)

            amp = vol * self.master_volume * self.envelope / n_freq
            outdata[:, 0] += wave_l * amp
            outdata[:, 1] += wave_r * amp

    def snapshot(self, n=512):
        with self._lock:
            freqs_snapshot = list(self.frequencies)
        if not freqs_snapshot: return np.zeros(n)
        t = np.arange(n) / SAMPLE_RATE
        w = sum(np.sin(2 * np.pi * f * t) * v for f, v in freqs_snapshot)
        return w / len(freqs_snapshot) * self.envelope


# --- Kivy Custom Widgets ---
class Particle:
    __slots__ = ('x', 'y', 'vx', 'vy', 'color', 'life', 'decay', 'size')
    def __init__(self, x, y, color, angle, speed):
        self.x, self.y = x, y
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        self.color = color
        self.life = 1.0
        self.decay = 0.004 + np.random.random() * 0.012
        self.size = 1.5 + np.random.random() * 4.5

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vx *= 0.992
        self.vy *= 0.992
        self.life -= self.decay

class VisualizationWidget(Widget):
    def __init__(self, audio, **kwargs):
        super().__init__(**kwargs)
        self.audio = audio
        self.particles = []
        self.t = 0.0
        self._pt = 0.0
        Clock.schedule_interval(self._tick, 1/60)

    def _tick(self, dt):
        self.t += dt
        self._pt += dt
        
        if self.audio.playing and self._pt > 0.04:
            self._pt = 0
            cx, cy = self.center_x, self.center_y
            with self.audio._lock:
                freqs = list(self.audio.frequencies)
            for i, (freq, vol) in enumerate(freqs):
                if np.random.random() < 0.6:
                    a = np.random.random() * 2 * math.pi
                    sp = 0.4 + vol * 2.5
                    self.particles.append(Particle(cx, cy, RING_COLORS_KIVY[i % 5], a, sp))

        for p in self.particles: p.update()
        self.particles = [p for p in self.particles if p.life > 0]
        self.update_canvas()

    def update_canvas(self):
        self.canvas.clear()
        with self.canvas:
            # Background
            Color(0.06, 0.06, 0.13, 1)
            Rectangle(pos=self.pos, size=self.size)
            
            cx, cy = self.center_x, self.center_y
            w, h = self.width, self.height

            # Particles
            for p in self.particles:
                Color(*p.color[:3], p.life * 0.7)
                Ellipse(pos=(p.x - p.size, p.y - p.size), size=(p.size*2, p.size*2))

            with self.audio._lock:
                freqs = list(self.audio.frequencies)
            freqs = [(f, v) for f, v in freqs if f > 0]
            n_freq = len(freqs)
            env = self.audio.envelope

            if not freqs:
                Color(1, 1, 1, 0.15)
                lbl = CoreLabel(text="Press PLAY to begin", font_size=sp(14))
                lbl.refresh()
                Rectangle(texture=lbl.texture, size=lbl.texture.size, 
                          pos=(cx - lbl.texture.size[0]/2, cy - lbl.texture.size[1]/2))
                return

            max_r = min(w, h) * 0.38

            for i, (freq, vol) in enumerate(freqs):
                col = RING_COLORS_KIVY[i % 5]
                base_r = (i + 1) * max_r / (n_freq + 1)
                pulse = math.sin(self.t * freq * 0.08) * 15 * vol * env
                radius = base_r + pulse

                # Glow
                Color(*col[:3], 0.1)
                Line(circle=(cx, cy, radius), width=dp(4))
                
                # Ring Shape
                Color(*col[:3], 0.8)
                points = []
                segs = 80
                for j in range(segs + 1):
                    a = j * 2 * math.pi / segs
                    d = math.sin(a * (int(freq) % 7 + 3) + self.t * freq * 0.04) * 8 * vol * env
                    r = radius + d
                    points.extend([cx + r * math.cos(a), cy + r * math.sin(a)])
                Line(points=points, width=dp(1.2), close=True)

                # Label
                Color(*col)
                lbl = CoreLabel(text=f"{int(freq)} Hz", font_size=sp(10))
                lbl.refresh()
                lx = cx + radius * math.cos(-0.78)
                ly = cy + radius * math.sin(-0.78)
                Rectangle(texture=lbl.texture, size=lbl.texture.size, pos=(lx+5, ly-5))

            # Center Core
            Color(0.7, 1.0, 0.9, env * 0.3)
            Ellipse(pos=(cx-30, cy-30), size=(60, 60))
            Color(1, 1, 1, env)
            Ellipse(pos=(cx-4, cy-4), size=(8, 8))

class WaveformWidget(Widget):
    def __init__(self, audio, **kwargs):
        super().__init__(**kwargs)
        self.audio = audio
        self.data = np.zeros(512)
        self.size_hint_y = None
        self.height = dp(80)
        Clock.schedule_interval(self.refresh, 1/60)

    def refresh(self, dt):
        self.data = self.audio.snapshot(512)
        self.update_canvas()

    def update_canvas(self):
        self.canvas.clear()
        with self.canvas:
            Color(0.04, 0.04, 0.08, 1)
            Rectangle(pos=self.pos, size=self.size)
            
            w, h = self.width, self.height
            Color(0.1, 0.1, 0.2, 1)
            Line(points=[0, h/2, w, h/2], width=dp(1))

            n = len(self.data)
            points = []
            for i in range(n):
                points.extend([i * w / n, h/2 - self.data[i] * h * 0.35])
            
            if len(points) > 3:
                Color(0, 1, 0.8, 0.2)
                Line(points=points, width=dp(2))
                Color(0, 1, 0.8, 0.8)
                Line(points=points, width=dp(1))

class PsychRadarWidget(Widget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dims = [0.0] * 8
        self.target_dims = [0.0] * 8

    def set_dimensions(self, dims): self.target_dims = dims[:8]

    def tick(self):
        changed = False
        for i in range(8):
            diff = self.target_dims[i] - self.dims[i]
            if abs(diff) > 0.005:
                self.dims[i] += diff * 0.12
                changed = True
        if changed: self.update_canvas()

    def update_canvas(self):
        self.canvas.clear()
        with self.canvas:
            Color(0.04, 0.04, 0.06, 1)
            Rectangle(pos=self.pos, size=self.size)
            
            cx, cy = self.center_x, self.center_y - dp(10)
            max_r = min(self.width, self.height) / 2 - dp(40)
            n, step, start = 8, 2 * math.pi / 8, -math.pi / 2

            Color(0.1, 0.1, 0.2, 1)
            for level in (0.25, 0.5, 0.75, 1.0):
                pts = []
                for i in range(n):
                    a = start + i * step
                    pts.extend([cx + max_r * level * math.cos(a), cy + max_r * level * math.sin(a)])
                Line(points=pts, close=True)

            if any(d > 0.01 for d in self.dims):
                pts = []
                for i in range(n):
                    a = start + i * step
                    r = max_r * max(self.dims[i], 0.02)
                    pts.extend([cx + r * math.cos(a), cy + r * math.sin(a)])
                Color(0, 1, 0.8, 0.2)
                Line(points=pts, close=True, width=dp(2))
                Color(0, 1, 0.8, 0.1)
                # Filled area (Kivy Line doesn't fill, so this is a visual approximation)
                
            for i in range(n):
                a = start + i * step
                lx = cx + (max_r + dp(15)) * math.cos(a)
                ly = cy + (max_r + dp(15)) * math.sin(a)
                Color(*DIM_COLORS_KIVY[i])
                lbl = CoreLabel(text=DIM_NAMES[i], font_size=sp(9))
                lbl.refresh()
                Rectangle(texture=lbl.texture, size=lbl.texture.size, pos=(lx-20, ly-5))

class BrainWaveWidget(Widget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.activations = [0.05] * 5
        self.targets = [0.05] * 5

    def set_activations(self, acts): self.targets = acts[:5]

    def tick(self):
        changed = False
        for i in range(5):
            diff = self.targets[i] - self.activations[i]
            if abs(diff) > 0.005:
                self.activations[i] += diff * 0.12
                changed = True
        if changed: self.update_canvas()

    def update_canvas(self):
        self.canvas.clear()
        with self.canvas:
            Color(0.04, 0.04, 0.06, 1)
            Rectangle(pos=self.pos, size=self.size)
            
            bar_h, spacing = dp(22), dp(8)
            x_bar, bar_w = dp(76), self.width - dp(130)
            y = self.top - dp(30)

            for i, (name, sym, lo, hi, color, desc) in enumerate(BRAIN_WAVES):
                act = self.activations[i]
                Color(*color)
                lbl = CoreLabel(text=f"{sym} {name}", font_size=sp(11))
                lbl.refresh()
                Rectangle(texture=lbl.texture, size=lbl.texture.size, pos=(dp(6), y - 5))

                Color(0.08, 0.08, 0.12, 1)
                Rectangle(pos=(x_bar, y), size=(bar_w, bar_h), radius=[dp(5)])
                
                fill_w = bar_w * act
                if fill_w > 2:
                    Color(*color)
                    Rectangle(pos=(x_bar, y), size=(fill_w, bar_h), radius=[dp(5)])

                Color(*color)
                lbl2 = CoreLabel(text=f"{act:.0%}", font_size=sp(10))
                lbl2.refresh()
                Rectangle(texture=lbl2.texture, size=lbl2.texture.size, pos=(x_bar + bar_w + dp(6), y))
                
                y -= (bar_h + spacing)

class PerceptionWidget(Widget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.scores = [0.0] * 6
        self.targets = [0.0] * 6

    def set_scores(self, scores): self.targets = scores[:6]

    def tick(self):
        changed = False
        for i in range(6):
            diff = self.targets[i] - self.scores[i]
            if abs(diff) > 0.005:
                self.scores[i] += diff * 0.12
                changed = True
        if changed: self.update_canvas()

    def update_canvas(self):
        self.canvas.clear()
        with self.canvas:
            Color(0.04, 0.04, 0.06, 1)
            Rectangle(pos=self.pos, size=self.size)
            
            bar_h, spacing = dp(18), dp(7)
            x_bar, bar_w = dp(88), self.width - dp(140)
            y = self.top - dp(30)

            for i in range(6):
                sc = self.scores[i]
                col = PERCEPTION_COLORS_KIVY[i]

                Color(*col)
                lbl = CoreLabel(text=PERCEPTION_CHANNELS[i], font_size=sp(10))
                lbl.refresh()
                Rectangle(texture=lbl.texture, size=lbl.texture.size, pos=(dp(6), y - 3))

                Color(0.08, 0.08, 0.12, 1)
                Rectangle(pos=(x_bar, y), size=(bar_w, bar_h), radius=[dp(4)])
                
                fill_w = bar_w * sc
                if fill_w > 2:
                    Color(*col)
                    Rectangle(pos=(x_bar, y), size=(fill_w, bar_h), radius=[dp(4)])

                Color(*col)
                lbl2 = CoreLabel(text=f"{sc:.0%}", font_size=sp(10))
                lbl2.refresh()
                Rectangle(texture=lbl2.texture, size=lbl2.texture.size, pos=(x_bar + bar_w + dp(6), y))
                
                y -= (bar_h + spacing)

class FreqRow(BoxLayout):
    def __init__(self, freq, label, color, on_change, on_remove, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'horizontal'
        self.size_hint_y = None
        self.height = dp(40)
        self.padding = [0, dp(2)]
        self.color = color
        self._on_change = on_change
        self._on_remove = on_remove

        self.add_widget(Label(text="●", color=color, font_size=sp(16), size_hint_x=None, width=dp(20)))
        self.add_widget(Label(text=label, color=(0.6, 0.6, 0.6, 1), font_size=sp(11), size_hint_x=None, width=dp(85)))
        
        self.slider = Slider(min=20, max=2000, value=freq, value_track_color=color)
        self.slider.bind(value=self._on_slider)
        self.add_widget(self.slider)

        self.vl = Label(text=f"{int(freq)} Hz", color=color, font_size=sp(13), bold=True, size_hint_x=None, width=dp(70))
        self.add_widget(self.vl)

        btn = Button(text="×", color=(0.4, 0.4, 0.4, 1), background_normal='', background_color=(0,0,0,0), font_size=sp(18), size_hint_x=None, width=dp(30))
        btn.bind(on_press=lambda x: self._on_remove(self))
        self.add_widget(btn)

    def _on_slider(self, instance, val):
        self.vl.text = f"{int(val)} Hz"
        self._on_change()

    def freq(self): return float(self.slider.value)


# --- Main Application ---
class SonicCalmApp(App):
    def build(self):
        self.audio = AudioEngine()
        self.rows = []
        self._active_construction = ""

        Window.clearcolor = (0.047, 0.047, 0.078, 1) # #0c0c14

        # Root Layout
        root = BoxLayout(orientation='vertical', padding=dp(10), spacing=dp(10))

        # Header
        header = BoxLayout(orientation='vertical', size_hint_y=None, height=dp(60))
        header.add_widget(Label(text="✦ S O N I C   C A L M ✦", font_size=sp(26), color=(0, 1, 0.78, 1), bold=False))
        header.add_widget(Label(text="Tonal Variation • Psychological Effects • Perception & Cognition", font_size=sp(10), color=(0.3, 0.3, 0.4, 1)))
        root.add_widget(header)

        # Content Row
        content_row = BoxLayout(orientation='horizontal', spacing=dp(10))

        # Left Column (Viz)
        left_col = BoxLayout(orientation='vertical', spacing=dp(5))
        self.viz = VisualizationWidget(self.audio)
        left_col.add_widget(self.viz)
        self.wave = WaveformWidget(self.audio)
        left_col.add_widget(self.wave)
        left_col.size_hint_x = 0.6 
        content_row.add_widget(left_col)

        # Right Column (Controls)
        scroll = ScrollView(do_scroll_x=False, size_hint_x=0.4)
        rpanel = BoxLayout(orientation='vertical', spacing=dp(10), size_hint_y=None)
        rpanel.bind(minimum_height=rpanel.setter('height'))

        # Presets
        preset_grid = GridLayout(cols=2, spacing=dp(5), size_hint_y=None, height=dp(180))
        for name, data in TONAL_CONSTRUCTIONS.items():
            b = Button(text=name, font_size=sp(12), background_normal='', background_color=(0.09, 0.09, 0.18, 1), color=(0.8, 0.8, 0.8, 1))
            b.bind(on_press=lambda instance, n=name: self._apply_construction(n))
            preset_grid.add_widget(b)
        rpanel.add_widget(preset_grid)

        self.const_label = Label(text="Select a preset or build your own.", font_size=sp(11), color=(0.5, 0.5, 0.5, 1), size_hint_y=None, height=dp(80), halign='left', valign='top', markup=True)
        rpanel.add_widget(self.const_label)

        # Frequencies
        self.fbox = BoxLayout(orientation='vertical', spacing=dp(5), size_hint_y=None)
        self.fbox.bind(minimum_height=self.fbox.setter('height'))
        rpanel.add_widget(self.fbox)

        ab = Button(text="+ Add Tone", font_size=sp(12), background_normal='', background_color=(0,0,0,0), color=(0.4, 0.4, 0.4, 1), outline_color=(0.2,0.2,0.2))
        ab.bind(on_press=lambda x: self._add())
        rpanel.add_widget(ab)

        # Binaural
        bin_box = BoxLayout(orientation='vertical', spacing=dp(5), size_hint_y=None, height=dp(100))
        self.b_check = CheckBox(active=False, color=(0, 1, 0.8, 1))
        bin_box.add_widget(BoxLayout(children=[self.b_check, Label(text="Enable Binaural (Headphones)", color=(0.6,0.6,0.6,1), font_size=sp(12))]))
        self.b_slider = Slider(min=1, max=40, value=6, disabled=True)
        self.b_check.bind(active=self._bin)
        self.b_slider.bind(value=self._bin)
        bin_box.add_widget(self.b_slider)
        self.b_lbl = Label(text="Δf = 0 Hz (Disabled)", font_size=sp(11), color=(0.5, 0.5, 0.5, 1), size_hint_y=None, height=dp(20))
        bin_box.add_widget(self.b_lbl)
        rpanel.add_widget(bin_box)

        # Volume
        vol_box = BoxLayout(orientation='horizontal', size_hint_y=None, height=dp(40))
        vol_box.add_widget(Label(text="🔊", font_size=sp(16)))
        self.vol_s = Slider(min=0, max=100, value=50)
        self.vol_s.bind(value=self._vol)
        vol_box.add_widget(self.vol_s)
        self.vol_v = Label(text="50 %", font_size=sp(12), color=(0.8,0.8,0.8,1), size_hint_x=None, width=dp(40))
        vol_box.add_widget(self.vol_v)
        rpanel.add_widget(vol_box)

        # Play
        self.play_btn = ToggleButton(text="▶   P L A Y", font_size=sp(16), background_normal='', background_color=(0.05, 0.15, 0.15, 1), color=(0, 1, 0.8, 1))
        self.play_btn.bind(on_press=self._toggle)
        rpanel.add_widget(self.play_btn)

        scroll.add_widget(rpanel)
        content_row.add_widget(scroll)

        root.add_widget(content_row)

        # Bottom Analysis
        bot_row = BoxLayout(orientation='horizontal', spacing=dp(10), size_hint_y=None, height=dp(200))
        self.radar = PsychRadarWidget()
        bot_row.add_widget(self.radar)
        self.brain_wv = BrainWaveWidget()
        bot_row.add_widget(self.brain_wv)
        self.perc_wv = PerceptionWidget()
        bot_row.add_widget(self.perc_wv)
        root.add_widget(bot_row)

        # Timers
        Clock.schedule_interval(self._update_analysis, 1/20)
        self._add_row(432, "Universal")
        self._add_row(528, "Love/Heal")

        return root

    def _add_row(self, freq, label=None):
        idx = len(self.rows)
        color = RING_COLORS_KIVY[idx % 5]
        if label is None:
            best = min(SOLFEGGIO, key=lambda k: abs(k - freq))
            label = SOLFEGGIO[best] if abs(best - freq) < 20 else f"Tone {idx + 1}"
        
        r = FreqRow(freq, label, color, self._sync, self._del)
        self.fbox.add_widget(r)
        self.rows.append(r)
        self._sync()

    def _add(self):
        if len(self.rows) >= 5: return
        used = {int(row.freq()) for row in self.rows}
        d = 432
        for f in SOLFEGGIO:
            if f not in used: d = f; break
        self._add_row(d)

    def _del(self, row):
        if len(self.rows) <= 1: return
        self.rows.remove(row)
        self.fbox.remove_widget(row)
        self._sync()

    def _sync(self):
        with self.audio._lock:
            self.audio.frequencies = [(r.freq(), 1.0) for r in self.rows]

    def _bin(self, instance, value):
        on = self.b_check.active
        self.b_slider.disabled = not on
        if on:
            v = float(self.b_slider.value)
            self.audio.binaural_offset = v
            tag = ("Delta" if v < 4 else "Theta" if v < 8 else "Alpha" if v < 13 else "Beta" if v < 30 else "Gamma")
            self.b_lbl.text = f"Δf = {v:.1f} Hz ({tag})"
        else:
            self.audio.binaural_offset = 0
            self.b_lbl.text = "Δf = 0 Hz (Disabled)"

    def _vol(self, instance, v):
        self.audio.master_volume = v / 100
        self.vol_v.text = f"{int(v)} %"

    def _toggle(self, instance):
        if self.play_btn.state == 'down':
            self.play_btn.text = "⏹   S T O P"
            self.play_btn.color = (1, 0.3, 0.4, 1)
            self.play_btn.background_color = (0.15, 0.05, 0.05, 1)
            self.audio.start()
        else:
            self.play_btn.text = "▶   P L A Y"
            self.play_btn.color = (0, 1, 0.8, 1)
            self.play_btn.background_color = (0.05, 0.15, 0.15, 1)
            self.audio.stop()

    def _apply_construction(self, name):
        data = TONAL_CONSTRUCTIONS.get(name)
        if not data: return
        self._active_construction = name

        for r in self.rows[:]:
            self.fbox.remove_widget(r)
        self.rows.clear()

        for f in data["freqs"]:
            best = min(SOLFEGGIO, key=lambda k: abs(k - f))
            label = SOLFEGGIO.get(f, SOLFEGGIO.get(best, f"Custom {f}"))
            self._add_row(f, label)

        if data["binaural"] > 0:
            self.b_check.active = True
            self.b_slider.value = int(data["binaural"])
        else:
            self.b_check.active = False

        self.const_label.text = f"[color=00ffc8][b]{name}[/b][/color]\n\n{data['desc']}"
        
        if self.play_btn.state != 'down':
            self.play_btn.state = 'down'
            self._toggle(self.play_btn)

    def _update_analysis(self, dt):
        freqs = [f for f, _ in self.audio.frequencies]

        dims = aggregate_dimensions(freqs)
        self.radar.set_dimensions(dims)
        self.radar.tick()

        brain = get_brain_wave_activation(self.audio.binaural_offset, freqs)
        self.brain_wv.set_activations(brain)
        self.brain_wv.tick()

        perc = aggregate_perception(freqs)
        self.perc_wv.set_scores(perc)
        self.perc_wv.tick()

        if not self._active_construction and freqs:
            self._update_auto_construction(freqs)

    def _update_auto_construction(self, freqs):
        lines = [f"[color=9470ff][b]Custom Mix[/b][/color]\n"]
        for f in freqs:
            name = "Custom"
            for sf, sn in SOLFEGGIO.items():
                if abs(sf - f) < 15: name = sn; break
            lines.append(f"\n{int(f)} Hz ({name})")
        
        dims = aggregate_dimensions(freqs)
        max_idx = dims.index(max(dims))
        lines.append(f"\n[b][color={DIM_COLORS_KIVY[max_idx]}]Primary:[/color][/b] {DIM_NAMES[max_idx]}")
        
        self.const_label.text = "".join(lines)

    def on_stop(self):
        self.audio.stop()
        if self.audio.stream:
            try: self.audio.stream.close()
            except: pass

if __name__ == "__main__":
    SonicCalmApp().run()