"""
UI _ codée à l'aide d'une IA générative
"""
from __future__ import annotations
 
import sys
import math
import numpy as np
 
import pyvista as pv
from pyvistaqt import QtInteractor
 
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QSplitter,
    QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSlider, QDoubleSpinBox, QGroupBox, QTabWidget,
    QFrame, QProgressBar, QSizePolicy,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
 
import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import matplotlib.ticker as ticker
 
# ══════════════════════════════════════════════════════════════════════════════
#  PALETTE — papier millimétré + phosphore vert oscilloscope
# ══════════════════════════════════════════════════════════════════════════════
BG          = "#F5F2EB"
BG2         = "#EDE9DF"
PANEL_L     = "#E8E4D9"
BORDER_L    = "#C8C2B0"
BORDER_D    = "#A09880"
 
PHOSPHOR    = "#39FF14"
PHOSPHOR_D  = "#1A7A00"
AMBER       = "#FF8C00"
CYAN_OSC    = "#007A7A"
RED_OSC     = "#CC2200"
 
INK         = "#1A1612"
INK2        = "#4A4540"
INK3        = "#8A8278"
GRID_COLOR  = "#C8C2B0"
 
FONT_MONO   = "Courier New"   # disponible sur Windows, macOS, Linux
N_RUNS      = 10
 
# ══════════════════════════════════════════════════════════════════════════════
#  QSS global
# ══════════════════════════════════════════════════════════════════════════════
GLOBAL_QSS = f"""
* {{
    font-family: '{FONT_MONO}', 'Liberation Mono', 'Courier New', monospace;
    font-size: 11px;
    color: {INK};
}}
QMainWindow, QWidget {{ background-color: {BG}; }}
QGroupBox {{
    background-color: {BG2};
    border: 1px solid {BORDER_L};
    border-top: 2px solid {BORDER_D};
    border-radius: 2px;
    margin-top: 18px;
    padding: 6px 4px 4px 4px;
    font-size: 10px; font-weight: bold;
    color: {INK2}; letter-spacing: 2px;
}}
QGroupBox::title {{
    subcontrol-origin: margin; left: 8px; top: -1px;
    padding: 0 4px; background: {BG2};
}}
QLabel {{ background: transparent; color: {INK}; }}
QDoubleSpinBox {{
    background: {BG}; border: 1px solid {BORDER_L};
    border-radius: 1px; padding: 2px 6px; color: {INK};
    min-width: 72px;
}}
QDoubleSpinBox:focus {{ border: 1px solid {BORDER_D}; }}
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
    width: 14px; background: {PANEL_L}; border: none;
}}
QSlider::groove:horizontal {{
    background: {PANEL_L}; border: 1px solid {BORDER_L};
    height: 3px; border-radius: 1px;
}}
QSlider::handle:horizontal {{
    background: {INK}; width: 10px; height: 10px;
    margin: -4px 0; border-radius: 5px;
}}
QSlider::sub-page:horizontal {{
    background: {PHOSPHOR_D}; border-radius: 1px;
}}
QPushButton#run_btn {{
    background: {INK}; color: {PHOSPHOR};
    border: none; border-radius: 2px;
    padding: 12px 20px; font-size: 12px;
    font-weight: bold; letter-spacing: 3px; min-height: 44px;
}}
QPushButton#run_btn:hover {{ background: {INK2}; }}
QPushButton#run_btn:disabled {{ background: {BORDER_L}; color: {INK3}; }}
QPushButton#sec_btn {{
    background: transparent; color: {INK3};
    border: 1px solid {BORDER_L}; border-radius: 2px;
    padding: 6px 12px; font-size: 10px; letter-spacing: 1px;
}}
QPushButton#sec_btn:hover {{ border-color: {BORDER_D}; color: {INK}; }}
QTabWidget::pane {{
    border: 1px solid {BORDER_L}; background: {BG}; border-radius: 0;
}}
QTabBar::tab {{
    background: {PANEL_L}; color: {INK3};
    border: 1px solid {BORDER_L}; border-bottom: none;
    padding: 6px 18px; font-size: 10px;
    letter-spacing: 1px; margin-right: 2px;
}}
QTabBar::tab:selected {{
    background: {BG}; color: {INK};
    border-top: 2px solid {PHOSPHOR_D};
}}
QTabBar::tab:hover:!selected {{ background: {BG2}; }}
QProgressBar {{
    background: {PANEL_L}; border: 1px solid {BORDER_L};
    border-radius: 1px; height: 6px; text-align: center;
    font-size: 9px; color: {INK3};
}}
QProgressBar::chunk {{ background: {PHOSPHOR_D}; border-radius: 1px; }}
QSplitter::handle {{ background: {BORDER_L}; }}
"""
 
MPL_STYLE = {
    "axes.facecolor":    BG2,
    "figure.facecolor":  BG,
    "axes.edgecolor":    BORDER_D,
    "axes.labelcolor":   INK2,
    "axes.labelsize":    9,
    "axes.titlesize":    10,
    "axes.titleweight":  "bold",
    "axes.titlecolor":   INK,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "xtick.color":       INK3,
    "ytick.color":       INK3,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "grid.color":        GRID_COLOR,
    "grid.linestyle":    "--",
    "grid.linewidth":    0.5,
    "grid.alpha":        1.0,
    "text.color":        INK,
    "font.family":       "monospace",
    "lines.linewidth":   1.8,
    "legend.fontsize":   8,
    "legend.framealpha": 0.9,
    "legend.edgecolor":  BORDER_L,
    "legend.facecolor":  BG,
}
 
 
# ══════════════════════════════════════════════════════════════════════════════
#  Worker thread
# ══════════════════════════════════════════════════════════════════════════════
class SimWorker(QThread):
    finished = pyqtSignal(object)
    error    = pyqtSignal(str)
    progress = pyqtSignal(int)
 
    def __init__(self, params: dict, n_runs: int = N_RUNS):
        super().__init__()
        self.params = params
        self.n_runs = n_runs
 
    def run(self):
        try:
            from .physics import Physique, Muon, Scintillateur, Rayon, Vecteur
            p = self.params
            self.progress.emit(5)
 
            W, H, D = p["scint_w"], p["scint_h"], p["scint_d"]
            Scint = Scintillateur(
                (0,0,0),(W,0,0),(W,H,0),(0,H,0),
                (0,0,-D),(W,0,-D),(W,H,-D),(0,H,-D),
            )
            self.progress.emit(10)
            theta = math.radians(p["theta"]); phi = math.radians(p["phi"])
            vx = math.sin(theta)*math.cos(phi)
            vy = math.sin(theta)*math.sin(phi)
            vz = -math.cos(theta)
            Ray  = Rayon(Vecteur(vx, vy, vz, (p["ox"], p["oy"], p["oz"])))
            muon = Muon(rayon=Ray, beta=p["beta"])
            self.progress.emit(15)
            physique = Physique(rayon=None, scintillateur=Scint, muon=muon)
            self.progress.emit(20)
 
            multi = physique.run_multi(
                n_runs=self.n_runs, n_points=400,
                progress_cb=lambda v: self.progress.emit(v)
            )
            self.progress.emit(92)
 
            ref = multi["runs"][0]
            physique.muon.beta = p["beta"]
            ref_grid = physique.subdivide_hexa(21, 21, 21)
            physique.muon.beta = p["beta"]
            _, _, _, _, ref_grid = physique.E_reçue(grid=ref_grid)
            self.progress.emit(98)
 
            self.finished.emit({
                "physique": physique, "grid": ref_grid, "multi": multi,
                "t": ref["t"], "E": ref["E"], "beta": ref["beta"],
                "photons": ref["photons"], "t_sig": ref["t_sig"], "sig": ref["sig"],
            })
            self.progress.emit(100)
        except Exception:
            import traceback
            self.error.emit(traceback.format_exc())
 
 
# ══════════════════════════════════════════════════════════════════════════════
#  ParamRow
# ══════════════════════════════════════════════════════════════════════════════
class ParamRow(QWidget):
    valueChanged = pyqtSignal(float)
 
    def __init__(self, label, unit, vmin, vmax, default, decimals=3, parent=None):
        super().__init__(parent)
        self._scale = 10**decimals
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 1, 0, 1); lay.setSpacing(6)
 
        lbl = QLabel(label)
        lbl.setFixedWidth(52)
        lbl.setStyleSheet(f"color:{INK2}; font-size:10px;")
 
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(int(vmin*self._scale))
        self.slider.setMaximum(int(vmax*self._scale))
        self.slider.setValue(int(default*self._scale))
 
        self.spin = QDoubleSpinBox()
        self.spin.setDecimals(decimals)
        self.spin.setMinimum(vmin); self.spin.setMaximum(vmax)
        self.spin.setValue(default)
        self.spin.setSingleStep((vmax-vmin)/200)
        self.spin.setFixedWidth(80)
        if unit: self.spin.setSuffix(f" {unit}")
 
        lay.addWidget(lbl); lay.addWidget(self.slider, 1); lay.addWidget(self.spin)
 
        self.slider.valueChanged.connect(self._from_slider)
        self.spin.valueChanged.connect(self._from_spin)
 
    def _from_slider(self, v):
        val = v / self._scale
        self.spin.blockSignals(True); self.spin.setValue(val); self.spin.blockSignals(False)
        self.valueChanged.emit(val)
 
    def _from_spin(self, v):
        self.slider.blockSignals(True); self.slider.setValue(int(v*self._scale)); self.slider.blockSignals(False)
        self.valueChanged.emit(v)
 
    def value(self): return self.spin.value()
    def setValue(self, v): self.spin.setValue(v)
 
 
# ══════════════════════════════════════════════════════════════════════════════
#  Panneau paramètres
# ══════════════════════════════════════════════════════════════════════════════
class ParameterPanel(QWidget):
    paramsChanged = pyqtSignal(dict)
    runRequested  = pyqtSignal(dict)
 
    DEFAULTS = dict(beta=0.90, theta=10.0, phi=0.0,
                    ox=0.05, oy=0.05, oz=0.10,
                    scint_w=0.10, scint_h=0.10, scint_d=0.01)
 
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(306)
        self.setStyleSheet(f"background:{PANEL_L};")
 
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 14, 12, 12)
        root.setSpacing(8)
 
        # En-tête
        hdr = QLabel("MUON / SCINTILLATOR\nSIMULATOR")
        hdr.setStyleSheet(
            f"font-size:13px; font-weight:bold; color:{INK};"
            f" letter-spacing:2px; line-height:160%;"
        )
        root.addWidget(hdr)
 
        sub = QLabel("Bethe-Bloch  ·  Stepping  ·  Scintillation")
        sub.setStyleSheet(f"color:{INK3}; font-size:9px; letter-spacing:1px;")
        root.addWidget(sub)
        root.addWidget(self._hr())
 
        # §1 Particule
        g1 = QGroupBox("§1 — Particule")
        v1 = QVBoxLayout(g1); v1.setSpacing(4)
        self.p_beta  = ParamRow("β",  "c",  0.10, 0.9999, self.DEFAULTS["beta"],  4)
        self.p_theta = ParamRow("θ",  "°",  0.0,  89.0,   self.DEFAULTS["theta"], 1)
        self.p_phi   = ParamRow("φ",  "°",  0.0,  360.0,  self.DEFAULTS["phi"],   1)
        for w in (self.p_beta, self.p_theta, self.p_phi): v1.addWidget(w)
        root.addWidget(g1)
 
        # Énergie cinétique calculée
        self.lbl_E = QLabel("E_cin  =  —  MeV")
        self.lbl_E.setStyleSheet(
            f"color:{PHOSPHOR_D}; font-size:10px; padding:3px 6px;"
            f" background:{BG}; border:1px solid {BORDER_L}; border-left:3px solid {PHOSPHOR_D};"
        )
        root.addWidget(self.lbl_E)
        self.p_beta.valueChanged.connect(self._upd_E)
        self._upd_E(self.DEFAULTS["beta"])
 
        # §2 Origine
        g2 = QGroupBox("§2 — Origine du rayon")
        v2 = QVBoxLayout(g2); v2.setSpacing(4)
        self.p_ox = ParamRow("x₀", "m", 0.00, 0.10, self.DEFAULTS["ox"], 3)
        self.p_oy = ParamRow("y₀", "m", 0.00, 0.10, self.DEFAULTS["oy"], 3)
        self.p_oz = ParamRow("z₀", "m", 0.00, 0.30, self.DEFAULTS["oz"], 3)
        for w in (self.p_ox, self.p_oy, self.p_oz): v2.addWidget(w)
        root.addWidget(g2)
 
        # §3 Détecteur
        g3 = QGroupBox("§3 — Détecteur  BC-408")
        v3 = QVBoxLayout(g3); v3.setSpacing(4)
        self.p_sw = ParamRow("L_x", "m", 0.02, 0.30, self.DEFAULTS["scint_w"], 3)
        self.p_sh = ParamRow("L_y", "m", 0.02, 0.30, self.DEFAULTS["scint_h"], 3)
        self.p_sd = ParamRow("L_z", "m", 0.002,0.05, self.DEFAULTS["scint_d"], 3)
        for w in (self.p_sw, self.p_sh, self.p_sd): v3.addWidget(w)
        root.addWidget(g3)
 
        # Indicateur preview
        self.lbl_pv = QLabel("○  Preview temps réel actif")
        self.lbl_pv.setStyleSheet(
            f"color:{PHOSPHOR_D}; font-size:9px; letter-spacing:1px;"
        )
        root.addWidget(self.lbl_pv)
 
        root.addStretch()
        root.addWidget(self._hr())
 
        # Boutons
        row = QHBoxLayout()
        self.btn_rst = QPushButton("RESET")
        self.btn_rst.setObjectName("sec_btn")
        self.btn_rst.clicked.connect(self._reset)
        row.addWidget(self.btn_rst)
        lbl_n = QLabel(f"N = {N_RUNS} runs")
        lbl_n.setStyleSheet(f"color:{INK3}; font-size:9px;")
        row.addWidget(lbl_n, 0, Qt.AlignRight)
        root.addLayout(row)
 
        self.btn_run = QPushButton("▶  RUN SIMULATION")
        self.btn_run.setObjectName("run_btn")
        self.btn_run.clicked.connect(
            lambda: self.runRequested.emit(self.get_params()))
        root.addWidget(self.btn_run)
 
        for w in self._all():
            w.valueChanged.connect(
                lambda _: self.paramsChanged.emit(self.get_params()))
 
    def _hr(self):
        f = QFrame(); f.setFrameShape(QFrame.HLine)
        f.setStyleSheet(f"background:{BORDER_L}; max-height:1px; border:none;")
        return f
 
    def _all(self):
        return [self.p_beta, self.p_theta, self.p_phi,
                self.p_ox, self.p_oy, self.p_oz,
                self.p_sw, self.p_sh, self.p_sd]
 
    def get_params(self):
        return dict(beta=self.p_beta.value(), theta=self.p_theta.value(),
                    phi=self.p_phi.value(), ox=self.p_ox.value(),
                    oy=self.p_oy.value(), oz=self.p_oz.value(),
                    scint_w=self.p_sw.value(), scint_h=self.p_sh.value(),
                    scint_d=self.p_sd.value())
 
    def _reset(self):
        d = self.DEFAULTS
        pairs = [('p_beta','beta'),('p_theta','theta'),('p_phi','phi'),
                 ('p_ox','ox'),('p_oy','oy'),('p_oz','oz'),
                 ('p_sw','scint_w'),('p_sh','scint_h'),('p_sd','scint_d')]
        for attr, key in pairs:
            getattr(self, attr).setValue(d[key])
 
    def _upd_E(self, beta):
        import scipy.constants as sc
        m_mu = 207 * sc.electron_mass
        g    = 1.0 / (1-beta**2)**0.5 if beta < 1 else float('inf')
        E_MeV = (g-1) * m_mu * sc.c**2 / (sc.electron_volt * 1e6)
        self.lbl_E.setText(f"E_cin  =  {E_MeV:>8.1f}  MeV")
 
    def set_running(self, running: bool):
        self.btn_run.setEnabled(not running)
        self.btn_run.setText(
            "⏳  CALCUL EN COURS…" if running else "▶  RUN SIMULATION")
        self.lbl_pv.setText(
            "●  Simulation active" if running else "○  Preview temps réel actif")
 
 
# ══════════════════════════════════════════════════════════════════════════════
#  Overlay chargement
# ══════════════════════════════════════════════════════════════════════════════
class LoadingOverlay(QWidget):
    _STEPS = [(5,"Initialisation…"),(15,"Construction du rayon…"),
              (20,"Subdivision hexaédrique…"),(40,"Stepping Bethe-Bloch…"),
              (70,"Tirages stochastiques Poisson…"),(90,"Signal de scintillation…"),
              (98,"Finalisation…")]
 
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background:rgba(245,242,235,220);")
        lay = QVBoxLayout(self); lay.setAlignment(Qt.AlignCenter); lay.setSpacing(14)
 
        self.lbl = QLabel("SIMULATION EN COURS")
        self.lbl.setStyleSheet(
            f"font-size:14px; font-weight:bold; color:{INK}; letter-spacing:4px;")
        self.lbl.setAlignment(Qt.AlignCenter)
 
        self.step = QLabel("Initialisation…")
        self.step.setStyleSheet(f"color:{INK3}; font-size:10px; letter-spacing:1px;")
        self.step.setAlignment(Qt.AlignCenter)
 
        self.bar = QProgressBar()
        self.bar.setRange(0,100); self.bar.setValue(0)
        self.bar.setFixedWidth(280); self.bar.setFixedHeight(6)
        self.bar.setTextVisible(False)
 
        self.pct = QLabel("0 %")
        self.pct.setStyleSheet(
            f"color:{PHOSPHOR_D}; font-size:12px; font-weight:bold;")
        self.pct.setAlignment(Qt.AlignCenter)
 
        for w in (self.lbl, self.step, self.bar, self.pct): lay.addWidget(w, 0, Qt.AlignCenter)
        self.hide()
 
    def set_progress(self, v):
        self.bar.setValue(v); self.pct.setText(f"{v} %")
        for thr, msg in reversed(self._STEPS):
            if v >= thr: self.step.setText(msg); break
 
    def show_overlay(self):
        self.bar.setValue(0); self.pct.setText("0 %"); self.step.setText("Initialisation…")
        self.show(); self.raise_()
 
    def resizeEvent(self, e):
        if self.parent(): self.setGeometry(self.parent().rect())
        super().resizeEvent(e)
 
 
# ══════════════════════════════════════════════════════════════════════════════
#  Preview 3D
# ══════════════════════════════════════════════════════════════════════════════
class Preview3D(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self); lay.setContentsMargins(0,0,0,0); lay.setSpacing(0)
 
        hdr = QLabel("  GÉOMÉTRIE — PREVIEW TEMPS RÉEL")
        hdr.setFixedHeight(26)
        hdr.setStyleSheet(
            f"background:{BG2}; color:{INK2}; font-size:9px; letter-spacing:2px;"
            f" border-bottom:1px solid {BORDER_L}; padding-left:8px;")
        lay.addWidget(hdr)
 
        self.plotter = QtInteractor(self)
        self.plotter.set_background(BG2)
        lay.addWidget(self.plotter.interactor)
 
        self._timer = QTimer(); self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._do_update)
        self._pending = None
 
    def schedule_update(self, params):
        self._pending = params; self._timer.start(250)
 
    def _do_update(self):
        if self._pending: self._draw(self._pending)
 
    def _draw(self, p):
        from .geometry import Scintillateur, Rayon, Vecteur
        self.plotter.clear()
        W, H, D = p["scint_w"], p["scint_h"], p["scint_d"]
        Scint = Scintillateur(
            (0,0,0),(W,0,0),(W,H,0),(0,H,0),
            (0,0,-D),(W,0,-D),(W,H,-D),(0,H,-D))
        theta = math.radians(p["theta"]); phi = math.radians(p["phi"])
        vx = math.sin(theta)*math.cos(phi)
        vy = math.sin(theta)*math.sin(phi)
        vz = -math.cos(theta)
        Ray = Rayon(Vecteur(vx, vy, vz, (p["ox"], p["oy"], p["oz"])))
        Scint.plot3DScint(plotter=self.plotter, color="#AACCAA",
                          opacity=0.25, show_edges=True, show_points=True)
        Scint.plot_intersections_red(ray=Ray, plotter=self.plotter)
        Ray.plot3DRay(plotter=self.plotter)
        self.plotter.add_axes(); self.plotter.reset_camera()
 
    def update_with_results(self, result):
        physique = result["physique"]; grid = result["grid"]
        self.plotter.clear()
        physique.scintillateur.plot3DScint(
            plotter=self.plotter, color="#AACCAA",
            opacity=0.18, show_edges=True, show_points=False)
        self.plotter.add_mesh(grid, style="wireframe", color=BORDER_L, opacity=0.3)
        hit = grid.threshold(0.5, scalars="atteinte")
        if hit.n_cells > 0:
            self.plotter.add_mesh(hit, scalars="Energie", cmap="YlOrRd",
                                  opacity=0.9, show_edges=True)
        physique.rayon.plot3DRay(plotter=self.plotter)
        self.plotter.add_axes(); self.plotter.reset_camera()
 
 
# ══════════════════════════════════════════════════════════════════════════════
#  Onglet Courbes — style oscilloscope / publication
# ══════════════════════════════════════════════════════════════════════════════
class ResultsTab(QWidget):
    # Couleurs des 4 canaux : phosphore, ambre, cyan, rouge
    _CH_COL = [PHOSPHOR_D, AMBER, CYAN_OSC, RED_OSC]
 
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6); lay.setSpacing(4)
 
        # Barre de stats style "mesures de labo"
        self.stats = QLabel(
            "  CH1 Signal  ·  CH2 Photons  ·  CH3 Énergie  ·  CH4 β   "
            "|   En attente de simulation…"
        )
        self.stats.setStyleSheet(
            f"background:{BG2}; color:{INK2}; font-size:9px;"
            f" padding:4px 8px; border:1px solid {BORDER_L}; letter-spacing:1px;")
        lay.addWidget(self.stats)
 
        self.fig = Figure(figsize=(14, 8), dpi=100)
        self.fig.patch.set_facecolor(BG)
 
        with matplotlib.rc_context(MPL_STYLE):
            self.ax_sig   = self.fig.add_subplot(2, 2, 1)
            self.ax_photo = self.fig.add_subplot(2, 2, 2)
            self.ax_E     = self.fig.add_subplot(2, 2, 3)
            self.ax_beta  = self.fig.add_subplot(2, 2, 4)
            self.fig.tight_layout(pad=2.5, h_pad=3.2, w_pad=2.5)
 
        self.canvas = FigureCanvasQTAgg(self.fig)
        lay.addWidget(self.canvas)
        self._placeholder()
 
    def _placeholder(self):
        for ax, title in [
            (self.ax_sig,   "CH1 — Signal de scintillation"),
            (self.ax_photo, "CH2 — Photons émis / voxel"),
            (self.ax_E,     "CH3 — Énergie cumulée"),
            (self.ax_beta,  "CH4 — β(t) du muon"),
        ]:
            ax.set_facecolor(BG2)
            ax.set_title(title, color=INK2, fontsize=9, pad=6,
                         fontfamily=FONT_MONO, loc='left')
            ax.text(0.5, 0.5, "awaiting simulation…", ha='center', va='center',
                    color=BORDER_D, fontsize=9, fontfamily=FONT_MONO,
                    transform=ax.transAxes, style='italic')
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values(): sp.set_color(BORDER_L)
        self.canvas.draw()
 
    def update(self, t_ref, E_ref, beta_ref, photons_ref,
               t_sig_ref, sig_ref, multi: dict):
        with matplotlib.rc_context(MPL_STYLE):
            self._draw_signal(multi)
            self._draw_photons(multi)
            self._draw_E(t_ref, E_ref)
            self._draw_beta(t_ref, beta_ref, multi)
            self.fig.tight_layout(pad=2.5, h_pad=3.2, w_pad=2.5)
            self.canvas.draw()
 
        n = multi["sig_runs"].shape[0]
        ph = multi["ph_runs"].sum(axis=1)
        self.stats.setText(
            f"  CH1 Signal  ·  CH2 Photons  ·  CH3 Énergie  ·  CH4 β   |  "
            f"N = {n} runs   <Nph> = {ph.mean():.0f} ± {ph.std():.0f}   "
            f"|   trait plein = moyenne  ·  zone = ±1σ"
        )
 
    # ── CH1 Signal ───────────────────────────────────────────────────────────
    def _draw_signal(self, multi):
        ax = self.ax_sig; ax.cla(); ax.set_facecolor(BG2)
        t = multi["t_common"]; sr = multi["sig_runs"]; n = sr.shape[0]
        col = self._CH_COL[0]
        for i in range(n):
            ax.plot(t, sr[i], color=col, alpha=0.18, lw=0.8)
        mean = sr.mean(0); std = sr.std(0)
        ax.fill_between(t, mean-std, mean+std, color=col, alpha=0.18)
        ax.plot(t, mean, color=col, lw=2.0, label=f"<S>  n={n}")
        ax.set_xlabel("t  (ns)"); ax.set_ylabel("dE  (J)")
        ax.set_title("CH1 — Signal de scintillation", color=INK,
                     fontsize=9, pad=6, fontfamily=FONT_MONO, loc='left')
        ax.legend(); ax.grid(True)
        self._style(ax, col)
 
    # ── CH2 Photons ──────────────────────────────────────────────────────────
    def _draw_photons(self, multi):
        ax = self.ax_photo; ax.cla(); ax.set_facecolor(BG2)
        t = multi["ph_t_common"]; pr = multi["ph_runs"]; n = pr.shape[0]
        col = self._CH_COL[1]
        for i in range(n):
            ax.scatter(t, pr[i], color=col, alpha=0.12, s=3, linewidths=0)
        mean = pr.mean(0); std = pr.std(0)
        w  = max(1, len(mean)//10)
        ms = np.convolve(mean, np.ones(w)/w, mode='same')
        ss = np.convolve(std,  np.ones(w)/w, mode='same')
        ax.fill_between(t, ms-ss, ms+ss, color=col, alpha=0.22)
        ax.plot(t, ms, color=col, lw=2.0, label=f"<Nph/vox>  n={n}")
        vals = pr[pr > 0]
        if len(vals):
            yc, ys = float(np.mean(vals)), float(np.std(vals))
            ax.set_ylim(max(0, yc-4*ys), yc+4*ys)
        ax.set_xlabel("t  (ns)"); ax.set_ylabel("Nγ")
        ax.set_title("CH2 — Photons émis / voxel  [Poisson]", color=INK,
                     fontsize=9, pad=6, fontfamily=FONT_MONO, loc='left')
        ax.legend(); ax.grid(True)
        self._style(ax, col)
 
    # ── CH3 Énergie ──────────────────────────────────────────────────────────
    def _draw_E(self, t_ref, E_ref):
        ax = self.ax_E; ax.cla(); ax.set_facecolor(BG2)
        col = self._CH_COL[2]
        t = np.array(t_ref)*1e9; E = np.array(E_ref)
        ax.fill_between(t, E, alpha=0.15, color=col)
        ax.plot(t, E, color=col, lw=2.0, label="E_dep cumulée")
        ax.set_xlabel("t  (ns)"); ax.set_ylabel("E  (J)")
        ax.set_title("CH3 — Énergie cumulée déposée", color=INK,
                     fontsize=9, pad=6, fontfamily=FONT_MONO, loc='left')
        ax.legend(); ax.grid(True)
        self._style(ax, col)
 
    # ── CH4 Beta ─────────────────────────────────────────────────────────────
    def _draw_beta(self, t_ref, beta_ref, multi):
        ax = self.ax_beta; ax.cla(); ax.set_facecolor(BG2)
        col = self._CH_COL[3]; runs = multi.get("runs", [])
        if len(runs) > 1:
            t0 = np.array(runs[0]["t"], dtype=float)*1e9
            mat = []
            for r in runs:
                tr = np.array(r["t"], dtype=float)*1e9
                br = np.array(r["beta"], dtype=float)
                if len(tr) == len(t0): mat.append(br)
                elif len(tr) > 1:
                    mat.append(np.interp(t0, tr, br, left=br[0], right=br[-1]))
            if mat:
                M = np.array(mat)
                for row in M: ax.plot(t0, row, color=col, alpha=0.15, lw=0.8)
                mean = M.mean(0); std = M.std(0)
                ax.fill_between(t0, mean-std, mean+std, color=col, alpha=0.15)
                ax.plot(t0, mean, color=col, lw=2.0, label=f"<beta>  n={len(runs)}")
        else:
            t = np.array(t_ref, dtype=float)*1e9
            ax.plot(t, beta_ref, color=col, lw=2.0, label="β(t)")
        ax.set_xlabel("t  (ns)"); ax.set_ylabel("β = v/c")
        ax.set_title("CH4 — β(t) du muon", color=INK,
                     fontsize=9, pad=6, fontfamily=FONT_MONO, loc='left')
        ax.legend(); ax.grid(True)
        self._style(ax, col)
 
    def _style(self, ax, accent):
        for sp in ax.spines.values():
            sp.set_color(BORDER_L); sp.set_linewidth(0.8)
        ax.spines['left'].set_color(accent); ax.spines['left'].set_linewidth(2.0)
        ax.tick_params(axis='both', length=3, color=BORDER_D, labelsize=8)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x,_: f"{x:.9g}"))
 
 
# ══════════════════════════════════════════════════════════════════════════════
#  Onglet Stepping 3D
# ══════════════════════════════════════════════════════════════════════════════
class SteppingTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self); lay.setContentsMargins(0,0,0,0); lay.setSpacing(0)
        hdr = QLabel("  STEPPING — VOXELS TRAVERSÉS  |  couleur ∝ énergie déposée")
        hdr.setFixedHeight(26)
        hdr.setStyleSheet(
            f"background:{BG2}; color:{INK2}; font-size:9px; letter-spacing:2px;"
            f" border-bottom:1px solid {BORDER_L}; padding-left:8px;")
        lay.addWidget(hdr)
        self.plotter = QtInteractor(self)
        self.plotter.set_background(BG2)
        lay.addWidget(self.plotter.interactor)
        self.plotter.add_text("Run a simulation to visualize stepping",
                              position="lower_edge", font_size=9, color=INK3)
 
    def update_with_results(self, result):
        physique = result["physique"]; grid = result["grid"]
        self.plotter.clear()
        physique.scintillateur.plot3DScint(
            plotter=self.plotter, color="#AACCAA",
            opacity=0.12, show_edges=True, show_points=False)
        self.plotter.add_mesh(grid, style="wireframe", color=BORDER_L, opacity=0.22)
        hit = grid.threshold(0.5, scalars="atteinte")
        if hit.n_cells > 0:
            self.plotter.add_mesh(
                hit, scalars="Energie", cmap="YlOrRd", opacity=0.95,
                show_edges=True,
                scalar_bar_args={"title":"E (J)","color":INK2,
                                 "title_font_size":9,"label_font_size":8,"fmt":"%.2e"})
        physique.rayon.plot3DRay(plotter=self.plotter)
        self.plotter.add_axes(); self.plotter.reset_camera()
 
 
# ══════════════════════════════════════════════════════════════════════════════
#  Fenêtre principale
# ══════════════════════════════════════════════════════════════════════════════
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(
            "Muon in Scintillator Simulator  ·  PHY473  ·  T. Adouani")
        self.resize(1640, 960)
        self.setStyleSheet(GLOBAL_QSS)
        self._worker = None
 
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0,0,0,0); root.setSpacing(0)
 
        self.param_panel = ParameterPanel()
        self.param_panel.paramsChanged.connect(
            lambda p: self.preview.schedule_update(p))
        self.param_panel.runRequested.connect(self._on_run)
        root.addWidget(self.param_panel)
 
        sep = QFrame(); sep.setFrameShape(QFrame.VLine)
        sep.setStyleSheet(f"background:{BORDER_L}; max-width:1px; border:none;")
        root.addWidget(sep)
 
        right = QSplitter(Qt.Vertical); right.setHandleWidth(4)
        self.preview = Preview3D()
        right.addWidget(self.preview)
 
        self.tabs = QTabWidget()
        self.results_tab  = ResultsTab()
        self.stepping_tab = SteppingTab()
        self.tabs.addTab(self.results_tab,  "Courbes")
        self.tabs.addTab(self.stepping_tab, "Stepping 3D")
        right.addWidget(self.tabs)
        right.setSizes([480, 440])
        root.addWidget(right, 1)
 
        self.overlay = LoadingOverlay(self)
        QTimer.singleShot(200, lambda: self.preview.schedule_update(
            self.param_panel.get_params()))
 
    def resizeEvent(self, e):
        super().resizeEvent(e)
        if hasattr(self, "overlay"): self.overlay.setGeometry(self.rect())
 
    def _on_run(self, params):
        if self._worker and self._worker.isRunning(): return
        self.param_panel.set_running(True)
        self.overlay.show_overlay(); self.overlay.setGeometry(self.rect())
        self._worker = SimWorker(params)
        self._worker.progress.connect(self.overlay.set_progress)
        self._worker.finished.connect(self._on_done)
        self._worker.error.connect(self._on_error)
        self._worker.start()
 
    def _on_done(self, result):
        self.overlay.set_progress(100)
        QTimer.singleShot(400, lambda: self._apply(result))
 
    def _apply(self, result):
        self.overlay.hide(); self.param_panel.set_running(False)
        self.preview.update_with_results(result)
        self.stepping_tab.update_with_results(result)
        self.results_tab.update(
            result["t"], result["E"], result["beta"],
            result["photons"], result["t_sig"], result["sig"],
            result["multi"])
        self.tabs.setCurrentIndex(0)
 
    def _on_error(self, tb):
        self.overlay.hide(); self.param_panel.set_running(False)
        from PyQt5.QtWidgets import QMessageBox
        mb = QMessageBox(self)
        mb.setWindowTitle("Erreur simulation")
        mb.setText("Une erreur s'est produite.")
        mb.setDetailedText(tb); mb.setStyleSheet(GLOBAL_QSS); mb.exec_()
 
 
# ══════════════════════════════════════════════════════════════════════════════
#  Point d'entrée
# ══════════════════════════════════════════════════════════════════════════════
def launch_ui(physique=None, grid=None, t=None, E=None, beta=None,
              t_sig=None, sig=None, photons_émis=None):
    app = QApplication.instance() or QApplication(sys.argv)
    win = MainWindow(); win.show()
 
    if physique is not None and grid is not None:
        t_ns_  = np.array(t,     dtype=float)*1e9 if (t     is not None and len(t))     else np.zeros(1)
        t_sig_ = np.asarray(t_sig, dtype=float)   if (t_sig is not None and len(t_sig)) else np.zeros(1)
        sig_   = np.asarray(sig,   dtype=float)   if (sig   is not None and len(sig))   else np.zeros(1)
        ph_    = np.array(photons_émis, dtype=float) if photons_émis else np.zeros(1)
        tc     = np.linspace(0.0, float(t_sig_.max()) if len(t_sig_)>1 else 1.0, 400)
        si     = np.interp(tc, t_sig_, sig_, left=0.0, right=0.0)
        mm = dict(t_common=tc*1e9, sig_runs=si[np.newaxis,:],
                  ph_t_common=t_ns_[:len(ph_)], ph_runs=ph_[np.newaxis,:],
                  runs=[dict(t=t,E=E,beta=beta,photons=photons_émis,
                             t_sig=t_sig,sig=sig)])
        result = dict(physique=physique, grid=grid, t=t, E=E, beta=beta,
                      photons=photons_émis, t_sig=t_sig, sig=sig, multi=mm)
        QTimer.singleShot(500, lambda: win._apply(result))
 
    app.exec_()