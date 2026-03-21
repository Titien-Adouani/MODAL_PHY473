"""
UI fondée sur PyQt
Codée avec l'aide d'une IA Générative 
"""

from __future__ import annotations

import sys
import math
import threading
import numpy as np

import pyvista as pv
from pyvistaqt import QtInteractor

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QSplitter,
    QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
    QPushButton, QSlider, QDoubleSpinBox, QGroupBox,
    QTabWidget, QStackedWidget, QSizePolicy, QFrame,
    QProgressBar,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QPropertyAnimation, QEasingCurve
from PyQt5.QtGui import QFont, QColor, QPalette, QLinearGradient

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

# ─────────────────────────────────────────────────────────────────────────────
#  Palette & styles
# ─────────────────────────────────────────────────────────────────────────────
DARK_BG     = "#0d0f14"
PANEL_BG    = "#13161e"
CARD_BG     = "#1a1e2a"
BORDER      = "#252a38"
ACCENT      = "#00e5ff"
ACCENT2     = "#7c4dff"
TEXT_PRI    = "#e8eaf6"
TEXT_SEC    = "#6b7280"
SUCCESS     = "#00e676"
WARNING     = "#ff9800"
DANGER      = "#f44336"

GLOBAL_QSS = f"""
QMainWindow, QWidget {{
    background-color: {DARK_BG};
    color: {TEXT_PRI};
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 12px;
}}
QGroupBox {{
    background-color: {CARD_BG};
    border: 1px solid {BORDER};
    border-radius: 8px;
    margin-top: 16px;
    padding: 8px;
    font-size: 11px;
    font-weight: bold;
    color: {ACCENT};
    letter-spacing: 1px;
    text-transform: uppercase;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 6px;
}}
QLabel {{
    color: {TEXT_PRI};
    background: transparent;
}}
QLabel#secondary {{
    color: {TEXT_SEC};
    font-size: 11px;
}}
QDoubleSpinBox, QSlider::groove:horizontal {{
    background-color: {PANEL_BG};
    border: 1px solid {BORDER};
    border-radius: 4px;
    color: {TEXT_PRI};
}}
QDoubleSpinBox {{
    padding: 4px 8px;
    min-width: 80px;
}}
QDoubleSpinBox:focus {{
    border: 1px solid {ACCENT};
}}
QSlider::groove:horizontal {{
    height: 4px;
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background: {ACCENT};
    border: none;
    width: 14px;
    height: 14px;
    margin: -5px 0;
    border-radius: 7px;
}}
QSlider::sub-page:horizontal {{
    background: {ACCENT};
    border-radius: 2px;
}}
QPushButton#run_btn {{
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 {ACCENT2}, stop:1 {ACCENT});
    color: #000000;
    font-weight: bold;
    font-size: 13px;
    letter-spacing: 2px;
    border: none;
    border-radius: 8px;
    padding: 14px 24px;
    min-height: 48px;
}}
QPushButton#run_btn:hover {{
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 {ACCENT}, stop:1 {ACCENT2});
}}
QPushButton#run_btn:disabled {{
    background: {BORDER};
    color: {TEXT_SEC};
}}
QPushButton#reset_btn {{
    background: transparent;
    color: {TEXT_SEC};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 8px 16px;
    font-size: 11px;
}}
QPushButton#reset_btn:hover {{
    border-color: {ACCENT};
    color: {ACCENT};
}}
QTabWidget::pane {{
    border: 1px solid {BORDER};
    background: {PANEL_BG};
    border-radius: 0 8px 8px 8px;
}}
QTabBar::tab {{
    background: {CARD_BG};
    color: {TEXT_SEC};
    border: 1px solid {BORDER};
    padding: 8px 20px;
    font-size: 11px;
    letter-spacing: 1px;
}}
QTabBar::tab:selected {{
    background: {PANEL_BG};
    color: {ACCENT};
    border-bottom: 2px solid {ACCENT};
}}
QProgressBar {{
    background-color: {CARD_BG};
    border: 1px solid {BORDER};
    border-radius: 4px;
    text-align: center;
    color: {TEXT_PRI};
    height: 8px;
}}
QProgressBar::chunk {{
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 {ACCENT2}, stop:1 {ACCENT});
    border-radius: 4px;
}}
QSplitter::handle {{
    background: {BORDER};
    width: 2px;
}}
QFrame#divider {{
    background: {BORDER};
    max-height: 1px;
}}
"""

MPL_STYLE = {
    "axes.facecolor":   PANEL_BG,
    "figure.facecolor": PANEL_BG,
    "axes.edgecolor":   BORDER,
    "axes.labelcolor":  TEXT_SEC,
    "xtick.color":      TEXT_SEC,
    "ytick.color":      TEXT_SEC,
    "grid.color":       BORDER,
    "grid.alpha":       0.5,
    "text.color":       TEXT_PRI,
    "lines.linewidth":  2.0,
}


# ─────────────────────────────────────────────────────────────────────────────
#  Worker thread — runs the simulation without blocking the UI
# ─────────────────────────────────────────────────────────────────────────────
N_RUNS = 10   # nombre de tirages stochastiques par simulation


class SimWorker(QThread):
    finished  = pyqtSignal(object)   # emits result dict
    error     = pyqtSignal(str)
    progress  = pyqtSignal(int)      # 0-100
    run_done  = pyqtSignal(int, int) # (run_index, n_total) — pour l'overlay

    def __init__(self, params: dict, n_runs: int = N_RUNS):
        super().__init__()
        self.params = params
        self.n_runs = n_runs

    def run(self):
        try:
            from .physics import Physique, Muon, Scintillateur, Rayon, Vecteur

            p = self.params
            self.progress.emit(5)

            # ── Scintillateur
            W, H, D = p["scint_w"], p["scint_h"], p["scint_d"]
            Scint = Scintillateur(
                (0,0,0), (W,0,0), (W,H,0), (0,H,0),
                (0,0,-D), (W,0,-D), (W,H,-D), (0,H,-D),
            )
            self.progress.emit(10)

            # ── Direction du muon
            theta = math.radians(p["theta"])
            phi   = math.radians(p["phi"])
            vx =  math.sin(theta) * math.cos(phi)
            vy =  math.sin(theta) * math.sin(phi)
            vz = -math.cos(theta)

            Ray  = Rayon(Vecteur(vx, vy, vz, (p["ox"], p["oy"], p["oz"])))
            muon = Muon(rayon=Ray, beta=p["beta"])
            self.progress.emit(15)

            physique = Physique(rayon=None, scintillateur=Scint, muon=muon)
            self.progress.emit(20)

            # ── Multi-runs stochastiques
            def _cb(v):
                self.progress.emit(v)

            multi = physique.run_multi(n_runs=self.n_runs,
                                       n_points=400,
                                       progress_cb=_cb)
            self.progress.emit(92)

            # Le premier run sert d'affichage 3D de référence
            ref = multi["runs"][0]

            # La grille de référence pour la visu 3D = dernier run de run_multi
            # On recrée une grille propre depuis le run de référence
            physique.muon.beta = p["beta"]   # reset beta
            ref_grid = physique.subdivide_hexa(21, 21, 21)
            physique.muon.beta = p["beta"]
            _, _, _, _, ref_grid = physique.E_reçue(grid=ref_grid)
            self.progress.emit(98)

            self.finished.emit({
                "physique" : physique,
                "grid"     : ref_grid,
                "multi"    : multi,
                # Données du run de référence (rétrocompat)
                "t"        : ref["t"],
                "E"        : ref["E"],
                "beta"     : ref["beta"],
                "photons"  : ref["photons"],
                "t_sig"    : ref["t_sig"],
                "sig"      : ref["sig"],
            })
            self.progress.emit(100)

        except Exception:
            import traceback
            self.error.emit(traceback.format_exc())


# ─────────────────────────────────────────────────────────────────────────────
#  Paramètre slider + spinbox couplés
# ─────────────────────────────────────────────────────────────────────────────
class ParamRow(QWidget):
    valueChanged = pyqtSignal(float)

    def __init__(self, label: str, vmin: float, vmax: float,
                 default: float, decimals: int = 3, step: float = None,
                 unit: str = "", parent=None):
        super().__init__(parent)
        self._min = vmin
        self._max = vmax
        self._decimals = decimals
        self._scale = 10 ** decimals

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(8)

        lbl = QLabel(label)
        lbl.setFixedWidth(90)
        lbl.setStyleSheet(f"color: {TEXT_SEC}; font-size: 11px;")

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(int(vmin * self._scale))
        self.slider.setMaximum(int(vmax * self._scale))
        self.slider.setValue(int(default * self._scale))

        self.spin = QDoubleSpinBox()
        self.spin.setDecimals(decimals)
        self.spin.setMinimum(vmin)
        self.spin.setMaximum(vmax)
        self.spin.setValue(default)
        self.spin.setSingleStep(step or (vmax - vmin) / 100)
        self.spin.setFixedWidth(90)
        if unit:
            self.spin.setSuffix(f" {unit}")

        layout.addWidget(lbl)
        layout.addWidget(self.slider, 1)
        layout.addWidget(self.spin)

        # Connexions croisées
        self.slider.valueChanged.connect(self._slider_changed)
        self.spin.valueChanged.connect(self._spin_changed)

    def _slider_changed(self, v):
        val = v / self._scale
        self.spin.blockSignals(True)
        self.spin.setValue(val)
        self.spin.blockSignals(False)
        self.valueChanged.emit(val)

    def _spin_changed(self, v):
        self.slider.blockSignals(True)
        self.slider.setValue(int(v * self._scale))
        self.slider.blockSignals(False)
        self.valueChanged.emit(v)

    def value(self) -> float:
        return self.spin.value()

    def setValue(self, v: float):
        self.spin.setValue(v)


# ─────────────────────────────────────────────────────────────────────────────
#  Panneau de paramètres (colonne gauche)
# ─────────────────────────────────────────────────────────────────────────────
class ParameterPanel(QWidget):
    paramsChanged = pyqtSignal(dict)
    runRequested  = pyqtSignal(dict)

    # Valeurs par défaut
    DEFAULTS = dict(
        beta=0.90, theta=10.0, phi=0.0,
        ox=0.05, oy=0.05, oz=0.10,
        scint_w=0.10, scint_h=0.10, scint_d=0.01,
    )

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(340)
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        # ── Titre
        title = QLabel("MUON SIMULATOR")
        title.setStyleSheet(
            f"color: {ACCENT}; font-size: 16px; font-weight: bold;"
            f" letter-spacing: 3px; padding: 8px 0;"
        )
        root.addWidget(title)

        sub = QLabel("Mini-GEANT4 · Scintillateur plastique")
        sub.setObjectName("secondary")
        root.addWidget(sub)

        div = QFrame(); div.setObjectName("divider")
        root.addWidget(div)

        # ── Groupe : Muon
        grp_muon = QGroupBox("Muon")
        g1 = QVBoxLayout(grp_muon)
        self.p_beta  = ParamRow("β (vitesse)",  0.10, 0.9999, self.DEFAULTS["beta"],  4, unit="c")
        self.p_theta = ParamRow("θ polaire",    0.0,  89.0,   self.DEFAULTS["theta"], 1, unit="°")
        self.p_phi   = ParamRow("φ azimutal",   0.0,  360.0,  self.DEFAULTS["phi"],   1, unit="°")
        for w in (self.p_beta, self.p_theta, self.p_phi):
            g1.addWidget(w)
        root.addWidget(grp_muon)

        # ── Groupe : Position d'entrée
        grp_pos = QGroupBox("Origine du rayon")
        g2 = QVBoxLayout(grp_pos)
        self.p_ox = ParamRow("x₀",  0.00, 0.10, self.DEFAULTS["ox"],  3, unit="m")
        self.p_oy = ParamRow("y₀",  0.00, 0.10, self.DEFAULTS["oy"],  3, unit="m")
        self.p_oz = ParamRow("z₀",  0.00, 0.30, self.DEFAULTS["oz"],  3, unit="m")
        for w in (self.p_ox, self.p_oy, self.p_oz):
            g2.addWidget(w)
        root.addWidget(grp_pos)

        # ── Groupe : Scintillateur
        grp_scint = QGroupBox("Scintillateur")
        g3 = QVBoxLayout(grp_scint)
        self.p_sw = ParamRow("Largeur",   0.02, 0.30, self.DEFAULTS["scint_w"], 3, unit="m")
        self.p_sh = ParamRow("Hauteur",   0.02, 0.30, self.DEFAULTS["scint_h"], 3, unit="m")
        self.p_sd = ParamRow("Épaisseur", 0.002, 0.05, self.DEFAULTS["scint_d"], 3, unit="m")
        for w in (self.p_sw, self.p_sh, self.p_sd):
            g3.addWidget(w)
        root.addWidget(grp_scint)

        # ── Live preview indicator
        self.preview_lbl = QLabel("⬤  Preview en direct")
        self.preview_lbl.setStyleSheet(f"color: {SUCCESS}; font-size: 11px;")
        root.addWidget(self.preview_lbl)

        root.addStretch()

        # ── Bouton Reset
        btn_reset = QPushButton("RESET DÉFAUTS")
        btn_reset.setObjectName("reset_btn")
        btn_reset.clicked.connect(self._reset)
        root.addWidget(btn_reset)

        # ── Bouton Lancer
        self.btn_run = QPushButton("▶  LANCER LA SIMULATION")
        self.btn_run.setObjectName("run_btn")
        self.btn_run.clicked.connect(self._on_run)
        root.addWidget(self.btn_run)

        # Connexions live
        for w in self._all_params():
            w.valueChanged.connect(self._on_change)

    def _all_params(self):
        return [self.p_beta, self.p_theta, self.p_phi,
                self.p_ox, self.p_oy, self.p_oz,
                self.p_sw, self.p_sh, self.p_sd]

    def get_params(self) -> dict:
        return dict(
            beta=self.p_beta.value(),
            theta=self.p_theta.value(),
            phi=self.p_phi.value(),
            ox=self.p_ox.value(),
            oy=self.p_oy.value(),
            oz=self.p_oz.value(),
            scint_w=self.p_sw.value(),
            scint_h=self.p_sh.value(),
            scint_d=self.p_sd.value(),
        )

    def _on_change(self, _):
        self.paramsChanged.emit(self.get_params())

    def _on_run(self):
        self.runRequested.emit(self.get_params())

    def _reset(self):
        defaults = self.DEFAULTS
        self.p_beta.setValue(defaults["beta"])
        self.p_theta.setValue(defaults["theta"])
        self.p_phi.setValue(defaults["phi"])
        self.p_ox.setValue(defaults["ox"])
        self.p_oy.setValue(defaults["oy"])
        self.p_oz.setValue(defaults["oz"])
        self.p_sw.setValue(defaults["scint_w"])
        self.p_sh.setValue(defaults["scint_h"])
        self.p_sd.setValue(defaults["scint_d"])

    def set_running(self, running: bool):
        self.btn_run.setEnabled(not running)
        self.btn_run.setText(
            "⏳  SIMULATION EN COURS…" if running else "▶  LANCER LA SIMULATION"
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Écran de chargement (overlay)
# ─────────────────────────────────────────────────────────────────────────────
class LoadingOverlay(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        self.setStyleSheet(f"background-color: rgba(13,15,20,210);")

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(20)

        self.lbl_title = QLabel("SIMULATION EN COURS")
        self.lbl_title.setStyleSheet(
            f"color: {ACCENT}; font-size: 18px; font-weight: bold; letter-spacing: 4px;"
        )
        self.lbl_title.setAlignment(Qt.AlignCenter)

        self.lbl_sub = QLabel("Calcul du stepping Bethe-Bloch…")
        self.lbl_sub.setObjectName("secondary")
        self.lbl_sub.setAlignment(Qt.AlignCenter)

        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.setValue(0)
        self.bar.setFixedWidth(320)
        self.bar.setFixedHeight(8)
        self.bar.setTextVisible(False)

        self.lbl_pct = QLabel("0 %")
        self.lbl_pct.setStyleSheet(f"color: {TEXT_SEC}; font-size: 11px;")
        self.lbl_pct.setAlignment(Qt.AlignCenter)

        layout.addWidget(self.lbl_title)
        layout.addWidget(self.lbl_sub)
        layout.addWidget(self.bar, 0, Qt.AlignCenter)
        layout.addWidget(self.lbl_pct)
        self.hide()

    def set_progress(self, v: int):
        self.bar.setValue(v)
        self.lbl_pct.setText(f"{v} %")
        steps = [
            (5,  "Initialisation du scintillateur…"),
            (20, "Construction du rayon muon…"),
            (30, "Subdivision hexaédrique…"),
            (50, "Stepping Bethe-Bloch…"),
            (75, "Calcul de l'énergie déposée…"),
            (90, "Calcul du signal de scintillation…"),
            (99, "Finalisation…"),
        ]
        for threshold, msg in reversed(steps):
            if v >= threshold:
                self.lbl_sub.setText(msg)
                break

    def show_overlay(self):
        self.bar.setValue(0)
        self.lbl_pct.setText("0 %")
        self.lbl_sub.setText("Initialisation…")
        self.show()
        self.raise_()

    def resizeEvent(self, event):
        if self.parent():
            self.setGeometry(self.parent().rect())
        super().resizeEvent(event)


# ─────────────────────────────────────────────────────────────────────────────
#  Vue 3D preview (mise à jour live sur changement de params)
# ─────────────────────────────────────────────────────────────────────────────
class Preview3D(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        hdr = QLabel("PREVIEW 3D — GÉOMÉTRIE")
        hdr.setStyleSheet(
            f"color: {ACCENT}; font-size: 11px; letter-spacing: 2px;"
            f" padding: 6px 10px; background: {CARD_BG};"
        )
        layout.addWidget(hdr)

        self.plotter = QtInteractor(self)
        self.plotter.set_background(DARK_BG)
        layout.addWidget(self.plotter.interactor)

        self._timer = QTimer()
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._do_update)
        self._pending_params = None

    def schedule_update(self, params: dict):
        """Debounce : attendre 300ms d'inactivité avant de redessiner."""
        self._pending_params = params
        self._timer.start(300)

    def _do_update(self):
        if self._pending_params is None:
            return
        p = self._pending_params
        self._draw(p)

    def _draw(self, p: dict):
        import math
        from .geometry import Scintillateur, Rayon, Vecteur

        self.plotter.clear()

        W, H, D = p["scint_w"], p["scint_h"], p["scint_d"]
        Scint = Scintillateur(
            (0,0,0), (W,0,0), (W,H,0), (0,H,0),
            (0,0,-D), (W,0,-D), (W,H,-D), (0,H,-D),
        )

        theta = math.radians(p["theta"])
        phi   = math.radians(p["phi"])
        vx = math.sin(theta) * math.cos(phi)
        vy = math.sin(theta) * math.sin(phi)
        vz = -math.cos(theta)
        Ray = Rayon(Vecteur(vx, vy, vz, (p["ox"], p["oy"], p["oz"])))

        Scint.plot3DScint(
            plotter=self.plotter, color="#00e5ff",
            opacity=0.18, show_edges=True, show_points=True
        )
        Scint.plot_intersections_red(ray=Ray, plotter=self.plotter)
        Ray.plot3DRay(plotter=self.plotter)
        self.plotter.add_axes()
        self.plotter.reset_camera()

    def update_with_results(self, result: dict):
        """Afficher le résultat de simulation (voxels colorés)."""
        physique = result["physique"]
        grid     = result["grid"]

        self.plotter.clear()
        physique.scintillateur.plot3DScint(
            plotter=self.plotter, color="#00e5ff",
            opacity=0.15, show_edges=True, show_points=False
        )
        self.plotter.add_mesh(grid, style="wireframe", color="#252a38",
                              show_edges=True, opacity=0.3)
        hit = grid.threshold(0.5, scalars="atteinte")
        if hit.n_cells > 0:
            self.plotter.add_mesh(hit, scalars="Energie",
                                  cmap="plasma", opacity=0.95,
                                  show_edges=True)
        physique.rayon.plot3DRay(plotter=self.plotter)
        self.plotter.add_axes()
        self.plotter.reset_camera()


# ─────────────────────────────────────────────────────────────────────────────
#  Onglet Résultats — courbes matplotlib
# ─────────────────────────────────────────────────────────────────────────────
class ResultsTab(QWidget):
    """
    Onglet courbes — N runs stochastiques (transparents) + moyenne surbrillance.
    Courbes : Signal scintillation | Photons/voxel | Énergie cumulée | β muon
    """
    _RUN_COLORS = [
        "#00e5ff","#7c4dff","#ff9800","#00e676","#f44336",
        "#40c4ff","#e040fb","#ffeb3b","#69f0ae","#ff6d00",
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # Barre de stats
        self.stats_bar = QLabel("En attente de simulation…")
        self.stats_bar.setStyleSheet(
            f"color:{TEXT_SEC};font-size:11px;padding:4px 8px;"
            f"background:{CARD_BG};border-radius:4px;"
        )
        layout.addWidget(self.stats_bar)

        self.fig = Figure(figsize=(13, 8), dpi=100)
        self.fig.patch.set_facecolor(PANEL_BG)
        with matplotlib.rc_context(MPL_STYLE):
            self.ax_sig   = self.fig.add_subplot(2, 2, 1)
            self.ax_photo = self.fig.add_subplot(2, 2, 2)
            self.ax_E     = self.fig.add_subplot(2, 2, 3)
            self.ax_beta  = self.fig.add_subplot(2, 2, 4)
            self.fig.tight_layout(pad=3.0)

        self.canvas = FigureCanvasQTAgg(self.fig)
        layout.addWidget(self.canvas)
        self._placeholder()

    def _placeholder(self):
        for ax, title in [
            (self.ax_sig,   "Signal de scintillation"),
            (self.ax_photo, "Photons émis / voxel"),
            (self.ax_E,     "Énergie cumulée déposée"),
            (self.ax_beta,  "β du muon"),
        ]:
            ax.set_facecolor(CARD_BG)
            ax.set_title(title, color=TEXT_SEC, fontsize=10, pad=8)
            ax.text(0.5, 0.5, "En attente de simulation…",
                    ha="center", va="center", color=TEXT_SEC,
                    fontsize=10, transform=ax.transAxes)
            ax.set_xticks([]); ax.set_yticks([])
        self.canvas.draw()

    # ── Point d'entrée appelé depuis MainWindow ───────────────────────────
    def update(self, t_ref, E_ref, beta_ref, photons_ref,
               t_sig_ref, sig_ref, multi: dict):
        with matplotlib.rc_context(MPL_STYLE):
            self._draw_signal(multi)
            self._draw_photons(multi)
            self._draw_E(t_ref, E_ref)
            self._draw_beta(t_ref, beta_ref, multi)
            self.fig.tight_layout(pad=3.0)
            self.canvas.draw()

        n_runs  = multi["sig_runs"].shape[0]
        mean_ph = multi["ph_runs"].sum(axis=1).mean()
        std_ph  = multi["ph_runs"].sum(axis=1).std()
        self.stats_bar.setText(
            f"  {n_runs} tirages stochastiques  |  "
            f"Photons totaux : {mean_ph:.0f} ± {std_ph:.0f}  |  "
            f"courbe épaisse = moyenne · zone ombrée = ±1σ"
        )

    # ── Signal de scintillation ───────────────────────────────────────────
    def _draw_signal(self, multi: dict):
        ax = self.ax_sig
        ax.cla(); ax.set_facecolor(CARD_BG)

        t_ns     = multi["t_common"]       # ns, shape (n_points,)
        sig_runs = multi["sig_runs"]       # (n_runs, n_points)
        n_runs   = sig_runs.shape[0]

        for i in range(n_runs):
            c = self._RUN_COLORS[i % len(self._RUN_COLORS)]
            ax.plot(t_ns, sig_runs[i], color=c, alpha=0.22, lw=0.9)

        mean = sig_runs.mean(axis=0)
        std  = sig_runs.std(axis=0)
        ax.fill_between(t_ns, mean - std, mean + std,
                        color=ACCENT, alpha=0.18, zorder=4)
        ax.plot(t_ns, mean, color=ACCENT, lw=2.5,
                label=f"Moyenne ({n_runs} runs)", zorder=5)

        ax.set_xlabel("t (ns)"); ax.set_ylabel("Signal (J)")
        ax.set_title(f"Signal de scintillation  [{n_runs} runs]",
                     color=TEXT_PRI, fontsize=10)
        ax.legend(fontsize=9, framealpha=0.15, labelcolor=TEXT_PRI)
        ax.grid(True, alpha=0.22)

    # ── Photons émis / voxel — scatter + moyenne ──────────────────────────
    def _draw_photons(self, multi: dict):
        ax = self.ax_photo
        ax.cla(); ax.set_facecolor(CARD_BG)

        t_ns    = multi["ph_t_common"]    # ns, shape (n_vox,)
        ph_runs = multi["ph_runs"]        # (n_runs, n_vox)
        n_runs  = ph_runs.shape[0]

        # Scatter des runs individuels — montre la dispersion Poisson
        for i in range(n_runs):
            c = self._RUN_COLORS[i % len(self._RUN_COLORS)]
            ax.scatter(t_ns, ph_runs[i], color=c, alpha=0.18,
                       s=4, linewidths=0)

        # Moyenne lissée + ±1σ
        mean = ph_runs.mean(axis=0)
        std  = ph_runs.std(axis=0)

        # Fenêtre glissante pour lisser la moyenne (variance Poisson pure)
        from numpy.lib.stride_tricks import sliding_window_view
        w = max(1, len(mean) // 10)   # fenêtre = 10% des voxels
        pad = w // 2
        mean_smooth = np.convolve(mean, np.ones(w)/w, mode='same')
        std_smooth  = np.convolve(std,  np.ones(w)/w, mode='same')

        ax.fill_between(t_ns, mean_smooth - std_smooth, mean_smooth + std_smooth,
                        color=ACCENT2, alpha=0.25, zorder=4)
        ax.plot(t_ns, mean_smooth, color=ACCENT2, lw=2.0,
                label=f"Moyenne lissée ({n_runs} runs)", zorder=5)

        # Axe Y centré sur la moyenne ±3σ pour voir la vraie dispersion
        all_vals = ph_runs[ph_runs > 0]
        if len(all_vals):
            yc  = float(np.mean(all_vals))
            ys  = float(np.std(all_vals))
            ax.set_ylim(max(0, yc - 4*ys), yc + 4*ys)

        ax.set_xlabel("t (ns)"); ax.set_ylabel("N photons / voxel")
        ax.set_title(f"Photons émis / voxel  (Poisson, {n_runs} runs)",
                     color=TEXT_PRI, fontsize=10)
        ax.legend(fontsize=9, framealpha=0.15, labelcolor=TEXT_PRI)
        ax.grid(True, alpha=0.22)

    # ── Énergie cumulée (run de référence — déterministe) ─────────────────
    def _draw_E(self, t_ref, E_ref):
        ax = self.ax_E
        ax.cla(); ax.set_facecolor(CARD_BG)
        t_ns = np.array(t_ref) * 1e9
        ax.plot(t_ns, E_ref, color=WARNING, lw=2)
        ax.fill_between(t_ns, E_ref, alpha=0.12, color=WARNING)
        ax.set_xlabel("t (ns)"); ax.set_ylabel("E (J)")
        ax.set_title("Énergie cumulée déposée  [run réf.]",
                     color=TEXT_PRI, fontsize=10)
        ax.grid(True, alpha=0.22)

    # ── β du muon — tous les runs + moyenne ───────────────────────────────
    def _draw_beta(self, t_ref, beta_ref, multi: dict):
        ax = self.ax_beta
        ax.cla(); ax.set_facecolor(CARD_BG)

        runs   = multi.get("runs", [])
        n_runs = len(runs)

        if n_runs > 1:
            # Aligner tous les runs sur le même axe t (même longueur par construction)
            t_ns_ref = np.array(runs[0]["t"], dtype=float) * 1e9

            beta_matrix = []
            for r in runs:
                t_r  = np.array(r["t"],    dtype=float) * 1e9
                b_r  = np.array(r["beta"], dtype=float)
                # Interpoler sur l'axe du run 0 si longueurs diffèrent
                if len(t_r) == len(t_ns_ref):
                    beta_matrix.append(b_r)
                elif len(t_r) > 1:
                    beta_matrix.append(np.interp(t_ns_ref, t_r, b_r,
                                                  left=b_r[0], right=b_r[-1]))

            if beta_matrix:
                beta_matrix = np.array(beta_matrix)   # (n_runs, n_vox)

                # Runs individuels
                for i, row in enumerate(beta_matrix):
                    c = self._RUN_COLORS[i % len(self._RUN_COLORS)]
                    ax.plot(t_ns_ref, row, color=c, alpha=0.22, lw=0.9)

                mean = beta_matrix.mean(axis=0)
                std  = beta_matrix.std(axis=0)
                ax.fill_between(t_ns_ref, mean - std, mean + std,
                                color=SUCCESS, alpha=0.18, zorder=4)
                ax.plot(t_ns_ref, mean, color=SUCCESS, lw=2.5,
                        label=f"Moyenne ({n_runs} runs)", zorder=5)
                ax.legend(fontsize=9, framealpha=0.15, labelcolor=TEXT_PRI)
        else:
            # Fallback : un seul run
            t_ns = np.array(t_ref, dtype=float) * 1e9
            ax.plot(t_ns, beta_ref, color=SUCCESS, lw=2)

        ax.set_ylim(bottom=0)
        ax.set_xlabel("t (ns)"); ax.set_ylabel("β = v/c")
        ax.set_title(f"β du muon  [{n_runs} runs]", color=TEXT_PRI, fontsize=10)
        ax.grid(True, alpha=0.22)


# ─────────────────────────────────────────────────────────────────────────────
#  Stepping tab (résultat 3D post-simulation)
# ─────────────────────────────────────────────────────────────────────────────
class SteppingTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        hdr = QLabel("STEPPING — VOXELS TRAVERSÉS  |  couleur = énergie déposée")
        hdr.setStyleSheet(
            f"color: {ACCENT}; font-size: 11px; letter-spacing: 1px;"
            f" padding: 6px 10px; background: {CARD_BG};"
        )
        layout.addWidget(hdr)

        self.plotter = QtInteractor(self)
        self.plotter.set_background(DARK_BG)
        layout.addWidget(self.plotter.interactor)

        self._empty_msg()

    def _empty_msg(self):
        self.plotter.add_text(
            "Lancez une simulation pour visualiser le stepping",
            position="lower_edge", font_size=10, color=TEXT_SEC,
        )

    def update_with_results(self, result: dict):
        physique = result["physique"]
        grid     = result["grid"]

        self.plotter.clear()
        physique.scintillateur.plot3DScint(
            plotter=self.plotter, color="#00e5ff",
            opacity=0.12, show_edges=True, show_points=False
        )
        self.plotter.add_mesh(grid, style="wireframe",
                              color=BORDER, show_edges=True, opacity=0.25)
        hit = grid.threshold(0.5, scalars="atteinte")
        if hit.n_cells > 0:
            self.plotter.add_mesh(
                hit, scalars="Energie", cmap="inferno",
                opacity=0.98, show_edges=True,
                scalar_bar_args={"title": "E déposée (J)",
                                 "color": TEXT_PRI,
                                 "title_font_size": 10,
                                 "label_font_size": 9}
            )
        physique.rayon.plot3DRay(plotter=self.plotter)
        self.plotter.add_axes()
        self.plotter.reset_camera()


# ─────────────────────────────────────────────────────────────────────────────
#  Fenêtre principale
# ─────────────────────────────────────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Muon in Scintillator Simulator  |  T. Adouani")
        self.resize(1600, 950)
        self.setStyleSheet(GLOBAL_QSS)

        self._worker: SimWorker | None = None

        # ── Layout racine : splitter horizontal
        central = QWidget()
        self.setCentralWidget(central)
        root_h = QHBoxLayout(central)
        root_h.setContentsMargins(0, 0, 0, 0)
        root_h.setSpacing(0)

        # Colonne gauche : panneau paramètres
        self.param_panel = ParameterPanel()
        self.param_panel.paramsChanged.connect(self._on_params_changed)
        self.param_panel.runRequested.connect(self._on_run)
        root_h.addWidget(self.param_panel)

        # Séparateur vertical
        sep = QFrame(); sep.setFrameShape(QFrame.VLine)
        sep.setStyleSheet(f"background:{BORDER}; max-width:1px;")
        root_h.addWidget(sep)

        # Colonne droite : splitter vertical (preview | tabs)
        right_splitter = QSplitter(Qt.Vertical)
        right_splitter.setHandleWidth(3)

        # Preview 3D (haut)
        self.preview = Preview3D()
        right_splitter.addWidget(self.preview)

        # Onglets résultats (bas)
        self.tabs = QTabWidget()
        self.results_tab  = ResultsTab()
        self.stepping_tab = SteppingTab()
        self.tabs.addTab(self.results_tab,  "📈  Courbes")
        self.tabs.addTab(self.stepping_tab, "🟠  Stepping 3D")
        right_splitter.addWidget(self.tabs)

        right_splitter.setSizes([500, 420])
        root_h.addWidget(right_splitter, 1)

        # Overlay de chargement (par-dessus la colonne droite)
        self.overlay = LoadingOverlay(self)

        # Preview initiale
        QTimer.singleShot(200, lambda: self.preview.schedule_update(
            self.param_panel.get_params()
        ))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "overlay"):
            self.overlay.setGeometry(self.rect())

    def _on_params_changed(self, params: dict):
        self.preview.schedule_update(params)

    def _on_run(self, params: dict):
        if self._worker and self._worker.isRunning():
            return

        self.param_panel.set_running(True)
        self.overlay.show_overlay()
        self.overlay.setGeometry(self.rect())

        self._worker = SimWorker(params)
        self._worker.progress.connect(self.overlay.set_progress)
        self._worker.finished.connect(self._on_sim_done)
        self._worker.error.connect(self._on_sim_error)
        self._worker.start()

    def _on_sim_done(self, result: dict):
        self.overlay.set_progress(100)
        QTimer.singleShot(400, lambda: self._apply_results(result))

    def _apply_results(self, result: dict):
        self.overlay.hide()
        self.param_panel.set_running(False)

        self.preview.update_with_results(result)
        self.stepping_tab.update_with_results(result)
        self.results_tab.update(
            result["t"], result["E"], result["beta"],
            result["photons"], result["t_sig"], result["sig"],
            result["multi"],
        )
        self.tabs.setCurrentIndex(0)

    def _on_sim_error(self, tb: str):
        self.overlay.hide()
        self.param_panel.set_running(False)
        from PyQt5.QtWidgets import QMessageBox
        msg = QMessageBox(self)
        msg.setWindowTitle("Erreur de simulation")
        msg.setText("Une erreur s'est produite pendant la simulation.")
        msg.setDetailedText(tb)
        msg.setStyleSheet(GLOBAL_QSS)
        msg.exec_()


# ─────────────────────────────────────────────────────────────────────────────
#  Point d'entrée
# ─────────────────────────────────────────────────────────────────────────────
def launch_ui(physique=None, grid=None, t=None, E=None, beta=None,
              t_sig=None, sig=None, photons_émis=None):
    """
    Peut être appelé sans arguments (nouveau mode interactif)
    ou avec des résultats pré-calculés (rétrocompatibilité).
    """
    app = QApplication.instance() or QApplication(sys.argv)
    win = MainWindow()
    win.show()

    # Si des résultats sont déjà fournis, les afficher directement
    if physique is not None and grid is not None:
        # Construire un multi minimal (1 run) pour la rétrocompatibilité
        t_ns_   = np.array(t, dtype=float) * 1e9 if (t is not None and len(t)) else np.zeros(1)
        t_sig_  = np.asarray(t_sig, dtype=float) if (t_sig is not None and len(t_sig)) else np.zeros(1)
        sig_    = np.asarray(sig,   dtype=float) if (sig   is not None and len(sig))   else np.zeros(1)
        ph_     = np.array(photons_émis, dtype=float) if photons_émis else np.zeros(1)

        t_common = np.linspace(0.0, float(t_sig_.max()) if len(t_sig_) > 1 else 1.0, 400)
        sig_i    = np.interp(t_common, t_sig_, sig_, left=0.0, right=0.0)

        minimal_multi = dict(
            t_common    = t_common * 1e9,
            sig_runs    = sig_i[np.newaxis, :],
            ph_t_common = t_ns_[:len(ph_)],
            ph_runs     = ph_[np.newaxis, :],
            runs        = [dict(t=t, E=E, beta=beta, photons=photons_émis,
                                t_sig=t_sig, sig=sig)],
        )
        result = dict(physique=physique, grid=grid,
                      t=t, E=E, beta=beta,
                      photons=photons_émis,
                      t_sig=t_sig, sig=sig,
                      multi=minimal_multi)
        QTimer.singleShot(500, lambda: win._apply_results(result))

    app.exec_()
