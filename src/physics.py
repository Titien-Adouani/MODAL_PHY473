from __future__ import annotations
from .geometry import Géométrie, Vecteur, Rayon, Scintillateur
from .ui import launch_ui
from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
import scipy.constants
from scipy.constants import physical_constants
import math
import pyvista as pv

#region constantes de maillage
nx_ = 101
ny_ = 101
nz_ = 101
#endregion

#region constantes physiques
r"""
Dans cette region, nous définissons les constantes physiques qui seront utilisées dans la suite du projet. 
Pour la cohérence de ces constantes et le choix des valeurs numériques pour le scintillateur plastique,
voir le notebook dans modal_phy473\src
"""

pi    = math.pi
c     = scipy.constants.c
eV    = scipy.constants.electron_volt
me    = scipy.constants.electron_mass
e     = scipy.constants.elementary_charge
E0    = scipy.constants.epsilon_0
h     = scipy.constants.Planck
r_e   = physical_constants["classical electron radius"][0]
hbar  = h / (2 * pi)
z_mu  = -1          

m_mu  = 207 * me    


rho   = 1.060       
A     = 12.0        
Z     = 6           
Na    = scipy.constants.Avogadro
I     = 68.7 * eV   
Z_rho_sur_A = Z * (rho * 1e6) / A   
# endregion

#region fonctions auxilliaires
def _index_3d(i, j, k, nx, ny, nz):
    return i + (nx + 1) * (j + (ny + 1) * k)
#endregion

#region Lois de la physique
"""
Dans ce programme, nous définissons plusieurs fonctions qui coderont la physique de nos systèmes :
-gamma(beta), qui déduit le facteur gamma relativiste d'un facteur beta
-E_photon(), qui tire une énergie aléatoiremenent dans le spectre du scintillateur plastique
-T_max(beta), qui donne l'énergie maximale transmissible à un électron libre par une particule chargée
-impulsion(beta), energie_relativiste(beta) qui retournent ces quantités relativistes.
-BetheBloch(beta) qui retourne le dE/dx donné par la formule de Bethe et Bloch
-exponential_decay(t, tau) qui simule une décroissance exponentielle
"""
def gamma(beta):
    if np.any(beta >= 1):
        raise ValueError("beta doit être < 1")
    return 1.0 / np.sqrt(1.0 - beta ** 2)
def E_photon():
    """Énergie d'un photon de scintillation (tirage uniforme 350–550 nm)."""
    lam = np.random.uniform(350e-9, 550e-9)
    return h * c / lam
def T_max(beta):
    """Énergie cinétique maximale transférable à un électron (SI, J)."""
    g = gamma(beta)
    num   = 2 * me * c**2 * beta**2 * g**2
    denom = 1 + 2 * g * (me / m_mu) + (me / m_mu)**2
    return num / denom
def impulsion(beta):
    return beta * gamma(beta) * c * m_mu
def energie_relativiste(beta):
    p = impulsion(beta)
    return np.sqrt((m_mu * c**2)**2 + (p * c)**2)
def BetheBloch(beta):
    """
    dE/dx en J/m (SI, unités cohérentes).
    rho en g/cm³, A en g/mol → rho_sur_A en mol/m³.
    K = 4π Na r_e² me c²  en J·m²/mol.
    """
    g  = gamma(beta)
    K  = 4 * pi * Na * r_e**2 * me * c**2          # J·m²/mol
    arg = (2 * me * c**2 * beta**2 * T_max(beta)) / (I**2 * (1 - beta**2))
    dEdx = (K * z_mu**2 * Z_rho_sur_A / beta**2) * (0.5 * np.log(arg) - beta**2)
    return dEdx   # J/m
def exponential_decay(dt, tau=2e-9):
    return np.exp(-dt / tau)
#endregion

#region Classe Muon
"""
Classe qui définit la physique d'un Muon. 
Un muon, dans notre simulation, est la donnée d'un objet géométrique Rayon, et d'un beta relativiste.
"""

@dataclass()
class Muon:
    rayon: Rayon
    beta:  float
    m = m_mu

    def gamma(self):
        return gamma(self.beta)

    def energie(self):
        return energie_relativiste(self.beta)

    def impulsion_relativiste(self):
        return impulsion(self.beta)

    def dE(self, dx):
        """Perte d'énergie moyenne sur un pas dx (m), en J."""
        return BetheBloch(self.beta) * dx

    def maj_beta(self, dx):
        """Met à jour beta après avoir traversé dx mètres."""
        dE_val  = self.dE(dx)
        E_new   = self.energie() - dE_val
        if E_new <= self.m * c**2:
            return 0.0
        p2      = (E_new**2 - (self.m * c**2)**2) / c**2
        beta    = np.sqrt(p2 * c**2) / E_new
        self.beta = float(beta)
        return self.beta
#endregion

#region Classe Stepping
"""
La classe stepping est la plus importante de notre simulation, puisqu'elle permet de décomposer un objet 
géométrique scintillateur en un ensemble de Voxel, et fournit des méthodes pour traiter ces voxels et leurs
interactions.
Elle fournit ainsi les méthodes :
-self.subdivide_hexa(nx, ny, nz) : avec la bibliothèque pyVista, transforme un objet Scintillateur 
en un objet de type pv.UnstructuredGrid, ensemble de voxels issus de la décomposition selon les axes
x, y, et z en nx, ny, nz
-...
"""

@dataclass()
class Stepping(Géométrie):

    def subdivide_hexa(self, nx: int, ny: int, nz: int) -> pv.UnstructuredGrid:
        if nx <= 0 or ny <= 0 or nz <= 0:
            raise ValueError("nx, ny, nz doivent être > 0")

        A1 = np.asarray(self.scintillateur.A1, float)
        u  = np.asarray(self.scintillateur.A2, float) - A1
        v  = np.asarray(self.scintillateur.A4, float) - A1
        w  = np.asarray(self.scintillateur.B1, float) - A1

        pts = np.empty(((nx+1)*(ny+1)*(nz+1), 3), float)
        idx = 0
        for k in range(nz+1):
            for j in range(ny+1):
                for i in range(nx+1):
                    pts[idx] = A1 + (i/nx)*u + (j/ny)*v + (k/nz)*w
                    idx += 1

        ncells = nx * ny * nz
        cells      = np.empty(ncells * 9, dtype=np.int64)
        celltypes  = np.full(ncells, pv.CellType.HEXAHEDRON, dtype=np.uint8)

        offset = 0
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    p000 = _index_3d(i,   j,   k,   nx, ny, nz)
                    p100 = _index_3d(i+1, j,   k,   nx, ny, nz)
                    p110 = _index_3d(i+1, j+1, k,   nx, ny, nz)
                    p010 = _index_3d(i,   j+1, k,   nx, ny, nz)
                    p001 = _index_3d(i,   j,   k+1, nx, ny, nz)
                    p101 = _index_3d(i+1, j,   k+1, nx, ny, nz)
                    p111 = _index_3d(i+1, j+1, k+1, nx, ny, nz)
                    p011 = _index_3d(i,   j+1, k+1, nx, ny, nz)
                    cells[offset]   = 8
                    cells[offset+1:offset+9] = [p000,p100,p110,p010,p001,p101,p111,p011]
                    offset += 9

        grid = pv.UnstructuredGrid(cells, celltypes, pts)
        grid.cell_data["atteinte"] = np.zeros(grid.n_cells, dtype=bool)
        grid.cell_data["Energie"]  = np.zeros(grid.n_cells, dtype=float)
        return grid

    def scint_from_grid_id(self, grid: pv.UnstructuredGrid, cid):
        cell      = grid.get_cell(cid)
        pts_cell  = cell.points
        return Scintillateur(
            A1=pts_cell[0], A2=pts_cell[1], A3=pts_cell[2], A4=pts_cell[3],
            B1=pts_cell[4], B2=pts_cell[5], B3=pts_cell[6], B4=pts_cell[7],
        )

    def plot_grid(self):
        plotter = self.plot()
        grid = self.subdivide_hexa(nx=nx_, ny=ny_, nz=nz_)
        plotter.add_mesh(grid, style="wireframe", color="black",
                         show_edges=True, opacity=0.8)
        nb_cell = grid.n_cells
        for cid in range(nb_cell):
            sub = self.scint_from_grid_id(grid, cid)
            hit, _, _ = sub.intersect_points_oriented(self.rayon)
            grid.cell_data["atteinte"][cid] = hit
        if "atteinte" in grid.cell_data:
            h = grid.threshold(0.5, scalars="atteinte")
            plotter.add_mesh(h, color="orange", opacity=0.9, show_edges=True)
        return plotter

    def show_grid(self):
        self.plot_grid().show()
#endregion

#region classe Physique
@dataclass()
class Physique(Stepping):
    muon: Muon

    def __post_init__(self):
        self.rayon = self.muon.rayon

    def ids_sorted_by_time(self, grid: pv.UnstructuredGrid):
        O  = np.array(self.muon.rayon.origine(), dtype=float)
        vd = self.rayon.vecteur_directeur()
        D  = np.array([vd.vx, vd.vy, vd.vz], dtype=float)

        A1   = np.array(self.scintillateur.A1, dtype=float)
        diag = np.linalg.norm(np.array(self.scintillateur.B3, dtype=float) - A1)
        P1   = O + (diag * 100) * D

        cell_ids = grid.find_cells_along_line(O, P1)
        if len(cell_ids) == 0:
            return np.empty((0, 4), dtype=float)

        ids_atteintes = []
        for cid in cell_ids:
            sub = self.scint_from_grid_id(cid=cid, grid=grid)
            atteinte, pin, pout = sub.intersect_points_oriented(ray=self.muon.rayon)
            if not atteinte:
                continue
            pin      = np.array(pin)
            pout     = np.array(pout)
            norme_in = np.dot(pin - O, D)
            dl_3d    = np.linalg.norm(pout - pin)
            ids_atteintes.append([cid, norme_in, dl_3d])

        if not ids_atteintes:
            return np.empty((0, 3), dtype=float)

        ids_atteintes = np.array(ids_atteintes)
        return ids_atteintes[np.argsort(ids_atteintes[:, 1])]
    
    def E_reçue(self, grid: pv.UnstructuredGrid | None):
        if grid is None:
            grid = self.subdivide_hexa(nx=nx_, ny=ny_, nz=nz_)

        grid.cell_data["t_entree"] = np.full(grid.n_cells, np.inf, dtype=float)

        ids_atteinte = self.ids_sorted_by_time(grid=grid)

        t, E_reçue_list, beta_muon, photons_émis = [], [], [], []

        for data_scint in ids_atteinte:
            cid      = int(data_scint[0])
            norme_in = data_scint[1]
            dl       = data_scint[2]

            dE = self.muon.dE(dl)

            grid.cell_data["Energie"][cid] += dE
            grid.cell_data["atteinte"][cid] = True

            t_voxel = norme_in / (self.muon.beta * c)
            t.append(t_voxel)
            grid.cell_data["t_entree"][cid] = t_voxel

            E_reçue_list.append(float(np.sum(grid.cell_data["Energie"])))

            E_pho = E_photon()
            n_ph  = int(np.random.poisson(abs(dE) / E_pho))
            photons_émis.append(n_ph)
            beta_muon.append(self.muon.beta)

            if self.muon.maj_beta(dl) <= 0:
                break

        if t:
            t = np.array(t) - t[0]

        return t, E_reçue_list, beta_muon, photons_émis, grid

    def signal_curve(self, grid: pv.UnstructuredGrid, n_points: int = 500,
                     tau_rise: float = 0.9e-9, tau_dec: float = 2.1e-9):
        t_in = grid.cell_data["t_entree"]
        dE   = grid.cell_data["Energie"]
        mask = ~np.isinf(t_in)

        if not np.any(mask):
            return np.array([]), np.array([])

        t_in_v = t_in[mask]
        dE_v   = dE[mask]

        t_start = t_in_v.min()
        t_end   = t_in_v.max() + 10 * tau_dec
        t_arr   = np.linspace(t_start, t_end, n_points)

        dt     = t_arr[:, None] - t_in_v[None, :]
        causal = dt >= 0
        sig    = np.where(
            causal,
            dE_v * (np.exp(-dt / tau_dec) - np.exp(-dt / tau_rise)),
            0.0,
        )
        return t_arr - t_start, np.maximum(sig.sum(axis=1), 0.0)


    """
    Méthode run_multi() : effectue plusieurs tirs d'un muon de même énergie sur le scintillateur
    pour pouvoir récupérer des données issues des tirages stochastiques à chaque passage.
    
    Méthode codée à l'aide d'une IA générative
    """
    
    def run_multi(self, n_runs: int = 10, n_points: int = 500,
                  progress_cb=None):
        """
        Relance n_runs fois la simulation avec le même muon (beta initial fixe).
        Seuls les tirages stochastiques (Poisson photons, E_photon) varient.

        Retourne un dict avec :
          - t_common   : axe temporel commun (ns), shape (n_points,)
          - runs        : liste de dicts par run :
                            t, E, beta, photons, t_sig, sig
          - photons_runs : array (n_runs, n_voxels_max) — photons/voxel
          - sig_runs     : array (n_runs, n_points)     — signal/run
          - ph_t_common  : axe temporel commun pour photons (ns)
          - ph_runs      : array (n_runs, n_voxels_max) — photons interpolés
        """
        beta0 = self.muon.beta   

        runs        = []
        sig_list    = []     
        ph_list     = []     
        t_ph_list   = []     

        for i in range(n_runs):
            self.muon.beta = beta0

            grid_i = self.subdivide_hexa(nx=nx_, ny=ny_, nz=nz_)
            t, E, beta_run, photons, grid_i = self.E_reçue(grid=grid_i)
            t_sig, sig = self.signal_curve(grid_i, n_points=n_points)

            runs.append(dict(t=t, E=E, beta=beta_run,
                             photons=photons, t_sig=t_sig, sig=sig))
            sig_list.append((t_sig, sig))
            t_ph_list.append(np.array(t, dtype=float) * 1e9)   # en ns
            ph_list.append(np.array(photons, dtype=float))

            if progress_cb:
                progress_cb(int(20 + 70 * (i + 1) / n_runs))

        t_max_sig = max(ts.max() for ts, _ in sig_list if len(ts) > 0)
        t_common  = np.linspace(0.0, t_max_sig, n_points)

        sig_runs = np.zeros((n_runs, n_points))
        for i, (ts, ss) in enumerate(sig_list):
            if len(ts) > 1:
                sig_runs[i] = np.interp(t_common, ts, ss, left=0.0, right=0.0)

        t_ph_max    = max(tt[-1] for tt in t_ph_list if len(tt) > 0)
        t_ph_min    = min(tt[0]  for tt in t_ph_list if len(tt) > 0)
        n_interp    = max(len(p) for p in ph_list)
        ph_t_common = np.linspace(t_ph_min, t_ph_max, n_interp)  # ns

        ph_runs = np.zeros((n_runs, n_interp))
        for i, (tt, pp) in enumerate(zip(t_ph_list, ph_list)):
            if len(tt) > 1:
                ph_runs[i] = np.interp(ph_t_common, tt, pp,
                                       left=pp[0], right=pp[-1])

        self.muon.beta = beta0

        return dict(
            t_common    = t_common * 1e9,   
            runs        = runs,
            sig_runs    = sig_runs,
            ph_t_common = ph_t_common,
            ph_runs     = ph_runs,
        )
# endregion

# region point d'entrée
if __name__ == "__main__":
    Scint = Scintillateur(
        (0, 0, 0),     (0.1, 0, 0),   (0.1, 0.1, 0), (0, 0.1, 0),
        (0, 0, -0.01), (0.1, 0, -0.01), (0.1, 0.1, -0.01), (0, 0.1, -0.01),
    )
    Ray   = Rayon(Vecteur(0, 0, -1, (0.05, 0.05, 0.1)))
    muon  = Muon(rayon=Ray, beta=0.9)

    physique = Physique(rayon=None, scintillateur=Scint, muon=muon)
    t, E, beta, photons, grid = physique.E_reçue(grid=None)

    print(f"Photons émis (premier voxel) : {photons[0] if photons else 'N/A'}")
    print(f"Total photons : {sum(photons)}")

    bb_check = BetheBloch(0.9) / eV / 1e6 * 1e-2
    print(f"BetheBloch(β=0.9) = {bb_check:.3f} MeV/cm  (attendu ≈ 2 MeV/cm)")

    t_sig, sig = physique.signal_curve(grid)
    launch_ui(physique=physique, grid=grid, t=t, E=E, beta=beta,
              t_sig=t_sig, sig=sig, photons_émis=photons)
#endregion