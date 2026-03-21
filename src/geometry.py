from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
import pyvista as pv


@dataclass(frozen=True)
class Vecteur:
    vx: float
    vy: float
    vz: float
    origin: Tuple[float, float, float]

    def produit_scalaire(self, a: "Vecteur") -> float:
        return self.vx * a.vx + self.vy * a.vy + self.vz * a.vz

    def norme(self) -> float:
        return float(np.sqrt(self.vx * self.vx + self.vy * self.vy + self.vz * self.vz))

    def normalize(self) -> "Vecteur":
        n = self.norme()
        if n == 0:
            raise ValueError("Impossible de normaliser un vecteur nul.")
        return Vecteur(self.vx / n, self.vy / n, self.vz / n, self.origin)

    def asTuple(self) -> Tuple[float, float, float]:
        return (self.vx, self.vy, self.vz)

    def plot3DVect(self, plotter: Optional[pv.Plotter] = None, *, scale: Optional[float] = None) -> pv.Plotter:
        """
        Affiche un vecteur sous forme de flèche dans PyVista.
        - plotter=None -> crée un Plotter
        - scale=None -> longueur automatique basée sur la norme
        """
        if plotter is None:
            plotter = pv.Plotter()

        o = np.array(self.origin, dtype=float)
        v = np.array([self.vx, self.vy, self.vz], dtype=float)
        n = np.linalg.norm(v)
        if n == 0:
            raise ValueError("Vecteur nul : rien à tracer.")

        # longueur de la flèche
        length = float(scale) if scale is not None else float(n)

        # PyVista : Arrow(start, direction, scale)
        # direction : vecteur directeur (pas forcément unitaire), scale ~ longueur
        direction = v / n  # unitaire
        arrow = pv.Arrow(start=o, direction=direction, scale=length)

        plotter.add_mesh(arrow, color="black")
        return plotter

    def showPlot3DVect(self, plotter: Optional[pv.Plotter] = None, *, scale: Optional[float] = None) -> None:
        plotter = self.plot3DVect(plotter=plotter, scale=scale)
        plotter.show()


@dataclass(frozen=True)
class Rayon:
    v: Vecteur

    def vecteur_directeur(self) -> Vecteur:
        return self.v.normalize()

    def origine(self) -> Tuple[float, float, float]:
        return self.v.origin

    def plot3DRay(self, plotter: Optional[pv.Plotter]) -> pv.Plotter:
        """
        Trace une demi-droite depuis l'origine, dans la direction du vecteur directeur,
        avec une longueur basée sur les bounds actuelles de la scène.
        """
        if plotter is None:
            raise ValueError("plotter is None : tracez le scintillateur (ou la scène) avant le rayon.")

        vdir = self.vecteur_directeur()
        P0 = np.array(self.origine(), dtype=float)
        d = np.array([vdir.vx, vdir.vy, vdir.vz], dtype=float)

        # Longueur automatique : 2x la diagonale des bounds si disponibles, sinon fallback
        b = plotter.bounds  # (xmin, xmax, ymin, ymax, zmin, zmax)
        if any(np.isinf(b)) or any(np.isnan(b)):
            L = 10.0
        else:
            dx = b[1] - b[0]
            dy = b[3] - b[2]
            dz = b[5] - b[4]
            diag = float(np.sqrt(dx * dx + dy * dy + dz * dz))
            L = 1 * diag if diag > 0 else 5

        P1 = P0 + L * d

        # PyVista line
        line = pv.Line(P0, P1)

        plotter.add_mesh(line, color="red", line_width=3)
        # Optionnel : petite flèche au départ pour montrer le sens
        arrow = pv.Arrow(start=P0, direction=d, scale=min(L * 0.15, 5.0))
        plotter.add_mesh(arrow, color="red")
        return plotter

    def showPlot3DRay(self, plotter: Optional[pv.Plotter]) -> None:
        plotter = self.plot3DRay(plotter)
        plotter.show()


@dataclass(frozen=True)
class Scintillateur:
    A1: Tuple[float, float, float]
    A2: Tuple[float, float, float]
    A3: Tuple[float, float, float]
    A4: Tuple[float, float, float]
    B1: Tuple[float, float, float]
    B2: Tuple[float, float, float]
    B3: Tuple[float, float, float]
    B4: Tuple[float, float, float]

    def points(self):
        return [self.A1, self.A2, self.A3, self.A4,
                self.B1, self.B2, self.B3, self.B4]

    def intersect_points_oriented(self, ray: "Rayon", eps: float = 1e-12):
    # --- Base du parallélépipède (monde -> local)
        A1 = np.asarray(self.A1, dtype=float)
        u  = np.asarray(self.A2, dtype=float) - A1
        v  = np.asarray(self.A4, dtype=float) - A1
        w  = np.asarray(self.B1, dtype=float) - A1

        M = np.column_stack([u, v, w])  # 3x3

        det = float(np.linalg.det(M))
        if abs(det) < eps:
            raise ValueError("Parallélépipède dégénéré: base non inversible (u,v,w coplanaires ou nuls).")

        Minv = np.linalg.inv(M)

        # --- Rayon (monde)
        O = np.asarray(ray.origine(), dtype=float)
        vd = ray.vecteur_directeur()
        D = np.asarray([vd.vx, vd.vy, vd.vz], dtype=float)

        # --- Rayon en coordonnées locales: p(t) = p0 + t * d
        p0 = Minv @ (O - A1)
        d  = Minv @ D

        # --- Intersection rayon avec AABB local [0,1]^3 (slabs)
        tmin = -np.inf
        tmax =  np.inf

        for i in range(3):
            if abs(d[i]) < eps:
                # Rayon parallèle aux plans de cette dimension
                if p0[i] < 0.0 or p0[i] > 1.0:
                    return False, None, None
            else:
                t1 = (0.0 - p0[i]) / d[i]
                t2 = (1.0 - p0[i]) / d[i]
                t_enter = min(t1, t2)
                t_exit  = max(t1, t2)

                tmin = max(tmin, t_enter)
                tmax = min(tmax, t_exit)

                if tmin > tmax:
                    return False, None, None

        # --- Demi-droite: on impose t >= 0
        if tmax < 0:
            return False, None, None

        t_enter_world = max(tmin, 0.0)
        t_exit_world  = tmax

        P_enter = O + t_enter_world * D
        P_exit  = O + t_exit_world  * D

        return True, tuple(P_enter), tuple(P_exit)

    def _as_hexahedron_grid(self) -> pv.UnstructuredGrid:
        """
        Construit un hexaèdre VTK à partir des 8 points dans l'ordre:
        [A1,A2,A3,A4,B1,B2,B3,B4]
        """
        pts = np.asarray(self.points(), dtype=float)

        # Cell connectivity for a single HEXAHEDRON:
        # [8, 0,1,2,3,4,5,6,7]
        cells = np.array([8, 0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int64)
        celltypes = np.array([pv.CellType.HEXAHEDRON], dtype=np.uint8)

        grid = pv.UnstructuredGrid(cells, celltypes, pts)
        return grid

    def plot_intersections_red(self, ray: "Rayon", plotter: Optional[pv.Plotter], *,
                           point_size: float = 4.0,
                           show_segment_inside: bool = True,
                           eps: float = 1e-12) -> pv.Plotter:

        if plotter is None:
            raise ValueError("plotter is None : créez d'abord la scène (ex: via plot3DScint).")

        hit, Pin, Pout = self.intersect_points_oriented(ray, eps=eps)
        if not hit:
            return plotter  # rien à afficher

        pts = np.array([Pin, Pout], dtype=float)

        # Points verts (sphères)
        pts_poly = pv.PolyData(pts)
        plotter.add_mesh(
            pts_poly,
            color="green",
            point_size=point_size,
            render_points_as_spheres=True
        )

        # (option) Segment intérieur
        if show_segment_inside:
            seg = pv.Line(pts[0], pts[1])
            plotter.add_mesh(seg, color="green", line_width=4)

        return plotter

    def plot3DScint(self, plotter: Optional[pv.Plotter], *,
                    show_points: bool = True,
                    show_edges: bool = True,
                    color: str = "lightblue",
                    opacity: float = 0.35) -> pv.Plotter:
        if plotter is None:
            plotter = pv.Plotter()

        grid = self._as_hexahedron_grid()

        # Surface (faces)
        surf = grid.extract_surface()

        plotter.add_mesh(
            surf,
            color=color,
            opacity=opacity,
            show_edges=show_edges,
            edge_color="black"
        )

        # Points (sommets)
        if show_points:
            pts_poly = pv.PolyData(np.asarray(self.points(), dtype=float))
            plotter.add_mesh(pts_poly, color="black", point_size=10, render_points_as_spheres=True)

        return plotter

    def showPlot3DScint(self, plotter: Optional[pv.Plotter] = None, **kwargs) -> None:
        plotter = self.plot3DScint(plotter=plotter, **kwargs)
        plotter.show()


@dataclass()
class Géométrie :
    scintillateur : Scintillateur
    rayon : Rayon
    def données_origine_rayon(self) :
        return self.rayon.origine()

    def donnée_rayon_vecteur_directeur(self) :
        return self.rayon.vecteur_directeur()

    def données_scint(self) :
        return self.scintillateur.points()

    def intersection_bool(self) :
        bool, point_entrée, point_sortie = self.scintillateur.intersect_points_oriented(ray=self.rayon)
        return bool

    def intersection_points(self) :
        bool, point_entrée, point_sortie = self.scintillateur.intersect_points_oriented(ray=self.rayon)
        return point_entrée, point_sortie

    def intersection_longueur(self) :
        if self.intersection_bool() :
            point_entrée, point_sortie = self.intersection_points()
            return Vecteur(point_sortie[0]-point_entrée[0],
                        point_sortie[1]-point_entrée[1],
                        point_sortie[2]-point_entrée[2],
                        origin=(0,0,0)).norme()
        else :
            return None

    def plot(self) :
        plotter = self.scintillateur.plot3DScint(plotter=None, color="lightblue", opacity=0.35, show_edges=True, show_points=True)
        plotter = self.scintillateur.plot_intersections_red(plotter=plotter, ray=self.rayon)
        plotter.add_axes()
        
        point_entrée, point_sortie = self.intersection_points()
        plotter.add_text(
        "Titien Adouani\n"
        "Muon in scintillator simulator\n"
        f"Intersection : {self.intersection_bool()}\n"
        f"Longueur de parcours : {self.intersection_longueur()}\n"
        f"Point d'entrée : {point_entrée}\n"
        f"Point de sortie : {point_sortie}",
        position="upper_right",
        font_size=8,
        color="black",
        shadow=True,
        viewport=True
        )
        return plotter

    def showplot(self) :
        plotter = self.plot()
        self.rayon.showPlot3DRay(plotter=plotter)
    
# =========================
# EXEMPLE 
# ATTENTION : dans la version actuelle, 
# fonctionne uniquement avec des scintillateurs
# parallépipédiques orientés dans le sens de la longueur 
# et de la largeur selon les axes ex et ey. 
# =========================

# Scint = Scintillateur(
#     (0,0,0),  # A1
#     (10,0,0),  # A2
#     (10,10,0),  # A3
#     (0,10,0),  # A4
#     (0,0,1),  # B1
#     (10,0,1),  # B2
#     (10,10,1),  # B3
#     (0,10,1)   # B4
# )

# Ray = Rayon(Vecteur(0, 0, -1, (5, 5, 10)))

# géométrie = Géométrie(scintillateur=Scint, rayon=Ray)
# géométrie.showplot()

        
    
    

        