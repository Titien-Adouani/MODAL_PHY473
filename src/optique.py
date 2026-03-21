from __future__ import annotations
from pprint import pprint
from .geometry import Géométrie, Vecteur, Rayon, Scintillateur
from .physics import Stepping, Physique, Muon
from .ui import launch_ui
from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
import scipy.constants
from scipy.constants import physical_constants
import math
import pyvista as pv

@dataclass()
class FaceOptique() :
    R : float
    points : np.ndarray
    normale : Tuple[float, float, float]
    nature : str
@dataclass()
class Optique(Physique) :
    def __post_init__(self):
        return super().__post_init__()
    @classmethod
    def from_physique(cls, physique: Physique, **kwargs) -> "Optique":
        return cls(rayon = physique.rayon, scintillateur = physique.scintillateur, muon = physique.muon, **kwargs)

    def compute_faces(self, Rexp = 0.5, Rbord = 0.8, Rpmt = 0) :
        grid = self.scintillateur._as_hexahedron_grid()
        faces = grid.extract_surface()
        normales_pv = faces.compute_normals(cell_normals=True, point_normals=False, consistent_normals=True, auto_orient_normals=True)
        points = [faces.get_cell(i).points for i in range(faces.n_cells)]

        normales = normales_pv.cell_data["Normals"]
        """
        convention adoptée : axes +-z : surfaces exposées, axes +-y : surfaces recouvertes de métal,
        axe +x : recouverte de métal, axe -x : entrée du PMT et photocathode
        axes 
        """
        dict_surface = {
            'bord' : {'FacesOptiques' : [], 'ids' : []},
            'exposée' : {'FacesOptiques' : [], 'ids' : []},
            'pmt' : {'FacesOptiques' : [], 'ids' : []}
        }
        
        for i in range(len(normales)) :
            normale = np.array(normales[i]).tolist()
            if normale == [1.,0.,0.] or normale == [0.,-1.,0.] or normale == [0.,1.,0.]:
                surface = FaceOptique(R = Rbord, points=points[i], normale = np.array(normale), nature = 'bord')
                dict_surface['bord']['FacesOptiques'].append(surface)
                dict_surface['bord']['ids'].append(i)
            if normale == [0.,0.,1.] or normale == [0.,0.,-1.] : 
                surface = FaceOptique(R = Rexp, points = points[i], normale = np.array(normale), nature = 'exposée')
                dict_surface['exposée']['FacesOptiques'].append(surface)
                dict_surface['exposée']['ids'].append(i)
            if normale == [-1.,0.,0.] : 
                surface = FaceOptique(R = Rpmt, points=points[i], normale = np.array(normale), nature = 'pmt')
                dict_surface['pmt']['FacesOptiques'].append(surface)
                dict_surface['pmt']['ids'].append(i)
        pprint(dict_surface)
        return faces, dict_surface

Scint = Scintillateur(
        (0, 0, 0),     (0.1, 0, 0),   (0.1, 0.1, 0), (0, 0.1, 0),
        (0, 0, -0.01), (0.1, 0, -0.01), (0.1, 0.1, -0.01), (0, 0.1, -0.01),
    )
Ray   = Rayon(Vecteur(0, 0, -1, (0.05, 0.05, 0.1)))
muon  = Muon(rayon=Ray, beta=0.9)

physique = Physique(rayon=None, scintillateur=Scint, muon=muon)
optique = Optique.from_physique(physique)
optique.compute_faces()

    