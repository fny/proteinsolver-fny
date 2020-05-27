from Bio.PDB import *
import warnings
import nglview as nv
import os
import math
import glob
import re
import matplotlib.pyplot as plt
import seaborn as sns


def flatten(lists):
    return [x for lst in lists for x in lst]

def degrees(rad_angle) :
    """Converts any angle in radians to degrees.

    If the input is None, then it returns None.
    For numerical input, the output is mapped to [-180,180]
    """
    if rad_angle is None :
        return None
    angle = rad_angle * 180 / math.pi
    while angle > 180 :
        angle = angle - 360
    while angle < -180 :
        angle = angle + 360
    return angle

def memoize(function):
    memo = {}
    def wrapper(*args):
        if args in memo:
            return memo[args]
        else:
            rv = function(*args)
            memo[args] = rv
            return rv
    return wrapper

class ProteinMixin:
    @memoize
    def structure(self):
        parser = PDBParser()
        return parser.get_structure(self.pdb_id, self.file_path)
    
    def render(self):
        return nv.show_biopython(self.structure())
        
    def add_to_view(self, view):
        view.add_component(self.structure())

    def num_residues(self):
        return len(self.residues())
    
    def contacts(self):
        return list(zip(self.contact_x, self.contact_y))
    @memoize
    def header(self):
        return parse_pdb_header(self.file_path)
    
    @memoize
    def residues(self):
        return list(next(self.structure().get_chains()).get_residues())

    def angles(self):
        polypeptides = PPBuilder().build_peptides(self.structure())
        return [(degrees(i), degrees(j)) for i, j in polypeptides[0].get_phi_psi_list()]
    
    def angles_binned(self, binsize = 10):
        def convert_angle(angle, binsize):
            if angle is None:
                return None
            return angle - angle % binsize
        
        return [(convert_angle(i, binsize), convert_angle(j, binsize)) for i, j in self.angles()]
    
    def rama_plot(self, binsize=None):
        if binsize:
            angles = self.angles_binned(binsize)
        else:
            angles = self.angles()

        phi, psi = [[ i for i, j in angles ],  [ j for i, j in angles ]] 
        return sns.jointplot(phi, psi, kind='scatter', stat_func=None)
    