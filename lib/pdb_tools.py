from tqdm import tqdm

from Bio.PDB import *
import warnings
import nglview as nv
import os
import math
import glob
import re
import matplotlib.pyplot as plt
import seaborn as sns
from .schema import Session, engine, Protein, Base

# AMINO_ACIDS = set('ALA|ASX|CYS|ASP|GLU|PHE|GLY|HIS|ILE|LYS|LEU|MET|ASN|PRO|GLN|ARG|SER|THR|SEC|VAL|TRP|XAA|TYR|GLX'.split('|'))
AMINO_ACIDS = {
    'ALA': 'A',
    'ARG': 'R',
    'ASN': 'N',
    'ASP': 'D',
    'CYS': 'C',
    'GLU': 'E',
    'GLN': 'Q',
    'GLY': 'G',
    'HIS': 'H',
    'ILE': 'I',
    'LEU': 'L',
    'LYS': 'K',
    'MET': 'M',
    'PHE': 'F',
    'PRO': 'P',
    'SER': 'S',
    'THR': 'T',
    'TRP': 'W',
    'TYR': 'Y',
    'VAL': 'V'
}
AMINO_ACIDS2 = {
    'A': 'ALA',
    'R': 'ARG',
    'N': 'ASN',
    'D': 'ASP',
    'C': 'CYS',
    'E': 'GLU',
    'Q': 'GLN',
    'G': 'GLY',
    'H': 'HIS',
    'I': 'ILE',
    'L': 'LEU',
    'K': 'LYS',
    'M': 'MET',
    'F': 'PHE',
    'P': 'PRO',
    'S': 'SER',
    'T': 'THR',
    'W': 'TRP',
    'Y': 'TYR',
    'V': 'VAL'
}



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


def pdb_view(file):
    parser = PDBParser()
    structure = parser.get_structure('', file)
    view = nv.show_biopython(structure)
    return view

# class Protein:
#     def __init__()


PDB_PATH = '/usr/project/xtmp/fny/cs590/pdb/'

import sys
from io import StringIO 

class NullIO(StringIO):
    def write(self, txt):
        pass


def silent(fn):
    """Decorator to silence functions."""
    def silent_fn(*args, **kwargs):
        saved_stdout = sys.stdout
        sys.stdout = NullIO()
        result = fn(*args, **kwargs)
        sys.stdout = saved_stdout
        return result
    return silent_fn


    
def download_family(family):
    session = Session()

    with open(f'../data/protein-families/{family}.txt') as f:
        ids = f.read().split(',')
        ids = [re.sub('_\d|', '', id_) for id_ in ids]

    for id_ in tqdm(ids):
        exists = session.query(Protein).filter_by(pdb_id=id_).scalar() is not None
        if exists:
            continue
        prot = ProteinMaker(id_, family).make()
        if prot == None or prot.length > 130:
            continue
        
        session.add(prot)
        session.commit()


def are_clean_residues(resnames):
    for resname in resnames:
        if resname in AMINO_ACIDS.keys():
            return False
    return True
    
class ProteinMaker:
    def __init__(self, id_, family=None):
        self.is_made = False
        self.protein = Protein(pdb_id = id_.upper(), family=family)
        self.id_ = id_
        self.bio_struct = None
        self.pdb_header = None
        
    def make(self):
        pdbl = PDBList()
        path = silent(pdbl.retrieve_pdb_file)(self.id_, file_format="pdb", pdir=PDB_PATH)
        self.pdb_header = parse_pdb_header(path)
        self.protein.has_missing_residues = self.pdb_header['has_missing_residues']
        self.protein.pdb_name = self.pdb_header['name']
        parser = PDBParser()
        with warnings.catch_warnings(record=True) as w:
            self.bio_struct = silent(parser.get_structure)(self.id_, path)
            if len(w) > 0:
                if 'Chain A is discontinuous' in str(w[0].message):
                    self.protein.has_discont = True
        
        self.protein.sequence = ''.join(self.resnames())
        self.protein.length = len(self.protein.sequence)
        self.protein.is_clean = are_clean_residues(self.resnames())
        
        polypeptides = PPBuilder().build_peptides(self.bio_struct)
        if len(polypeptides) == 0:
            return None
        self.protein.phi_angles, self.protein.psi_angles = self.angles() 
        self.protein.pdb_data = self.pdb_data(path)
        self.protein.file_path = path
        
        self.is_made = True
        return self.protein
    
    def length(self):
        return len(self.protein.sequence)
    
    def pdb_data(self, path):
        with open(path, 'r') as f:
            content = f.read()
        return content
        
    def residues(self):
        residues = []
        for chain in self.bio_struct.get_chains():
            if chain.id == 'A':
                for r in chain.get_residues():
                    het, _, _, = r.get_id()
                    if het == ' ' and r.get_resname() in AMINO_ACIDS.keys():
                        residues.append(r)
                break
        return residues

    def resnames(self):
        return [AMINO_ACIDS[res.get_resname()] for res in self.residues()]

    def angles(self):
        polypeptides = PPBuilder().build_peptides(self.bio_struct)
        phi = [degrees(i) for i, j in polypeptides[0].get_phi_psi_list()]
        psi = [degrees(j) for i, j in polypeptides[0].get_phi_psi_list()]
        return phi, psi

        