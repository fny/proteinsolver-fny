import sys
import os
import project_path
from multiprocessing import Pool, cpu_count
from lib.schema import Protein, ProteinSolverResult, Session
import random
import subprocess
import os
import sys
from pathlib import Path
import tqdm
import torch
import torch_geometric
import matplotlib.pyplot as plt
import pandas as pd
import heapq
from IPython.display import display, HTML, Image

# Warning: proteinsolver is very buggy!
# You may need to edit files manually if you get any errors.
import proteinsolver
# Warning: kmbio is buggy too!
# Some of the imports need to be redirected to biopython
import Bio
from kmbio import PDB
from kmtools import sci_tools, structure_tools

# Add project root to path
module_path = os.path.abspath(os.path.join(os.pardir))
if module_path not in sys.path:
    sys.path.append(module_path)

from lib import viz
from lib.schema import Protein, ProteinSolverResult, Session


def f(pdb_id):
    session = Session()
    exists = session.query(ProteinSolverResult).filter_by(pdb_id=pdb_id).scalar()
    if exists:
        print(f"DIDZ {pdb_id}")
        return
    #!/usr/bin/env python
    # coding: utf-8

    # # ProteinSolver Demo
    # 
    # Here, we load the ProteinSolver network and use it to design sequences that match the geometry of the PDZ domain.

    # In[1]:


    # In[2]:


    #
    # Globals
    #

    # PICK YOUR PROTEIN HERE!
    PDB_ID = pdb_id #os.environ.get('PDB_ID', '2HE4')

    DATA_PATH = "/home/home3/fny/cs590/data/proteinsolver"
    PDB_PATH = Bio.PDB.PDBList().retrieve_pdb_file(PDB_ID, file_format="pdb", pdir=DATA_PATH)
    STRUCTURE = PDB.Structure(PDB_ID + "_A", PDB.load(PDB_PATH)[0].extract('A'))
    MODEL_ID = "191f05de"
    MODEL_STATE = "protein_train/191f05de/e53-s1952148-d93703104.state"

    print('Protein ID:', PDB_ID)


    # The following should return True indicating GPUs are available.

    # In[3]:


    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    torch.cuda.is_available()


    # ## Load Model

    # In[4]:


    torch.cuda.empty_cache()


    # In[5]:


    get_ipython().run_line_magic('run', '../protein_train/{MODEL_ID}/model.py')


    # In[6]:


    # Model configuration
    batch_size = 1
    num_features = 20
    adj_input_size = 2
    hidden_size = 128
    frac_present = 0.5
    frac_present_valid = frac_present
    info_size= 1024
    state_file = MODEL_STATE

    net = Net(
        x_input_size=num_features + 1, adj_input_size=adj_input_size, hidden_size=hidden_size, output_size=num_features
    )
    net.load_state_dict(torch.load('../' + state_file, map_location=device))
    net.eval()
    net = net.to(device)


    # ## Data Preprocessing

    # Many proteins from the PDB did not work due to functional groups being placed at the residue locations. This portion of the script corrects for that.

    # In[7]:


    from typing import NamedTuple

    class ProteinData(NamedTuple):
        sequence: str
        row_index: torch.LongTensor
        col_index: torch.LongTensor
        distances: torch.FloatTensor

    def extract_seq_and_adj(structure, chain_id):
        domain, result_df = get_interaction_dataset_wdistances(
            StructureWrapper(structure), 0, chain_id, r_cutoff=12
        )
        domain_sequence = structure_tools.get_chain_sequence(domain)
        assert max(result_df["residue_idx_1"].values) < len(domain_sequence)
        assert max(result_df["residue_idx_2"].values) < len(domain_sequence)

        data = ProteinData(
            domain_sequence,
            result_df["residue_idx_1"].values,
            result_df["residue_idx_2"].values,
            result_df["distance"].values
        )
        return data


    def get_interaction_dataset_wdistances(
        structure, model_id, chain_id, r_cutoff=100
    ):
        chain = structure[0][chain_id]
        num_residues = len(list(chain.residues))
        dd = structure_tools.DomainDef(model_id, chain_id, 1, num_residues)
        domain = structure_tools.extract_domain(structure, [dd])
        distances_core = structure_tools.get_distances(
            domain.to_dataframe(), r_cutoff, groupby="residue"
        )
        assert (distances_core["residue_idx_1"] <= distances_core["residue_idx_2"]).all()
        return domain, distances_core


    class StructureWrapper(object):
        def __init__(self, structure):
            self.structure = structure

        def __getitem__(self, item):
             return StructureWrapper(self.structure[item])

        def __getattr__(self, name):
            if name == 'residues':
                rs = []
                for residue in STRUCTURE.residues:
                    x, _, _ = residue.id
                    if x == ' ':
                        rs.append(residue)
                return rs
            return getattr(self.structure, name)

    def preprocess(structure):
        return extract_seq_and_adj(StructureWrapper(STRUCTURE), 'A')

    STRUCTURE_SUMMARY = preprocess(STRUCTURE)


    # ## Searching for Designs

    # The model returns probabilities for every amino acid for each residue in the sequence. One method to search the space is using uniform cost search (i.e. single-source, greedy Djikstra's).
    # 
    # We start with the initial sequence and run it through the model. We then find the amino acid with the highest score for each residue, create a series of new chains with those residues updated, and place the newly created chains back in the prioirty queue which is ordered by score.

    # In[8]:


    @torch.no_grad()
    def frontier(net, x, x_score, edge_index, edge_attr, cutoff):
        index_array = torch.arange(len(x))
        mask = x == 20

        # Compute the output
        output = torch.softmax(net(x, edge_index, edge_attr), dim=1)[mask]
        # Select missing positions
        index_array = index_array[mask]

        # Find the entry with the highest probability
        max_score, max_index = output.max(dim=1)[0].max(dim=0)
        row_with_max_score = output[max_index]

        # Build nodes to search where each node updates one
        # probability from the maximum found
        nodes = []
        for i, p in enumerate(row_with_max_score):
            x_clone = x.clone()
            x_score_clone = x_score.clone()
            x_clone[index_array[max_index]] = i
            x_score_clone[index_array[max_index]] = torch.log(p)
            nodes.append((x_clone, x_score_clone))
        return nodes

    @torch.no_grad()
    def protein_search(net, x, edge_index, edge_attr, candidates, cutoff, max_iters = 1000000, verbose=False):
        x_score = torch.ones_like(x).to(torch.float) * cutoff
        heap = [(0, torch.randn(1), x, x_score)]

        iters = tqdm.tqdm(range(max_iters)) if verbose else range(max_iters)

        for i in iters:        
            p, tiebreaker, x, x_score = heapq.heappop(heap)        
            n_missing = torch.sum(x == 20)
            if verbose and i % 1000 == 0:
                print(i, p,
                    "- Heap:", len(heap),
                    f", Results:", len(candidates),
                    f", Missing: {n_missing}/{x.shape[0]}"                
                )
            if n_missing == 0:
                candidates.append((p.cpu(), x.data.cpu().numpy(), x_score.data.cpu().numpy()))
                continue
            for x, x_score in frontier(net, x, x_score, edge_index, edge_attr, cutoff):
                pre_p = -x_score.sum()
                heapq.heappush(heap, (-x_score.sum(), torch.randn(1), x, x_score))
            if len(heap) > 1_000_000:
                heap = heap[:700_000]
                heapq.heapify(heap)
        return candidates

    # Convert protein data and load it into the to GPU
    row_data = proteinsolver.datasets.protein.row_to_data(STRUCTURE_SUMMARY)
    data = proteinsolver.datasets.protein.transform_edge_attr(row_data)
    data.to(device)
    data.y = data.x

    candidates = []
    try:
        protein_search(net, torch.ones_like(data.x) * 20, data.edge_index, data.edge_attr, candidates=candidates, cutoff=np.log(0.15), verbose=False, max_iters=5000)
    except KeyboardInterrupt:
        pass


    # ## Results

    # In[9]:


    df = pd.DataFrame([
      (
          ''.join(proteinsolver.utils.AMINO_ACIDS[i] for i in candidate[1]),
          candidate[2].sum(),
          candidate[2].sum() / len(candidate[1]),
          float((candidate[1] == data.x.data.cpu().numpy()).sum().item()) / data.x.size(0)
      ) for candidate in candidates
    ], columns = ["sequence", "log_prob_sum", "log_prob_avg", "seq_identity"])


    # In[10]:


    df = df.sort_values("log_prob_avg", ascending=False).iloc[:200_000]


    # In[11]:


    df


    # In[12]:


    result = ProteinSolverResult(
        pdb_id=PDB_ID,
        n_results=df.shape[0],
        max_prob_avg=df['log_prob_avg'].max(),
        sequences=df['sequence'].values,
        log_prob_sums=df['log_prob_sum'].values,
        log_prob_avgs=df['log_prob_avg'].values,
        seq_identities=df['seq_identity'].values,
    )

    exists = session.query(ProteinSolverResult).filter_by(pdb_id=PDB_ID).scalar()
    if not exists:
        session.add(result)
        session.commit()

    print(f"DONE {pdb_id}")

session = Session()
session.query(ProteinSolverResult).all()
pdb_ids = [i[0] for i in session.query(Protein.pdb_id).all()]
n_procs = int(max(cpu_count() / 2 - 1, 1))
random.shuffle(pdb_ids)

for pdb_id in pdb_ids:
    try:
        f(pdb_id)
    except:
        print("Unexpected error:", sys.exc_info()[0])
#@with Pool(n_procs) as p:
#    print(p.mapp(f, pdb_ids))
