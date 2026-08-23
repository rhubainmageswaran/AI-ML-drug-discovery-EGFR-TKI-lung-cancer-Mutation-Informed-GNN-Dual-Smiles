#!/usr/bin/env python3
"""
Prediction script for the simplified base1d model.

The model keeps the ligand GNN, site GNN and site physicochemical branches
independent until one explicit pre-attention fusion:
    site GNN (256) + ligand (256) + physchem (64) -> fused site token (256)
Then one site self-attention block models relationships among the 8 fused sites,
followed by learned attention pooling and the two multitask prediction heads.

This script mirrors adv_physchem_gnn_base1d_simple.py exactly for the GNN,
physchem feature generation, scaling, checkpoint format, and downstream head.
It must load the checkpoint produced by that training script.
"""

import os
import sys
import pickle
import argparse
from functools import lru_cache
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from loguru import logger
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, spearmanr
from numpy.linalg import norm

# RDKit imports
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, MolSurf, GraphDescriptors, Fragments
from rdkit.Chem import AllChem, rdMolDescriptors
from rdkit import DataStructs
from rdkit import RDLogger

# ---- PyTorch / torch_geometric imports ----
# NOTE: no TensorFlow/Keras import at all -- the downstream head is a plain PyTorch
# nn.Module now (see GNNCrossAttentionSetTransformerHead below), loaded from a single
# torch.save() checkpoint, not a Keras .h5 file.
import torch
import torch.nn as nn
import torch.nn.functional as F

TORCH_GEOMETRIC_AVAILABLE = False
try:
    from torch_geometric.nn import MessagePassing, global_mean_pool, global_add_pool, global_max_pool
    from torch_geometric.utils import add_self_loops
    from torch_geometric.data import Data, Batch
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    print("\n" + "!"*80)
    print("WARNING: torch_geometric not installed!")
    print("To install: pip install torch-geometric torch-scatter torch-sparse")
    print("!"*80 + "\n")
    MessagePassing = nn.Module
    global_mean_pool = None
    global_add_pool = None
    global_max_pool = None
    add_self_loops = None
    Data = None
    Batch = None

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')
os.environ['CUDA_VISIBLE_DEVICES'] = ''
np.random.seed(42)
torch.manual_seed(42)

# Logger configuration
logger.remove()
logger.add(sys.stderr, level="INFO")

print("="*80)
print("GNN CROSS-ATTENTION + JOINT-1024D SET-TRANSFORMER MODEL -- PREDICTION (base1d_simple)")
print("Loads the fine-tuned GNN backbone + downstream head (SAB+PMA @512d + parallel")
print("1024-d joint ligand+site cross-attention branch) from a single checkpoint")
print("8-site mechanistic mutation order: FULL -> ATP_POCKET -> P_LOOP -> C_HELIX ->")
print("  DEL19 -> HINGE_LOOP(T790M/C797S) -> DFG_A_LOOP -> HRD_CAT")
print("="*80)

# Mutation sites used throughout, IN MECHANISTIC ORDER (must match the training
# script's df_train column names exactly -- see "Load Data" there).
MUTATION_SITES = [
    ('FULL_SMILES', 'smiles_full_sequence_egfr_manual'),
    ('ATP_POCKET', 'smiles_sequence_atp_ pocket'),
    ('P_LOOP', 'smiles_sequence_p_loop_constant'),
    ('C_HELIX', 'smiles_sequence_c_helix_constant'),
    ('DEL19', 'smiles_sequence_19_deletions'),
    ('HINGE_LOOP_T790M_C797S', 'smiles_sequence_hinge_loop_t790m_c797s'),
    ('DFG_A_LOOP', 'smiles_sequence_a_loop_dfg'),
    ('HRD_CAT', 'smiles_sequence_hrd_constant'),
]

# ---- OGB-style feature vocabularies (safe_index maps unseen values to a
#      trailing "misc" bucket instead of crashing, matching ogb.utils.features) ----

def safe_index(vocab_list, value):
    """Return index of value in vocab_list, or the last ('misc') index if absent."""
    try:
        return vocab_list.index(value)
    except ValueError:
        return len(vocab_list) - 1

ATOM_VOCAB = {
    'atomic_num': list(range(1, 119)) + ['misc'],                      # 118 + misc = 119
    'chirality': ['CHI_UNSPECIFIED', 'CHI_TETRAHEDRAL_CW',
                  'CHI_TETRAHEDRAL_CCW', 'CHI_OTHER'],                  # 4
    'degree': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],               # 12
    'formal_charge': [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 'misc'],    # 12
    'num_h': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],                       # 10
    'num_radical_e': [0, 1, 2, 3, 4, 'misc'],                           # 6
    'hybridization': ['SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'misc'],     # 6
    'is_aromatic': [False, True],                                      # 2
    'is_in_ring': [False, True],                                       # 2
}

BOND_VOCAB = {
    'bond_type': ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC', 'misc'],    # 5
    'stereo': ['STEREONONE', 'STEREOZ', 'STEREOE', 'STEREOCIS',
               'STEREOTRANS', 'STEREOANY', 'misc'],                    # 7
    'is_conjugated': [False, True],                                    # 2
    'is_in_ring': [False, True],                                       # 2
}

ATOM_FEATURE_DIMS = [len(ATOM_VOCAB[k]) for k in
                      ['atomic_num', 'chirality', 'degree', 'formal_charge',
                       'num_h', 'num_radical_e', 'hybridization',
                       'is_aromatic', 'is_in_ring']]
BOND_FEATURE_DIMS = [len(BOND_VOCAB[k]) for k in
                      ['bond_type', 'stereo', 'is_conjugated', 'is_in_ring']]
NUM_ATOM_PHYSCHEM = 4   # Gasteiger charge, Crippen logP contrib, Crippen MR contrib, TPSA contrib


if TORCH_GEOMETRIC_AVAILABLE:

    class AtomEncoderRich(nn.Module):
        """
        Encodes the 9 OGB-style categorical atom features (each via its own
        nn.Embedding, summed -- the standard AtomEncoder pattern from
        Hu et al. 2020) plus a continuous physicochemical context vector
        (Gasteiger partial charge, Crippen logP/MR atomic contributions,
        TPSA atomic contribution). The categorical sum and the projected
        continuous vector are added together, so every node embedding is
        anchored in the same physicochemical axes used by the hierarchical
        branch (electrostatics, hydrophobicity, polar surface area).
        """
        def __init__(self, emb_dim, num_physchem=NUM_ATOM_PHYSCHEM):
            super(AtomEncoderRich, self).__init__()
            self.atom_embeddings = nn.ModuleList()
            for dim in ATOM_FEATURE_DIMS:
                emb = nn.Embedding(dim, emb_dim)
                nn.init.xavier_uniform_(emb.weight.data)
                self.atom_embeddings.append(emb)
            # Continuous physchem context -> projected into the same space
            self.physchem_proj = nn.Sequential(
                nn.Linear(num_physchem, emb_dim),
                nn.ReLU(),
                nn.Linear(emb_dim, emb_dim)
            )

        def forward(self, x_cat, x_physchem):
            h = 0
            for i, emb in enumerate(self.atom_embeddings):
                h = h + emb(x_cat[:, i])
            h = h + self.physchem_proj(x_physchem)
            return h

    class BondEncoderRich(nn.Module):
        """Encodes the 4 OGB-style categorical bond features (type, stereo,
        conjugation, ring membership), each via its own nn.Embedding, summed."""
        def __init__(self, emb_dim):
            super(BondEncoderRich, self).__init__()
            self.bond_embeddings = nn.ModuleList()
            for dim in BOND_FEATURE_DIMS:
                emb = nn.Embedding(dim, emb_dim)
                nn.init.xavier_uniform_(emb.weight.data)
                self.bond_embeddings.append(emb)

        def forward(self, edge_attr_cat):
            h = 0
            for i, emb in enumerate(self.bond_embeddings):
                h = h + emb(edge_attr_cat[:, i])
            return h

    class GINEConvV2(MessagePassing):
        """
        Upgraded GINEConv: trainable epsilon (Xu et al. 2019 generalized GIN
        update: h_v' = MLP((1+eps)*h_v + sum_u (h_u + e_uv))) with a BondEncoderRich
        producing the edge embeddings, and a 2-layer MLP update with BatchNorm
        (deeper/better-normalized update function than the base1b MLP).
        """
        def __init__(self, emb_dim, train_eps=True):
            super(GINEConvV2, self).__init__(aggr='add')
            self.mlp = nn.Sequential(
                nn.Linear(emb_dim, 2 * emb_dim),
                nn.LayerNorm(2 * emb_dim),
                nn.ReLU(),
                nn.Linear(2 * emb_dim, emb_dim)
            )
            self.bond_encoder = BondEncoderRich(emb_dim)
            self.eps = nn.Parameter(torch.zeros(1)) if train_eps else torch.zeros(1)
            self.train_eps = train_eps

        def forward(self, x, edge_index, edge_attr_cat):
            # Self-loop bond feature: bond_type uses the dedicated "misc" bucket
            # (mirrors base1b's convention of a distinct self-loop bond-type
            # code) while stereo/conjugation/ring-membership use their neutral
            # "none/false" index (0) since a self-loop has no real stereochemistry.
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
            n = x.size(0)
            self_loop_attr = torch.stack([
                torch.full((n,), len(BOND_VOCAB['bond_type']) - 1, dtype=torch.long, device=x.device),
                torch.zeros(n, dtype=torch.long, device=x.device),
                torch.zeros(n, dtype=torch.long, device=x.device),
                torch.zeros(n, dtype=torch.long, device=x.device),
            ], dim=1)
            edge_attr_cat = torch.cat([edge_attr_cat, self_loop_attr], dim=0)
            edge_embeddings = self.bond_encoder(edge_attr_cat)
            out = self.propagate(edge_index, x=x, edge_attr=edge_embeddings)
            return self.mlp((1 + self.eps.to(x.device)) * x + out)

        def message(self, x_j, edge_attr):
            return F.relu(x_j + edge_attr)

        def update(self, aggr_out):
            return aggr_out

    class GINVirtualNet(nn.Module):
        """
        GIN + Virtual Node + Jumping-Knowledge molecular graph encoder.

        This replaces the base1b MolCLR GINet with the established
        higher-capacity GIN configuration ("GIN-VN") used as the reference
        GIN baseline on the OGB ogbg-mol* leaderboards:
          - richer 9-feature atom / 4-feature bond encoding (AtomEncoderRich /
            BondEncoderRich), with atomic physchem context on every node
          - trainable-epsilon GINEConv message passing
          - a per-graph virtual node that is broadcast to all atoms and
            refreshed by sum-pooling atom states after every layer, giving
            the network cheap access to whole-molecule context
          - Jumping-Knowledge concatenation over all layer depths before
            pooling, so both local and global substructure signal reaches
            the final embedding
          - mean+max graph pooling concatenation for a more discriminative
            graph-level readout than mean pooling alone
        """
        def __init__(self, num_layer=5, emb_dim=300, feat_dim=512,
                     drop_ratio=0.1, jk_mode='concat'):
            super(GINVirtualNet, self).__init__()
            assert num_layer >= 2, "Need >=2 GIN layers for virtual-node updates to be meaningful"
            self.num_layer = num_layer
            self.emb_dim = emb_dim
            self.feat_dim = feat_dim
            self.drop_ratio = drop_ratio
            self.jk_mode = jk_mode

            self.atom_encoder = AtomEncoderRich(emb_dim)
            self.gnns = nn.ModuleList([GINEConvV2(emb_dim, train_eps=True) for _ in range(num_layer)])
            self.batch_norms = nn.ModuleList([nn.LayerNorm(emb_dim) for _ in range(num_layer)])

            # Virtual node: shared learnable init + per-layer refresh MLP
            self.virtualnode_embedding = nn.Embedding(1, emb_dim)
            nn.init.constant_(self.virtualnode_embedding.weight.data, 0)
            self.mlp_virtualnode_list = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(emb_dim, 2 * emb_dim), nn.LayerNorm(2 * emb_dim), nn.ReLU(),
                    nn.Linear(2 * emb_dim, emb_dim), nn.LayerNorm(emb_dim), nn.ReLU()
                ) for _ in range(num_layer - 1)
            ])

            jk_dim = emb_dim * num_layer if jk_mode == 'concat' else emb_dim
            self.feat_lin = nn.Linear(jk_dim * 2, feat_dim)  # *2 for mean+max pooling concat

        def forward(self, data):
            x_cat, x_physchem = data.x, data.physchem
            edge_index, edge_attr_cat, batch = data.edge_index, data.edge_attr, data.batch
            num_graphs = int(batch[-1].item()) + 1 if batch.numel() > 0 else 1

            virtualnode_embedding = self.virtualnode_embedding(
                torch.zeros(num_graphs, dtype=torch.long, device=x_cat.device)
            )

            h = self.atom_encoder(x_cat, x_physchem)
            h_list = [h]
            for layer in range(self.num_layer):
                # Broadcast virtual node context onto every atom in its graph
                h_list[layer] = h_list[layer] + virtualnode_embedding[batch]

                h = self.gnns[layer](h_list[layer], edge_index, edge_attr_cat)
                h = self.batch_norms[layer](h)
                if layer == self.num_layer - 1:
                    h = F.dropout(h, self.drop_ratio, training=self.training)
                else:
                    h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
                h_list.append(h)

                # Refresh virtual node from this layer's atom states
                if layer < self.num_layer - 1:
                    vn_temp = global_add_pool(h_list[layer], batch, size=num_graphs) + virtualnode_embedding
                    virtualnode_embedding = F.dropout(
                        self.mlp_virtualnode_list[layer](vn_temp), self.drop_ratio, training=self.training
                    )

            # Jumping-Knowledge: use every layer's atom representations (skip
            # the raw input embedding at index 0), or just the last layer.
            if self.jk_mode == 'concat':
                node_repr = torch.cat(h_list[1:], dim=-1)
            else:
                node_repr = h_list[-1]

            mean_pooled = global_mean_pool(node_repr, batch, size=num_graphs)
            max_pooled = global_max_pool(node_repr, batch, size=num_graphs)
            graph_repr = torch.cat([mean_pooled, max_pooled], dim=-1)
            return self.feat_lin(graph_repr)

        def load_pretrained(self, state_dict):
            """Shape-checked partial load: copies only params whose name AND
            shape match (the richer encoder means most MolCLR-checkpoint
            tensors won't match shape -- those are skipped with a warning
            instead of crashing, and the rest of the network stays randomly
            initialized for training from scratch)."""
            own_state = self.state_dict()
            loaded, skipped = 0, 0
            for name, param in state_dict.items():
                if name not in own_state:
                    skipped += 1
                    continue
                if isinstance(param, nn.parameter.Parameter):
                    param = param.data
                if own_state[name].shape != param.shape:
                    skipped += 1
                    continue
                own_state[name].copy_(param)
                loaded += 1
            logger.info(f"GINVirtualNet.load_pretrained: {loaded} tensors loaded, {skipped} skipped (name/shape mismatch)")

    # Backwards-compatible alias so downstream code can keep calling `GINet(...)`
    GINet = GINVirtualNet

else:
    # Fallback: Simple MLP-based molecular embedding (no graph structure)
    class GINet(nn.Module):
        """Fallback MLP embedder when torch_geometric not available."""
        def __init__(self, num_layer=5, emb_dim=300, feat_dim=512, drop_ratio=0.0, pool='mean'):
            super(GINet, self).__init__()
            self.feat_dim = feat_dim
            # Simple MLP that produces fixed-size embeddings from molecular fingerprints
            self.mlp = nn.Sequential(
                nn.Linear(2048, 512),  # Morgan fingerprint size
                nn.ReLU(),
                nn.Dropout(drop_ratio),
                nn.Linear(512, feat_dim),
                nn.ReLU()
            )
            
        def forward(self, fingerprints):
            """Accept fingerprint tensor directly instead of graph data."""
            return self.mlp(fingerprints)
        
        def load_pretrained(self, state_dict):
            pass  # No pretrained weights for fallback


def atom_to_feature_vector(atom):
    """9 OGB-style categorical atom features -- see ATOM_VOCAB header comment
    for the physicochemical meaning of each dimension."""
    return [
        safe_index(ATOM_VOCAB['atomic_num'], atom.GetAtomicNum()),
        safe_index(ATOM_VOCAB['chirality'], str(atom.GetChiralTag())),
        safe_index(ATOM_VOCAB['degree'], atom.GetTotalDegree()),
        safe_index(ATOM_VOCAB['formal_charge'], atom.GetFormalCharge()),
        safe_index(ATOM_VOCAB['num_h'], atom.GetTotalNumHs()),
        safe_index(ATOM_VOCAB['num_radical_e'], atom.GetNumRadicalElectrons()),
        safe_index(ATOM_VOCAB['hybridization'], str(atom.GetHybridization())),
        safe_index(ATOM_VOCAB['is_aromatic'], atom.GetIsAromatic()),
        safe_index(ATOM_VOCAB['is_in_ring'], atom.IsInRing()),
    ]


def bond_to_feature_vector(bond):
    """4 OGB-style categorical bond features -- bond type, stereo,
    conjugation and ring membership map onto the same rigidity / pi-stacking
    axes used by generate_lig_intra_features / generate_mut_intra_features."""
    return [
        safe_index(BOND_VOCAB['bond_type'], str(bond.GetBondType())),
        safe_index(BOND_VOCAB['stereo'], str(bond.GetStereo())),
        safe_index(BOND_VOCAB['is_conjugated'], bond.GetIsConjugated()),
        safe_index(BOND_VOCAB['is_in_ring'], bond.IsInRing()),
    ]


def atom_physchem_context(mol):
    """Per-atom continuous physicochemical context: Gasteiger partial charge,
    Crippen logP/MR atomic contributions, TPSA atomic contribution.

    These are the exact atom-level quantities that the hierarchical branch
    aggregates into molecule-level descriptors (MaxPartialCharge/MinPartialCharge,
    Crippen MolLogP/MolMR, TPSA) -- giving the GNN node features direct,
    physically-meaningful grounding rather than purely structural indices.
    Returns an (n_atoms, 4) float array; any RDKit failure falls back to zeros
    for that contribution so a single ill-conditioned atom can't crash featurization.
    """
    n_atoms = mol.GetNumAtoms()
    charges = np.zeros(n_atoms, dtype=np.float32)
    try:
        AllChem.ComputeGasteigerCharges(mol)
        for i, atom in enumerate(mol.GetAtoms()):
            val = atom.GetDoubleProp('_GasteigerCharge')
            charges[i] = val if np.isfinite(val) else 0.0
    except Exception:
        pass

    logp_contribs = np.zeros(n_atoms, dtype=np.float32)
    mr_contribs = np.zeros(n_atoms, dtype=np.float32)
    try:
        contribs = rdMolDescriptors._CalcCrippenContribs(mol)
        for i, (logp, mr) in enumerate(contribs):
            logp_contribs[i] = logp
            mr_contribs[i] = mr
    except Exception:
        pass

    tpsa_contribs = np.zeros(n_atoms, dtype=np.float32)
    try:
        contribs = rdMolDescriptors._CalcTPSAContribs(mol)
        for i, val in enumerate(contribs):
            tpsa_contribs[i] = val
    except Exception:
        pass

    return np.stack([charges, logp_contribs, mr_contribs, tpsa_contribs], axis=1)


def smiles_to_graph(smiles):
    """Convert SMILES string to a PyTorch Geometric Data object with the
    rich OGB-style categorical features (data.x, data.edge_attr) plus a
    continuous per-atom physicochemical context vector (data.physchem)."""
    if not TORCH_GEOMETRIC_AVAILABLE:
        return None  # Use fingerprint-based approach in fallback mode

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    if mol.GetNumAtoms() == 0:
        return None

    # ---- Node features ----
    atom_features = [atom_to_feature_vector(atom) for atom in mol.GetAtoms()]
    x = torch.tensor(atom_features, dtype=torch.long)
    physchem = torch.tensor(atom_physchem_context(mol), dtype=torch.float)

    # ---- Edge features (both directions, undirected molecular graph) ----
    edge_index = []
    edge_attr = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feat = bond_to_feature_vector(bond)
        edge_index.extend([[i, j], [j, i]])
        edge_attr.extend([feat, feat])

    if len(edge_index) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, len(BOND_FEATURE_DIMS)), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.long)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, physchem=physchem)


def smiles_to_fingerprint(smiles, nBits=2048):
    """Convert SMILES to Morgan fingerprint for fallback mode."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(nBits)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=nBits)
    arr = np.zeros(nBits)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def build_graph_cache(unique_smiles, existing_cache=None):
    """
    Parse every SMILES string in `unique_smiles` into a torch_geometric.data.Data
    object ONCE (RDKit parsing + the OGB-style atom/bond featurization + the atomic
    physchem context -- see smiles_to_graph) and keep it around for the rest of the
    run. This is safe to reuse across EVERY epoch and EVERY forward pass: graph
    STRUCTURE is a fixed function of the SMILES string alone and never changes,
    unlike the *embedding* produced from that structure, which changes every
    optimizer step once the GNN backbone is being fine-tuned (see embed_batch()
    below). This replaces the old weight-fingerprinted embedding cache -- there is
    no fingerprint here because this cache does not depend on the model's weights
    at all.

    `existing_cache`: optional dict (e.g. loaded from disk) to extend in place --
    only SMILES not already present are parsed.

    Returns (cache: dict[str, Data], bad_smiles: set[str]). Any SMILES RDKit can't
    parse into a graph is reported in bad_smiles instead of raising, so the caller
    can drop the offending rows from common_valid_indices up front.
    """
    cache = existing_cache if existing_cache is not None else {}
    bad = set()
    to_parse = [s for s in unique_smiles if s not in cache]
    logger.info(f"build_graph_cache: {len(unique_smiles)} unique SMILES requested, "
                f"{len(unique_smiles) - len(to_parse)} already cached, {len(to_parse)} to parse")
    for i, smi in enumerate(to_parse):
        if i % 500 == 0:
            print(f"  Building graph cache: {i}/{len(to_parse)}")
        g = smiles_to_graph(smi)
        if g is None:
            bad.add(smi)
            logger.warning(f"Failed to convert SMILES to graph: {smi}")
        else:
            cache[smi] = g
    logger.success(f"build_graph_cache complete: {len(cache)} graphs cached, {len(bad)} failed to parse")
    return cache, bad


def load_graph_cache(cache_path):
    """Load the on-disk SMILES->graph-structure cache, or an empty dict if none
    exists yet. Unlike the old embedding cache this file name is NOT keyed to the
    GNN's weights (graph structure doesn't depend on weights), so it is reused
    across every run/every set of fine-tuned weights."""
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            cache = pickle.load(f)
        logger.info(f"Loaded graph structure cache: {cache_path} ({len(cache)} SMILES cached)")
        return cache
    logger.info(f"No existing graph structure cache at {cache_path} -- starting fresh")
    return {}


def save_graph_cache(cache, cache_path):
    """Persist the (mutated-in-place) graph structure cache dict back to disk."""
    with open(cache_path, 'wb') as f:
        pickle.dump(cache, f)
    logger.info(f"Saved graph structure cache: {cache_path} ({len(cache)} SMILES cached)")


def embed_batch(smiles_batch, graph_cache, gnn_model, device, training):
    """
    Differentiable, batched GNN forward pass for a list of SMILES strings -- used for
    BOTH the ligand column and each of the n_sites mutation-site columns during
    end-to-end fine-tuning. This is the direct replacement for the old
    get_gnn_embeddings(): it still deduplicates identical SMILES WITHIN one batch
    (mutation-site SMILES repeat constantly -- every row sharing a `tkd` mutant label
    has byte-identical SMILES for all 8 mutation-site columns, so a batch spanning
    only a handful of distinct mutants pays for a handful of forward passes on the
    mutant side, not one per row), but UNLIKE the old function it never wraps the
    forward pass in torch.no_grad() when training=True, and it never caches the
    resulting embedding beyond this one call -- the whole point is that the GNN's
    weights move every optimizer step, so any embedding computed before the current
    step is stale and must be recomputed.

    Every SMILES in `smiles_batch` MUST already have an entry in `graph_cache`
    (built up front via build_graph_cache over the full common_valid_indices set --
    see main()), so this function never needs to parse RDKit on the fly.

    training=True  -> caller has already put gnn_model in .train() mode; gradients
                       are tracked so the returned tensor can be backpropagated through.
    training=False -> runs under torch.no_grad() for validation/inference; caller is
                       responsible for having called gnn_model.eval() beforehand.

    Returns a torch.Tensor of shape (len(smiles_batch), gnn_model.feat_dim) on `device`.
    """
    unique_smiles = list(dict.fromkeys(smiles_batch))
    graphs = [graph_cache[s] for s in unique_smiles]

    def _forward():
        batch = Batch.from_data_list(graphs).to(device)
        return gnn_model(batch)

    if training:
        unique_embs = _forward()
    else:
        with torch.no_grad():
            unique_embs = _forward()

    smi_to_row = {s: i for i, s in enumerate(unique_smiles)}
    gather_idx = torch.tensor([smi_to_row[s] for s in smiles_batch],
                               dtype=torch.long, device=unique_embs.device)
    return unique_embs.index_select(0, gather_idx)


def iterate_batches(idx_array, batch_size, shuffle=False, rng=None):
    """Yield successive `batch_size` slices of positional indices (0..N-1) into the
    common_valid_indices-ordered arrays used throughout main() (physchem_scaled,
    y_train_scaled1/2, lig_smiles_arr, site_smiles_arrs, ...). This is the manual-
    training-loop replacement for what Keras' model.fit(batch_size=...) did
    internally; `shuffle=True` (used for the training split each epoch) draws a
    fresh permutation from `rng` (a numpy.random.Generator/RandomState) so batch
    composition varies epoch to epoch, matching Keras' default shuffle=True."""
    order = idx_array.copy()
    if shuffle:
        perm = rng.permutation(len(order))
        order = order[perm]
    for start in range(0, len(order), batch_size):
        yield order[start:start + batch_size]


def safe_divide(numerator, denominator, default=0.0):
    if isinstance(denominator, (int, float)):
        return numerator / denominator if denominator != 0 else default
    return np.where(denominator != 0, numerator / denominator, default)

@lru_cache(maxsize=200_000)
def generate_lig_inter_features(smiles): #Intermolecular Ligand, input smiles ligand, returns np array
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    features = []
    
    try:
        #Hydrogen Bonding,
        features.append(Lipinski.NumHDonors(mol))
        features.append(Lipinski.NumHAcceptors(mol))
        features.append(Lipinski.NHOHCount(mol))
        features.append(Lipinski.NOCount(mol))
        features.append(rdMolDescriptors.CalcNumHBD(mol)) #includes N, O, and S (manual edit)
        features.append(rdMolDescriptors.CalcNumHBA(mol)) #includes N, O, and S (manual edit)
        
        #Electrostatic bonding
        #Partial charge = the small positive or negative charge assigned to each atom due to unequal sharing of electrons in bonds (like in polar bond
        features.append(Descriptors.MaxPartialCharge(mol)) #highest partial charge among all atoms in the molecule (most positive atom), (likely electrophilic)
        features.append(Descriptors.MinPartialCharge(mol)) #highest partial charge among all atoms in the molecule (most negative atom), (likely nucleophilic)
        features.append(Descriptors.MaxAbsPartialCharge(mol)) #largest absolute value of partial charge among all atom, strongest charge polarization within the molecule
        features.append(Descriptors.MaxPartialCharge(mol) - Descriptors.MinPartialCharge(mol)) #difference between most positive and most negative atoms), larger values mean stronger polarity within the molecule.
        features.append(Descriptors.MinAbsPartialCharge(mol))
        
        #Polar surface
        features.append(MolSurf.TPSA(mol))#The sum of the surface areas of all polar atoms (mostly oxygen and nitrogen) and their attached hydrogens.High TPSA â†’ more polar, less membrane permeable, more soluble, Low TPSA â†’ less polar, more membrane permeable (good for oral drugs)
        
        features.append(MolSurf.LabuteASA(mol))#An approximation of the total solvent-accessible surface area (SASA) of the molecule, calculated using Labuteâ€™s algorithm. #Reflects molecular size and hydrophobic surface exposure
        #Aiâ€‹=Siâ€‹â‹…Piâ€‹, S = 4Ï€r^2i , Piâ€‹=1âˆ’jâˆâ€‹(1âˆ’fij
        # S=total spherical surface area of atom , Pi=atomic solvation parameter, fij=fraction of atom i's surface area in contact with atom j

        features.append(Crippen.MolMR(mol))#The Ghose-Crippen formula is an atom-contribution method used to estimate the octanol-water partition coefficient (log P) and molar refractivity (MR) of a molecule.
        #Molar refractivity, a measure of the polarizability of the molecule
        
        #Size & Rigidity  (#May overlap with others)
        features.append(Descriptors.MolWt(mol)) #molecular weight
        features.append(Lipinski.HeavyAtomCount(mol))
        features.append(rdMolDescriptors.CalcNumRotatableBonds(mol))
        
        features.append(Crippen.MolLogP(mol))
        features.append(Descriptors.FractionCSP3(mol)) # fraction of SP3 hybridised carbons
        features.append(Lipinski.NumAromaticRings(mol))
        aromatic_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
        features.append(aromatic_atoms)
        
        # Pi-Pi stacking 
        features.append(Descriptors.NumAromaticCarbocycles(mol))
        features.append(Descriptors.NumAromaticHeterocycles(mol))

        #Halogen
        features.append(Fragments.fr_halogen(mol))

        #Flexibility
        features.append(Lipinski.NumRotatableBonds(mol))

        return np.array(features)
        
    except Exception as e:
        print(f"Error in lig_inter: {str(e)}")
        return None


@lru_cache(maxsize=200_000)
def generate_mut_inter_features(smiles): #Intermolecular Mutation, input smiles mutation, returns np array
    return generate_lig_inter_features(smiles)


#Subsequent priority of descriptors to capture features from intramolecular forces 
@lru_cache(maxsize=200_000)
def generate_lig_intra_features(smiles): #Intramolecular Ligand
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    features = []
    
    try:
        #Covalent bond
        num_bonds = mol.GetNumBonds()
        features.append(num_bonds)
        
        #higher order bonds favours intramolecular forces within molecule, bond order indicates strength of bond
        single_bonds = sum(1 for bond in mol.GetBonds() if bond.GetBondTypeAsDouble() == 1.0)
        double_bonds = sum(1 for bond in mol.GetBonds() if bond.GetBondTypeAsDouble() == 2.0)
        triple_bonds = sum(1 for bond in mol.GetBonds() if bond.GetBondTypeAsDouble() == 3.0)
        aromatic_bonds = sum(1 for bond in mol.GetBonds() if bond.GetIsAromatic())
        features.extend([single_bonds, double_bonds, triple_bonds, aromatic_bonds])
        
        avg_bond_order = np.mean([bond.GetBondTypeAsDouble() for bond in mol.GetBonds()]) if num_bonds > 0 else 0
        features.append(avg_bond_order)
        
        #Rigidity (May Overlap), flexibility indicates less intramolecular forces within molecule
        features.append(Lipinski.NumRotatableBonds(mol))
        features.append(Lipinski.RingCount(mol))
        features.append(Lipinski.NumAromaticRings(mol))
        
        rigid_bonds = sum(1 for bond in mol.GetBonds() if bond.IsInRing())
        fraction_rigid = rigid_bonds / num_bonds if num_bonds > 0 else 0
        features.append(fraction_rigid)
        
        # Pi-Pi bonding 
        features.append(Descriptors.NumAromaticCarbocycles(mol))
        features.append(Descriptors.NumAromaticHeterocycles(mol))
        
        #Hybridization (May Overlap), branching indicates less intramolecular forces within molecule
        sp2_carbons = sum(1 for atom in mol.GetAtoms() if atom.GetHybridization() == Chem.HybridizationType.SP2)
        sp3_carbons = sum(1 for atom in mol.GetAtoms() if atom.GetHybridization() == Chem.HybridizationType.SP3)
        sp_carbons = sum(1 for atom in mol.GetAtoms() if atom.GetHybridization() == Chem.HybridizationType.SP)
        features.extend([sp_carbons, sp2_carbons, sp3_carbons])
        
        #Ring strain
        ring_sizes = [len(ring) for ring in mol.GetRingInfo().AtomRings()]
        avg_ring_size = np.mean(ring_sizes) if ring_sizes else 0
        min_ring_size = min(ring_sizes) if ring_sizes else 0
        features.extend([avg_ring_size, min_ring_size])
        
        three_member_rings = sum(1 for size in ring_sizes if size == 3)
        four_member_rings = sum(1 for size in ring_sizes if size == 4)
        features.extend([three_member_rings, four_member_rings])
        
        #Complexity
        features.append(GraphDescriptors.BertzCT(mol))
        features.append(GraphDescriptors.Kappa1(mol))
        features.append(GraphDescriptors.Kappa2(mol))
        features.append(GraphDescriptors.Kappa3(mol))
        features.append(rdMolDescriptors.CalcNumBridgeheadAtoms(mol))
        features.append(rdMolDescriptors.CalcNumSpiroAtoms(mol))
        
        return np.array(features)
        
    except Exception as e:
        print(f"Error in lig_intra: {str(e)}")
        return None


@lru_cache(maxsize=200_000)
def generate_mut_intra_features(smiles): # Intramolecular Mutation
    return generate_lig_intra_features(smiles)

def calculate_similarity_metrics(vec1, vec2): #input np arrays, returns a dict with math metrics
    # 1. Calculate Cosine Similarity with safety check
    norm1 = norm(vec1)
    norm2 = norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        # If either vector has zero norm, return default values
        return {
            'cosine_similarity': 0.0,
            'sine_dissimilarity': 0.0,
            'dot_product': 0.0
        }
    
    cosine_sim = np.dot(vec1, vec2) / (norm1 * norm2)
    sine_of_angle = np.sqrt(1 - cosine_sim**2)

    #logger.info(f"Cosine Similarity: {cosine_sim}, Sine dssimilarity: {sine_of_angle}, Dot Product: {np.dot(vec1, vec2)}")
    
    return {
        'cosine_similarity': cosine_sim,
        'sine_dissimilarity': sine_of_angle,
        'dot_product': np.dot(vec1, vec2)
    }


def calculate_fp_metrics(smiles1, smiles2): #input smiles, returns dict with rdkit datastructs similarity
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    
    if mol1 is None or mol2 is None:
        return {'dice_sim': 0.0, 'tanimato': 0.0}
    
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)
    
    dice_sim = DataStructs.DiceSimilarity(fp1, fp2)
    tanimato = DataStructs.TanimotoSimilarity(fp1, fp2)

    #logger.info(f"Dice Similarity: {dice_sim}, Tanimoto Similarity: {tanimato}")

    
    return {
        'dice_sim': dice_sim,
        'tanimato': tanimato,
    }


def generate_inter_interaction_features(lig_inter, mut_inter): #similarity on intermolecular interactions ligand and mutation
    features = []
    metrics = calculate_similarity_metrics(lig_inter, mut_inter)
    
    features.append(metrics['cosine_similarity'])
    features.append(metrics['sine_dissimilarity'])
    
    return np.array(features)


def generate_intra_interaction_features(lig_intra, mut_intra): #similarity on intramolecular interactions ligand and mutation
    features = []
    metrics = calculate_similarity_metrics(lig_intra, mut_intra)
    
    features.append(metrics['cosine_similarity'])
    features.append(metrics['sine_dissimilarity'])
    
    return np.array(features)


@lru_cache(maxsize=200_000)
def generate_final_interaction_features(lig_smiles, mut_smiles): # fingerprints, morgan fingerprints dominate
    features = []
    
    fp_inter_metrics = calculate_fp_metrics(lig_smiles, mut_smiles)
    features.extend([fp_inter_metrics['dice_sim'], fp_inter_metrics['tanimato']])
    
    return np.array(features)


def generate_custom_features(lig_inter, mut_inter, lig_intra, mut_intra): 
    """Generate custom intermolecular and intramolecular features with safe division"""
    lig_mut_inter = []
    lig_mut_intra = []
    lig_mut_mix_inter_intra = []
    
    # H attraction ligand , H = lig_hbd . mut_hba / mut_hbd 
    # (#assumption: ligand moves to mut (mut is fixed position), lig hbd and mut hba attracts ligand), 
    # mut hbd repels favouring intra bond within mut, ignoring intra bond repelling from ligand
    H_linear_lipinski = safe_divide(lig_inter[0] * mut_inter[1], mut_inter[0], default=0.0)
    lig_mut_inter.append(H_linear_lipinski)
    
    H_linear_total = safe_divide(lig_inter[4] * mut_inter[5], mut_inter[4], default=0.0)
    lig_mut_inter.append(H_linear_total)
    
    # H attraction ligand , H = lig_hbd . mut_hba / mut_hbd with weighted mut bond path (Kappa)
    # (#assumption: ligand moves to mut (mut is fixed position), lig hbd and mut hba attracts ligand), 
    # mut hbd repels favouring intra bond within mut
    H_path = safe_divide(safe_divide(lig_inter[0] * mut_inter[1], mut_inter[0], default=0.0), mut_intra[21], default=0.0)
    lig_mut_mix_inter_intra.append(H_path)
    
    # Streght H bond in intermolecular lig to mut minus mut intra bond within mut Lig(x1,y1) Mut(x2,y2)
    # total attraction H_stregth , Lig(x1y1) Mut(x2,y2) , (lig_x1 * mut_y2 / lig_x2) + (lig_x2 * mut_y1 / mut_y2)
    # inter bond attarct x1y2 , intra bond forming assumed as repelled, x1/y1 , assumed no repelling inter H bonds
    H_strength = safe_divide(lig_inter[0] * mut_inter[1], lig_inter[1], default=0.0) + safe_divide(lig_inter[1] * mut_inter[0], mut_inter[1], default=0.0)
    lig_mut_inter.append(H_strength)
    
    H_strength_total = safe_divide(lig_inter[4] * mut_inter[5], lig_inter[4], default=0.0) + safe_divide(lig_inter[5] * mut_inter[4], mut_inter[5], default=0.0)
    lig_mut_inter.append(H_strength_total)
    
    # Lig donating stregght + Mut accepting Stregth , ligand movving to mut
    H_frac_lipinski = safe_divide(lig_inter[0], lig_inter[1], default=0.0) + safe_divide(mut_inter[1], mut_inter[0], default=0.0)
    lig_mut_inter.append(H_frac_lipinski)
    
    H_frac_total = safe_divide(lig_inter[4], lig_inter[5], default=0.0) + safe_divide(mut_inter[5], mut_inter[4], default=0.0)
    lig_mut_inter.append(H_frac_total)
    

    #using max positive and max negative charge, and length and size is simple number of bonds  (q1q2/r2)
    # Attraction opp site charge lig(q1/r1) * mut(q2/r2), q1 is max positive and q2 is max neg
    # size options include: Molwt, number of bonds, Euclidean distance . radius of gyration (rdMolDescriptors.CalcRadiusOfGyration(mol))

    # Assumption: non moving mutant, only ligand moving to mutant through attraction charge Only, (taking max abs postive and min ngeative)
    # only Attraction intermolecular forces, assuming no intrabond attraction within molecule. Assumed no repelling intermolecule same charge
    #A c_linear q1 pos to q2 neg / r1r2 
    # B c_linear q1 neg to q2 pos/r1r2
    #total & ratio

    # assuming got positive charges ligand and negative charge mut with weighted size sp3
    c_linear1_size1 = safe_divide(lig_inter[6], lig_intra[14], default=0.0) * safe_divide(mut_inter[7], mut_intra[14], default=0.0)
    lig_mut_mix_inter_intra.append(c_linear1_size1)
    
    c_linear2_size1 = safe_divide(lig_inter[7], lig_intra[14], default=0.0) * safe_divide(mut_inter[6], mut_intra[14], default=0.0)
    lig_mut_mix_inter_intra.append(c_linear2_size1)
    
    c_total = (c_linear1_size1 ** 2) + (c_linear2_size1 ** 2) #bringing out magnitude of each attarction parts
    lig_mut_mix_inter_intra.append(c_total)
    
    #difference between pos lig neg mut to neg mut pos lig
    c_diff = ((lig_inter[6]) - (mut_inter[7])) - ((mut_inter[6]) - (lig_inter[7]))
    lig_mut_inter.append(c_diff)
    
    #difference between pos lig neg mut to neg mut pos lig
    c_tpsa_diff = lig_inter[11] - mut_inter[11]
    lig_mut_inter.append(c_tpsa_diff)
    
    c_crip_logh = lig_inter[17] - mut_inter[17]
    lig_mut_inter.append(c_crip_logh)
    
    frac_tpsa_logH = safe_divide(lig_inter[11] * mut_inter[11], lig_inter[17] * mut_inter[17], default=0.0)
    lig_mut_inter.append(frac_tpsa_logH)
    
    #pi-pi stacking ratio
    pi_pi_ratio1 = safe_divide(lig_inter[21] + lig_inter[22] + mut_inter[21] + mut_inter[22], lig_intra[15] + mut_intra[15], default=0.0)
    lig_mut_mix_inter_intra.append(pi_pi_ratio1)
    
    pi_pi_ratio2 = safe_divide(lig_inter[21] + lig_inter[22] + mut_inter[21] + mut_inter[22], lig_intra[22] + mut_intra[22], default=0.0)
    lig_mut_mix_inter_intra.append(pi_pi_ratio2)
    
    #Bringing out difference between a more rigid/loose ligand 

    #double/triple bond ratio increasing
    # bond rigid total double, triple n aromatic over total num of bonds (tighter intra lig and intra mut strength as a total)
    # bond single (looser intra lig and intra mut strength)
    bond_rigid = safe_divide(lig_intra[2] + lig_intra[3] + lig_intra[4], lig_intra[0], default=0.0) + safe_divide(mut_intra[2] + mut_intra[3] + mut_intra[4], mut_intra[0], default=0.0)
    bond_single = safe_divide(lig_intra[1], lig_intra[0], default=0.0) + safe_divide(mut_intra[1], mut_intra[0], default=0.0)
    bond_diff = (bond_single - bond_rigid) ** 2
    lig_mut_intra.append(bond_diff)
    
    #spsp2/sp3 ratio
    # fraction of spsp2/sp3 between ligand and mutant
    # bigger difference indicate mutants more loose, ligands are same
    hybridisation_lig = safe_divide(lig_intra[12] + lig_intra[13], lig_intra[14] + lig_intra[12] + lig_intra[13], default=0.0)
    hybridisation_mut = safe_divide(mut_intra[12] + mut_intra[13], mut_intra[14] + mut_intra[12] + mut_intra[13], default=0.0)
    hybridisation_diff = (hybridisation_mut - hybridisation_lig) ** 2
    lig_mut_intra.append(hybridisation_diff)
    
    kappa_ratio = safe_divide(lig_intra[21], mut_intra[21], default=0.0)
    lig_mut_intra.append(kappa_ratio)
    
    return lig_mut_inter, lig_mut_intra, lig_mut_mix_inter_intra

def generate_hierarchical_features(ligand_smiles_series, mutation_smiles_series):
    print('\nGenerating hierarchical features...')
    ligand_cache, mutation_cache, interaction_cache = {}, {}, {}
    
    results = {k: [] for k in ['lig_inter', 'mut_inter', 'inter_interaction', 'lig_intra', 
                                'mut_intra', 'intra_interaction', 'lig_mut_mix_inter_intra', 
                                'final_fp_interaction']}
    valid_indices = []
    
    for idx, (lig_smi, mut_smi) in enumerate(zip(ligand_smiles_series, mutation_smiles_series)):
        if idx % 50 == 0:
            print(f'  Processing sample {idx}/{len(ligand_smiles_series)}...')
        
        # Cache ligand features
        if lig_smi in ligand_cache:
            lig_inter, lig_intra = ligand_cache[lig_smi]
        else:
            lig_inter = generate_lig_inter_features(lig_smi)
            lig_intra = generate_lig_intra_features(lig_smi)
            ligand_cache[lig_smi] = (lig_inter, lig_intra)
        
        # Cache mutation features
        if mut_smi in mutation_cache:
            mut_inter, mut_intra = mutation_cache[mut_smi]
        else:
            mut_inter = generate_mut_inter_features(mut_smi)
            mut_intra = generate_mut_intra_features(mut_smi)
            mutation_cache[mut_smi] = (mut_inter, mut_intra)
        
        if any(f is None for f in [lig_inter, mut_inter, lig_intra, mut_intra]):
            continue
        
        # Cache interaction features
        pair_key = (lig_smi, mut_smi)
        if pair_key in interaction_cache:
            lig_mut_mix, inter_int, intra_int, fp_int = interaction_cache[pair_key]
        else:
            lig_mut_inter, lig_mut_intra, lig_mut_mix = generate_custom_features(
                lig_inter, mut_inter, lig_intra, mut_intra)
            inter_int = generate_inter_interaction_features(lig_inter, mut_inter)
            intra_int = generate_intra_interaction_features(lig_intra, mut_intra)
            if lig_mut_inter:
                inter_int = np.concatenate([np.array(lig_mut_inter), inter_int])
            if lig_mut_intra:
                intra_int = np.concatenate([np.array(lig_mut_intra), intra_int])
            fp_int = generate_final_interaction_features(lig_smi, mut_smi)
            interaction_cache[pair_key] = (lig_mut_mix, inter_int, intra_int, fp_int)
        
        results['lig_inter'].append(lig_inter)
        results['mut_inter'].append(mut_inter)
        results['inter_interaction'].append(inter_int)
        results['lig_intra'].append(lig_intra)
        results['mut_intra'].append(mut_intra)
        results['intra_interaction'].append(intra_int)
        results['lig_mut_mix_inter_intra'].append(np.array(lig_mut_mix))
        results['final_fp_interaction'].append(fp_int)
        valid_indices.append(idx)
    
    print(f'  Successfully generated features for {len(valid_indices)} samples')
    result_dict = {k: np.array(v) for k, v in results.items()}
    result_dict['valid_indices'] = valid_indices
    return result_dict


# =============================================================================
# Physicochemical feature combination + GNN cross-attention + Set-Transformer model
# =============================================================================

PHYSCHEM_GROUP_KEYS = ['lig_inter', 'mut_inter', 'inter_interaction', 'lig_intra',
                        'mut_intra', 'intra_interaction', 'lig_mut_mix_inter_intra',
                        'final_fp_interaction']


def combine_physchem_features(feature_dict):
    """
    Flatten the 8 physicochemical descriptor groups produced by
    generate_hierarchical_features() into a single per-sample vector.

    PSEUDOCODE:
        combined[i] = concat( lig_inter[i], mut_inter[i], inter_interaction[i],
                               lig_intra[i], mut_intra[i], intra_interaction[i],
                               lig_mut_mix_inter_intra[i], final_fp_interaction[i] )

    The width is computed dynamically (not hardcoded) so this keeps working if any
    of the 8 descriptor-generating functions above is ever extended.
    Returns shape (n_samples, D_physchem).
    """
    return np.concatenate([feature_dict[k] for k in PHYSCHEM_GROUP_KEYS], axis=1).astype(np.float32)


def scale_site_stack(arr3d, scaler=None, fit_indices=None):
    """
    Fit (if scaler is None) or apply a StandardScaler to a (N, n_sites, D) array by
    POOLING across the site axis -- i.e. all n_sites (8) sites are treated as draws
    from the same distribution and share ONE scaler. This is consistent with the
    shared-weight physchem projection (nn.Linear applied identically across the
    site axis, the PyTorch equivalent of Keras' TimeDistributed(Dense)) inside
    GNNCrossAttentionSetTransformerHead: a shared-weight layer expects a consistent
    input distribution regardless of which of the 8 sites the row came from.

    `fit_indices` (base1d addition): if provided, the scaler is FIT only on rows
    arr3d[fit_indices] (i.e. the scaffold-split TRAIN rows) but still applied
    (transform) to every row in arr3d, train and validation alike. This is what
    makes the scaffold split leak-free: fitting the scaler on the full dataset
    (train+val) before splitting would let validation-set statistics quietly
    influence the scaling every training row is trained against. Ignored (with a
    warning) if `scaler` is already provided, since in that case no fitting happens
    here at all -- this is the code path the predict script uses.

    PSEUDOCODE:
        flat        = reshape(arr3d, (N * n_sites, D))
        fit_rows    = arr3d[fit_indices].reshape(-1, D)  if fit_indices is not None else flat
        scaler.fit(fit_rows)                             if scaler is None
        flat_scaled = scaler.transform(flat)
        return reshape(flat_scaled, (N, n_sites, D)), scaler

    Returns (scaled_arr3d, fitted_scaler).
    """
    n, n_sites, d = arr3d.shape
    flat = arr3d.reshape(n * n_sites, d)
    if scaler is None:
        scaler = StandardScaler()
        if fit_indices is not None:
            fit_rows = arr3d[fit_indices].reshape(-1, d)
            scaler.fit(fit_rows)
        else:
            scaler.fit(flat)
        flat_scaled = scaler.transform(flat)
    else:
        if fit_indices is not None:
            logger.warning("scale_site_stack: fit_indices given but a pre-fit scaler was "
                            "also passed -- ignoring fit_indices, only transforming.")
        flat_scaled = scaler.transform(flat)
    return flat_scaled.reshape(n, n_sites, d), scaler


class MultiHeadAttentionKeyDim(nn.Module):
    """
    Faithful PyTorch port of tf.keras.layers.MultiHeadAttention(num_heads, key_dim):
    projects query/key/value to (num_heads * key_dim), does scaled dot-product
    attention, then projects the concatenated heads back to the QUERY's own last
    dimension. Unlike torch.nn.MultiheadAttention, this does not require embed_dim
    to be divisible by num_heads, and query/key/value may have DIFFERENT last-dim
    widths (needed below: the site<->site SAB block operates on the wide
    site_dim = 3*gnn_dim + physchem_proj_dim space, while the earlier ligand<->site
    cross-attention operates on native gnn_dim -- both share the same
    num_heads/key_dim for their internal attention computation regardless).
    """
    def __init__(self, query_dim, key_value_dim, num_heads, key_dim, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.key_dim = key_dim
        inner_dim = num_heads * key_dim
        self.q_proj = nn.Linear(query_dim, inner_dim)
        self.k_proj = nn.Linear(key_value_dim, inner_dim)
        self.v_proj = nn.Linear(key_value_dim, inner_dim)
        self.out_proj = nn.Linear(inner_dim, query_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.scale = key_dim ** -0.5

    def forward(self, query, key, value):
        B, Lq, _ = query.shape
        Lk = key.shape[1]
        q = self.q_proj(query).view(B, Lq, self.num_heads, self.key_dim).transpose(1, 2)
        k = self.k_proj(key).view(B, Lk, self.num_heads, self.key_dim).transpose(1, 2)
        v = self.v_proj(value).view(B, Lk, self.num_heads, self.key_dim).transpose(1, 2)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        out = torch.matmul(attn_weights, v)                      # (B, heads, Lq, key_dim)
        out = out.transpose(1, 2).contiguous().view(B, Lq, self.num_heads * self.key_dim)
        return self.out_proj(out)


class GNNCrossAttentionSetTransformerHead(nn.Module):
    """
    Simplified base1d head.

    Design principle:
      1. Ligand GNN, site GNN and site physicochemical descriptors are encoded
         independently.
      2. The three representations are fused ONCE, before any attention.
      3. One site self-attention block lets the 8 ligand-conditioned site tokens
         exchange information with each other.
      4. A lightweight learned attention-pooling layer produces one global site
         summary.
      5. A compact shared MLP feeds the activity and docking heads.

    This deliberately removes the previous bidirectional cross-attention,
    1600-d raw-skip fusion, SAB+PMA stack, and parallel 1024-d raw GNN branch.
    The physicochemical branch remains independent until the explicit
    ligand + site-GNN + physchem fusion immediately before self-attention.

    Default shapes (batch dimension omitted):
        ligand GNN                 : (512,)
        8 site GNNs                : (8, 512)
        8 site physchem vectors    : (8, physchem_dim)
        ligand projection          : (256,)
        site GNN projection        : (8, 256)
        physchem projection        : (8, 64)
        pre-attention fusion       : (8, 576) -> (8, 256)
        site self-attention        : (8, 256)
        attention pooling          : (256,)
        prediction trunk           : 256 -> 128 -> 64 -> 32
        outputs                    : activity (1), docking (1)
    """
    def __init__(self, gnn_dim=512, n_sites=8, physchem_dim=None,
                 fusion_dim=256, physchem_proj_dim=64,
                 num_heads=4, key_dim=32, dropout=0.15):
        super().__init__()
        if physchem_dim is None:
            raise ValueError(
                "physchem_dim must be provided (width of the concatenated "
                "8-group physchem vector -- see combine_physchem_features)"
            )
        self.n_sites = n_sites
        self.gnn_dim = gnn_dim
        self.fusion_dim = fusion_dim
        self.physchem_proj_dim = physchem_proj_dim

        # ------------------------------------------------------------------
        # STEP 1: independent encoders -- NO cross-modal fusion yet.
        # ------------------------------------------------------------------
        self.lig_proj = nn.Sequential(
            nn.Linear(gnn_dim, fusion_dim),
            nn.LeakyReLU(0.1),
            nn.LayerNorm(fusion_dim),
        )
        self.site_gnn_proj = nn.Sequential(
            nn.Linear(gnn_dim, fusion_dim),
            nn.LeakyReLU(0.1),
            nn.LayerNorm(fusion_dim),
        )
        self.physchem_proj = nn.Sequential(
            nn.Linear(physchem_dim, physchem_proj_dim),
            nn.LeakyReLU(0.1),
            nn.LayerNorm(physchem_proj_dim),
        )

        # ------------------------------------------------------------------
        # STEP 2: ONE explicit multimodal fusion BEFORE attention.
        # Every site receives:
        #   site-GNN representation + ligand representation + physchem.
        # The fusion output is the token consumed by attention.
        # ------------------------------------------------------------------
        fusion_input_dim = fusion_dim + fusion_dim + physchem_proj_dim
        self.pre_attention_fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_dim),
            nn.LeakyReLU(0.1),
            nn.LayerNorm(fusion_dim),
            nn.Dropout(dropout),
        )

        # ------------------------------------------------------------------
        # STEP 3: ONE site self-attention block.
        # The 8 site tokens are now already ligand + GNN + physchem aware.
        # ------------------------------------------------------------------
        self.site_attention = MultiHeadAttentionKeyDim(
            fusion_dim, fusion_dim, num_heads, key_dim, dropout=dropout
        )
        self.site_attn_ln = nn.LayerNorm(fusion_dim)
        self.site_ff = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim * 2, fusion_dim),
        )
        self.site_ff_ln = nn.LayerNorm(fusion_dim)

        # ------------------------------------------------------------------
        # STEP 4: lightweight learned attention pooling over the 8 sites.
        # This replaces a separate SAB/PMA stack with a single scalar score
        # per site and a weighted sum.
        # ------------------------------------------------------------------
        self.pool_score = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.Tanh(),
            nn.Linear(fusion_dim // 2, 1),
        )
        self.pool_ln = nn.LayerNorm(fusion_dim)

        # ------------------------------------------------------------------
        # STEP 5: compact multitask prediction trunk.
        # ------------------------------------------------------------------
        self.dense0 = nn.Linear(fusion_dim, 128)
        self.bn0 = nn.LayerNorm(128)
        self.drop0 = nn.Dropout(0.25)
        self.dense1 = nn.Linear(128, 64)
        self.bn1 = nn.LayerNorm(64)
        self.drop1 = nn.Dropout(0.15)
        self.dense2 = nn.Linear(64, 32)
        self.drop2 = nn.Dropout(0.10)

        self.activity_head = nn.Sequential(
            nn.Linear(32, 16), nn.ReLU(), nn.Dropout(0.10), nn.Linear(16, 1)
        )
        self.docking_head = nn.Sequential(
            nn.Linear(32, 16), nn.ReLU(), nn.Dropout(0.10), nn.Linear(16, 1)
        )

    def forward(self, lig_gnn_embed, mut_gnn_embed_seq, site_physchem_features):
        # STEP 1: encode each modality independently.
        lig = self.lig_proj(lig_gnn_embed)                         # (B, 256)
        site_gnn = self.site_gnn_proj(mut_gnn_embed_seq)          # (B, 8, 256)
        physchem = self.physchem_proj(site_physchem_features)     # (B, 8, 64)

        # STEP 2: fuse ligand + site GNN + physchem BEFORE attention.
        lig_tiled = lig.unsqueeze(1).expand(-1, self.n_sites, -1)
        fused_input = torch.cat([site_gnn, lig_tiled, physchem], dim=-1)
        site_tokens = self.pre_attention_fusion(fused_input)      # (B, 8, 256)

        # STEP 3: one site self-attention block.
        attn_out = self.site_attention(site_tokens, site_tokens, site_tokens)
        site_tokens = self.site_attn_ln(site_tokens + attn_out)
        ff_out = self.site_ff(site_tokens)
        site_tokens = self.site_ff_ln(site_tokens + ff_out)       # (B, 8, 256)

        # STEP 4: learned attention pooling over the 8 site tokens.
        scores = self.pool_score(site_tokens)                     # (B, 8, 1)
        weights = torch.softmax(scores, dim=1)
        pooled = torch.sum(weights * site_tokens, dim=1)          # (B, 256)
        pooled = self.pool_ln(pooled)

        # STEP 5: compact shared trunk -> two task-specific heads.
        x = self.drop0(F.relu(self.bn0(self.dense0(pooled))))
        x = self.drop1(F.relu(self.bn1(self.dense1(x))))
        x = self.drop2(F.relu(self.dense2(x)))

        activity_output = self.activity_head(x)
        docking_output = self.docking_head(x)
        return activity_output, docking_output


def evaluate_and_plot(df_results, output_dir, model_name):
    """
    Evaluate predictions and generate comprehensive plots with statistics

    - Save both Pearson and Spearman metrics to CSV table
    - Create individual plots for each mutation
    - Generate correlation plots per mutation (both Pearson and Spearman)
    - Save all plots separately
    """

    print("\n" + "=" * 80)
    print("EVALUATION METRICS")
    print("=" * 80)

    # Create metrics directory
    metrics_dir = os.path.join(output_dir, 'metrics')
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # ========================================================================
    # 1. OVERALL METRICS
    # ========================================================================
    mae_act = mean_absolute_error(df_results['actual_activity'], df_results['predicted_activity'])
    rmse_act = np.sqrt(mean_squared_error(df_results['actual_activity'], df_results['predicted_activity']))
    mae_dock = mean_absolute_error(df_results['actual_docking'], df_results['predicted_docking'])
    rmse_dock = np.sqrt(mean_squared_error(df_results['actual_docking'], df_results['predicted_docking']))

    # Calculate correlations for overall
    pearson_act, pval_act_p = pearsonr(df_results['actual_activity'], df_results['predicted_activity'])
    pearson_dock, pval_dock_p = pearsonr(df_results['actual_docking'], df_results['predicted_docking'])
    spearman_act, pval_act_s = spearmanr(df_results['actual_activity'], df_results['predicted_activity'])
    spearman_dock, pval_dock_s = spearmanr(df_results['actual_docking'], df_results['predicted_docking'])

    print("\nOverall Performance:")
    print(f"  Activity  - MAE: {mae_act:.4f}, RMSE: {rmse_act:.4f}, Pearson R: {pearson_act:.4f}, Spearman \u03c1: {spearman_act:.4f}")
    print(f"  Docking   - MAE: {mae_dock:.4f}, RMSE: {rmse_dock:.4f}, Pearson R: {pearson_dock:.4f}, Spearman \u03c1: {spearman_dock:.4f}")

    # ========================================================================
    # 2. PER-MUTATION METRICS
    # ========================================================================
    print("\n" + "=" * 80)
    print("PER-MUTATION METRICS")
    print("=" * 80)

    metrics_data = []

    metrics_data.append({
        'Mutation': 'Overall',
        'N_Samples': len(df_results),
        'Activity_MAE': mae_act,
        'Activity_RMSE': rmse_act,
        'Activity_Pearson_R': pearson_act,
        'Activity_Pearson_pval': pval_act_p,
        'Activity_Spearman_rho': spearman_act,
        'Activity_Spearman_pval': pval_act_s,
        'Docking_MAE': mae_dock,
        'Docking_RMSE': rmse_dock,
        'Docking_Pearson_R': pearson_dock,
        'Docking_Pearson_pval': pval_dock_p,
        'Docking_Spearman_rho': spearman_dock,
        'Docking_Spearman_pval': pval_dock_s
    })

    mutations = sorted(df_results['tkd'].unique())

    for mutation in mutations:
        mut_data = df_results[df_results['tkd'] == mutation]
        n_samples = len(mut_data)

        if n_samples < 2:
            print(f"\n{mutation}: Insufficient data (n={n_samples}), skipping")
            continue

        mae_a = mean_absolute_error(mut_data['actual_activity'], mut_data['predicted_activity'])
        rmse_a = np.sqrt(mean_squared_error(mut_data['actual_activity'], mut_data['predicted_activity']))
        mae_d = mean_absolute_error(mut_data['actual_docking'], mut_data['predicted_docking'])
        rmse_d = np.sqrt(mean_squared_error(mut_data['actual_docking'], mut_data['predicted_docking']))

        try:
            pearson_a, pval_a_p = pearsonr(mut_data['actual_activity'], mut_data['predicted_activity'])
            pearson_d, pval_d_p = pearsonr(mut_data['actual_docking'], mut_data['predicted_docking'])
            spearman_a, pval_a_s = spearmanr(mut_data['actual_activity'], mut_data['predicted_activity'])
            spearman_d, pval_d_s = spearmanr(mut_data['actual_docking'], mut_data['predicted_docking'])
        except Exception:
            pearson_a, pval_a_p = np.nan, np.nan
            pearson_d, pval_d_p = np.nan, np.nan
            spearman_a, pval_a_s = np.nan, np.nan
            spearman_d, pval_d_s = np.nan, np.nan

        print(f"\n{mutation} (n={n_samples}):")
        print(f"  Activity  - MAE: {mae_a:.4f}, RMSE: {rmse_a:.4f}, Pearson R: {pearson_a:.4f}, Spearman \u03c1: {spearman_a:.4f}")
        print(f"  Docking   - MAE: {mae_d:.4f}, RMSE: {rmse_d:.4f}, Pearson R: {pearson_d:.4f}, Spearman \u03c1: {spearman_d:.4f}")

        metrics_data.append({
            'Mutation': mutation,
            'N_Samples': n_samples,
            'Activity_MAE': mae_a,
            'Activity_RMSE': rmse_a,
            'Activity_Pearson_R': pearson_a,
            'Activity_Pearson_pval': pval_a_p,
            'Activity_Spearman_rho': spearman_a,
            'Activity_Spearman_pval': pval_a_s,
            'Docking_MAE': mae_d,
            'Docking_RMSE': rmse_d,
            'Docking_Pearson_R': pearson_d,
            'Docking_Pearson_pval': pval_d_p,
            'Docking_Spearman_rho': spearman_d,
            'Docking_Spearman_pval': pval_d_s
        })

    # ========================================================================
    # 3. SAVE METRICS TO CSV
    # ========================================================================
    metrics_df = pd.DataFrame(metrics_data)
    metrics_csv_path = os.path.join(metrics_dir, f'{model_name}_metrics_summary.csv')
    metrics_df.to_csv(metrics_csv_path, index=False, float_format='%.6f')
    print(f"\n\u2713 Metrics saved to: {metrics_csv_path}")

    # ========================================================================
    # 4. CREATE OVERALL COMBINED PLOT (2x2 layout)
    # ========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    axes[0, 0].scatter(df_results['actual_activity'], df_results['predicted_activity'],
                      alpha=0.5, s=20, edgecolors='k', linewidths=0.5)
    axes[0, 0].plot([df_results['actual_activity'].min(), df_results['actual_activity'].max()],
                   [df_results['actual_activity'].min(), df_results['actual_activity'].max()],
                   'r--', lw=2, label='Perfect prediction')
    axes[0, 0].set_xlabel('Actual Activity', fontsize=11)
    axes[0, 0].set_ylabel('Predicted Activity', fontsize=11)
    axes[0, 0].set_title(f'Activity (Overall)\nRMSE={rmse_act:.4f}, PR={pearson_act:.3f}, SR={spearman_act:.3f}', fontsize=12)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].scatter(df_results['actual_docking'], df_results['predicted_docking'],
                      alpha=0.5, s=20, edgecolors='k', linewidths=0.5)
    axes[0, 1].plot([df_results['actual_docking'].min(), df_results['actual_docking'].max()],
                   [df_results['actual_docking'].min(), df_results['actual_docking'].max()],
                   'r--', lw=2, label='Perfect prediction')
    axes[0, 1].set_xlabel('Actual Docking Score', fontsize=11)
    axes[0, 1].set_ylabel('Predicted Docking Score', fontsize=11)
    axes[0, 1].set_title(f'Docking (Overall)\nRMSE={rmse_dock:.4f}, PR={pearson_dock:.3f}, SR={spearman_dock:.3f}', fontsize=12)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    mutation_colors = plt.cm.tab10(np.linspace(0, 1, len(mutations)))
    for idx, mutation in enumerate(mutations):
        mut_data = df_results[df_results['tkd'] == mutation]
        residuals = mut_data['actual_activity'] - mut_data['predicted_activity']
        axes[1, 0].scatter(mut_data['predicted_activity'], residuals,
                         label=mutation, alpha=0.6, s=20, c=[mutation_colors[idx]])
    axes[1, 0].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[1, 0].set_xlabel('Predicted Activity', fontsize=11)
    axes[1, 0].set_ylabel('Residuals (Actual - Predicted)', fontsize=11)
    axes[1, 0].set_title('Activity Residuals by Mutation', fontsize=12)
    axes[1, 0].legend(fontsize='small', loc='best')
    axes[1, 0].grid(True, alpha=0.3)

    for idx, mutation in enumerate(mutations):
        mut_data = df_results[df_results['tkd'] == mutation]
        residuals = mut_data['actual_docking'] - mut_data['predicted_docking']
        axes[1, 1].scatter(mut_data['predicted_docking'], residuals,
                         label=mutation, alpha=0.6, s=20, c=[mutation_colors[idx]])
    axes[1, 1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[1, 1].set_xlabel('Predicted Docking Score', fontsize=11)
    axes[1, 1].set_ylabel('Residuals (Actual - Predicted)', fontsize=11)
    axes[1, 1].set_title('Docking Residuals by Mutation', fontsize=12)
    axes[1, 1].legend(fontsize='small', loc='best')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    overall_plot_file = os.path.join(plots_dir, f'{model_name}_overall_combined.png')
    plt.savefig(overall_plot_file, dpi=300, bbox_inches='tight')
    print(f"\u2713 Overall combined plot saved to: {overall_plot_file}")
    plt.close()

    # ========================================================================
    # 5. CREATE INDIVIDUAL PLOTS FOR EACH MUTATION
    # ========================================================================
    print("\nGenerating individual mutation plots...")

    for mutation in mutations:
        mut_data = df_results[df_results['tkd'] == mutation]

        if len(mut_data) < 2:
            continue

        mut_metrics = metrics_df[metrics_df['Mutation'] == mutation].iloc[0]

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle(f'Mutation: {mutation} (n={len(mut_data)})', fontsize=14, fontweight='bold')

        axes[0, 0].scatter(mut_data['actual_activity'], mut_data['predicted_activity'],
                          alpha=0.6, s=50, edgecolors='k', linewidths=0.8, c='steelblue')
        min_val = min(mut_data['actual_activity'].min(), mut_data['predicted_activity'].min())
        max_val = max(mut_data['actual_activity'].max(), mut_data['predicted_activity'].max())
        axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
        axes[0, 0].set_xlabel('Actual Activity', fontsize=11)
        axes[0, 0].set_ylabel('Predicted Activity', fontsize=11)
        axes[0, 0].set_title(f'Activity Prediction\nMAE={mut_metrics["Activity_MAE"]:.4f}, RMSE={mut_metrics["Activity_RMSE"]:.4f}', fontsize=11)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].scatter(mut_data['actual_docking'], mut_data['predicted_docking'],
                          alpha=0.6, s=50, edgecolors='k', linewidths=0.8, c='darkorange')
        min_val = min(mut_data['actual_docking'].min(), mut_data['predicted_docking'].min())
        max_val = max(mut_data['actual_docking'].max(), mut_data['predicted_docking'].max())
        axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
        axes[0, 1].set_xlabel('Actual Docking Score', fontsize=11)
        axes[0, 1].set_ylabel('Predicted Docking Score', fontsize=11)
        axes[0, 1].set_title(f'Docking Prediction\nMAE={mut_metrics["Docking_MAE"]:.4f}, RMSE={mut_metrics["Docking_RMSE"]:.4f}', fontsize=11)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        residuals_act = mut_data['actual_activity'] - mut_data['predicted_activity']
        axes[1, 0].scatter(mut_data['predicted_activity'], residuals_act,
                          alpha=0.6, s=50, edgecolors='k', linewidths=0.8, c='steelblue')
        axes[1, 0].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[1, 0].set_xlabel('Predicted Activity', fontsize=11)
        axes[1, 0].set_ylabel('Residuals (Actual - Predicted)', fontsize=11)
        axes[1, 0].set_title('Activity Residuals', fontsize=11)
        axes[1, 0].grid(True, alpha=0.3)

        residuals_dock = mut_data['actual_docking'] - mut_data['predicted_docking']
        axes[1, 1].scatter(mut_data['predicted_docking'], residuals_dock,
                          alpha=0.6, s=50, edgecolors='k', linewidths=0.8, c='darkorange')
        axes[1, 1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[1, 1].set_xlabel('Predicted Docking Score', fontsize=11)
        axes[1, 1].set_ylabel('Residuals (Actual - Predicted)', fontsize=11)
        axes[1, 1].set_title('Docking Residuals', fontsize=11)
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        mutation_safe = mutation.replace('/', '_').replace('\\\\', '_')
        mutation_plot_file = os.path.join(plots_dir, f'{model_name}_mutation_{mutation_safe}.png')
        plt.savefig(mutation_plot_file, dpi=300, bbox_inches='tight')
        print(f"  \u2713 Saved: {mutation_safe}.png")
        plt.close()

    # ========================================================================
    # 6. CREATE CORRELATION PLOTS FOR EACH MUTATION
    # ========================================================================
    print("\nGenerating correlation plots...")

    corr_dir = os.path.join(plots_dir, 'correlations')
    os.makedirs(corr_dir, exist_ok=True)

    for mutation in mutations:
        mut_data = df_results[df_results['tkd'] == mutation]

        if len(mut_data) < 2:
            continue

        mut_metrics = metrics_df[metrics_df['Mutation'] == mutation].iloc[0]

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f'Correlations - {mutation} (n={len(mut_data)})', fontsize=14, fontweight='bold')

        axes[0].scatter(mut_data['actual_activity'], mut_data['predicted_activity'],
                       alpha=0.6, s=50, edgecolors='k', linewidths=0.8, c='steelblue')

        z = np.polyfit(mut_data['actual_activity'], mut_data['predicted_activity'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(mut_data['actual_activity'].min(), mut_data['actual_activity'].max(), 100)
        axes[0].plot(x_line, p(x_line), "g-", linewidth=2, label=f'Fit: y={z[0]:.3f}x+{z[1]:.3f}')

        min_val = min(mut_data['actual_activity'].min(), mut_data['predicted_activity'].min())
        max_val = max(mut_data['actual_activity'].max(), mut_data['predicted_activity'].max())
        axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')

        axes[0].set_xlabel('Actual Activity', fontsize=12)
        axes[0].set_ylabel('Predicted Activity', fontsize=12)
        axes[0].set_title(f'Activity\nPearson R = {mut_metrics["Activity_Pearson_R"]:.3f}, Spearman \u03c1 = {mut_metrics["Activity_Spearman_rho"]:.3f}',
                         fontsize=11)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].scatter(mut_data['actual_docking'], mut_data['predicted_docking'],
                       alpha=0.6, s=50, edgecolors='k', linewidths=0.8, c='darkorange')

        z = np.polyfit(mut_data['actual_docking'], mut_data['predicted_docking'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(mut_data['actual_docking'].min(), mut_data['actual_docking'].max(), 100)
        axes[1].plot(x_line, p(x_line), "g-", linewidth=2, label=f'Fit: y={z[0]:.3f}x+{z[1]:.3f}')

        min_val = min(mut_data['actual_docking'].min(), mut_data['predicted_docking'].min())
        max_val = max(mut_data['actual_docking'].max(), mut_data['predicted_docking'].max())
        axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')

        axes[1].set_xlabel('Actual Docking Score', fontsize=12)
        axes[1].set_ylabel('Predicted Docking Score', fontsize=12)
        axes[1].set_title(f'Docking\nPearson R = {mut_metrics["Docking_Pearson_R"]:.3f}, Spearman \u03c1 = {mut_metrics["Docking_Spearman_rho"]:.3f}',
                         fontsize=11)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        mutation_safe = mutation.replace('/', '_').replace('\\\\', '_')
        corr_plot_file = os.path.join(corr_dir, f'{model_name}_correlations_{mutation_safe}.png')
        plt.savefig(corr_plot_file, dpi=300, bbox_inches='tight')
        print(f"  \u2713 Saved: correlations_{mutation_safe}.png")
        plt.close()

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    print(f"\u2713 Metrics CSV: {metrics_csv_path}")
    print(f"\u2713 Overall plot: {overall_plot_file}")
    print(f"\u2713 Individual mutation plots: {plots_dir}")
    print(f"\u2713 Correlation plots: {corr_dir}")
    print("=" * 80)


def load_models_and_scalers(model_dir, device):
    """
    Load the fine-tuned GNN backbone + downstream head from the SINGLE checkpoint
    the training script saves (gnn_cross_attention_settransformer_simple_
    finetuned.pt -- contains gnn_state_dict, head_state_dict, and hyperparams
    together, since the two were fine-tuned jointly and are only meaningful loaded
    together), plus the physchem/target scalers.

    hp.get(key, default) is used (not hp[key]) for the four new STEP-6 hyperparams
    (num_pma_seeds, joint_dim, joint_num_heads, joint_key_dim) so that a checkpoint
    saved by an older training script without those keys still loads, falling back
    to this script's defaults -- though in practice those defaults MUST match
    whatever the checkpoint was actually trained with, or head_model.load_state_dict()
    below will fail with a shape mismatch (a loud crash, not silently wrong weights).

    Unlike the old (frozen-embedding) predict script, there is no separate
    'gin_vn_pretrained.pth' + Keras '.h5' pair to load, and feature_scalers.pkl
    contains only 'physchem_scaler' (no lig_gnn_scaler/mut_gnn_scaler -- see
    the training script's "FINE-TUNING UPGRADE NOTES").
    """
    print("\nLoading models and scalers...")

    checkpoint_path = os.path.join(model_dir, 'gnn_cross_attention_settransformer_simple_finetuned.pt')
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    hp = checkpoint['hyperparams']
    print(f"  Checkpoint hyperparams: {hp}")
    print(f"  Checkpoint epoch={checkpoint.get('epoch')}, val_loss={checkpoint.get('val_loss', float('nan')):.4f}")

    gnn_model = GINVirtualNet(num_layer=5, emb_dim=300, feat_dim=hp['gnn_dim'],
                               drop_ratio=0.1, jk_mode='concat')
    gnn_model.load_state_dict(checkpoint['gnn_state_dict'])
    gnn_model.to(device)
    gnn_model.eval()
    print(f"  \u2713 Loaded fine-tuned GNN backbone from {checkpoint_path}")

    head_model = GNNCrossAttentionSetTransformerHead(
        gnn_dim=hp['gnn_dim'], n_sites=hp['n_sites'], physchem_dim=hp['physchem_dim'],
        fusion_dim=hp.get('fusion_dim', 256),
        physchem_proj_dim=hp.get('physchem_proj_dim', 64),
        num_heads=hp.get('num_heads', 4),
        key_dim=hp.get('key_dim', 32),
        dropout=hp.get('dropout', 0.15))
    head_model.load_state_dict(checkpoint['head_state_dict'])
    head_model.to(device)
    head_model.eval()
    print(f"  \u2713 Loaded fine-tuned simplified multimodal head (fusion -> site self-attention -> learned pooling) "
          f"from {checkpoint_path}")

    with open(os.path.join(model_dir, 'feature_scalers.pkl'), 'rb') as f:
        feature_scalers = pickle.load(f)
    print(f"  \u2713 Loaded feature scalers: {list(feature_scalers.keys())}")

    with open(os.path.join(model_dir, 'y_scalers.pkl'), 'rb') as f:
        y_scalers = pickle.load(f)
    y_scaler1 = y_scalers['y_scaler1']
    y_scaler2 = y_scalers['y_scaler2']
    print("  \u2713 Loaded y scalers")

    return gnn_model, head_model, feature_scalers, y_scaler1, y_scaler2, hp


def make_predictions(input_csv, model_dir='.', output_dir='.'):
    """
    Make predictions using the fine-tuned GNN cross-attention + gated-physchem +
    Set-Transformer model (base1d_512d).

    Parameters
    ----------
    input_csv : str
        Path to input CSV file with 'smiles' and 'tkd' columns.
        Optionally can include the 8 mutation-site SMILES columns, IN MECHANISTIC
        ORDER (see MUTATION_SITES above):
        - smiles_full_sequence_egfr_manual
        - smiles_sequence_atp_ pocket
        - smiles_sequence_p_loop_constant
        - smiles_sequence_c_helix_constant
        - smiles_sequence_19_deletions
        - smiles_sequence_hinge_loop_t790m_c797s
        - smiles_sequence_a_loop_dfg
        - smiles_sequence_hrd_constant
        If these columns are absent, the mutant's 8-site profile is looked up
        from 'mutation_profiles.csv' (saved by the training script) by 'tkd'.
    model_dir : str
        Directory containing model files (default: current directory)
    output_dir : str
        Directory to save prediction outputs (default: current directory)

    Returns
    -------
    df_results : pd.DataFrame
        DataFrame with predictions
    """
    print(f"Loading prediction data from: {input_csv}")
    df_pred = pd.read_csv(input_csv)
    df_pred.columns = df_pred.columns.str.strip()

    required_cols = ['smiles', 'tkd']
    if not all(col in df_pred.columns for col in required_cols):
        raise ValueError(f"Input CSV must contain 'smiles' and 'tkd' columns. Found: {df_pred.columns.tolist()}")

    has_ground_truth = 'standard value' in df_pred.columns and 'dock' in df_pred.columns

    mutation_site_cols = [col for _, col in MUTATION_SITES]
    has_mutation_sites = all(col in df_pred.columns for col in mutation_site_cols)

    if has_mutation_sites:
        print("Using mutation site SMILES from input CSV")
        unique_mutation_profiles = df_pred[mutation_site_cols + ['tkd']].drop_duplicates(subset=['tkd']).reset_index(drop=True)
    else:
        print("Mutation site SMILES not in input CSV. Loading from saved profiles...")
        try:
            unique_mutation_profiles = pd.read_csv(os.path.join(model_dir, 'mutation_profiles.csv'))
            unique_mutation_profiles.columns = unique_mutation_profiles.columns.str.strip()
            print(f"Loaded {len(unique_mutation_profiles)} mutation profiles")
        except FileNotFoundError:
            print("\n" + "="*80)
            print("ERROR: Mutation site SMILES data not found!")
            print("="*80)
            print("\nYou have two options:")
            print("\n1. Include mutation site SMILES columns in your input CSV, in mechanistic order:")
            for _, col in MUTATION_SITES:
                print(f"   - {col}")
            print("\n2. Ensure 'mutation_profiles.csv' exists in model directory")
            print("   (This is saved during training)")
            print("="*80)
            return None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    try:
        gnn_model, head_model, feature_scalers, y_scaler1, y_scaler2, hp = load_models_and_scalers(model_dir, device)
    except FileNotFoundError as e:
        print(f"Error loading models/scalers: {e}")
        return None

    physchem_scaler = feature_scalers['physchem_scaler']
    n_sites = hp['n_sites']
    if n_sites != len(MUTATION_SITES):
        logger.warning(f"Checkpoint hyperparams say n_sites={n_sites} but this script's "
                        f"MUTATION_SITES has {len(MUTATION_SITES)} entries -- these MUST match "
                        f"the training run's mutation_sites list, or predictions will be wrong.")

    print(f"Total prediction samples: {len(df_pred)}")
    unique_mutations = df_pred['tkd'].unique()
    print(f"Found {len(unique_mutations)} unique mutations to process")

    # ---- Graph-structure cache: weight-independent, so training's cache is
    # directly reusable -- every mutation-site SMILES (and any ligand SMILES
    # that also appeared during training) typically hits it already. Build it
    # ONCE up front for everything this whole prediction run will need (mirrors
    # the training script's Stage-1 structure), rather than incrementally per
    # mutation group. ----
    print("\nBuilding SMILES -> graph-structure cache for this prediction run...")
    graph_cache_path = os.path.join(model_dir, 'smiles_graph_structure_cache.pkl')
    graph_cache = load_graph_cache(graph_cache_path)

    all_needed_smiles = set(df_pred['smiles'].astype(str))
    for mutation_name in unique_mutations:
        mut_profile = unique_mutation_profiles[unique_mutation_profiles['tkd'] == mutation_name]
        if len(mut_profile) == 0:
            continue
        mut_profile = mut_profile.iloc[0]
        for _, col in MUTATION_SITES:
            val = mut_profile[col]
            if pd.notna(val) and val != '':
                all_needed_smiles.add(str(val))

    graph_cache, bad_smiles = build_graph_cache(sorted(all_needed_smiles), existing_cache=graph_cache)
    save_graph_cache(graph_cache, graph_cache_path)
    if bad_smiles:
        print(f"  {len(bad_smiles)} SMILES failed graph parsing -- affected rows will be dropped below")

    predict_batch_size = 64
    all_results = []

    for mutation_name in unique_mutations:
        print(f"\nProcessing mutation: {mutation_name}")
        mut_data = df_pred[df_pred['tkd'] == mutation_name].reset_index(drop=True)
        print(f"  Compounds: {len(mut_data)}")

        mut_profile = unique_mutation_profiles[unique_mutation_profiles['tkd'] == mutation_name]
        if len(mut_profile) == 0:
            print(f"  Warning: Mutation '{mutation_name}' not found in training data. Skipping.")
            continue
        mut_profile = mut_profile.iloc[0]

        mut_site_smiles = [mut_profile[col] for _, col in MUTATION_SITES]
        if any(pd.isna(smi) or smi == '' for smi in mut_site_smiles):
            print(f"  Warning: Missing mutation site SMILES for '{mutation_name}'. Skipping.")
            continue
        mut_site_smiles = [str(s) for s in mut_site_smiles]

        if any(s in bad_smiles for s in mut_site_smiles):
            print(f"  Warning: A mutation-site SMILES for '{mutation_name}' failed graph parsing. Skipping.")
            continue

        # ---- Physchem features for each of the 8 sites, IN MECHANISTIC ORDER,
        # reusing the SAME generate_hierarchical_features() the training script
        # calls (scoped to this one mutation's compounds) -- this is what
        # guarantees train/predict feature parity, rather than re-deriving the
        # same logic by hand a second time. ----
        all_feature_dicts = []
        all_valid_indices = []
        for site_idx, (site_name, _) in enumerate(MUTATION_SITES):
            site_smiles_series = pd.Series([mut_site_smiles[site_idx]] * len(mut_data))
            feature_dict = generate_hierarchical_features(mut_data['smiles'], site_smiles_series)
            all_feature_dicts.append(feature_dict)
            all_valid_indices.append(set(feature_dict['valid_indices']))

        # Also exclude any compound whose own SMILES failed graph parsing
        lig_str_all = mut_data['smiles'].astype(str)
        graph_ok_idx = set(i for i, s in enumerate(lig_str_all) if s not in bad_smiles)
        all_valid_indices.append(graph_ok_idx)

        common_valid_indices = sorted(set.intersection(*all_valid_indices))
        if len(common_valid_indices) == 0:
            print(f"  Warning: No valid compounds for '{mutation_name}'. Skipping.")
            continue

        # Filter each site's descriptor groups down to common_valid_indices
        # (mirrors the training script's Stage-1 filtering step exactly)
        for i, feature_dict in enumerate(all_feature_dicts):
            site_valid_idx = feature_dict['valid_indices']
            mask = np.isin(site_valid_idx, common_valid_indices)
            for key in PHYSCHEM_GROUP_KEYS:
                all_feature_dicts[i][key] = feature_dict[key][mask]

        physchem_per_site = [combine_physchem_features(all_feature_dicts[site_idx])
                              for site_idx in range(n_sites)]
        physchem_stack = np.stack(physchem_per_site, axis=1)             # (n_valid, n_sites, D)
        # scaler=physchem_scaler is provided -> scale_site_stack only TRANSFORMS
        # (using the stats fit on TRAIN rows during training), never re-fits.
        physchem_scaled, _ = scale_site_stack(physchem_stack, scaler=physchem_scaler)
        physchem_tensor_full = torch.tensor(physchem_scaled, dtype=torch.float32, device=device)

        compound_smiles_valid = mut_data['smiles'].astype(str).iloc[common_valid_indices].tolist()

        # ---- GNN embeddings + prediction, gradient-free, chunked for memory
        # safety on large mutation groups (see embed_batch docstring: dedup
        # means the n_sites mutant-side forward passes cost effectively nothing
        # extra per chunk, since all rows in a chunk share the same mutant). ----
        gnn_model.eval()
        head_model.eval()
        activity_chunks, docking_chunks = [], []
        with torch.no_grad():
            for batch_pos in iterate_batches(np.arange(len(common_valid_indices)),
                                              predict_batch_size, shuffle=False):
                batch_compound_smiles = [compound_smiles_valid[p] for p in batch_pos]
                lig_emb = embed_batch(batch_compound_smiles, graph_cache, gnn_model, device, training=False)
                mut_emb_stack = torch.stack(
                    [embed_batch([site_smi] * len(batch_compound_smiles), graph_cache, gnn_model, device, training=False)
                     for site_smi in mut_site_smiles],
                    dim=1)                                                # (b, n_sites, gnn_dim)
                physchem_batch = physchem_tensor_full[batch_pos]
                activity_pred, docking_pred = head_model(lig_emb, mut_emb_stack, physchem_batch)
                activity_chunks.append(activity_pred.cpu().numpy())
                docking_chunks.append(docking_pred.cpu().numpy())

        activity_pred_all = np.concatenate(activity_chunks, axis=0)
        docking_pred_all = np.concatenate(docking_chunks, axis=0)

        # Inverse transform: activity was log1p'd then StandardScaler'd during
        # training (see main()), so undo in the same order; docking only went
        # through StandardScaler.
        pred_activity = np.expm1(y_scaler1.inverse_transform(activity_pred_all.reshape(-1, 1)).flatten())
        pred_docking = y_scaler2.inverse_transform(docking_pred_all.reshape(-1, 1)).flatten()

        for i, idx in enumerate(common_valid_indices):
            row = mut_data.iloc[idx]
            res = {
                'smiles': row['smiles'],
                'tkd': row['tkd'],
                'predicted_activity': pred_activity[i],
                'predicted_docking': pred_docking[i]
            }
            if has_ground_truth:
                res['actual_activity'] = row['standard value']
                res['actual_docking'] = row['dock']
            for col in df_pred.columns:
                if col not in res and col not in ['smiles', 'tkd', 'standard value', 'dock']:
                    res[col] = row[col]
            all_results.append(res)

        print(f"  \u2713 Predicted {len(common_valid_indices)} compounds for {mutation_name}")

    if not all_results:
        print("\n" + "="*80)
        print("ERROR: No valid predictions generated.")
        print("="*80)
        return None

    df_results = pd.DataFrame(all_results)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_path = os.path.join(output_dir, 'predictions_gnn_cross_attention_settransformer_simple.csv')
    df_results.to_csv(output_path, index=False)
    print(f"\n\u2713 Predictions saved to: {output_path}")
    print(f"\u2713 Total predictions: {len(df_results)}")
    print(f"\u2713 Mutations processed: {df_results['tkd'].nunique()}")

    if has_ground_truth and len(df_results) > 0:
        evaluate_and_plot(df_results, output_dir, 'gnn_cross_attention_settransformer_simple')

    return df_results

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Predictions for the fine-tuned GNN + pre-attention multimodal fusion + site self-attention model (base1d_simple)')
    parser.add_argument('--input', type=str, required=True, help='Input CSV file')
    parser.add_argument('--model_dir', type=str, default='.', help='Model directory')
    parser.add_argument('--output_dir', type=str, default='.', help='Output directory')

    args = parser.parse_args()

    results = make_predictions(args.input, args.model_dir, args.output_dir)

    if results is not None:
        print(f"\n\u2713 Complete! Total predictions: {len(results)}")
    else:
        sys.exit(1)

# to run script:
# python predict_adv_physchem_gnn_base1d_simple_working_random_split.py --input validated_july_2026_testset_valid_tki.csv --model_dir . --output_dir ./prediction_output
