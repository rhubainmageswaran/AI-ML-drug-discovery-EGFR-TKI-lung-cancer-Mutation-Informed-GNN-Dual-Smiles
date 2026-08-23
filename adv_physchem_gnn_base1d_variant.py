
import os
import sys
import copy
import pickle
from functools import lru_cache
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from loguru import logger
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
from numpy.linalg import norm

# RDKit imports
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, MolSurf, GraphDescriptors, Fragments
from rdkit.Chem import AllChem, rdMolDescriptors
from rdkit import DataStructs
from rdkit import RDLogger

# ---- PyTorch / torch_geometric imports ----
# NOTE (base1d fine-tuning upgrade): TensorFlow/Keras is no longer imported at all.
# The old pipeline computed GNN embeddings once under torch.no_grad() (PyTorch) and
# then fed the *detached* numpy result into a separately-trained Keras model -- two
# different autograd frameworks glued together at a numpy boundary, which makes it
# IMPOSSIBLE for gradients to flow from the activity/docking loss back into the GNN's
# atom/bond embeddings or message-passing weights, no matter how the Keras side is
# built. True fine-tuning requires one differentiable graph end-to-end, so the entire
# downstream head (cross-attention + gated physchem fusion + Set-Transformer SAB/PMA)
# has been ported to PyTorch (see GNNCrossAttentionSetTransformerHead below) and is
# now trained jointly with the GNN backbone in a single optimizer.step() per batch.
# See the "FINE-TUNING UPGRADE NOTES" block below for the full rationale.
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
    print("Running in FALLBACK mode with simple MLP embeddings")
    print("!"*80 + "\n")
    # Fallback placeholders
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
logger.add(sys.stderr, level="DEBUG", format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")
logger.add("adv_physchem_gnn_{time}.txt", rotation="500 MB", retention="10 days", 
           compression="zip", level="DEBUG", 
           format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}")



print("="*80)
print("GIN-VN-GNN SIMPLIFIED MULTIMODAL ATTENTION MODEL (base1d_simple, FINE-TUNED)")
print("GIN backbone: GINEConv(train-eps) + Virtual Node + Jumping-Knowledge")
print("+ OGB-style 9-feature atoms / 4-feature bonds + atomic physchem context")
print("Independent ligand/site/physchem encoders + pre-attention multimodal fusion")
print("+ One 256-d site self-attention block + learned attention pooling")
print("+ Physchem stays independent until ligand + site-GNN + physchem fusion")
print("8-site mechanistic mutation order: FULL -> ATP_POCKET -> P_LOOP -> C_HELIX ->")
print("  DEL19 -> HINGE_LOOP(T790M/C797S) -> DFG_A_LOOP -> HRD_CAT")
print("GNN backbone is FINE-TUNED end-to-end with the downstream head (single PyTorch")
print("autograd graph, single optimizer) -- no frozen/precomputed embeddings.")
print("="*80)


# =============================================================================
# GNN UPGRADE NOTES (base1b -> base1c)
# =============================================================================
# base1b used the original MolCLR GINEConv: 2 categorical atom features
# (atomic number, chirality) and 2 categorical bond features (bond type,
# bond direction), mean-pooled after 5 plain GIN layers.
#
# This version upgrades the GIN family to the architecture that is currently
# the established strong baseline for molecular graph learning (the "GIN+VN"
# configuration from Hu et al. 2020, "Open Graph Benchmark", used as the
# reference GIN baseline on ogbg-mol* leaderboards, and the pre-training
# scheme of Hu et al. 2019/2020 "Strategies for Pre-training GNNs"):
#
#   1. Richer, chemically-meaningful atom/bond features (OGB-style):
#        Atoms (9 categorical): atomic num, chirality, degree, formal charge,
#        #H, #radical electrons, hybridization, aromaticity, ring membership.
#        Bonds (4 categorical): bond type, stereo, conjugation, ring membership.
#      These map directly onto the physicochemical axes already used by the
#      hierarchical branch (electrostatics -> formal charge/partial charge,
#      rigidity -> ring/hybridization, pi-stacking -> aromaticity/conjugation),
#      so the GNN node/edge features now carry the same physical meaning as
#      the hand-crafted descriptors instead of two opaque integer codes.
#   2. Atom-level physicochemical context features (continuous): Gasteiger
#      partial charge, Crippen logP/MR atomic contributions and TPSA atomic
#      contribution are concatenated into the atom encoder. This gives every
#      node direct access to the same electrostatic/hydrophobic/polar-surface
#      signal that the hierarchical branch aggregates at the whole-molecule
#      level (MaxPartialCharge, Crippen MolLogP/MolMR, TPSA), so the GNN and
#      hierarchical branches are complementary rather than redundant.
#   3. Trainable-epsilon GINEConv (Xu et al. 2019 GIN, generalized epsilon).
#   4. Virtual Node (GIN-VN): a per-graph global token that is broadcast to
#      every atom and refreshed by sum-pooling atom states each layer -- the
#      standard, benchmark-proven way to give a GIN access to whole-molecule
#      context without extra message-passing hops.
#   5. Jumping-Knowledge concatenation (Xu et al. 2018) over all layer depths
#      before graph readout, so both short-range (early layers) and
#      long-range (later layers) substructure information reach the pooled
#      embedding, instead of only the last layer's over-smoothed output.
#   6. Mean+Max graph pooling concatenation for a more discriminative readout
#      than mean pooling alone.
#
# NOTE ON PRETRAINED WEIGHTS: the original MolCLR checkpoint
# (checkpoints/molclr_pretrained.pth) was trained with the old 2-feature atom
# / 2-feature bond encoding and will NOT shape-match this richer encoder.
# load_pretrained() below only copies parameters whose name AND shape match,
# and skips (with a warning) anything that doesn't -- so this script degrades
# gracefully to random initialization for the encoder layers rather than
# crashing, exactly like a fresh model would if no checkpoint is supplied.
# A partial/random init is now just the STARTING POINT for fine-tuning (see below)
# rather than the final, frozen embedding source.
# =============================================================================

# =============================================================================
# FUSION UPGRADE NOTES (base1c -> base1d)
# =============================================================================
# The GIN-VN backbone and the ligand<->mutant-site cross-attention + gated-
# physchem fusion (STEPS 1-4 of the downstream head) are architecturally
# UNCHANGED from base1c. Two things changed:
#
#   1. FUSION HEAD: the BiLSTM(128->64) + BiGRU(128->64) "RNN" stage that
#      consumed the fused site vectors as a sequence has been replaced with
#      a Set-Transformer-style block (Lee et al. 2019, "Set Transformer: A
#      Framework for Attention-based Permutation-Invariant Neural Networks"):
#        - a Set Attention Block (SAB): one more self-attention layer, this
#          time letting the 8 sites attend to EACH OTHER (something the old
#          model never did -- STEP 2's cross-attention is site<->ligand only,
#          sites never exchange information directly), so e.g. a P-loop
#          change and a DFG-motif change can be recognised as jointly
#          conferring resistance instead of only being combined implicitly
#          through the shared ligand token.
#        - Pooling by Multihead Attention (PMA): a small number of trainable
#          seed/query vectors attend over the 8 (now site-aware) tokens to
#          produce a FIXED-SIZE, PERMUTATION-INVARIANT pooled vector.
#      This is the architecturally correct replacement for "flatten an 8-step
#      RNN's final hidden state": the mutation sites are an unordered set of
#      structural motifs, not a temporal sequence, and a BiLSTM/BiGRU silently
#      imposes whatever order the `mutation_sites` Python list happens to use
#      -- an order the rest of the architecture (the joint, symmetric cross-
#      attention in STEP 2) never assumed in the first place. The mechanistic
#      order (see "MUTATION-SEQUENCE UPGRADE NOTES" below) is still used to
#      build `mutation_sites` for readability/consistency with base1c, but the
#      SAB+PMA head's pooled output is mathematically invariant to that order.
#      As a side effect the new head also has substantially fewer parameters
#      than the two stacked bidirectional RNN stacks it replaces.
#
#   2. TRAIN/VAL SPLIT: `model.fit(..., validation_split=0.2)` sliced off the
#      last 20% of the (index-ordered) array -- not a random split, and
#      certainly not a chemistry-aware one. This is now replaced with an
#      explicit reproducible RANDOM TRAIN/VALIDATION SPLIT (`random_split_indices`)
#      using a seeded permutation of the post-filtered rows. Related ligand
#      scaffolds are therefore allowed to appear in both train and validation.
#      The physchem scaler is still fit on TRAIN INDICES ONLY and applied
#      transform-only to validation -- fitting a scaler on the full dataset
#      before splitting would leak validation statistics into training.
# =============================================================================

# =============================================================================
# FINE-TUNING UPGRADE NOTES (frozen-embedding base1d -> fine-tuned base1d)
# =============================================================================
# Previously (see the old TODO #1, and the old assumption #2 that a fine-tuning
# pass "is sufficient... for initial testing" without it): get_gnn_embeddings()
# ran the GIN-VN backbone under torch.no_grad() with model.eval(), cached the
# resulting embedding for every unique SMILES to disk (namespaced by a hash of
# the backbone's weights), and the downstream fusion model was a SEPARATE Keras
# graph consuming those already-detached numpy embeddings. That means the GNN's
# atom encoder, GINEConvV2 message-passing layers, and virtual-node MLPs never
# received a gradient from the activity/docking loss -- only the Keras head was
# ever trained. That whole two-framework split is removed:
#   1. The downstream head (STEPs 1-5 below) is now a plain PyTorch nn.Module
#      (GNNCrossAttentionSetTransformerHead) instead of a Keras Model, so it
#      lives in the SAME autograd graph as the GIN-VN backbone.
#   2. get_gnn_embeddings()/the weight-fingerprinted embedding-cache functions
#      are replaced by build_graph_cache()/embed_batch(): a cache of PARSED
#      GRAPH STRUCTURE (SMILES -> torch_geometric.data.Data), which is a fixed,
#      weight-independent function of the SMILES string and stays valid for the
#      whole run, plus a batched forward pass that runs the GNN WITH gradients
#      enabled every single training step. The embedding itself is therefore
#      recomputed (not cached) every forward pass, exactly as fine-tuning requires.
#   3. main() now runs an explicit PyTorch training loop (forward -> composite
#      MSE loss -> backward -> one Adam step over BOTH the GNN backbone's and
#      the head's parameters) instead of a single Keras model.fit() call, with
#      manually-implemented early stopping / best-checkpoint restore mirroring
#      Keras's EarlyStopping(restore_best_weights=True) + ModelCheckpoint, on
#      top of the random-split train/val arrays from the Fusion Upgrade above.
#   4. The external StandardScaler previously fit on the (frozen) ligand/mutant
#      GNN embeddings (lig_gnn_scaler / mut_gnn_scaler) is removed: a scaler
#      fit once on the embedding distribution before any gradient step would go
#      stale the moment the backbone weights move, and the head already applies
#      LayerNorm immediately after each embedding's projection (STEP 1), which
#      is the standard way to normalize a fine-tuned backbone's output. The
#      physchem descriptor scaler (still fit on TRAIN rows only) is untouched --
#      those are static RDKit features, unaffected by GNN fine-tuning.
#   5. Because forward passes are recomputed every step, this is meaningfully
#      more compute per epoch than the old frozen-embedding pipeline. The one
#      cost-saving carried over from the old cache is dedup: embed_batch()
#      still forwards each DISTINCT SMILES in a minibatch only once (mutation-
#      site SMILES repeat constantly -- every row sharing a `tkd` mutant label
#      has byte-identical SMILES across all 8 mutation-site columns) and
#      index_selects the shared embedding into every row that needs it, which
#      correctly accumulates gradients from every one of those rows back into
#      that single shared forward pass.
# =============================================================================

# =============================================================================
# SIMPLIFIED HEAD NOTES (base1d_1024d -> base1d_simple)
# =============================================================================
# The downstream head is intentionally reduced to one coherent pathway:
#   independent ligand GNN encoder
#   independent site GNN encoder
#   independent site physicochemical encoder
#          -> explicit fusion (ligand + site GNN + physchem)
#          -> one site self-attention block
#          -> learned attention pooling
#          -> compact multitask MLP
#
# No raw-skip 1600-d representation, bidirectional cross-attention, separate
# SAB/PMA stack, or parallel 1024-d raw-GNN branch is used. The physchem branch
# remains independent until the single multimodal fusion immediately before
# attention. This keeps the architecture interpretable and reduces duplicated
# processing of the same ligand/site GNN information.
# =============================================================================

# =============================================================================
# MUTATION-SEQUENCE UPGRADE NOTES (6 sites -> 8-site mechanistic order)
# =============================================================================
# The mutant-site columns now follow the mechanistic order in which an EGFR
# tyrosine-kinase-domain inhibitor actually encounters/depends on each region
# of the kinase domain, rather than the previous ad hoc 6-column layout (which
# also merged the P-loop and hinge loop into one column):
#     FULL_SMILES -> ATP_POCKET -> P_LOOP -> C_HELIX -> DEL19 ->
#     HINGE_LOOP(T790M/C797S) -> DFG_A_LOOP -> HRD_CAT
# This is now 8 sites (n_sites=8 everywhere below), not 6. See the "Load Data"
# section for the exact df_train column names. As noted in the Fusion Upgrade
# section above, this order matters for how `mutation_sites` is built and for
# comparability with base1c, but is NOT required by the SAB+PMA head itself,
# which pools the 8 sites as a permutation-invariant set.
# =============================================================================

#Assumptions:
#1. GNN embeddings (ligand + 8 mutant sites) are the primary representation; the priority-gated
#   hierarchical cascade over hand-crafted descriptors has been removed.
#2. The GIN-VN backbone is fine-tuned jointly with the downstream head (see "FINE-TUNING
#   UPGRADE NOTES" above) -- there is no frozen/precomputed embedding step anymore.
#3. A joint bidirectional cross-attention block (ligand token <-> all 8 mutant-site tokens,
#   shared projection weights across sites) enriches the GNN embeddings more effectively than
#   concatenation alone, and produces the 8 fused site tokens the Set-Transformer head consumes
#   as an unordered SET (not a sequence -- see "Fusion Upgrade Notes" above).
#4. The 8 physicochemical descriptor groups still add signal on top of the GNN, so they are kept,
#   but fused in via a learned sigmoid gate rather than the old priority-gated cascade -- the
#   gated physchem vector is concatenated onto the (already fully-formed) fused GNN site vector,
#   so physchem can only ever be an additive supplement and can never suppress the GNN pathway.
#5. The whole pipeline (GIN-VN backbone -> projection -> cross-attention -> gated physchem fusion
#   -> Set-Transformer SAB + PMA pooling) is one end-to-end trainable PyTorch nn.Module, trained
#   in a single manual training loop, rather than the old two-stage flow of 6 independently-
#   trained per-site models followed by a separately-trained sequence model (and rather than the
#   frozen-GNN + separate-Keras-model split that immediately preceded this version).
#6. The 8 mutation sites (full pocket, ATP pocket, P-loop, C-helix, exon-19 deletions, hinge loop
#   T790M/C797S, DFG/activation loop, HRD catalytic motif) are treated as a permutation-invariant
#   SET rather than an ordered sequence. Train/validation assignment is a reproducible random row
#   split across the complete post-filtered dataset.

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

    `fit_indices`: if provided, the scaler is FIT only on rows
    arr3d[fit_indices] (i.e. the random-split TRAIN rows) but still applied
    (transform) to every row in arr3d, train and validation alike. This is what
    makes the random split leak-free: fitting the scaler on the full dataset
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


# =============================================================================
# Random train/validation split
# =============================================================================
def random_split_indices(n_samples, val_frac=0.2, seed=42):
    """
    Reproducible random train/validation split.

    The split is performed on row positions after all samples have passed the
    common-validity checks. A seeded permutation keeps the split deterministic
    while allowing chemically related ligands/scaffolds to appear in both
    training and validation, as requested.

    Returns (train_indices, val_indices) as sorted numpy arrays of positional
    indices into the post-filtered dataset.
    """
    if not 0.0 < val_frac < 1.0:
        raise ValueError(f"val_frac must be between 0 and 1; got {val_frac}")
    if n_samples < 2:
        raise ValueError("At least 2 samples are required for a train/validation split.")

    rng = np.random.RandomState(seed)
    shuffled = rng.permutation(n_samples)
    n_val = max(1, int(round(val_frac * n_samples)))
    n_val = min(n_val, n_samples - 1)

    val_indices = np.sort(shuffled[:n_val])
    train_indices = np.sort(shuffled[n_val:])

    logger.info('=' * 80)
    logger.info('RANDOM TRAIN/VALIDATION SPLIT')
    logger.info(f'  Total rows           : {n_samples}')
    logger.info(f'  Train rows           : {len(train_indices)} ({len(train_indices)/n_samples*100:.1f}%)')
    logger.info(f'  Validation rows      : {len(val_indices)} ({len(val_indices)/n_samples*100:.1f}%)')
    logger.info(f'  Seed={seed}, val_frac={val_frac}')
    logger.info('  Ligand scaffolds are NOT constrained to one split.')
    logger.info('=' * 80)

    return train_indices, val_indices


# =============================================================================
# PyTorch downstream head (replaces the old Keras TileToSites/PMASeedVectors
# custom layers and build_gnn_cross_attention_settransformer_model() -- see
# "FINE-TUNING UPGRADE NOTES" near the top of this file). Broadcasting the
# ligand's global context onto every mutant site is just a `.expand()` call in
# PyTorch, and a trainable "seed vector" is just an nn.Parameter, so neither
# custom layer is needed the way Keras required one for safe save/reload.
# =============================================================================

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


# =============================================================================
# Load Data
# =============================================================================
# Mechanistic reaction order for EGFR (see "MUTATION-SEQUENCE UPGRADE NOTES" above):
#   full -> ATP pocket -> P-loop -> C-helix -> 19-deletions -> hinge loop (T790M/C797S)
#   -> A-loop DFG -> HRD catalytic motif
print("\nLoading datasets...")
script_dir = os.path.dirname(os.path.abspath(__file__))
#df_train = pd.read_csv(os.path.join(script_dir, 'trainset_valid_n_nonvalid_tki.csv'))
df_train = pd.read_csv("/mnt/c/back_up_drive/d/Publications/project_insilico_human/project_physicochem_activity/main_july_2026_new_mechanistic_sequence_scripts/validated_july_2026_trainset_valid_n_nonvalid_tki.csv")

df_train.columns = df_train.columns.str.strip()

# Substructure smiles for feature capture of mutation protein -- 8-site mechanistic order
ligand_smiles = df_train['smiles']
full_smiles = df_train['smiles_full_sequence_egfr_manual']
mutation_smiles = df_train['smiles_sequence_atp_ pocket']
mut_p_loop = df_train['smiles_sequence_p_loop_constant']
mut_helix = df_train['smiles_sequence_c_helix_constant']
mut_19_del = df_train['smiles_sequence_19_deletions']
mut_hinge_loop = df_train['smiles_sequence_hinge_loop_t790m_c797s']
mut_dfg_a_loop = df_train['smiles_sequence_a_loop_dfg']
mut_hrd_cat = df_train['smiles_sequence_hrd_constant']

mutant = df_train['tkd']

activity_values = df_train['standard value']  # y_train1
docking_values = df_train['dock']              # y_train2

print(f"Training samples: {len(ligand_smiles)}")

# Data Validation
print("\n" + "="*80)
print("DATA VALIDATION")
print("="*80)

mutation_site_columns = {
    'Full_SMILES': full_smiles,
    'ATP_POCKET': mutation_smiles,
    'P_LOOP': mut_p_loop,
    'C_HELIX': mut_helix,
    'DEL19': mut_19_del,
    'HINGE_LOOP_T790M_C797S': mut_hinge_loop,
    'DFG_A_LOOP': mut_dfg_a_loop,
    'HRD_CAT': mut_hrd_cat
}

for site_name, site_series in mutation_site_columns.items():
    missing_count = site_series.isna().sum()
    print(f"  {site_name:24s}: {missing_count:4d} missing SMILES ({missing_count/len(site_series)*100:.2f}%)")

# Filter to valid samples
valid_mask = ~(
    ligand_smiles.isna() |
    full_smiles.isna() |
    mutation_smiles.isna() |
    mut_p_loop.isna() |
    mut_helix.isna() |
    mut_19_del.isna() |
    mut_hinge_loop.isna() |
    mut_dfg_a_loop.isna() |
    mut_hrd_cat.isna() |
    activity_values.isna() |
    docking_values.isna()
)

valid_sample_count = valid_mask.sum()
print(f"\n✓ Valid samples: {valid_sample_count}/{len(df_train)} ({valid_sample_count/len(df_train)*100:.2f}%)")

if valid_sample_count == 0:
    print("\n✗ ERROR: No complete samples found!")
    sys.exit(1)

df_train_valid = df_train[valid_mask].copy().reset_index(drop=True)

ligand_smiles_valid = df_train_valid['smiles']
full_smiles_valid = df_train_valid['smiles_full_sequence_egfr_manual']
mutation_smiles_valid = df_train_valid['smiles_sequence_atp_ pocket']
mut_p_loop_valid = df_train_valid['smiles_sequence_p_loop_constant']
mut_helix_valid = df_train_valid['smiles_sequence_c_helix_constant']
mut_19_del_valid = df_train_valid['smiles_sequence_19_deletions']
mut_hinge_loop_valid = df_train_valid['smiles_sequence_hinge_loop_t790m_c797s']
mut_dfg_a_loop_valid = df_train_valid['smiles_sequence_a_loop_dfg']
mut_hrd_cat_valid = df_train_valid['smiles_sequence_hrd_constant']
mutant_valid = df_train_valid['tkd']

activity_values_valid = df_train_valid['standard value']
activity_values2_valid = df_train_valid['dock'].values

# Create unique mutation profiles (8-site mechanistic order + mutant label)
mutation_profile_columns = [
    'smiles_full_sequence_egfr_manual',
    'smiles_sequence_atp_ pocket',
    'smiles_sequence_p_loop_constant',
    'smiles_sequence_c_helix_constant',
    'smiles_sequence_19_deletions',
    'smiles_sequence_hinge_loop_t790m_c797s',
    'smiles_sequence_a_loop_dfg',
    'smiles_sequence_hrd_constant',
    'tkd'
]

unique_mutation_profiles = df_train_valid[mutation_profile_columns].drop_duplicates(subset=['tkd']).reset_index(drop=True)
print(f"Unique mutation profiles: {len(unique_mutation_profiles)}")


# ============================================================================
# MAIN WORKFLOW
# ============================================================================

def main():
    print("\n" + "="*80)
    print("STAGE 0: INITIALIZE GNN BACKBONE (will be FINE-TUNED, not frozen)")
    print("="*80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    gnn_model = GINVirtualNet(num_layer=5, emb_dim=300, feat_dim=512,
                               drop_ratio=0.1, jk_mode='concat')
    gnn_model.to(device)
    print("OK GINVirtualNet (GIN + VirtualNode + JumpingKnowledge) model initialized")

    # NOTE: the original MolCLR checkpoint was trained against a 2-feature atom /
    # 2-feature bond encoder and will not shape-match this richer encoder --
    # load_pretrained() only copies tensors whose name AND shape match, so
    # mismatched layers are skipped with a log message instead of crashing. This
    # is now purely an INITIALIZATION choice (a warm start for fine-tuning) --
    # the loaded weights are no longer treated as fixed/frozen afterward.
    pretrained_path = "checkpoints/gin_vn_pretrained.pth"
    if os.path.exists(pretrained_path):
        logger.info(f"Loading pre-trained GIN-VN weights from {pretrained_path} (warm start)")
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        gnn_model.load_pretrained(state_dict)
        logger.success("Pre-trained GIN-VN weights loaded (name+shape matches only)!")
    else:
        logger.warning(f"Pre-trained weights not found at {pretrained_path}")
        logger.warning("Continuing with randomly initialized GNN weights (fine-tuning "
                        "from scratch is fine, just slower to converge).")

    print("\n" + "="*80)
    print("STAGE 1: GENERATE PHYSCHEM FEATURES FOR ALL 8 MUTATION SITES "
          "(mechanistic order)")
    print("="*80)

    mutation_sites = [
        ('FULL_SMILES', full_smiles_valid),
        ('ATP_POCKET', mutation_smiles_valid),
        ('P_LOOP', mut_p_loop_valid),
        ('C_HELIX', mut_helix_valid),
        ('DEL19', mut_19_del_valid),
        ('HINGE_LOOP_T790M_C797S', mut_hinge_loop_valid),
        ('DFG_A_LOOP', mut_dfg_a_loop_valid),
        ('HRD_CAT', mut_hrd_cat_valid),
    ]
    n_sites = len(mutation_sites)

    all_feature_dicts = []
    all_valid_indices = []

    for site_name, mut_smiles_series in mutation_sites:
        print(f"\n{'='*80}")
        print(f"Processing {site_name}")
        print(f"{'='*80}")

        feature_dict = generate_hierarchical_features(ligand_smiles_valid, mut_smiles_series)
        all_feature_dicts.append(feature_dict)
        all_valid_indices.append(set(feature_dict['valid_indices']))

    # Graph-parseability check (see build_graph_cache): intersected in alongside the
    # per-site physchem valid-index sets so a row is only ever "common valid" if
    # BOTH its physchem descriptors AND its molecular graphs (ligand + all 8 sites)
    # could be built. In practice this rarely removes anything beyond what the
    # physchem RDKit parsing already excluded, but it keeps the guarantee explicit
    # rather than assumed.
    print("\nBuilding SMILES -> graph-structure cache (ligand + 8 mutation sites)...")
    graph_cache_path = 'smiles_graph_structure_cache.pkl'
    graph_cache = load_graph_cache(graph_cache_path)
    all_smiles_full = set(ligand_smiles_valid.astype(str))
    for _, mut_smiles_series in mutation_sites:
        all_smiles_full |= set(mut_smiles_series.astype(str))
    graph_cache, bad_smiles = build_graph_cache(sorted(all_smiles_full), existing_cache=graph_cache)
    save_graph_cache(graph_cache, graph_cache_path)

    if bad_smiles:
        lig_str = ligand_smiles_valid.astype(str)
        graph_valid_idx = set(range(len(df_train_valid)))
        graph_valid_idx -= set(np.where(lig_str.isin(bad_smiles))[0].tolist())
        for _, mut_smiles_series in mutation_sites:
            mut_str = mut_smiles_series.astype(str)
            graph_valid_idx -= set(np.where(mut_str.isin(bad_smiles))[0].tolist())
        all_valid_indices.append(graph_valid_idx)
        print(f"  {len(bad_smiles)} SMILES failed graph parsing -> "
              f"{len(df_train_valid) - len(graph_valid_idx)} rows will be dropped")

    # Find common valid indices (physchem descriptors -- and now graphs -- can fail
    # for a small number of lig/mutant SMILES pairs; this keeps only the samples
    # where every site succeeds on every check)
    common_valid_indices = set.intersection(*all_valid_indices)
    common_valid_indices = sorted(list(common_valid_indices))

    print(f"\n{'='*80}")
    print(f"Common valid samples across all sites: {len(common_valid_indices)}")
    print(f"{'='*80}")

    if len(common_valid_indices) == 0:
        print("\n✗ ERROR: No samples remain after filtering!")
        sys.exit(1)

    # Filter physchem features to common indices
    for i, feature_dict in enumerate(all_feature_dicts):
        site_valid_idx = feature_dict['valid_indices']
        mask = np.isin(site_valid_idx, common_valid_indices)
        for key in PHYSCHEM_GROUP_KEYS:
            all_feature_dicts[i][key] = feature_dict[key][mask]

    # ===== RANDOM TRAIN/VALIDATION SPLIT =====
    # Computed immediately after common_valid_indices is finalized and BEFORE any
    # scaler (feature or target) is fit. The split is random and reproducible;
    # related ligand scaffolds may occur in both train and validation sets.
    n_common = len(common_valid_indices)
    train_idx, val_idx = random_split_indices(n_common, val_frac=0.2, seed=42)

    # Get y_train for activity (IC50/Ki values)
    y_train1 = activity_values_valid.iloc[common_valid_indices].values
    y_train1 = np.log1p(y_train1)
    y_scaler1 = StandardScaler()
    y_scaler1.fit(y_train1[train_idx].reshape(-1, 1))          # base1d: TRAIN rows only
    y_train_scaled1 = y_scaler1.transform(y_train1.reshape(-1, 1)).flatten().astype(np.float32)

    # Get y_train2 for docking scores
    y_train2 = activity_values2_valid[common_valid_indices]
    y_scaler2 = StandardScaler()
    y_scaler2.fit(y_train2[train_idx].reshape(-1, 1))          # base1d: TRAIN rows only
    y_train_scaled2 = y_scaler2.transform(y_train2.reshape(-1, 1)).flatten().astype(np.float32)

    # Row-order-aligned SMILES arrays (same row order as y_train_scaled1/2 and the
    # physchem stack below) -- these feed embed_batch() every minibatch during
    # fine-tuning instead of a one-off, precomputed embedding lookup.
    lig_smiles_arr = ligand_smiles_valid.iloc[common_valid_indices].astype(str).to_numpy()
    site_smiles_arrs = [mut_smiles_series.iloc[common_valid_indices].astype(str).to_numpy()
                         for _, mut_smiles_series in mutation_sites]
    print(f"  Ligand SMILES: {lig_smiles_arr.shape}   "
          f"Mutant-site SMILES: {n_sites} columns x {len(common_valid_indices)} rows")

    # Physchem descriptor vector per site (unchanged -- these are static RDKit
    # features, independent of the GNN's weights, so they're computed once)
    physchem_per_site = [combine_physchem_features(all_feature_dicts[site_idx])
                          for site_idx in range(n_sites)]
    physchem_stack = np.stack(physchem_per_site, axis=1)        # (n_valid, 8, D)
    physchem_dim = physchem_stack.shape[-1]
    print(f"\nStacked physchem tensor: {physchem_stack.shape}")

    # Physchem gets its own scaler, pooled across all 8 sites (matches the shared-
    # weight physchem projection inside GNNCrossAttentionSetTransformerHead), fit
    # on TRAIN ROWS ONLY (train_idx from the random split above) and then
    # applied (transform) to the full array -- fitting on the full dataset before
    # splitting would leak validation-set statistics into the scaling every
    # training row is trained against. The GNN-embedding scalers from the old
    # frozen-embedding version (lig_gnn_scaler/mut_gnn_scaler) are intentionally
    # REMOVED here -- see "FINE-TUNING UPGRADE NOTES" near the top of this file for
    # why a static scaler fit on a moving embedding distribution is both stale and
    # redundant with the LayerNorm already applied right after each embedding's
    # projection inside the head.
    physchem_scaled, physchem_scaler = scale_site_stack(physchem_stack, fit_indices=train_idx)

    # ===== STAGE 2: FINE-TUNE GNN BACKBONE + GATED-PHYSCHEM CROSS-ATTENTION + SET-TRANSFORMER =====
    # ONE end-to-end PyTorch model (GNN backbone + GNNCrossAttentionSetTransformerHead)
    # trained with ONE optimizer over BOTH sets of parameters -- this is what makes
    # it fine-tuning rather than the old frozen-embedding + separate-Keras-model split.
    print("\n" + "="*80)
    print("STAGE 2: FINE-TUNE GNN BACKBONE + PRE-ATTENTION MULTIMODAL FUSION + SITE SELF-ATTENTION")
    print("="*80)

    head_model = GNNCrossAttentionSetTransformerHead(
        gnn_dim=512, n_sites=n_sites, physchem_dim=physchem_dim).to(device)
    optimizer = torch.optim.Adam(
        list(gnn_model.parameters()) + list(head_model.parameters()), lr=0.001)
    loss_weights = {'activity': 1.0, 'docking': 0.7}

    n_params_gnn = sum(p.numel() for p in gnn_model.parameters())
    n_params_head = sum(p.numel() for p in head_model.parameters())
    logger.info(f"Trainable parameters -- GNN backbone: {n_params_gnn:,}  "
                f"downstream head: {n_params_head:,}  total: {n_params_gnn + n_params_head:,}")

    epochs = 150
    batch_size = 32
    patience = 40
    rng = np.random.RandomState(42)

    history = {'loss': [], 'val_loss': [], 'activity_output_mae': [], 'val_activity_output_mae': []}
    best_val_loss = np.inf
    best_state = None
    patience_counter = 0
    checkpoint_path = 'gnn_cross_attention_settransformer_simple_finetuned.pt'

    def run_batches(idx_array, training):
        """One pass (train or val) over idx_array; returns (mean_loss, mean_activity_mae).
        Shared by both the train and validation loops below -- the only difference is
        `training` (gradients on/off + shuffling) and whether optimizer.step() runs."""
        total_loss, total_mae, n_seen = 0.0, 0.0, 0
        batches = iterate_batches(idx_array, batch_size, shuffle=training, rng=rng)
        for batch_pos in batches:
            lig_batch = lig_smiles_arr[batch_pos].tolist()
            site_batches = [arr[batch_pos].tolist() for arr in site_smiles_arrs]
            physchem_batch = torch.tensor(physchem_scaled[batch_pos], dtype=torch.float32, device=device)
            y1_batch = torch.tensor(y_train_scaled1[batch_pos], dtype=torch.float32, device=device).view(-1, 1)
            y2_batch = torch.tensor(y_train_scaled2[batch_pos], dtype=torch.float32, device=device).view(-1, 1)

            lig_emb = embed_batch(lig_batch, graph_cache, gnn_model, device, training=training)
            mut_emb_stack = torch.stack(
                [embed_batch(sb, graph_cache, gnn_model, device, training=training) for sb in site_batches],
                dim=1)  # (B, n_sites, gnn_dim)

            activity_pred, docking_pred = head_model(lig_emb, mut_emb_stack, physchem_batch)
            loss_activity = F.mse_loss(activity_pred, y1_batch)
            loss_docking = F.mse_loss(docking_pred, y2_batch)
            loss = loss_weights['activity'] * loss_activity + loss_weights['docking'] * loss_docking

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            b = len(batch_pos)
            total_loss += loss.item() * b
            total_mae += F.l1_loss(activity_pred, y1_batch).item() * b
            n_seen += b
        return total_loss / n_seen, total_mae / n_seen

    logger.info('='*80)
    logger.info('TRAINING GNN SIMPLIFIED MULTIMODAL ATTENTION MODEL (fine-tuned)')
    logger.info('='*80)

    for epoch in range(epochs):
        gnn_model.train()
        head_model.train()
        train_loss, train_mae = run_batches(train_idx, training=True)

        gnn_model.eval()
        head_model.eval()
        with torch.no_grad():
            val_loss, val_mae = run_batches(val_idx, training=False)

        history['loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['activity_output_mae'].append(train_mae)
        history['val_activity_output_mae'].append(val_mae)

        logger.info(f"Epoch {epoch+1}/{epochs} - loss: {train_loss:.4f} - "
                    f"val_loss: {val_loss:.4f} - activity_mae: {train_mae:.4f} - "
                    f"val_activity_mae: {val_mae:.4f}")

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {
                'gnn_state_dict': copy.deepcopy(gnn_model.state_dict()),
                'head_state_dict': copy.deepcopy(head_model.state_dict()),
            }
            torch.save({
                'gnn_state_dict': best_state['gnn_state_dict'],
                'head_state_dict': best_state['head_state_dict'],
                'epoch': epoch,
                'val_loss': best_val_loss,
                'hyperparams': {'gnn_dim': 512, 'n_sites': n_sites, 'physchem_dim': physchem_dim,
                                 'fusion_dim': 256, 'physchem_proj_dim': 64,
                                 'num_heads': 4, 'key_dim': 32, 'dropout': 0.15},
            }, checkpoint_path)
            logger.info(f"  -> new best val_loss {best_val_loss:.4f}, checkpoint saved to {checkpoint_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1} "
                            f"(no val_loss improvement for {patience} epochs; best={best_val_loss:.4f})")
                break

    # restore_best_weights=True equivalent
    if best_state is not None:
        gnn_model.load_state_dict(best_state['gnn_state_dict'])
        head_model.load_state_dict(best_state['head_state_dict'])
        logger.info(f"Restored best weights (val_loss={best_val_loss:.4f})")

    # Persist the random train/validation row assignment for reproducibility/debugging.
    # This is not required for prediction, but makes the exact random split auditable.
    np.savez('random_split_indices.npz', train_idx=train_idx, val_idx=val_idx)

    # ===== SAVE SCALERS =====
    feature_scalers = {
        'physchem_scaler': physchem_scaler
        # lig_gnn_scaler / mut_gnn_scaler removed -- see FINE-TUNING UPGRADE NOTES
    }
    with open('feature_scalers.pkl', 'wb') as f: pickle.dump(feature_scalers, f)
    with open('y_scalers.pkl', 'wb') as f: pickle.dump({'y_scaler1': y_scaler1, 'y_scaler2': y_scaler2}, f)
    unique_mutation_profiles.to_csv('mutation_profiles.csv', index=False)

    # ===== PLOT TRAINING HISTORY =====
    print("\n" + "="*80)
    print("PLOTTING TRAINING HISTORY")
    print("="*80)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss plot
    axes[0].plot(history['loss'], label='Train Loss', linewidth=2, color='#2E86AB')
    axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2, color='#A23B72')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss (MSE)', fontsize=12)
    axes[0].set_title('GNN Cross-Attention + Joint 1024d Set-Transformer Model (fine-tuned) - Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # MAE plot
    axes[1].plot(history['activity_output_mae'], label='Train Activity MAE', linewidth=2, color='#2E86AB')
    axes[1].plot(history['val_activity_output_mae'], label='Val Activity MAE', linewidth=2, color='#A23B72')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('MAE', fontsize=12)
    axes[1].set_title('GNN Cross-Attention + Joint 1024d Set-Transformer Model (fine-tuned) - MAE', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('gnn_simple_settransformer_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("✓ Training history plot saved: gnn_simple_settransformer_training_history.png")
    print(f"✓ Fine-tuned checkpoint (GNN backbone + head, jointly trained): {checkpoint_path}")


if __name__ == "__main__":
    main()

#TODO
# 1. [DONE] Fine-tune the GIN-VN backbone end-to-end with the downstream head (single
#    PyTorch optimizer over both) instead of using frozen/precomputed embeddings --
#    see "FINE-TUNING UPGRADE NOTES" near the top of this file.
# 2. [DONE] Encode ligand GNN, site GNN and site physchem independently, fuse them once
#    before attention, then model cross-site dependencies with one self-attention block.
# 3. [DONE] Experiment with different GNN architectures -- upgraded to GIN + Virtual Node
#    + Jumping-Knowledge + OGB-style rich atom/bond features + atomic physchem context.
# 4. Pre-train this GIN-VN architecture from scratch on ChEMBL/ZINC (contrastive or masked-atom
#    objective, following the MolCLR / Hu et al. pre-training recipe) as an initialization
#    for fine-tuning, since the original MolCLR checkpoint is not shape-compatible with the
#    richer encoder.
# 5. [DONE] Simplified the attention head to one site self-attention block plus learned
#    attention pooling. The 8 mutation sites remain a permutation-invariant set.
# 6. [DONE] Replaced the old random split with a reproducible random 80/20
#    train/validation split. The physchem scaler is still fit on TRAIN rows only.
# 7. Next candidate GNN-backbone experiments (not yet implemented as a numbered base1x script):
#    swap GIN-VN for GATv2 (learned per-neighbor attention) or DMPNN/Chemprop, and/or replace the
#    per-site "mutation SMILES" featurization with actual protein-language-model embeddings
#    (e.g. ESM-2) on the mutated kinase-domain sequence, which recent resistance-prediction
#    literature favors over encoding pocket motifs as small-molecule graphs.
# 8. Consider adding optional per-site auxiliary activity/docking heads on fused_sites (deep
#    supervision) if end-to-end training alone doesn't give each site's representation enough
#    direct gradient signal -- discussed but left out of this rearchitecture for simplicity.
# 9. [DONE] Removed the parallel 1024-d raw GNN branch and the 1600-d raw-skip fusion.
#    The current head has one explicit multimodal fusion before one site-attention block.
# 10. See adv_physchem_gnn_base1d_esm.py for the companion ESM2 experiment.
#     GNN encoder for a fine-tuned ESM2 protein-language-model embedding on the actual mutant
#     amino-acid sequence (item 7 above), while keeping this file's STEP-6 joint-head upgrade.
