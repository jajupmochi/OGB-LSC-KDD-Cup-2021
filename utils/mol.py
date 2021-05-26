#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  5 15:04:51 2021

@author: ljia
"""

def smiles2nxgraph_pysmiles(smiles_string):
	"""Converts SMILES string to NetworkX graph object by pysmiles library.

	Parameters
	----------
	smiles_string : string
		SMILES string.

	Returns
	-------
	Graph object.
	"""
	from pysmiles import read_smiles
	mol = read_smiles(smiles_string)
	return mol


def smiles2nxgraph_rdkit(smiles_string):
	"""Converts SMILES string to NetworkX graph object by RDKit library.

	Parameters
	----------
	smiles_string : string
		SMILES string.

	Returns
	-------
	Graph object.
	"""
	from rdkit import Chem
	mol = Chem.MolFromSmiles(smiles_string)
	mol = mol_to_nx(mol)
	return mol


def mol_to_nx(mol):
	"""Converts RDKit mol to NetworkX graph object.

	Parameters
	----------
	mol : RDKit mol
		RDKit molecule.

	Returns
	-------
	G : Networkx Graph.
		Graph object.

	References
	----------
	`keras-molecules <https://github.com/dakoner/keras-molecules/blob/dbbb790e74e406faa70b13e8be8104d9e938eba2/convert_rdkit_to_networkx.py#L17>`__
	"""
	import networkx as nx
	G = nx.Graph()

	for atom in mol.GetAtoms():
		G.add_node(atom.GetIdx(),
				   atomic_num=atom.GetAtomicNum(),
				   formal_charge=atom.GetFormalCharge(),
				   chiral_tag=atom.GetChiralTag(),
				   hybridization=atom.GetHybridization(),
				   num_explicit_hs=atom.GetNumExplicitHs(),
				   is_aromatic=atom.GetIsAromatic())
	for bond in mol.GetBonds():
		G.add_edge(bond.GetBeginAtomIdx(),
				   bond.GetEndAtomIdx(),
				   bond_type=bond.GetBondType())
	return G


def smiles2nxgraph_ogb(smiles_string):
	"""Converts SMILES string to OGB graph object and then to NetworkX graph
	object.

	Parameters
	----------
	smiles_string : string
		SMILES string.

	Returns
	-------
	Graph object.
	"""
	from ogb.utils import smiles2graph
	mol = smiles2graph(smiles_string)
	mol = graph2nx(mol)
	return mol



def graph2nx(graph_obj):
	"""Converts OGB graph object to NetworkX graph object.

	Parameters
	----------
	graph_obj : dict
		OGB graph object.

	Returns
	-------
	NetworkX graph object.
	"""
	import networkx as nx
	G = nx.Graph()

	for n in range(0, graph_obj['num_nodes']):
		feats = {}
		for i, feat in enumerate(graph_obj['node_feat'][n]):
			feats['label_' + str(i)] = feat
		G.add_node(n, **feats)

	idx_edge = graph_obj['edge_index'].shape[1]
	for e in range(0, idx_edge, 2): # @todo: make sure '2' is correct.
		feats_e = {}
		for i, feat_e in enumerate(graph_obj['edge_feat'][e]):
			feats_e['label_' + str(i)] = feat_e
		G.add_edge(graph_obj['edge_index'][0][e], graph_obj['edge_index'][1][e], **feats_e)

	return G
