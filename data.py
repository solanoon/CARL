import os, csv
import numpy as np
from rdkit import Chem, RDConfig, rdBase
from rdkit.Chem import AllChem, ChemicalFeatures
import argparse
import pandas as pd

chem_feature_factory = ChemicalFeatures.BuildFeatureFactory(os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef'))
from tqdm import tqdm
import os, sys
import numpy as np
from torch_geometric.data import Data
import torch

atom_list = ['Li','B','C','N','O','F','Na','P','S','Cl','K','Fe','Br','Pd','I','Cs']
charge_list = [1, 2, -1, 0]
degree_list = [1, 2, 3, 4, 0]
valence_list = [1, 2, 3, 4, 5, 6, 0]
hybridization_list = ['SP','SP2','SP3','SP3D','SP3D2','S']
hydrogen_list = [1, 2, 3, 0]
ringsize_list = [3, 4, 5, 6, 7, 8]
bond_list = ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC']

dummy = {'n_node':1, 'n_edge':0, 'node_attr':np.zeros((1, len(atom_list) + len(charge_list) + len(degree_list) + len(hybridization_list) + len(hydrogen_list) + len(valence_list) + len(ringsize_list) + 1))}

def mol_dict():
    return {'n_node': [],
            'n_edge': [],
            'node_attr': [],
            'edge_attr': [],
            'src': [],
            'dst': []}

def return_mol_feature(mol):
    
    mol_dict = {}

    def _DA(mol):

        D_list, A_list = [], []
        for feat in chem_feature_factory.GetFeaturesForMol(mol):
            if feat.GetFamily() == 'Donor': D_list.append(feat.GetAtomIds()[0])
            if feat.GetFamily() == 'Acceptor': A_list.append(feat.GetAtomIds()[0])
        
        return D_list, A_list

    def _chirality(atom):
        
        if atom.HasProp('Chirality'):
            c_list = [(atom.GetProp('Chirality') == 'Tet_CW'), (atom.GetProp('Chirality') == 'Tet_CCW')] 
        else:
            c_list = [0, 0]

        return c_list
        
    def _stereochemistry(bond):
        
        if bond.HasProp('Stereochemistry'):
            s_list = [(bond.GetProp('Stereochemistry') == 'Bond_Cis'), (bond.GetProp('Stereochemistry') == 'Bond_Trans')] 
        else:
            s_list = [0, 0]

        return s_list     
        

    n_node = mol.GetNumAtoms()
    n_edge = mol.GetNumBonds() * 2
    
    D_list, A_list = _DA(mol)  
    atom_fea1 = np.eye(len(atom_list), dtype = bool)[[atom_list.index(a.GetSymbol()) for a in mol.GetAtoms()]]
    atom_fea2 = np.eye(len(charge_list), dtype = bool)[[charge_list.index(a.GetFormalCharge()) for a in mol.GetAtoms()]][:,:-1]
    atom_fea3 = np.eye(len(degree_list), dtype = bool)[[degree_list.index(a.GetDegree()) for a in mol.GetAtoms()]][:,:-1]
    atom_fea4 = np.eye(len(hybridization_list), dtype = bool)[[hybridization_list.index(str(a.GetHybridization())) for a in mol.GetAtoms()]][:,:-1]
    atom_fea5 = np.eye(len(hydrogen_list), dtype = bool)[[hydrogen_list.index(a.GetTotalNumHs(includeNeighbors = True)) for a in mol.GetAtoms()]][:,:-1]
    atom_fea6 = np.eye(len(valence_list), dtype = bool)[[valence_list.index(a.GetTotalValence()) for a in mol.GetAtoms()]][:,:-1]
    atom_fea7 = np.array([[(j in D_list), (j in A_list)] for j in range(mol.GetNumAtoms())], dtype = bool)
    atom_fea8 = np.array([[a.GetIsAromatic(), a.IsInRing()] for a in mol.GetAtoms()], dtype = bool)
    atom_fea9 = np.array([[a.IsInRingSize(s) for s in ringsize_list] for a in mol.GetAtoms()], dtype = bool)
    atom_fea10 = np.array([_chirality(a) for a in mol.GetAtoms()], dtype = bool)
    
    node_attr = np.hstack([atom_fea1, atom_fea2, atom_fea3, atom_fea4, atom_fea5, atom_fea6, atom_fea7, atom_fea8, atom_fea9, atom_fea10])

    mol_dict['n_node'] = n_node
    mol_dict['n_edge'] = n_edge
    mol_dict['node_attr'] = node_attr

    if n_edge > 0:

        bond_fea1 = np.eye(len(bond_list), dtype = bool)[[bond_list.index(str(b.GetBondType())) for b in mol.GetBonds() if str(b.GetBondType()) != 'UNSPECIFIED']]
        bond_fea2 = np.array([[b.IsInRing(), b.GetIsConjugated()] for b in mol.GetBonds() if str(b.GetBondType()) != 'UNSPECIFIED'], dtype = bool)
        bond_fea3 = np.array([_stereochemistry(b) for b in mol.GetBonds() if str(b.GetBondType()) != 'UNSPECIFIED'], dtype = bool)
        
        edge_attr = np.hstack([bond_fea1, bond_fea2, bond_fea3])
        edge_attr = np.vstack([edge_attr, edge_attr])
        
        bond_loc = np.array([[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] for b in mol.GetBonds() if str(b.GetBondType()) != 'UNSPECIFIED'], dtype=int)
        src = np.hstack([bond_loc[:,0], bond_loc[:,1]])
        dst = np.hstack([bond_loc[:,1], bond_loc[:,0]])
    
        mol_dict['edge_attr'] = edge_attr
        mol_dict['src'] = src
        mol_dict['dst'] = dst
    
    return mol_dict



def dict_list_to_numpy(mol_dict):

    mol_dict['n_node'] = np.array(mol_dict['n_node']).astype(int)
    mol_dict['n_edge'] = np.array(mol_dict['n_edge']).astype(int)
    mol_dict['node_attr'] = np.vstack(mol_dict['node_attr']).astype(bool)
    if np.sum(mol_dict['n_edge']) > 0:
        mol_dict['edge_attr'] = np.vstack(mol_dict['edge_attr']).astype(bool)
        mol_dict['src'] = np.hstack(mol_dict['src']).astype(int)
        mol_dict['dst'] = np.hstack(mol_dict['dst']).astype(int)
    else:
        mol_dict['edge_attr'] = np.empty((0, len(bond_list) + 2)).astype(bool)
        mol_dict['src'] = np.empty(0).astype(int)
        mol_dict['dst'] = np.empty(0).astype(int)

    return mol_dict


class ReactionDataset():

    def __init__(self, dataname):

        self.dataname = dataname
        self.label_name = 'yield'
        self.ori_df = pd.read_csv(f'dataset/{self.dataname}.csv')
        self.ori_df_col = list(self.ori_df.columns)
        self.label_ind = self.ori_df_col.index(self.label_name)
        self.data_num = len(self.ori_df)

        self.reactant_inds = [i for i in range(len(self.ori_df_col)) if 'reactant' in self.ori_df_col[i]]
        self.condition_inds = [i for i in range(len(self.ori_df_col)) if 'reactants' not in self.ori_df_col[i] and 'reactant' not in self.ori_df_col[i] and 'product' not in self.ori_df_col[i] and 'products' not in self.ori_df_col[i] and 'yield' not in self.ori_df_col[i]]

    def load_data(self):
        if os.path.isfile(f'dataset/{self.dataname}.pt'):
          self.processed = torch.load(f'dataset/{self.dataname}.pt')
        else:
          self.process_data()

    def __getitem__(self, idx):
        return self.processed[idx]
        
    def __len__(self):
        return self.y.shape[0]

    def return_feature(self, smi):
        mol = Chem.MolFromSmiles(smi)
        ps = Chem.FindPotentialStereo(mol)
        for element in ps:
            if str(element.type) == 'Atom_Tetrahedral' and str(element.specified) == 'Specified': mol.GetAtomWithIdx(element.centeredOn).SetProp('Chirality', str(element.descriptor))
            elif str(element.type) == 'Bond_Double' and str(element.specified) == 'Specified': mol.GetBondWithIdx(element.centeredOn).SetProp('Stereochemistry', str(element.descriptor))
        return return_mol_feature(mol)

    def process_data(self):

        reactant_feat_dict = {}
        condition_feat_dict = {}
        product_feat_dict = {}

        dataset = []

        for i in tqdm(range(len(self.ori_df))):

            r = self.ori_df.iloc[i,:]

            r_x = []
            a_x = []
            p_x = []
            r_edge_in = []
            r_edge_out = []
            a_edge_in = []
            a_edge_out = []
            p_edge_in = []
            p_edge_out = []
            r_edge_attr = []
            a_edge_attr = []
            p_edge_attr = []

            reactants = r['reactants'].split('.')
            conditions = [r[ind] for ind in self.condition_inds]
            products = [r[self.ori_df_col.index('product')]]
            yld = r[self.label_ind]*100

            r_n_node_sum = 0
            for j, smi in enumerate(reactants):
                if smi not in reactant_feat_dict:
                    reactant_feat_dict[smi] = self.return_feature(smi)  

                if type(smi) == str:
                    r_x.append(reactant_feat_dict[smi]['node_attr'])

                    r_edge_in.append(reactant_feat_dict[smi]['src'] + r_n_node_sum)
                    r_edge_out.append(reactant_feat_dict[smi]['dst'] + r_n_node_sum)
                    r_edge_attr.append(reactant_feat_dict[smi]['edge_attr'])
                    r_n_node_sum += reactant_feat_dict[smi]['n_node']

                else:
                    r_x.append(dummy['node_attr'])
                    r_n_node_sum += dummy['n_node']

            r_x = np.concatenate(r_x)
            r_edge_in = np.concatenate(r_edge_in)
            r_edge_out = np.concatenate(r_edge_out)
            r_edge_attr = np.concatenate(r_edge_attr)
            r_graph = Data(x = torch.tensor(r_x, dtype = torch.float),
                edge_index = torch.tensor([r_edge_in, r_edge_out], dtype = torch.long),
                edge_attr = torch.tensor(r_edge_attr, dtype = torch.float))

            p_n_node_sum = 0
            for j, smi in enumerate(products):
                if smi not in product_feat_dict:
                    product_feat_dict[smi] = self.return_feature(smi)

                if type(smi) == str:
                    p_x.append(product_feat_dict[smi]['node_attr'])
                    p_edge_in.append(product_feat_dict[smi]['src'] + p_n_node_sum)
                    p_edge_out.append(product_feat_dict[smi]['dst'] + p_n_node_sum)
                    p_edge_attr.append(product_feat_dict[smi]['edge_attr'])

                    p_n_node_sum += product_feat_dict[smi]['n_node']
                else:
                    p_x.append(dummy['node_attr'])
                    p_n_node_sum += dummy['n_node']
            
            p_x = np.concatenate(p_x)
            p_edge_in = np.concatenate(p_edge_in)
            p_edge_out = np.concatenate(p_edge_out)
            p_edge_attr = np.concatenate(p_edge_attr)
            p_graph = Data(x = torch.tensor(p_x, dtype = torch.float),
                        edge_index = torch.tensor([p_edge_in, p_edge_out], dtype = torch.long),
                        edge_attr = torch.tensor(p_edge_attr, dtype = torch.float))
            
            a_n_node_sum = 0
            for j, smi in enumerate(conditions):

                if smi not in condition_feat_dict:
                    condition_feat_dict[smi] = self.return_feature(smi)

                if type(smi) == str:
                    a_x.append(condition_feat_dict[smi]['node_attr'])
                    a_edge_in.append(condition_feat_dict[smi]['src'] + a_n_node_sum)
                    a_edge_out.append(condition_feat_dict[smi]['dst'] + a_n_node_sum)
                    a_edge_attr.append(condition_feat_dict[smi]['edge_attr'])

                    a_n_node_sum += condition_feat_dict[smi]['n_node']
                else:
                    a_x.append(dummy['node_attr'])
                    a_n_node_sum += dummy['n_node']
            
            a_x = np.concatenate(a_x)
            a_edge_in = np.concatenate(a_edge_in)
            a_edge_out = np.concatenate(a_edge_out)
            a_edge_attr = np.concatenate(a_edge_attr)
            a_graph = Data(x = torch.tensor(a_x, dtype = torch.float),
                        edge_index = torch.tensor([a_edge_in, a_edge_out], dtype = torch.long),
                        edge_attr = torch.tensor(a_edge_attr, dtype = torch.float))

            
            dataset.append([r_graph, a_graph, p_graph, yld])
            
        torch.save(dataset, f'dataset/{self.dataname}.pt')
        self.processed = dataset

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataname', type=str, help='Dataset argument')
    args = parser.parse_args()

    data = ReactionDataset(dataname)
    data.process_data()