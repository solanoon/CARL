import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
import os
from torch_geometric.nn import Set2Set


class CARL(nn.Module):
    def __init__(self,
                device,
                node_hidden_dim=16,
                drug_layer_num=3,
                dropout = 0,
                ):
        
        super(CARL, self).__init__()

        node_input_dim = 49
        edge_input_dim = 8

        self.device_ = device
        self.node_input_dim = node_input_dim
        self.node_hidden_dim = node_hidden_dim
        self.edge_input_dim = edge_input_dim
        self.drug_layer_num = drug_layer_num

        self.drug_encoder = GINE(self.node_input_dim, self.edge_input_dim, 
                            self.node_hidden_dim, self.drug_layer_num,
                            )

        self.g2t = Set2Set(self.node_hidden_dim, 2)
        self.cong2t = Set2Set(2*self.node_hidden_dim, 2)
        self.dropout = nn.Dropout(dropout)

        self.compressor = nn.Sequential(
            nn.Linear(2 * self.node_hidden_dim, self.node_hidden_dim),
            nn.BatchNorm1d(self.node_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.node_hidden_dim, 1)
            )
        self.set2set_a2c = Set2Set(2 * node_hidden_dim, 2)          
        self.b_out_layer = nn.Linear(4*self.node_hidden_dim, 2*self.node_hidden_dim)

        self.a_emb = nn.Linear(2*self.node_hidden_dim, self.node_hidden_dim)
        self.b_emb = nn.Linear(2*self.node_hidden_dim, self.node_hidden_dim)
        self.c_emb = nn.Linear(2*self.node_hidden_dim, self.node_hidden_dim)
        
        self.fc1 = nn.Linear(10 * self.node_hidden_dim, 4 * self.node_hidden_dim)
        self.relu = nn.ReLU()
        self.out = nn.Linear(4 * self.node_hidden_dim, 2)
        self.init_model()
    

    def compress(self, solute_features):
        
        p = self.compressor(solute_features)
        temperature = 1.0
        bias = 0.0 + 0.0001 
        eps = (bias - (1 - bias)) * torch.rand(p.size()) + (1 - bias)
        gate_inputs = torch.log(eps) - torch.log(1 - eps)
        gate_inputs = gate_inputs.to(self.device_)
        gate_inputs = (gate_inputs + p) / temperature
        gate_inputs = torch.sigmoid(gate_inputs).squeeze()

        return gate_inputs, p


    def init_model(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)
    

    def condition_module(self, reactant, condition):

        normalized_reactant = F.normalize(reactant, dim = 1) 
        normalized_condition = F.normalize(condition, dim = 1)

        len_map = torch.sparse.mm(self.mapA.t(), self.mapB)

        interaction_map = torch.mm(normalized_reactant, normalized_condition.t())
        interaction_map = interaction_map * len_map.to_dense()

        condition_prime = torch.mm(interaction_map.t(), normalized_reactant)
        reactant_prime = torch.mm(interaction_map, normalized_condition)

        reactant = torch.cat((normalized_reactant, reactant_prime), dim=1)
        condition = torch.cat((normalized_condition, condition_prime), dim=1) 

        return reactant, condition
    


    def forward(self, data, test=False):

        A, B, C, _, self.mapA, self.mapB = data
        reactant_g = self.drug_encoder(A)
        condition_g = self.drug_encoder(B)
        drugC_g = self.drug_encoder(C)

        reactantconB_g, conditionconA_g = self.condition_module(reactant_g, condition_g)

        reactant = self.g2t(reactant_g, A.batch)
        condition = self.g2t(condition_g, B.batch)
        drugC = self.g2t(drugC_g, C.batch)
        reactantconB = self.cong2t(reactantconB_g, A.batch)
        conditionconA = self.cong2t(conditionconA_g, B.batch)


        lambda_pos, self.importance = self.compress(reactantconB_g)  
        lambda_pos = lambda_pos.reshape(-1, 1)
        lambda_neg = 1 - lambda_pos

        pos_to_product = reactantconB_g.clone().detach()
        node_feature_mean = scatter_mean(pos_to_product, A.batch, dim = 0)[A.batch]
        node_feature_std = scatter_std(pos_to_product, A.batch, dim = 0)[A.batch]

        noisy_node_feature_mean = lambda_pos * reactantconB_g + lambda_neg * node_feature_mean
        noisy_node_feature_std = lambda_neg * node_feature_std

        to_product = noisy_node_feature_mean + torch.rand_like(noisy_node_feature_mean) * noisy_node_feature_std
        w_reagent = lambda_neg * reactantconB_g

        to_product_s2s = self.set2set_a2c(to_product, A.batch)
        b_pred = global_mean_pool(w_reagent, A.batch)

        c_out = conditionconA
        c_pred = to_product_s2s
        b_out = self.b_out_layer(conditionconA)

        x = torch.cat((reactantconB, conditionconA, drugC), 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.out(x)

        if test:
          return x[:,0], x[:,1]

        return b_out, b_pred, c_out, c_pred, x[:,0], x[:,1]