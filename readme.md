# Code of Submission 8720

In this work, we introduce CARL, a condition-aware relational learning model designed to enhance the accuracy of chemical reaction yield predictions. 

- CARL systematically models the relations between the different classes of compounds involved in a reaction by categorizing them into three distinct entities: reactants, products, and condition molecules. 
- CARL employs relational learning to capture the directed and complex interactions among these entities. 
- CARL is built around two core modules, the condition-aware interaction module and the reactor module, and incorporates two novel loss functions, reconstruction loss and equilibrium loss, specifically tailored to address the challenges of multi-entity relational learning in chemical reaction.

### 1. Environment

```
python == 3.8.19
torch == 2.0.0
rdkit == 2022.9.5
```

### 2. Data

- Benchmark datsets downloaded from [https://github.com/rxn4chemistry/rxn_yields](https://github.com/rxn4chemistry/rxn_yields)

- data pre-processing:

  ```
  python data.py --dataname buchwald_hartwig
  ```

### 3. Train CARL

  ```
  CUDA_VISIBLE_DEVICES=0 python main_prob.py --batch_size 64 --lr 0.001 --dataname buchwald_hartwig --dataloader rxnabc --hidden_dim 256 --alpha 0.5 --beta 0.5
  ```