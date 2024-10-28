import torch
import copy
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from model import CARL

class Trainer:
    def __init__(self, args, train_loader, valid_loader, test_loader, repeat):

        self.args = args
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.device = args.device
        self.random_state = repeat

        self.train_y_mean = np.mean([train_loader.dataset[i][3] for i in range(len(train_loader.dataset))])
        self.train_y_std = np.std([train_loader.dataset[i][3] for i in range(len(train_loader.dataset))])

        self.model = CARL(device = self.device, node_hidden_dim=self.args.hidden_dim).to(self.device)
        self.optimizer = optim.Adam(params = self.model.parameters(), lr = self.args.lr, weight_decay = self.args.weight_decay)
        self.scheduler = MultiStepLR(self.optimizer, milestones = [400, 450], gamma = 0.1, verbose = False)

        self.best_val_mae = 10000

    def train(self):        

        loss_fn = nn.MSELoss(reduction = 'none')
        loss_fn_1 = nn.MSELoss(reduction = 'mean')
        loss_fn_2 = nn.MSELoss(reduction = 'mean')

        for epoch in range(1, self.args.epochs + 1):
            self.model.train()
            self.train_loss = 0
            self.interact_loss = 0
            self.similarity_loss = 0
            preserve = 0


            for bc, samples in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                masks = create_batch_mask(samples)
                
                labels = (samples[3] - self.train_y_mean) / self.train_y_std
                labels = labels.to(self.device)

                gb, gb_pred, gc, gc_pred, pred, logvar = self.model([i.to(self.device) for i in samples] + [i.to(self.device) for i in masks])

                loss_predict = loss_fn(pred, labels)
                loss_predict = (1 - 0.1) * loss_predict.mean() + 0.1 * (loss_predict * torch.exp(-logvar) + logvar ).mean()

                loss_reconstruct = loss_fn_1(gc_pred, gc)
                loss_equilibrium = loss_fn_2(gb_pred, gb)

                loss = loss_predict*self.args.l1 +  loss_reconstruct* self.args.l3 + loss_equilibrium * self.args.l2

                loss.backward()
                self.optimizer.step()
                self.train_loss += loss
                self.equilibrium_loss += loss_equilibrium
                self.reconstruct_loss += loss_reconstruct
            
            self.scheduler.step()
            

            if (epoch + 1) % self.args.monitor_per_epoch == 0:
                self.model.eval()
                val_mae, val_rmse, val_r2 = self.inference(self.valid_loader)
                self.val_mae_score = val_mae

                if val_mae < self.best_val_mae:
                    self.best_model = copy.deepcopy(self.model)

            test_mae, test_rmse, test_r2, test_pred, test_y = self.inference(self.test_loader, return_pred = True)



    def inference(self, dataloader, n_forward_pass = 30, return_pred = False):

        test_y = np.array([dataloader.dataset[i][3] for i in range(len(dataloader.dataset))])

        batch_size = dataloader.batch_size
        train_y_mean = self.train_y_mean
        train_y_std = self.train_y_std
        
        self.model.eval()
        MC_dropout(self.model)
        
        test_y_mean = []
        test_y_var = []
        
        with torch.no_grad():
            for bc, samples in enumerate(dataloader):

                masks = create_batch_mask(samples)

                mean_list = []
                var_list = []
                
                for _ in range(n_forward_pass):
                    mean, logvar = self.model([i.to(self.device) for i in samples] + [i.to(self.device) for i in masks], test = True)
                    mean_list.append(mean.cpu().numpy())
                    var_list.append(np.exp(logvar.cpu().numpy()))

                test_y_mean.append(np.array(mean_list).transpose())
                test_y_var.append(np.array(var_list).transpose())

        test_y_mean = np.vstack(test_y_mean) * train_y_std + train_y_mean
        test_y_var = np.vstack(test_y_var) * train_y_std ** 2
        
        test_y_pred = np.mean(test_y_mean, 1)
        test_y_epistemic = np.var(test_y_mean, 1)
        test_y_aleatoric = np.mean(test_y_var, 1)

        test_mae = mean_absolute_error(test_y, test_y_pred)
        test_rmse = mean_squared_error(test_y, test_y_pred) ** 0.5
        test_r2 = r2_score(test_y, test_y_pred)

        if return_pred:
            return test_mae, test_rmse, test_r2, test_y_pred, test_y
        return test_mae, test_rmse, test_r2
