import numpy as np
import pandas as pd
import torch
import gpytorch
import math
import os
import scanpy as sc
from .plot import plot_3d

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood,lengthscale_prior=None,outputscale_prior=None):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            base_kernel=gpytorch.kernels.RBFKernel(
                lengthscale_prior=lengthscale_prior
            ),
            outputscale_prior=outputscale_prior
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class GPRmodel():
    def __init__(self,expr,loc,loc_resample,used_genes,use_gpu=False,output_dir=None,**kwargs):
        
        if 'subset' not in kwargs.keys():
            self.subset = 10000
        else:
            self.subset = kwargs['subset']
            
        if 'lengthscale_prior' not in kwargs.keys():
            self.lengthscale_prior = None
        else:
            self.lengthscale_prior = kwargs['lengthscale_prior']
            
        if 'outputscale_prior' not in kwargs.keys():
            self.outputscale_prior = None
        else:
            self.outputscale_prior = kwargs['outputscale_prior']
            
        if 'noise_prior' not in kwargs.keys():
            self.noise_prior = None
        else:
            self.noise_prior = kwargs['noise_prior']
        
        if output_dir is not None:
            self.output_dir = output_dir
        else:
            self.output_dir = './gpr_models'
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        self.model = None
        self.flat_model = None
        self.loc_resample = torch.tensor(loc_resample,dtype=torch.float32)
        self.train_x, self.subset_y = self.prepare_gpr_data(loc,expr.values,self.subset)
        self.train_y = None
        self.used_genes = used_genes
        self.g_ind = [list(expr.columns).index(g) for g in self.used_genes]
        self.gene_ind_map = pd.DataFrame(self.g_ind,index=self.used_genes,columns=['g_ind'])
        self.log_bf = pd.DataFrame(index=self.used_genes,columns=['log_bf'])
        self.use_gpu = use_gpu
        if self.use_gpu:
            print('Using GPU accelerate')
            self.train_x = self.train_x.cuda()
            self.loc_resample = self.loc_resample.cuda()
    
    @staticmethod   
    def prepare_gpr_data(X,y,subset):
        subset_ind = np.random.permutation(X.shape[0])[:subset]
        subset_x = X[subset_ind]
        subset_y = y[subset_ind]
        subset_x = torch.tensor(subset_x,dtype=torch.float32)
        subset_y = torch.tensor(subset_y,dtype=torch.float32)
        return subset_x, subset_y
    
    def prepare_gpr_model(self, lengthscale_prior=None,outputscale_prior=None,noise_prior=None,bayesian_alter=False):

        if bayesian_alter:
            if self.flat_model is None:
                likelihood = gpytorch.likelihoods.GaussianLikelihood(
                    noise_prior=noise_prior
                )
                self.flat_model = ExactGPModel(self.train_x, self.train_y, likelihood, lengthscale_prior=lengthscale_prior,outputscale_prior=outputscale_prior)
            # self.init_model(self.flat_model,lengthscale=torch.tensor(99999999))
            self.init_model(self.flat_model,lengthscale=torch.inf)
        else:
            if self.model is None:
                likelihood = gpytorch.likelihoods.GaussianLikelihood(
                    noise_prior=noise_prior
                )
                self.model = ExactGPModel(self.train_x, self.train_y, likelihood, lengthscale_prior=lengthscale_prior,outputscale_prior=outputscale_prior)
            self.init_model(self.model)
            
    
    def init_model(self,model,noise=None,lengthscale=None,outputscale=None,constant=None):
        if noise is None:
            noise = self.train_y.std()/2
        if lengthscale is None:
            lengthscale = self.train_y.std()
        if outputscale is None:
            outputscale = torch.tensor(4)
        if constant is None:
            constant = torch.tensor(self.train_y.mean())
            
        hypers = {
            'likelihood.noise_covar.noise': noise,
            'covar_module.base_kernel.lengthscale': lengthscale,
            'covar_module.outputscale': outputscale,
            'mean_module.constant': constant
        }

        model.initialize(**hypers)
        model.train_targets = self.train_y
        model.zero_grad()
        # print('outputscale: %.3f   lengthscale: %.3f   noise: %.3f' % (
        #     model.covar_module.outputscale.item(),
        #     model.covar_module.base_kernel.lengthscale.item(),
        #     model.likelihood.noise.item()
        # ))
    
    def train_single_model(self,model,lr=1,training_iter=500,save=False,save_path=None,optimize_method='Adam'):
        # Find optimal model hyperparameters
        model.train()
        model.likelihood.train()
        
        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
        best_loss=np.inf
        
        if optimize_method=='Adam':
            optimizer = torch.optim.Adam(model.parameters(),lr=lr)
            for i in range(training_iter):
                optimizer.zero_grad()
                output = model(self.train_x)
                loss = -mll(output, self.train_y)
                loss.backward()
                optimizer.step()
                # print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
                #     i + 1, training_iter, loss.item(),
                #     model.covar_module.base_kernel.lengthscale.item(),
                #     model.likelihood.noise.item()
                # ))
                if loss.item() < best_loss:
                    best_model_state = model.state_dict()
                    best_loss = loss.item()
        # Bugs need to be solved
        elif optimize_method=='LBGFS':
            optimizer = torch.optim.LBFGS(model.parameters(),line_search_fn='strong_wolfe', lr=lr)
            def closure():
                optimizer.zero_grad()
                output = model(self.train_x)
                loss = -mll(output, self.train_y)
                loss.backward()
                return loss
            for i in range(training_iter):
                loss = optimizer.step(closure)
                # print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
                #     i + 1, training_iter, loss.item(),
                #     model.covar_module.base_kernel.lengthscale.item(),
                #     model.likelihood.noise.item()
                # ))
                if loss.item() < best_loss:
                    best_model_state = model.state_dict()
                    best_loss = loss.item()
        else:
            raise ValueError('Invalid optimize method: ', optimize_method)

        # Get into evaluation (predictive posterior) mode
        model.eval()
        model.likelihood.eval()
        # for param_name, param in model.named_parameters():
        #     print(f'Parameter name: {param_name:42} value = {param.item()}')
        if save:
            torch.save(best_model_state, save_path)
        print('Best model loss:', best_loss)
        return best_loss
    
    def optim_lengthscale(self,model,lr=1,l_range=torch.arange(1,12,1),optimize_method='Adam'):

        model.train()
        model.likelihood.train()
        loss_list = []
        print('Optimize lenthscale...')
        for lenthscale_alpha in l_range:
            self.init_model(model,lengthscale=lenthscale_alpha*self.train_y.std())
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
            
            if optimize_method=='Adam':
                optimizer = torch.optim.Adam(model.parameters(),lr=lr)
                optimizer.zero_grad()
                output = model(self.train_x)
                loss = -mll(output, self.train_y)
                loss.backward()
                optimizer.step()
            elif optimize_method=='LBGFS':
                optimizer = torch.optim.LBFGS(model.parameters(),line_search_fn='strong_wolfe', lr=lr)
                def closure():
                    optimizer.zero_grad()
                    output = model(self.train_x)
                    loss = -mll(output, self.train_y)
                    loss.backward()
                    return loss
                loss = optimizer.step(closure)
            else:
                raise ValueError('Invalid optimize method: ', optimize_method)
            # print(' - Lengthscale weight: %.3f   Loss: %.3f' % (
            #     lenthscale_alpha,
            #     loss.item(),
            # ))
            loss_list.append(loss.item())
        best_l_alpha = l_range[np.argmin(loss_list)]
        print('Initialize lenthscale alpha as %.3f' % best_l_alpha)
        self.init_model(model,lengthscale=best_l_alpha*self.train_y.std())
        model.eval()
        model.likelihood.eval()
    
    def train(self,lr=1,training_iter=500,save_model=True,save_pred=False,cal_bf=False,optim_l=False,optimize_method='Adam'):
        for g,g_i in zip(self.used_genes,self.g_ind):
            print(f'Modeling {g}')
            self.train_y = self.subset_y[:,g_i]
            self.prepare_gpr_model(lengthscale_prior=self.lengthscale_prior,outputscale_prior=self.outputscale_prior,noise_prior=self.noise_prior) 
            if self.use_gpu:
                self.train_y = self.train_y.cuda()
                self.model = self.model.cuda()
                self.model.likelihood = self.model.likelihood.cuda()
                
            if optim_l:
                self.optim_lengthscale(self.model,lr=lr,optimize_method=optimize_method)
            
            mll = self.train_single_model(
                self.model,
                lr=lr,
                optimize_method=optimize_method,
                training_iter=training_iter,
                save=save_model,
                save_path=os.path.join(self.output_dir,f'{g}_iter{training_iter}_lr{lr}_model_state.pth'))
            
            if save_pred:
                resampled_pred = self.predict_resampled_spot()
                np.save(os.path.join(self.output_dir,f'{g}_iter{training_iter}_lr{lr}_resampled_pred.npy'),resampled_pred)
            
            if cal_bf:
                self.prepare_gpr_model(lengthscale_prior=self.lengthscale_prior,outputscale_prior=self.outputscale_prior,noise_prior=self.noise_prior,bayesian_alter=True) 
                if self.use_gpu:
                    self.flat_model = self.flat_model.cuda()
                    self.flat_model.likelihood = self.flat_model.likelihood.cuda()
                flat_mll = self.train_single_model(self.flat_model,lr=lr,training_iter=training_iter,save=save_model,save_path=os.path.join(self.output_dir,f'{g}_iter{training_iter}_lr{lr}_flat_model_state.pth'))
                log_bf = -mll+flat_mll
                self.log_bf.loc[g,'log_bf'] = log_bf
                print(log_bf)

    def load_gene_model(self,gene,training_iter,lr):
        g_i = self.gene_ind_map.loc[gene,'g_ind']
        self.train_y = self.subset_y[:,g_i]
        if self.use_gpu:
            self.train_y = self.train_y.cuda()
        if self.model is None:
            self.prepare_gpr_model()
        state_dict = torch.load(os.path.join(self.output_dir,f'{gene}_iter{training_iter}_lr{lr}_model_state.pth'))
        self.model.load_state_dict(state_dict)
        self.model.train_targets = self.train_y

    def eval_model(self):
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = self.model.likelihood(self.model(self.train_x))
        return observed_pred
    
    def predict_resampled_spot(self, gene=None, data=None, training_iter=None, lr=None, save_pred=False, save_pred_path=None, n_in_batch=30000):
        if gene is not None:
            self.load_gene_model(gene, training_iter, lr)
        obs_pred_list = []
        for i in range(int(np.ceil(data.shape[0]/n_in_batch))):
            low_bound = i*n_in_batch
            high_bound = min((i+1)*n_in_batch,data.shape[0])
            data_batch = data[low_bound:high_bound,:]
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                observed_pred_batch = self.model.likelihood(self.model(torch.tensor(data_batch)))
            obs_pred_list.append(observed_pred_batch.mean.cpu().numpy())
        obs_pred_concated = np.concatenate(obs_pred_list)
        if save_pred:
            if save_pred_path is None:
                save_pred_path = os.path.join(self.output_dir,f'{gene}_iter{training_iter}_lr{lr}_resampled_pred.npy')
            np.save(save_pred_path,obs_pred_concated)
        return obs_pred_concated
    
    def plot_gpr_expr(
        self,
        gene,
        training_iter,
        lr,
        data=None,
        pred_path=None,
        save=False,
        save_path=None,
        save_pred=False,
        save_pred_path=None,
        save_dpi=150,
        return_expr=True,
        *args,**kwargs
    ):
        if data is None:
            data = self.loc_resample
        else:
            data = torch.tensor(data,dtype=torch.float32)
            if self.use_gpu:
                data = data.cuda()
        # if pred_path is None:
        #     pred_path = os.path.join(self.output_dir,f'{gene}_iter{training_iter}_lr{lr}_resampled_pred.npy')
        if save and save_path is None:
            save_path = os.path.join(self.output_dir,f'{gene}_iter{training_iter}_lr{lr}_resampled_pred.png')
        if (pred_path is not None) and os.path.exists(pred_path):
            resampled_pred = np.load(os.path.join(self.output_dir,f'{gene}_iter{training_iter}_lr{lr}_resampled_pred.npy'))
        else:
            resampled_pred = self.predict_resampled_spot(gene,data,training_iter,lr,save_pred,save_pred_path)
        plot_3d(data.cpu().numpy(), val=resampled_pred,save_path=save_path,save_dpi=save_dpi,*args,**kwargs)
        if return_expr:
            return resampled_pred
