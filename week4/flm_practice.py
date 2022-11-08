import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from torch.optim import Adam
from torch.nn import Module
import scipy
from torch.nn.functional import softmax

class FiniteLinearModel(Module):
    """Class for Finite mixtures of linear models"""

    def __init__(self, G: int, data: torch.Tensor):
        """
        Constructor

        :param G: Number of components
        :type G: int
        :param data: Data matrix contatining n elements each containing d values
        :type data: torch.Tensor
        """
        super(FiniteLinearModel, self).__init__()

        # define constants
        self.n = data.shape[0]
        self.G = G

        # define data
        self.X = data[:, 1:]
        self.X = torch.cat((self.X, torch.ones(self.n, 1)), axis=1)
        self.y = data[:, 0] 

        # number of covariates
        self.p = self.X.shape[1]

        # define parameters
        self.betas = torch.rand(G, self.p)
        self.sigmas = torch.rand(G, 1).abs()
        self.w = torch.rand(G, 1)

        # set gradient tracking
        self.w.requires_grad = True
        self.betas.requires_grad = True
        self.sigmas.requires_grad = True

    def compute_weights(self):
        """Compute mixing proportions"""
        return softmax(self.w[:, 0], dim=0)

    def log_density(self) -> torch.Tensor:
        """
            Take in covariate X and response y,
            compute n * G matrix of log densities

            :return: n * G matrix of log densities
            :rtype: torch.Tensor
        """
        ldens = torch.zeros(self.n, self.G)

        # iterate through components
        for g in range(self.G):

            # compute linear model
            # element wise multiplication then sum inner axis
            y_hats = (self.betas[g] * self.X).sum(-1)

            # compute log of gaussian density
            term1 = - 0.5 * torch.Tensor([2 * np.pi]).log()
            term2 = - self.sigmas[g].log()
            term3 = - 0.5 * ((y_hats - self.y) / self.sigmas[g]).pow(2)
            ldens[:, g] = term1 + term2 + term3
        
        return ldens

    def objective_function(self) -> torch.Tensor:
        """
        Return negative log likelihood objective function to minimize

        :returns: Objective function
        :rtype: torch.Tensor
        """
        # G * 1
        W = self.compute_weights()
        # n * G 
        dens = self.log_density().exp()

        # What is `dens` has extremely small values?
        output = - ((dens * W).sum(-1).log()).sum()

        return output

    def train(self, lr=1e-3, max_iterations=1000):
        """
            Train using Adam optimizer
        """
        optim = Adam([self.w, self.betas, self.sigmas], lr=lr)

        # Track loss
        loss = np.zeros(max_iterations)

        # Training loop
        for it in tqdm(range(max_iterations), desc="Model Training"):
            optim.zero_grad()
            cost = self.objective_function()
            cost.backward()
            optim.step()

            loss[it] = cost.data.numpy()

            # Log the loss
            if it % 100 == 0:
                tqdm.write(f"Loss: {loss[it]}")

    def fit(self) -> torch.Tensor:
        """
        Return ys which is a (n x G) matrix

        :return: ys which is a (n x G) matrix
        :rtype: torch.Tensor
        """
        ys = torch.zeros(self.X.shape[0], self.betas.shape[0])

        for g in range(self.G):
            ys[:, g] = (self.betas[g] * self.X).sum(-1)

        return ys

    def BIC(self) -> float:
        """Return Bayesian Information criteria"""
        rho = self.betas.numel() + self.sigmas.numel() + self.w.numel()
        bic =  -2 * self.objective_function() - rho * np.log(self.y.shape[0])
        return float(bic)

    def plot(self, col):
        """Plot given column"""
        plot_df = torch.cat((self.y.unsqueeze(-1), self.X), dim=-1)
        plot_df = pd.DataFrame(plot_df.numpy())
        plot_df = plot_df[[0, col]]

        sns.scatterplot(x=plot_df[col], y=plot_df[0], color='grey', s=2.0)

        y_fits = self.fit()

        for g in range(self.G):
            plot_df['y_fits'] = y_fits[:,g].detach().numpy()
            sns.scatterplot(x=plot_df[col], y=plot_df['y_fits'], color='red', s=2.0)

        plt.savefig("practice_flm_fit.png")
        plt.clf()

    def plot_colors(self, col, labs):
        """
            col: reference column
            labs: integer np.array
        """

        color_palette = sns.color_palette("bright", int(np.max(labs)) + 1)

        plot_df = torch.cat((self.y.unsqueeze(-1), self.X), dim=-1)
        plot_df = pd.DataFrame(plot_df.numpy())
        plot_df = plot_df[[0, col]]
        plot_df['colors'] = pd.Series(labs).apply(lambda x: color_palette[x])

        sns.scatterplot(x=plot_df[col], y=plot_df[0], color=plot_df['colors'], s=2.0)

        plt.savefig("practice_flm_fit_colors.png")
        plt.clf()

    def Estep(self) -> torch.Tensor:
        """Compute the expectation step and calculate weights"""
        # Disable automatic gradient calculation
        with torch.no_grad():
            dens = self.log_density().exp()
            W = self.compute_weights()
            dens = dens * W
            d_sum = dens.sum(-1).unsqueeze(-1)
            dens = dens / d_sum
            return dens

    def MAP(self) -> torch. Tensor:
        """Calculate labels using maximum a posteriori"""

        dens = self.Estep()
        labs = dens.argmax(-1)

        labs = labs.numpy()
        labs = labs.astype(int)

        return labs

    def plot_colors_MAP(self):
        """
            Plot MAP results
        """

        predictions = self.MAP()
        color_palette = sns.color_palette("bright", int(np.max(predictions)) + 1)

        plot_df = torch.cat((self.y.unsqueeze(-1), self.X), dim=-1)
        plot_df = pd.DataFrame(plot_df.numpy())
        plot_df = plot_df[[0, 1]]

        plot_df['colors'] = pd.Series(predictions).apply(lambda x: color_palette[x])

        sns.scatterplot(x=plot_df[1], y=plot_df[0], color=plot_df['colors'], s=2.0)

        plt.savefig("practice_flm_fit_colors_map.png")
        plt.clf()