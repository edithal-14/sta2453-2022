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

class FiniteLinearModelEM():
    """Finite linear model using EM algo"""
    def __init__(self, G: int, data: np.ndarray):
        """
        :param G: number of components
        :type G: int
        :param data: data to be fitted
        :type data: np.ndarray
        """

        if isinstance(data, torch.Tensor):
            data = data.numpy()

        # define constants
        self.n = data.shape[0]
        self.G = G

        # define data
        self.X = data[:, 1:]
        self.X = np.concatenate((self.X, np.ones(shape=(self.n, 1))), axis=1)
        self.y = data[:, 0] 

        # number of covariates
        self.p = self.X.shape[1]

        # define parameters
        self.betas = np.random.normal(size=(self.G, self.p))
        self.sigmas = np.abs(np.random.normal(size=(self.G,)))
        # mixing ratios
        self.w = np.abs(np.random.normal(size=(self.G,)))
        # w's should sum up to one for all i's and a given j
        self.w = scipy.special.softmax(self.w)
        # latent variable matrix of dimension = n * G
        self.z = np.random.normal(size=(self.n, self.G))
        # Z_ij should sum up to one for all i's and a given j
        self.z = scipy.special.softmax(self.z, axis=1)

    def Estep(self) -> np.ndarray:
        """Compute the expectation step and calculate weights"""
        dens = np.exp(self.log_density())
        dens1 = dens * self.w
        d_sum = dens1.sum(-1)[..., None]
        self.z = dens1 / d_sum

    def Mstep(self):
        """
        Estimate parameters to maximize likelihood
        """
        self.w = self.z.sum(axis=0)/self.z.shape[0]
        for g in range(self.G):
            self.sigmas[g] = np.sqrt((
                self.z[:, g] * np.power((self.y - self.X @ self.betas[g, :].T), 2)
            ).sum() / self.z[:, g].sum())
        # beta_new = (X' * W * X)^-1 * (X' * W * y)
        for g in range(self.G):
            term1 = np.linalg.inv(self.X.T @ np.diag(self.z[:,g]) @ self.X)
            self.betas[g,:] = term1 @ (self.X.T @ np.diag(self.z[:,g]) @ self.y)

    def fit(self, max_iter=10):
        """Run EM algorithm"""
        for it in tqdm(range(max_iter), desc="Running EM algo"):
            self.Estep()
            self.Mstep()

            if it % 1 == 0:
                tqdm.write(f"Loss: {self.objective_function()}")

    def objective_function(self) -> np.ndarray:
        """
        Return negative log likelihood objective function to minimize

        :returns: Objective function
        :rtype: np.ndarray
        """
        # n * G 
        dens = np.exp(self.log_density())

        # What if `dens` has extremely small values?
        output = - np.log((dens * self.w).sum(-1)).sum()

        return output

    def log_density(self) -> np.ndarray:
        """
            Take in covariate X and response y,
            compute n * G matrix of log densities

            :return: n * G matrix of log densities
            :rtype: np.ndarray 
        """
        ldens = np.zeros(shape=(self.n, self.G))

        # iterate through components
        for g in range(self.G):

            # compute linear model
            # element wise multiplication then sum inner axis
            y_hats = (self.betas[g] * self.X).sum(-1)

            # compute log of gaussian density
            term1 = - 0.5 * np.log([2 * np.pi])
            term2 = - np.log(self.sigmas[g])
            term3 = - 0.5 * np.power(((y_hats - self.y) / self.sigmas[g]), 2)
            ldens[:, g] = term1 + term2 + term3
        
        return ldens

    def MAP(self) -> np.ndarray:
        """Calculate labels using maximum a posteriori"""

        return np.argmax(self.z, axis=1).astype(int)

    def plot_colors_MAP(self):
        """
            Plot MAP results
        """

        predictions = self.MAP()
        color_palette = sns.color_palette("bright", int(np.max(predictions)) + 1)

        plot_df = np.concatenate((self.y[..., None], self.X), axis=-1)
        plot_df = pd.DataFrame(plot_df)
        plot_df = plot_df[[0, 1]]

        plot_df['colors'] = pd.Series(predictions).apply(lambda x: color_palette[x])

        sns.scatterplot(x=plot_df[1], y=plot_df[0], color=plot_df['colors'], s=2.0)

        plt.savefig("practice_flm_fit_colors_map.png")
        plt.clf()

    def BIC(self) -> float:
        """Return Bayesian Information criteria"""
        rho = self.betas.numel() + self.sigmas.numel() + self.w.numel()
        bic =  -2 * self.objective_function() - rho * np.log(self.y.shape[0])
        return float(bic)

    def plot(self):
        """Plot given column"""
        plot_df = np.concatenate((self.y[..., None], self.X), axis=-1)
        plot_df = pd.DataFrame(plot_df)
        plot_df = plot_df[[0, 1]]

        predictions = self.MAP()
        color_palette = sns.color_palette("bright", int(np.max(predictions)) + 1)
        plot_df['colors'] = pd.Series(predictions).apply(lambda x: color_palette[x])

        sns.scatterplot(x=plot_df[1], y=plot_df[0], color=plot_df['colors'], s=2.0)

        for g in range(self.G):
            y_fits = (self.betas[g] * self.X).sum(-1)
            plot_df['y_fits'] = y_fits
            sns.scatterplot(x=plot_df[1], y=plot_df['y_fits'], color='red', s=2.0)

        plt.savefig("practice_flm_fit.png")
        plt.clf()

    def plot_colors(self, col, labs):
        """
            col: reference column
            labs: integer np.array
        """

        color_palette = sns.color_palette("bright", int(np.max(labs)) + 1)

        plot_df = np.concatenate((self.y[..., None], self.X), axis=-1)
        plot_df = pd.DataFrame(plot_df)
        plot_df = plot_df[[0, col]]
        plot_df['colors'] = pd.Series(labs).apply(lambda x: color_palette[x])

        sns.scatterplot(x=plot_df[col], y=plot_df[0], color=plot_df['colors'], s=2.0)

        plt.savefig("practice_flm_fit_colors.png")
        plt.clf()

def load_french_motor():
    """Load french motor dataset"""
    df = pd.read_csv("../week3/french_motor.csv")
    # Discard first column
    df = df.iloc[:,1:]
    return df

if __name__ == "__main__":
    G = 3
    data_df = load_french_motor()
    data_s = data_df[['y_log', 'dens']].to_numpy()
    data_s = torch.Tensor(data_s)

    # Standardization
    data_s = (data_s - data_s.mean())/data_s.std()

    flm = FiniteLinearModelEM(G=G, data=data_s)

    # Run EM algo for 1000 iterations, then plot colors
    flm.fit(max_iter=10)
    flm.plot()