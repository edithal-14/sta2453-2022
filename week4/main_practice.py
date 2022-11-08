import pandas as pd
import numpy as np
import torch
from flm_em_practice import FiniteLinearModelEM

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

    # Run some tests on the model
    # test_lg = flm.log_density()
    # test_obj = flm.objective_function()

    # flm.train(lr=1e-3, max_iterations=1000)

    # flm.plot(1)

    # flm.plot_colors(1, data_df["labs"])

    # flm.plot_colors_MAP()

    # Run EM algo for 1000 iterations, then plot colors
    flm.fit(max_iter=10)
    flm.plot()