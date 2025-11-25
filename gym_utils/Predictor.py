import torch
import torch.nn as nn
import numpy as np
from gym_utils.define_models import MLP


class Predictor:
    def __init__(self) -> None:
        self.device = torch.device('cpu')

        # info for data normalization
        self.mean = np.array([6.5101062e+02, -7.2337663e-01, 4.4961038e-01])
        self.std = np.array([1.4219507e+02, 1.6561491e+00, 6.6634133e-02])

        # create cad vector and tensor for appending to data
        cad = np.arange(-360, 360, 0.1)
        self.cad_plt = cad
        self.res = 720 / self.cad_plt.size      # resolution of pressure trace
        self.cad = np.reshape(self.cad_plt, [1, -1, 1])

        # engine parameters for IMEP calc.
        b = 79 / 1000  # bore
        s = 86 / 1000  # stroke
        CR = 17.19  # compression ratio
        l = 160 / 1000  # connecting rod length
        a = s / 2  # crank radius
        delta = 0.6 / 1000  # piston pin offset
        R = l / a
        self.Vs = np.pi * ((b / 2) ** 2) * (2 * a)  # stroke volume
        Vc = self.Vs / (CR - 1)  # clearance volume
        self.V = Vc * (1 + (0.5 * (CR - 1)) * (R + 1 - np.cos(np.radians(self.cad_plt)) - np.sqrt(
            R ** 2 - (np.sin(np.radians(self.cad_plt)) + delta) ** 2)))  # Total Volume vector
        self.V = self.V[1800:-1800]
        self.V_p = self.V[1:]
        self.V_m = self.V[:-1]

        # miscellaneous
        self.ones = np.ones([1, 7200, 3])

    def init_model(self,
                   input_size: int,
                   num_layers: int,
                   layer_exp: int,
                   out_size: int,
                   dropout: float,
                   weights_path: str):
        # create model and load weights
        self.model = MLP(
            input_dim=input_size,
            output_dim=out_size,
            num_hidden=num_layers,
            hidden_exp=layer_exp,
            dropout=dropout,
        )
        self.model = self.model.to(self.device)
        # self.model = torch.compile(self.model, mode="max-autotune")
        self.model.load_state_dict(torch.load(weights_path))
        self.model.eval()

    def format_data(self, data):
        data = (data - self.mean) / self.std
        data = np.float32(data)
        data = torch.tensor(data)
        return data

    def model_predict(self, values, *, noise_in_percent=None):
        # case of no injection
        if values[2] == 0:
            p = np.zeros(self.cad_plt.size)

        # if injection
        else:
            # get data in correct format
            data = self.format_data(values)
            # make prediction and format it
            p = self.model(data)
            p = np.squeeze(p.detach().cpu().numpy())

        # work calculations
        p_p = p[1:]
        p_m = p[:-1]
        work = np.sum((p_p + p_m) / 2 * (self.V_p - self.V_m))  # net work
        imep = work / self.Vs

        # MPRR calculation
        prr = np.diff(p, 1)*1/self.res
        mprr = max(prr)

        # add noise if desired
        if noise_in_percent is not None:
            imep = imep + np.random.normal(0, imep * noise_in_percent / 100)
            mprr = mprr + np.random.normal(0, mprr * noise_in_percent / 100)
        return p, imep, mprr, self.cad_plt

# For RL, the states would have to be (assuming only IMEP matters) the current IMEP and the desired IMEP, and the
# reward would be a negative of the MAE between current and desired IMEP.

# Start with SARSA, then build from there

# Define episode as 10 time steps
