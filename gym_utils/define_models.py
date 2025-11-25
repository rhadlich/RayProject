import numpy as np
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 num_hidden: int,
                 hidden_exp: int,
                 dropout: float
                 ):
        super(MLP, self).__init__()
        self.dropout = nn.Dropout(dropout)
        hidden_dim = int((2 ** hidden_exp) / (1-dropout))
        self.fc_in = nn.Linear(input_dim, hidden_dim)
        self.fc_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.silu = nn.SiLU()
        self.network = nn.Sequential(
            self.fc_in, self.silu,
            *[nn.Sequential(self.fc_hidden, self.silu, self.dropout) for _ in range(num_hidden)],
            self.fc_out,
        )

    def forward(self, x):
        return self.network(x)


class MLP_with_GRU_head(nn.Module):
    def __init__(self,
                 input_dim,
                 num_hidden_MLP,
                 hidden_exp_MLP,
                 hidden_dim_GRU,
                 seq_len,
                 ):
        super(MLP_with_GRU_head, self).__init__()
        # ----- Encoder ----------------------------------------
        hidden_dim = int(2 ** hidden_exp_MLP)
        self.fc_in = nn.Linear(input_dim, hidden_dim)
        self.fc_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, hidden_dim_GRU)
        self.silu = nn.SiLU()
        self.enc = nn.Sequential(
            self.fc_in, self.silu,
            *[nn.Sequential(self.fc_hidden, self.silu) for _ in range(num_hidden_MLP)],
            self.fc_out,
        )

        # ----- Decoder ----------------------------------------
        self.gru = nn.GRU(input_size=1, hidden_size=hidden_dim_GRU, batch_first=True)
        self.to_p = nn.Linear(hidden_dim_GRU, 1)
        self.seq_len = seq_len

    def forward(self, input_tuple: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        Perform forward pass through encoder and decoder networks. Will be using
        teacher-forced sequence for training, i.e. feeding previous true pressure
        value as the input for the next prediction step. In inference this will
        instead be the previous prediction that will be used as input for the next.

        Inputs:
            input_tuple (tuple): contains the following:
                source (torch.Tensor): input data (batch_size, input_dim)
                p_in (torch.Tensor): crank-resolved pressure (batch_size, seq_len)

        Output:
            prediction (torch.Tensor): prediction (batch_size, seq_len)

        Nomenclature for dimensions:
            B: batch_size
            T: seq_len
        """
        source, p_in = input_tuple
        # ----- Encode the operating point ---------------------------
        h0 = torch.tanh(self.enc(source))           # (B, hidden_dim_GRU)
        h0 = h0.unsqueeze(0)                        # (1, B, hidden_dim_GRU)

        # ----- Decode into sequence ---------------------------------
        y, _ = self.gru(p_in.unsqueeze(-1), h0)     # y: (B, T, hidden_dim_GRU), p_in.unsqueeze(-1): (B, T, 1)
        p_hat = self.to_p(y).squeeze(-1)            # (B, T)
        return p_hat

    @torch.no_grad()
    def inference_no_grad(self, x_scalar: torch.Tensor):
        # get encoded data point
        h0 = torch.tanh(self.enc(x_scalar))
        h = h0.unsqueeze(0)

        # perform decoding
        p_prev = torch.zeros(x_scalar.size(0), 1, device=x_scalar.device)    # maybe an estimate of initial pressure?
        outputs = []
        for _ in range(self.seq_len):
            y, h = self.gru(p_prev.unsqueeze(-1), h)
            p_prev = self.to_p(y).squeeze(-1)
            outputs.append(p_prev)

        p_hat = torch.cat(outputs, dim=-1)
        return p_hat


class MSEWithDp(nn.Module):
    def __init__(self, alpha: float = 0.01, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        if reduction not in ("mean", "sum"):
            raise ValueError(f"Reduction {reduction} is not supported, must be 'mean' or 'sum'")
        self.reduction = reduction

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # difference in pressure
        diff = (target - output)**2

        # central finite difference to get dp and difference in dp
        dp_hat = output[:, 2:] - output[:, :-2]
        dp = target[:, 2:] - target[:, :-2]
        diff_dp = (dp_hat - dp)**2

        if self.reduction == "mean":
            return diff.mean() + self.alpha * diff_dp.mean()
        if self.reduction == "sum":
            return diff.sum() + self.alpha * diff_dp.sum()

