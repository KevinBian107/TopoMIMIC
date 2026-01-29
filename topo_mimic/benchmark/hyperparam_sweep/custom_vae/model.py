"""RNN-VAE model for behavioral sequence encoding."""

import torch
import torch.nn as nn
from torch import Tensor


class RNNVAE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 30,
        hidden_dim: int = 256,
        rnn_layers: int = 1,
        rnn_type: str = "GRU",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.rnn_layers = rnn_layers
        self.rnn_type = rnn_type

        rnn_class = nn.GRU if rnn_type == "GRU" else nn.LSTM
        self.encoder_rnn = rnn_class(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=rnn_layers,
            batch_first=True,
            bidirectional=False,
        )

        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.fc_decode_init = nn.Linear(latent_dim, hidden_dim)
        self.decoder_rnn = rnn_class(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=rnn_layers,
            batch_first=True,
            bidirectional=False,
        )
        self.fc_output = nn.Linear(hidden_dim, input_dim)

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        batch_size = x.size(0)

        if self.rnn_type == "GRU":
            _, h_n = self.encoder_rnn(x)
            h_final = h_n[-1]
        else:
            _, (h_n, _) = self.encoder_rnn(x)
            h_final = h_n[-1]

        mu = self.fc_mu(h_final)
        logvar = self.fc_logvar(h_final)
        return mu, logvar

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: Tensor, seq_len: int, target: Tensor | None = None) -> Tensor:
        batch_size = z.size(0)

        h_init = self.fc_decode_init(z)
        h_init = h_init.unsqueeze(0).repeat(self.rnn_layers, 1, 1)

        if self.rnn_type == "LSTM":
            c_init = torch.zeros_like(h_init)
            hidden = (h_init, c_init)
        else:
            hidden = h_init

        if target is not None:
            decoder_input = target
        else:
            decoder_input = torch.zeros(batch_size, seq_len, self.input_dim, device=z.device)

        output, _ = self.decoder_rnn(decoder_input, hidden)
        recon = self.fc_output(output)
        return recon

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z, x.size(1), target=x)
        return recon_x, mu, logvar

    def encode_sequence(self, x: Tensor) -> Tensor:
        mu, _ = self.encode(x)
        return mu
