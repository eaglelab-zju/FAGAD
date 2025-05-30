import torch
from torch_geometric.nn import MLP
from torch_geometric.utils import to_dense_adj

from .done import DONEBase


class AdONEBase(torch.nn.Module):
    """
    Adversarial Outlier Aware Attributed Network Embedding

    AdONE consists of an attribute autoencoder and a structure
    autoencoder. It estimates five loss to optimize the model,
    including an attribute proximity loss, an attribute homophily loss,
    a structure proximity loss, a structure homophily loss, and an
    alignment loss. It calculates three outlier scores, and averages
    them as an overall score. This model is transductive only.

    See :cite:`bandyopadhyay2020outlier` for details.

    Parameters
    ----------
    x_dim : int
        Input dimension of attribute.
    s_dim : int
        Input dimension of structure.
    hid_dim :  int, optional
        Hidden dimension of model. Default: ``64``.
    num_layers : int, optional
        Total number of layers in model. A half (floor) of the layers
        are for the encoder, the other half (ceil) of the layers are for
        decoders. Default: ``4``.
    dropout : float, optional
        Dropout rate. Default: ``0.``.
    weight_decay : float, optional
        Weight decay (L2 penalty). Default: ``0.``.
    act : callable activation function or None, optional
        Activation function if not None.
        Default: ``torch.nn.functional.relu``.
    w1 : float, optional
        Weight of structure proximity loss. Default: ``0.2``.
    w2 : float, optional
        Weight of structure homophily loss. Default: ``0.2``.
    w3 : float, optional
        Weight of attribute proximity loss. Default: ``0.2``.
    w4 : float, optional
        Weight of attribute homophily loss. Default: ``0.2``.
    w5 : float, optional
        Weight of alignment loss. Default: ``0.2``.
    **kwargs
        Other parameters for the backbone.
    """

    def __init__(self,
                 x_dim,
                 s_dim,
                 hid_dim=64,
                 num_layers=4,
                 dropout=0.0,
                 act=torch.nn.functional.relu,
                 w1=0.2,
                 w2=0.2,
                 w3=0.2,
                 w4=0.2,
                 w5=0.2,
                 **kwargs):
        super(AdONEBase, self).__init__()

        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.w4 = w4
        self.w5 = w5

        self.generator = DONEBase(x_dim=x_dim,
                                  s_dim=s_dim,
                                  hid_dim=hid_dim,
                                  num_layers=num_layers,
                                  dropout=dropout,
                                  act=act,
                                  w1=self.w1,
                                  w2=self.w2,
                                  w3=self.w3,
                                  w4=self.w4,
                                  w5=self.w5,
                                  **kwargs)

        self.discriminator = MLP(
            in_channels=hid_dim,
            hidden_channels=int(hid_dim / 2),
            out_channels=1,
            num_layers=2,
            dropout=dropout,
            act=torch.tanh,
        )
        self.emb = None
        self.inner = self.discriminator
        self.outer = self.generator

    def forward(self, x, s, edge_index):
        """
        Forward computation.

        Parameters
        ----------
        x : torch.Tensor
            Input attribute embeddings.
        s : torch.Tensor
            Input structure embeddings.
        edge_index : torch.Tensor
            Edge index.

        Returns
        -------
        x_ : torch.Tensor
            Reconstructed attribute embeddings.
        s_ : torch.Tensor
            Reconstructed structure embeddings.
        h_a : torch.Tensor
            Attribute hidden embeddings.
        h_s : torch.Tensor
            Structure hidden embeddings.
        dna : torch.Tensor
            Attribute neighbor distance.
        dns : torch.Tensor
            Structure neighbor distance.
        dis_a : torch.Tensor
            Attribute discriminator score.
        dis_s : torch.Tensor
            Structure discriminator score.
        """
        x_, s_, h_a, h_s, dna, dns = self.generator(x, s, edge_index)
        self.emb = (h_a, h_s)

        return x_, s_, h_a, h_s, dna, dns

    def loss_func_g(self, x, x_, s, s_, h_a, h_s, dna, dns):
        """
        Generator loss function for AdONE.

        Parameters
        ----------
        x : torch.Tensor
            Input attribute embeddings.
        x_ : torch.Tensor
            Reconstructed attribute embeddings.
        s : torch.Tensor
            Input structure embeddings.
        s_ : torch.Tensor
            Reconstructed structure embeddings.
        h_a : torch.Tensor
            Attribute hidden embeddings.
        h_s : torch.Tensor
            Structure hidden embeddings.
        dna : torch.Tensor
            Attribute neighbor distance.
        dns : torch.Tensor
            Structure neighbor distance.

        Returns
        -------
        loss : torch.Tensor
            Loss value.
        oa : torch.Tensor
            Attribute outlier scores.
        os : torch.Tensor
            Structure outlier scores.
        oc : torch.Tensor
            Combined outlier scores.
        """
        # equation 9 is based on the official implementation, and it
        # is slightly different from the paper
        dx = torch.sum(torch.pow(x - x_, 2), 1)
        tmp = self.w3 * dx + self.w4 * dna
        oa = tmp / torch.sum(tmp)

        # equation 8 is based on the official implementation, and it
        # is slightly different from the paper
        ds = torch.sum(torch.pow(s - s_, 2), 1)
        tmp = self.w1 * ds + self.w2 * dns
        os = tmp / torch.sum(tmp)

        # equation 10
        dc = torch.sum(torch.pow(h_a - h_s, 2), 1)
        oc = dc / torch.sum(dc)

        # equation 4
        loss_prox_a = torch.mean(torch.log(torch.pow(oa, -1)) * dx)

        # equation 5
        loss_hom_a = torch.mean(torch.log(torch.pow(oa, -1)) * dna)

        # equation 2
        loss_prox_s = torch.mean(torch.log(torch.pow(os, -1)) * ds)

        # equation 3
        loss_hom_s = torch.mean(torch.log(torch.pow(os, -1)) * dns)

        dis_a = torch.sigmoid(self.discriminator(h_a))
        dis_s = torch.sigmoid(self.discriminator(h_s))
        # equation 12
        loss_alg = torch.mean(
            torch.log(torch.pow(oc, -1)) *
            (torch.log(1 - dis_a) + torch.log(dis_s)))

        # equation 13
        loss = (self.w3 * loss_prox_a + self.w4 * loss_hom_a +
                self.w1 * loss_prox_s + self.w2 * loss_hom_s +
                self.w5 * loss_alg)

        return loss, oa, os, oc

    def loss_func_d(self, h_a, h_s):
        """
        Discriminator loss function for AdONE.

        Parameters
        ----------
        h_a : torch.Tensor
            Attribute hidden embeddings.
        h_s : torch.Tensor
            Structure hidden embeddings.

        Returns
        -------
        loss : torch.Tensor
            Loss value.
        """
        # equation 11
        dis_a = torch.sigmoid(self.discriminator(h_a))
        dis_s = torch.sigmoid(self.discriminator(h_s))
        loss = -torch.mean(torch.log(1 - dis_a) + torch.log(dis_s))
        return loss

    @staticmethod
    def process_graph(data):
        """
        Obtain the dense adjacency matrix of the graph.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Input graph.
        """
        data.s = to_dense_adj(data.edge_index)[0]
