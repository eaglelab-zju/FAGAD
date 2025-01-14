"""Graph Augmentation
Adapted from https://github.com/PyGCL/PyGCL/blob/main/GCL/augmentors/augmentor.py
"""

from copy import deepcopy
from typing import List

import dgl
import torch
from dgl import BaseTransform

from .transforms import AddEdge
from .transforms import DropEdge
from .transforms import DropNode
from .transforms import FeatureDropout
from .transforms import NodeShuffle
from .transforms import RandomMask


# pylint:disable=no-else-return,eval-used
class ComposeAug(BaseTransform):
    """Execute graph augments in sequence.

    Parameters
    ----------
    augs : List[BaseTransform]
        graphs augments using DGL tansform
    cross : bool, optional
        if use cross graph augments, by default True
    """

    def __init__(self, augs: List[BaseTransform], cross: bool = True) -> None:
        super().__init__()
        self.augs = augs
        self.cross = cross

    def __call__(self, g: dgl.DGLGraph):
        """Execute augments on graph

        Parameters
        ----------
        g : dgl.DGLGraph
            raw graph

        Returns
        -------
        if cross == True:
            return cross augmented graph
        else:
            return multiple augmented graphs
        """
        if self.cross:
            for aug in self.augs:
                g = aug(g)
            return g
        else:
            graphs = []
            tmpg = deepcopy(g)
            for aug in self.augs:
                newg = aug(tmpg)
                tmpg = deepcopy(g)
                graphs.append(newg)
            return graphs


class RandomChoiceAug(BaseTransform):
    """Execute graph augments in random.

    Parameters
    ----------
    augs : List[BaseTransform]
        graphs augments using DGL tansform
    n_choices : int
        number of choice aug types
    cross : bool, optional
        if use cross graph augments, by default True
    """

    def __init__(self,
                 augs: List[BaseTransform],
                 n_choices: int,
                 cross: bool = True) -> None:
        super().__init__()
        assert n_choices <= len(augs), "n_choices should <= augs length"
        self.augs = augs
        self.n_choices = n_choices
        self.cross = cross

    def __call__(self, g):
        """Execute augments on graph

        Parameters
        ----------
        g : dgl.DGLGraph
            raw graph

        Returns
        -------
        if cross == True:
            return cross augmented graph
        else:
            return multiple augmented graphs
        """
        n_augs = len(self.augs)
        perm = torch.randperm(n_augs)
        idx = perm[:self.n_choices]

        if self.cross:
            for i in idx:
                aug = self.augs[i]
                g = aug(g)
            return g
        else:
            graphs = []
            tmpg = deepcopy(g)
            for i in idx:
                aug = self.augs[i]
                newg = aug(tmpg)
                tmpg = deepcopy(g)
                graphs.append(newg)
            return graphs


# pylint: disable=unused-argument
class aug_none:
    """none aug"""

    def __call__(self, graph):
        return graph


aug_maps = {
    "add_edge": AddEdge,
    "drop_edge": DropEdge,
    "drop_node": DropNode,
    "feat_dropout": FeatureDropout,
    "node_shuffle": NodeShuffle,
    "random_mask": RandomMask,
    "none": aug_none,
}


def get_augments(aug_types: List = None):
    """Generate augments list.

    Args:
        aug_types (List): str type list. Defaults to None.
            e.g. ['random_mask:p=0.2','node_shuffle:is_use=True']

    Return:
        augs (List): augs list
    """
    augs = []
    for aug in aug_types:
        t = aug.split(":")
        if len(t) > 1:
            d = {
                i.split("=")[0]: eval(i.split("=")[1])
                for i in t[1].split(",")
            }
            augs.append(aug_maps[t[0]](**d))
        else:
            augs.append(aug_maps[t[0]]())

    return augs


# if __name__=='__main__':
#     import dgl
#     g = dgl.rand_graph(4,2)
#     g.ndata['feat'] = torch.rand((4,5))
#     print(g.ndata['feat'])
#     # 'random_mask','0.3','drop_edge','0.2'
#     # 'random_mask','drop_edge'
#     # 'random_mask','drop_edge','0.2'
#     # 'random_mask','node_shuffle','True'
#     # 'random_mask:p=0.2', 'node_shuffle:is_use=True'
#     augsss=get_augments(['random_mask:p=0.2', 'node_shuffle:is_use=True'])
#     transform = ComposeAug(augsss,cross=False)
#     gs = transform(g)
#     print(gs)
