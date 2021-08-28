from dgllife.model.gnn import GAT, AttentiveFPGNN
import dgl.function as fn
import torch
torch.set_default_tensor_type('torch.FloatTensor')
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import edge_softmax
import dgl
from dgllife.model.readout.weighted_sum_and_max import WeightedSumAndMax


class FC(nn.Module):
    def __init__(self, d_graph_layer, d_FC_layer, n_FC_layer, dropout, n_tasks):
        super(FC, self).__init__()
        self.d_graph_layer = d_graph_layer
        self.d_FC_layer = d_FC_layer
        self.n_FC_layer = n_FC_layer
        self.dropout = dropout
        self.predict = nn.ModuleList()
        for j in range(self.n_FC_layer):
            if j == 0:
                self.predict.append(nn.Linear(self.d_graph_layer, self.d_FC_layer))
                self.predict.append(nn.Dropout(self.dropout))
                self.predict.append(nn.LeakyReLU())
                self.predict.append(nn.BatchNorm1d(d_FC_layer))
            if j == self.n_FC_layer - 1:
                self.predict.append(nn.Linear(self.d_FC_layer, n_tasks))
            else:
                self.predict.append(nn.Linear(self.d_FC_layer, self.d_FC_layer))
                self.predict.append(nn.Dropout(self.dropout))
                self.predict.append(nn.LeakyReLU())
                self.predict.append(nn.BatchNorm1d(d_FC_layer))

    def forward(self, h):
        for layer in self.predict:
            h = layer(h)
        # return torch.sigmoid(h)
        return h


class AttentiveGRU1(nn.Module):
    """Update node features with attention and GRU.

    This will be used for incorporating the information of edge features
    into node features for message passing.

    Parameters
    ----------
    node_feat_size : int
        Size for the input node features.
    edge_feat_size : int
        Size for the input edge (bond) features.
    edge_hidden_size : int
        Size for the intermediate edge (bond) representations.
    dropout : float
        The probability for performing dropout.
    """

    def __init__(self, node_feat_size, edge_feat_size, edge_hidden_size, dropout):
        super(AttentiveGRU1, self).__init__()

        self.edge_transform = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(edge_feat_size, edge_hidden_size)
        )
        self.gru = nn.GRUCell(edge_hidden_size, node_feat_size)

    def forward(self, g, edge_logits, edge_feats, node_feats):
        """Update node representations.

        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs
        edge_logits : float32 tensor of shape (E, 1)
            The edge logits based on which softmax will be performed for weighting
            edges within 1-hop neighborhoods. E represents the number of edges.
        edge_feats : float32 tensor of shape (E, edge_feat_size)
            Previous edge features.
        node_feats : float32 tensor of shape (V, node_feat_size)
            Previous node features. V represents the number of nodes.

        Returns
        -------
        float32 tensor of shape (V, node_feat_size)
            Updated node features.
        """
        g = g.local_var()
        g.edata['e'] = edge_softmax(g, edge_logits) * self.edge_transform(edge_feats)
        g.update_all(fn.copy_edge('e', 'm'), fn.sum('m', 'c'))
        context = F.elu(g.ndata['c'])
        return F.relu(self.gru(context, node_feats))


class AttentiveGRU2(nn.Module):
    """Update node features with attention and GRU.

    This will be used in GNN layers for updating node representations.

    Parameters
    ----------
    node_feat_size : int
        Size for the input node features.
    edge_hidden_size : int
        Size for the intermediate edge (bond) representations.
    dropout : float
        The probability for performing dropout.
    """

    def __init__(self, node_feat_size, edge_hidden_size, dropout):
        super(AttentiveGRU2, self).__init__()

        self.project_node = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(node_feat_size, edge_hidden_size)
        )
        self.gru = nn.GRUCell(edge_hidden_size, node_feat_size)

    def forward(self, g, edge_logits, node_feats):
        """Update node representations.

        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs
        edge_logits : float32 tensor of shape (E, 1)
            The edge logits based on which softmax will be performed for weighting
            edges within 1-hop neighborhoods. E represents the number of edges.
        node_feats : float32 tensor of shape (V, node_feat_size)
            Previous node features. V represents the number of nodes.

        Returns
        -------
        float32 tensor of shape (V, node_feat_size)
            Updated node features.
        """
        g = g.local_var()
        g.edata['a'] = edge_softmax(g, edge_logits)
        g.ndata['hv'] = self.project_node(node_feats)

        g.update_all(fn.src_mul_edge('hv', 'a', 'm'), fn.sum('m', 'c'))
        context = F.elu(g.ndata['c'])
        return F.relu(self.gru(context, node_feats))


class GetContext(nn.Module):
    """Generate context for each node by message passing at the beginning.

    This layer incorporates the information of edge features into node
    representations so that message passing needs to be only performed over
    node representations.

    Parameters
    ----------
    node_feat_size : int
        Size for the input node features.
    edge_feat_size : int
        Size for the input edge (bond) features.
    graph_feat_size : int
        Size of the learned graph representation (molecular fingerprint).
    dropout : float
        The probability for performing dropout.
    """

    def __init__(self, node_feat_size, edge_feat_size, graph_feat_size, dropout):
        super(GetContext, self).__init__()

        self.project_node = nn.Sequential(
            nn.Linear(node_feat_size, graph_feat_size),
            nn.LeakyReLU()
        )
        self.project_edge1 = nn.Sequential(
            nn.Linear(node_feat_size + edge_feat_size, graph_feat_size),
            nn.LeakyReLU()
        )
        self.project_edge2 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2 * graph_feat_size, 1),
            nn.LeakyReLU()
        )
        self.attentive_gru = AttentiveGRU1(graph_feat_size, graph_feat_size,
                                           graph_feat_size, dropout)

    def apply_edges1(self, edges):
        """Edge feature update.

        Parameters
        ----------
        edges : EdgeBatch
            Container for a batch of edges

        Returns
        -------
        dict
            Mapping ``'he1'`` to updated edge features.
        """
        return {'he1': torch.cat([edges.src['hv'], edges.data['he']], dim=1)}

    def apply_edges2(self, edges):
        """Edge feature update.

        Parameters
        ----------
        edges : EdgeBatch
            Container for a batch of edges

        Returns
        -------
        dict
            Mapping ``'he2'`` to updated edge features.
        """
        return {'he2': torch.cat([edges.dst['hv_new'], edges.data['he1']], dim=1)}

    def forward(self, g, node_feats, edge_feats):
        """Incorporate edge features and update node representations.

        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs.
        node_feats : float32 tensor of shape (V, node_feat_size)
            Input node features. V for the number of nodes.
        edge_feats : float32 tensor of shape (E, edge_feat_size)
            Input edge features. E for the number of edges.

        Returns
        -------
        float32 tensor of shape (V, graph_feat_size)
            Updated node features.
        """
        g = g.local_var()
        g.ndata['hv'] = node_feats
        g.ndata['hv_new'] = self.project_node(node_feats)
        g.edata['he'] = edge_feats

        g.apply_edges(self.apply_edges1)
        g.edata['he1'] = self.project_edge1(g.edata['he1'])
        g.apply_edges(self.apply_edges2)
        logits = self.project_edge2(g.edata['he2'])

        return self.attentive_gru(g, logits, g.edata['he1'], g.ndata['hv_new'])


class GNNLayer(nn.Module):
    """GNNLayer for updating node features.

    This layer performs message passing over node representations and update them.

    Parameters
    ----------
    node_feat_size : int
        Size for the input node features.
    graph_feat_size : int
        Size for the graph representations to be computed.
    dropout : float
        The probability for performing dropout.
    """

    def __init__(self, node_feat_size, graph_feat_size, dropout):
        super(GNNLayer, self).__init__()

        self.project_edge = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2 * node_feat_size, 1),
            nn.LeakyReLU()
        )
        self.attentive_gru = AttentiveGRU2(node_feat_size, graph_feat_size, dropout)
        self.bn_layer = nn.BatchNorm1d(graph_feat_size)

    def apply_edges(self, edges):
        """Edge feature generation.

        Generate edge features by concatenating the features of the destination
        and source nodes.

        Parameters
        ----------
        edges : EdgeBatch
            Container for a batch of edges.

        Returns
        -------
        dict
            Mapping ``'he'`` to the generated edge features.
        """
        return {'he': torch.cat([edges.dst['hv'], edges.src['hv']], dim=1)}

    def forward(self, g, node_feats):
        """Perform message passing and update node representations.

        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs.
        node_feats : float32 tensor of shape (V, node_feat_size)
            Input node features. V for the number of nodes.

        Returns
        -------
        float32 tensor of shape (V, graph_feat_size)
            Updated node features.
        """
        g = g.local_var()
        g.ndata['hv'] = node_feats
        g.apply_edges(self.apply_edges)
        logits = self.project_edge(g.edata['he'])

        return self.bn_layer(self.attentive_gru(g, logits, node_feats))


class ModifiedAttentiveFPGNNV2(nn.Module):
    """`Pushing the Boundaries of Molecular Representation for Drug Discovery with the Graph
    Attention Mechanism <https://www.ncbi.nlm.nih.gov/pubmed/31408336>`__

    This class performs message passing in AttentiveFP and returns the updated node representations.

    Parameters
    ----------
    node_feat_size : int
        Size for the input node features.
    edge_feat_size : int
        Size for the input edge features.
    num_layers : int
        Number of GNN layers. Default to 2.
    graph_feat_size : int
        Size for the graph representations to be computed. Default to 200.
    dropout : float
        The probability for performing dropout. Default to 0.
    """

    def __init__(self,
                 node_feat_size,
                 edge_feat_size,
                 num_layers=2,
                 graph_feat_size=200,
                 dropout=0.):
        super(ModifiedAttentiveFPGNNV2, self).__init__()

        self.init_context = GetContext(node_feat_size, edge_feat_size, graph_feat_size, dropout)
        self.gnn_layers = nn.ModuleList()
        self.sum_node_feats = 0
        for _ in range(num_layers - 1):
            self.gnn_layers.append(GNNLayer(graph_feat_size, graph_feat_size, dropout))

    def forward(self, g, node_feats, edge_feats):
        """Performs message passing and updates node representations.

        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs.
        node_feats : float32 tensor of shape (V, node_feat_size)
            Input node features. V for the number of nodes.
        edge_feats : float32 tensor of shape (E, edge_feat_size)
            Input edge features. E for the number of edges.

        Returns
        -------
        node_feats : float32 tensor of shape (V, graph_feat_size)
            Updated node representations.
        """
        node_feats = self.init_context(g, node_feats, edge_feats)
        self.sum_node_feats = node_feats
        for gnn in self.gnn_layers:
            node_feats = gnn(g, node_feats)
            self.sum_node_feats = self.sum_node_feats + node_feats
        return self.sum_node_feats


class ModifiedAttentiveFPPredictorV2(nn.Module):
    """AttentiveFP for regression and classification on graphs.

    AttentiveFP is introduced in `Pushing the Boundaries of Molecular Representation for Drug
    Discovery with the Graph Attention Mechanism.
     <https://www.ncbi.nlm.nih.gov/pubmed/31408336>`__

    Parameters
    ----------
    node_feat_size : int
        Size for the input node features.
    edge_feat_size : int
        Size for the input edge features.
    num_layers : int
        Number of GNN layers. Default to 2.
    graph_feat_size : int
        Size for the learned graph representations. Default to 200.
    dropout : float
        Probability for performing the dropout. Default to 0.
    """

    def __init__(self,
                 node_feat_size,
                 edge_feat_size,
                 num_layers=2,
                 graph_feat_size=200,
                 dropout=0.):
        super(ModifiedAttentiveFPPredictorV2, self).__init__()

        self.gnn = ModifiedAttentiveFPGNNV2(node_feat_size=node_feat_size,
                                            edge_feat_size=edge_feat_size,
                                            num_layers=num_layers,
                                            graph_feat_size=graph_feat_size,
                                            dropout=dropout)
        self.predict = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(graph_feat_size, 1)
        )

    def forward(self, g, node_feats, edge_feats):
        """Graph-level regression/soft classification.

        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs.
        node_feats : float32 tensor of shape (V, node_feat_size)
            Input node features. V for the number of nodes.
        edge_feats : float32 tensor of shape (E, edge_feat_size)
            Input edge features. E for the number of edges.

        Returns
        -------
        float32 tensor of shape (G, n_tasks)
            Prediction for the graphs in the batch. G for the number of graphs.
        node_weights : list of float32 tensor of shape (V, 1), optional
            This is returned when ``get_node_weight`` is ``True``.
            The list has a length ``num_timesteps`` and ``node_weights[i]``
            gives the node weights in the i-th update.
        """
        sum_node_feats = self.gnn(g, node_feats, edge_feats)
        return sum_node_feats


class ModifiedAttentiveFPGNNV3(nn.Module):
    """`Pushing the Boundaries of Molecular Representation for Drug Discovery with the Graph
    Attention Mechanism <https://www.ncbi.nlm.nih.gov/pubmed/31408336>`__

    This class performs message passing in AttentiveFP and returns the updated node representations.

    Parameters
    ----------
    node_feat_size : int
        Size for the input node features.
    edge_feat_size : int
        Size for the input edge features.
    num_layers : int
        Number of GNN layers. Default to 2.
    graph_feat_size : int
        Size for the graph representations to be computed. Default to 200.
    dropout : float
        The probability for performing dropout. Default to 0.
    """

    def __init__(self,
                 node_feat_size,
                 edge_feat_size,
                 num_layers=2,
                 graph_feat_size=200,
                 dropout=0.):
        super(ModifiedAttentiveFPGNNV3, self).__init__()

        self.init_context = GetContext(node_feat_size, edge_feat_size, graph_feat_size, dropout)
        self.gnn_layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.gnn_layers.append(GNNLayer(graph_feat_size, graph_feat_size, dropout))
        self.predict = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(graph_feat_size, 1)
        )
        self.sum_predictions = 0
        self.num_layers = num_layers

    def forward(self, g, node_feats, edge_feats):
        """Performs message passing and updates node representations.

        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs.
        node_feats : float32 tensor of shape (V, node_feat_size)
            Input node features. V for the number of nodes.
        edge_feats : float32 tensor of shape (E, edge_feat_size)
            Input edge features. E for the number of edges.

        Returns
        -------
        node_feats : float32 tensor of shape (V, graph_feat_size)
            Updated node representations.
        """
        node_feats = self.init_context(g, node_feats, edge_feats)
        for gnn in self.gnn_layers:
            node_feats = gnn(g, node_feats)
            self.sum_predictions = self.sum_predictions + self.predict(node_feats)
        return self.sum_predictions / (self.num_layers - 1)


class ModifiedAttentiveFPPredictorV3(nn.Module):
    """AttentiveFP for regression and classification on graphs.

    AttentiveFP is introduced in `Pushing the Boundaries of Molecular Representation for Drug
    Discovery with the Graph Attention Mechanism.
     <https://www.ncbi.nlm.nih.gov/pubmed/31408336>`__

    Parameters
    ----------
    node_feat_size : int
        Size for the input node features.
    edge_feat_size : int
        Size for the input edge features.
    num_layers : int
        Number of GNN layers. Default to 2.
    graph_feat_size : int
        Size for the learned graph representations. Default to 200.
    dropout : float
        Probability for performing the dropout. Default to 0.
    """

    def __init__(self,
                 node_feat_size,
                 edge_feat_size,
                 num_layers=2,
                 graph_feat_size=200,
                 dropout=0.):
        super(ModifiedAttentiveFPPredictorV3, self).__init__()

        self.gnn = ModifiedAttentiveFPGNNV3(node_feat_size=node_feat_size,
                                            edge_feat_size=edge_feat_size,
                                            num_layers=num_layers,
                                            graph_feat_size=graph_feat_size,
                                            dropout=dropout)

    def forward(self, g, node_feats, edge_feats):
        """Graph-level regression/soft classification.

        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs.
        node_feats : float32 tensor of shape (V, node_feat_size)
            Input node features. V for the number of nodes.
        edge_feats : float32 tensor of shape (E, edge_feat_size)
            Input edge features. E for the number of edges.

        Returns
        -------
        float32 tensor of shape (G, n_tasks)
            Prediction for the graphs in the batch. G for the number of graphs.
        node_weights : list of float32 tensor of shape (V, 1), optional
            This is returned when ``get_node_weight`` is ``True``.
            The list has a length ``num_timesteps`` and ``node_weights[i]``
            gives the node weights in the i-th update.
        """
        predictions = self.gnn(g, node_feats, edge_feats)
        return predictions


class ModifiedGATPredictor(nn.Module):
    r"""GAT-based model for regression and classification on graphs.

    GAT is introduced in `Graph Attention Networks <https://arxiv.org/abs/1710.10903>`__.
    This model is based on GAT and can be used for regression and classification on graphs.

    After updating node representations, we perform a weighted sum with learnable
    weights and max pooling on them and concatenate the output of the two operations,
    which is then fed into an MLP for final prediction.

    For classification tasks, the output will be logits, i.e.
    values before sigmoid or softmax.

    Parameters
    ----------
    in_feats : int
        Number of input node features
    hidden_feats : list of int
        ``hidden_feats[i]`` gives the output size of an attention head in the i-th GAT layer.
        ``len(hidden_feats)`` equals the number of GAT layers. By default, we use ``[32, 32]``.
    num_heads : list of int
        ``num_heads[i]`` gives the number of attention heads in the i-th GAT layer.
        ``len(num_heads)`` equals the number of GAT layers. By default, we use 4 attention heads
        for each GAT layer.
    feat_drops : list of float
        ``feat_drops[i]`` gives the dropout applied to the input features in the i-th GAT layer.
        ``len(feat_drops)`` equals the number of GAT layers. By default, this will be zero for
        all GAT layers.
    attn_drops : list of float
        ``attn_drops[i]`` gives the dropout applied to attention values of edges in the i-th GAT
        layer. ``len(attn_drops)`` equals the number of GAT layers. By default, this will be zero
        for all GAT layers.
    alphas : list of float
        Hyperparameters in LeakyReLU, which are the slopes for negative values. ``alphas[i]``
        gives the slope for negative value in the i-th GAT layer. ``len(alphas)`` equals the
        number of GAT layers. By default, this will be 0.2 for all GAT layers.
    residuals : list of bool
        ``residual[i]`` decides if residual connection is to be used for the i-th GAT layer.
        ``len(residual)`` equals the number of GAT layers. By default, residual connection
        is performed for each GAT layer.
    agg_modes : list of str
        The way to aggregate multi-head attention results for each GAT layer, which can be either
        'flatten' for concatenating all-head results or 'mean' for averaging all-head results.
        ``agg_modes[i]`` gives the way to aggregate multi-head attention results for the i-th
        GAT layer. ``len(agg_modes)`` equals the number of GAT layers. By default, we flatten
        multi-head results for intermediate GAT layers and compute mean of multi-head results
        for the last GAT layer.
    activations : list of activation function or None
        ``activations[i]`` gives the activation function applied to the aggregated multi-head
        results for the i-th GAT layer. ``len(activations)`` equals the number of GAT layers.
        By default, ELU is applied for intermediate GAT layers and no activation is applied
        for the last GAT layer.
    """

    def __init__(self, in_feats, hidden_feats=None, num_heads=None, feat_drops=None,
                 attn_drops=None, alphas=None, residuals=None, agg_modes=None, activations=None):
        super(ModifiedGATPredictor, self).__init__()

        self.gnn = GAT(in_feats=in_feats,
                       hidden_feats=hidden_feats,
                       num_heads=num_heads,
                       feat_drops=feat_drops,
                       attn_drops=attn_drops,
                       alphas=alphas,
                       residuals=residuals,
                       agg_modes=agg_modes,
                       activations=activations)

        if self.gnn.agg_modes[-1] == 'flatten':
            gnn_out_feats = self.gnn.hidden_feats[-1] * self.gnn.num_heads[-1]
        else:
            gnn_out_feats = self.gnn.hidden_feats[-1]
        self.predict = nn.Sequential(nn.Linear(gnn_out_feats, 1))

    def forward(self, bg, feats):
        """Graph-level regression/soft classification.

        Parameters
        ----------
        bg : DGLGraph
            DGLGraph for a batch of graphs.
        feats : FloatTensor of shape (N, M1)
            * N is the total number of nodes in the batch of graphs
            * M1 is the input node feature size, which must match
              in_feats in initialization

        Returns
        -------
        FloatTensor of shape (B, n_tasks)
            * Predictions on graphs
            * B for the number of graphs in the batch
        """
        node_feats = self.gnn(bg, feats)
        return self.predict(node_feats)


class ModifiedAttentiveFPPredictor(nn.Module):
    """AttentiveFP for regression and classification on graphs.

    AttentiveFP is introduced in `Pushing the Boundaries of Molecular Representation for Drug
    Discovery with the Graph Attention Mechanism.
     <https://www.ncbi.nlm.nih.gov/pubmed/31408336>`__

    Parameters
    ----------
    node_feat_size : int
        Size for the input node features.
    edge_feat_size : int
        Size for the input edge features.
    num_layers : int
        Number of GNN layers. Default to 2.
    graph_feat_size : int
        Size for the learned graph representations. Default to 200.
    dropout : float
        Probability for performing the dropout. Default to 0.
    """

    def __init__(self,
                 node_feat_size,
                 edge_feat_size,
                 num_layers=2,
                 graph_feat_size=200,
                 dropout=0.):
        super(ModifiedAttentiveFPPredictor, self).__init__()

        self.gnn = AttentiveFPGNN(node_feat_size=node_feat_size,
                                  edge_feat_size=edge_feat_size,
                                  num_layers=num_layers,
                                  graph_feat_size=graph_feat_size,
                                  dropout=dropout)
        self.predict = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(graph_feat_size, 1)
        )

    def forward(self, g, node_feats, edge_feats):
        """Graph-level regression/soft classification.

        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs.
        node_feats : float32 tensor of shape (V, node_feat_size)
            Input node features. V for the number of nodes.
        edge_feats : float32 tensor of shape (E, edge_feat_size)
            Input edge features. E for the number of edges.

        Returns
        -------
        float32 tensor of shape (G, n_tasks)
            Prediction for the graphs in the batch. G for the number of graphs.
        node_weights : list of float32 tensor of shape (V, 1), optional
            This is returned when ``get_node_weight`` is ``True``.
            The list has a length ``num_timesteps`` and ``node_weights[i]``
            gives the node weights in the i-th update.
        """
        node_feats = self.gnn(g, node_feats, edge_feats)
        return self.predict(node_feats)


class DTIConvGraph3(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(DTIConvGraph3, self).__init__()
    # the MPL for update the edge state
        self.mpl = nn.Sequential(nn.Linear(in_dim, out_dim),
                                 nn.LeakyReLU(),
                                 nn.Linear(out_dim, out_dim),
                                 nn.LeakyReLU(),
                                 nn.Linear(out_dim, out_dim),
                                 nn.LeakyReLU())

    def EdgeUpdate(self, edges):
        return {'e': self.mpl(torch.cat([edges.data['e'], edges.data['m']], dim=1))}

    def forward(self, bg, atom_feats, bond_feats):
        bg.ndata['h'] = atom_feats
        bg.edata['e'] = bond_feats
        with bg.local_scope():
            bg.apply_edges(dgl.function.u_add_v('h', 'h', 'm'))
            bg.apply_edges(self.EdgeUpdate)
            return bg.edata['e']


class DTIConvGraph3Layer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout):  # in_dim = graph module1 output dim + 1
        super(DTIConvGraph3Layer, self).__init__()
        # the MPL for update the edge state
        self.grah_conv = DTIConvGraph3(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.bn_layer = nn.BatchNorm1d(out_dim)

    def forward(self, bg, atom_feats, bond_feats):
        new_feats = self.grah_conv(bg, atom_feats, bond_feats)
        return self.bn_layer(self.dropout(new_feats))


class EdgeWeightAndSum(nn.Module):
    """
    for normal use, please delete the 'temporary version' line and meanwhile recover the 'normal version'
    """
    def __init__(self, in_feats):
        super(EdgeWeightAndSum, self).__init__()
        self.in_feats = in_feats
        self.atom_weighting = nn.Sequential(
            nn.Linear(in_feats, 1),
            nn.Tanh()
        )

    def forward(self, g, edge_feats):
        with g.local_scope():
            g.edata['e'] = edge_feats
            g.edata['w'] = self.atom_weighting(g.edata['e'])
            # weights = g.edata['w']  # temporary version
            h_g_sum = dgl.sum_edges(g, 'e', 'w')
        return h_g_sum  # normal version
        # return h_g_sum, weights  # temporary version


class EdgeWeightedSumAndMax(nn.Module):
    """
    for normal use, please delete the 'temporary version' line and meanwhile recover the 'normal version'
    """
    def __init__(self, in_feats):
        super(EdgeWeightedSumAndMax, self).__init__()
        self.weight_and_sum = EdgeWeightAndSum(in_feats)

    def forward(self, bg, edge_feats):
        h_g_sum = self.weight_and_sum(bg, edge_feats)  # normal version
        # h_g_sum, weights = self.weight_and_sum(bg, edge_feats)  # temporary version
        with bg.local_scope():
            bg.edata['e'] = edge_feats
            h_g_max = dgl.max_edges(bg, 'e')
        h_g = torch.cat([h_g_sum, h_g_max], dim=1)
        return h_g  # normal version
        # return h_g, weights  # temporary version


class DTIPredictorV4(nn.Module):
    def __init__(self, node_feat_size, edge_feat_size, num_layers, graph_feat_size, outdim_g3,
                 d_FC_layer, n_FC_layer, dropout, n_tasks):
        super(DTIPredictorV4, self).__init__()
        # graph layers for ligand and protein
        self.cov_graph = ModifiedAttentiveFPPredictorV2(node_feat_size, edge_feat_size, num_layers, graph_feat_size, dropout)

        # graph layers for ligand and protein interaction
        self.noncov_graph = DTIConvGraph3Layer(graph_feat_size+1, outdim_g3, dropout)

        # MLP predictor
        self.FC = FC(outdim_g3*2, d_FC_layer, n_FC_layer, dropout, n_tasks)
        # read out
        self.readout = EdgeWeightedSumAndMax(outdim_g3)

    def forward(self, bg1, bg2, bg3):
        atom_feats1 = bg1.ndata.pop('h')
        bond_feats1 = bg1.edata.pop('e')
        atom_feats2 = bg2.ndata.pop('h')
        bond_feats2 = bg2.edata.pop('e')
        atom_feats1 = self.cov_graph(bg1, atom_feats1, bond_feats1)
        atom_feats2 = self.cov_graph(bg2, atom_feats2, bond_feats2)
        bg1.ndata['h'] = atom_feats1
        bg2.ndata['h'] = atom_feats2
        bg1_ls = dgl.unbatch(bg1)
        bg2_ls = dgl.unbatch(bg2)
        bg3_ls = dgl.unbatch(bg3)
        for i in range(len(bg1_ls)):
            bg3_ls[i].ndata['h'] = torch.cat([bg1_ls[i].ndata['h'], bg2_ls[i].ndata['h']], dim=0)
        bg3 = dgl.batch(bg3_ls)
        atom_feats3 = bg3.ndata['h']
        bond_feats3 = bg3.edata['e']
        bond_feats3 = self.noncov_graph(bg3, atom_feats3, bond_feats3)
        readouts = self.readout(bg3, bond_feats3)
        return self.FC(readouts)


class DTIPredictorV4Cat(nn.Module):
    def __init__(self, node_feat_size, edge_feat_size, num_layers, graph_feat_size, outdim_g3,
                 d_FC_layer, n_FC_layer, dropout, n_tasks, n_global_feats):
        super(DTIPredictorV4Cat, self).__init__()
        # graph layers for ligand and protein
        self.cov_graph = ModifiedAttentiveFPPredictorV2(node_feat_size, edge_feat_size, num_layers, graph_feat_size, dropout)

        # graph layers for ligand and protein interaction
        self.noncov_graph = DTIConvGraph3Layer(graph_feat_size+1, outdim_g3, dropout)

        # MLP predictor
        self.FC = FC(outdim_g3*2+n_global_feats, d_FC_layer, n_FC_layer, dropout, n_tasks)
        # read out
        self.readout = EdgeWeightedSumAndMax(outdim_g3)

    def forward(self, bg1, bg2, bg3, global_feats):
        atom_feats1 = bg1.ndata.pop('h')
        bond_feats1 = bg1.edata.pop('e')
        atom_feats2 = bg2.ndata.pop('h')
        bond_feats2 = bg2.edata.pop('e')
        atom_feats1 = self.cov_graph(bg1, atom_feats1, bond_feats1)
        atom_feats2 = self.cov_graph(bg2, atom_feats2, bond_feats2)
        bg1.ndata['h'] = atom_feats1
        bg2.ndata['h'] = atom_feats2
        bg1_ls = dgl.unbatch(bg1)
        bg2_ls = dgl.unbatch(bg2)
        bg3_ls = dgl.unbatch(bg3)
        for i in range(len(bg1_ls)):
            bg3_ls[i].ndata['h'] = torch.cat([bg1_ls[i].ndata['h'], bg2_ls[i].ndata['h']], dim=0)
        bg3 = dgl.batch(bg3_ls)
        atom_feats3 = bg3.ndata['h']
        bond_feats3 = bg3.edata['e']
        bond_feats3 = self.noncov_graph(bg3, atom_feats3, bond_feats3)
        readouts = self.readout(bg3, bond_feats3)
        total_feats = torch.cat([readouts, global_feats], dim=1)
        return self.FC(total_feats)


class DTIPredictorV4VS(nn.Module):
    def __init__(self, node_feat_size, edge_feat_size, num_layers, graph_feat_size, outdim_g3,
                 d_FC_layer, n_FC_layer, dropout, n_tasks):
        super(DTIPredictorV4VS, self).__init__()
        # graph layers for ligand and protein
        self.cov_graph = ModifiedAttentiveFPPredictorV2(node_feat_size, edge_feat_size, num_layers, graph_feat_size, dropout)

        # graph layers for ligand and protein interaction
        self.noncov_graph = DTIConvGraph3Layer(graph_feat_size+1, outdim_g3, dropout)

        # MLP predictor
        self.FC = FC(outdim_g3*2, d_FC_layer, n_FC_layer, dropout, n_tasks)
        # read out
        self.readout = EdgeWeightedSumAndMax(outdim_g3)

    def forward(self, bg, bg3):
        atom_feats = bg.ndata.pop('h')
        bond_feats = bg.edata.pop('e')
        atom_feats = self.cov_graph(bg, atom_feats, bond_feats)
        bond_feats3 = bg3.edata['e']
        bond_feats3 = self.noncov_graph(bg3, atom_feats, bond_feats3)
        readouts = self.readout(bg3, bond_feats3)
        return torch.sigmoid(self.FC(readouts))


class DTIPredictorV4VSLigand(nn.Module):
    def __init__(self, node_feat_size, edge_feat_size, num_layers, graph_feat_size, outdim_g3,
                 d_FC_layer, n_FC_layer, dropout, n_tasks):
        super(DTIPredictorV4VSLigand, self).__init__()
        # graph layers for ligand and protein
        self.cov_graph = ModifiedAttentiveFPPredictorV2(node_feat_size, edge_feat_size, num_layers, graph_feat_size, dropout)

        # graph layers for ligand and protein interaction
        self.noncov_graph = DTIConvGraph3Layer(graph_feat_size+1, outdim_g3, dropout)

        # MLP predictor
        self.FC = FC(outdim_g3*2, d_FC_layer, n_FC_layer, dropout, n_tasks)
        # read out
        self.readout = EdgeWeightedSumAndMax(outdim_g3)

    def forward(self, bg, bg3, mask):
        atom_feats = bg.ndata.pop('h')
        bond_feats = bg.edata.pop('e')
        atom_feats = self.cov_graph(bg, atom_feats, bond_feats)
        atom_feats = atom_feats * mask
        bond_feats3 = bg3.edata['e']
        bond_feats3 = self.noncov_graph(bg3, atom_feats, bond_feats3)
        readouts = self.readout(bg3, bond_feats3)
        return torch.sigmoid(self.FC(readouts))


class DTIPredictorV4_V2(nn.Module):
    def __init__(self, node_feat_size, edge_feat_size, num_layers, graph_feat_size, outdim_g3,
                 d_FC_layer, n_FC_layer, dropout, n_tasks):
        super(DTIPredictorV4_V2, self).__init__()
        # graph layers for ligand and protein
        self.cov_graph = ModifiedAttentiveFPPredictorV2(node_feat_size, edge_feat_size, num_layers, graph_feat_size, dropout)

        # graph layers for ligand and protein interaction
        self.noncov_graph = DTIConvGraph3Layer(graph_feat_size+1, outdim_g3, dropout)

        # MLP predictor
        self.FC = FC(outdim_g3*2, d_FC_layer, n_FC_layer, dropout, n_tasks)
        # read out
        self.readout = EdgeWeightedSumAndMax(outdim_g3)

    def forward(self, bg, bg3):
        atom_feats = bg.ndata.pop('h')
        bond_feats = bg.edata.pop('e')
        atom_feats = self.cov_graph(bg, atom_feats, bond_feats)
        bond_feats3 = bg3.edata['e']
        bond_feats3 = self.noncov_graph(bg3, atom_feats, bond_feats3)
        readouts = self.readout(bg3, bond_feats3)
        return self.FC(readouts)


class DTIPredictorV4_V2_Hidden_State(nn.Module):
    def __init__(self, node_feat_size, edge_feat_size, num_layers, graph_feat_size, outdim_g3,
                 d_FC_layer, n_FC_layer, dropout, n_tasks):
        super(DTIPredictorV4_V2_Hidden_State, self).__init__()
        # graph layers for ligand and protein
        self.cov_graph = ModifiedAttentiveFPPredictorV2(node_feat_size, edge_feat_size, num_layers, graph_feat_size, dropout)

        # graph layers for ligand and protein interaction
        self.noncov_graph = DTIConvGraph3Layer(graph_feat_size+1, outdim_g3, dropout)

        # MLP predictor
        self.FC = FC(outdim_g3*2, d_FC_layer, n_FC_layer, dropout, n_tasks)
        # read out
        self.readout = EdgeWeightedSumAndMax(outdim_g3)

    def forward(self, bg, bg3):
        atom_feats = bg.ndata.pop('h')
        bond_feats = bg.edata.pop('e')
        atom_feats = self.cov_graph(bg, atom_feats, bond_feats)
        bond_feats3 = bg3.edata['e']
        bond_feats3 = self.noncov_graph(bg3, atom_feats, bond_feats3)
        readouts, weights = self.readout(bg3, bond_feats3)
        return self.FC(readouts), bond_feats3, readouts, weights


class DTIPredictorV4_V2_InTra_Inter(nn.Module):
    """
    模型同时考虑节点特征（共价相互作用）和边特征（非共价相互作用）
    """
    def __init__(self, node_feat_size, edge_feat_size, num_layers, graph_feat_size, outdim_g3,
                 d_FC_layer, n_FC_layer, dropout, n_tasks, initial_inter=0.8, initial_intra=0.2):
        super(DTIPredictorV4_V2_InTra_Inter, self).__init__()
        # graph layers for ligand and protein
        self.cov_graph = ModifiedAttentiveFPPredictorV2(node_feat_size, edge_feat_size, num_layers, graph_feat_size, dropout)

        # graph layers for ligand and protein interaction
        self.noncov_graph = DTIConvGraph3Layer(graph_feat_size+1, outdim_g3, dropout)

        # MLP predictor
        self.FC_inter = FC(outdim_g3*2, d_FC_layer, n_FC_layer, dropout, n_tasks)
        self.FC_intra = FC(graph_feat_size*2, d_FC_layer, n_FC_layer, dropout, n_tasks)

        # read out
        self.readout_inter = EdgeWeightedSumAndMax(outdim_g3)
        self.readout_intra = WeightedSumAndMax(graph_feat_size)

        # the weights of intermolecular Interactions and Intramolecular Interactions
        self.w_inter = nn.Parameter(torch.tensor(initial_inter))
        self.w_intra = nn.Parameter(torch.tensor(initial_intra))

    def forward(self, bg, bg3):
        atom_feats = bg.ndata.pop('h')
        bond_feats = bg.edata.pop('e')
        atom_feats = self.cov_graph(bg, atom_feats, bond_feats)
        bond_feats3 = bg3.edata['e']
        bond_feats3 = self.noncov_graph(bg3, atom_feats, bond_feats3)
        readouts_inter = self.readout_inter(bg3, bond_feats3)
        readouts_intra = self.readout_intra(bg, atom_feats)
        return self.w_inter*self.FC_inter(readouts_inter) + self.w_intra*self.FC_intra(readouts_intra)


class DTIPredictorV4_V2_Mask(nn.Module):
    def __init__(self, node_feat_size, edge_feat_size, num_layers, graph_feat_size, outdim_g3,
                 d_FC_layer, n_FC_layer, dropout, n_tasks):
        super(DTIPredictorV4_V2_Mask, self).__init__()
        # graph layers for ligand and protein
        self.cov_graph = ModifiedAttentiveFPPredictorV2(node_feat_size, edge_feat_size, num_layers, graph_feat_size, dropout)

        # graph layers for ligand and protein interaction
        self.noncov_graph = DTIConvGraph3Layer(graph_feat_size+1, outdim_g3, dropout)

        # MLP predictor
        self.FC = FC(outdim_g3*2, d_FC_layer, n_FC_layer, dropout, n_tasks)
        # read out
        self.readout = EdgeWeightedSumAndMax(outdim_g3)

    def forward(self, bg, bg3, mask):
        atom_feats = bg.ndata.pop('h')
        bond_feats = bg.edata.pop('e')
        atom_feats = self.cov_graph(bg, atom_feats, bond_feats)
        atom_feats = atom_feats * mask
        bond_feats3 = bg3.edata['e']
        bond_feats3 = self.noncov_graph(bg3, atom_feats, bond_feats3)
        readouts = self.readout(bg3, bond_feats3)
        return self.FC(readouts)


class DTIConvGraph3Test(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(DTIConvGraph3Test, self).__init__()
        self.W1 = nn.Linear((in_dim - 1) * 2 + 1, out_dim)
        self.W2 = nn.Linear(in_dim-1, out_dim)

    def EdgeInit(self, edges):
        return {'init': torch.cat([edges.src['h'], edges.dst['h'], edges.data['e']], dim=1)}

    def EdgeUpdate(self, edges):
        return {'e': F.leaky_relu(self.W1(edges.data['init']) + self.W2(edges.data['m']))}

    def forward(self, bg, atom_feats, bond_feats):
        bg.ndata['h'] = atom_feats
        bg.edata['e'] = bond_feats
        with bg.local_scope():
            bg.apply_edges(self.EdgeInit)
            bg.apply_edges(dgl.function.u_add_v('h', 'h', 'm'))
            bg.apply_edges(self.EdgeUpdate)
            return bg.edata['e']


class DTIConvGraph3LayerTest(nn.Module):
    def __init__(self, in_dim, out_dim, dropout):  # in_dim = graph module1 output dim + 1
        super(DTIConvGraph3LayerTest, self).__init__()
        # the MPL for update the edge state
        self.grah_conv = DTIConvGraph3Test(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.bn_layer = nn.BatchNorm1d(out_dim)

    def forward(self, bg, atom_feats, bond_feats):
        new_feats = self.grah_conv(bg, atom_feats, bond_feats)
        return self.bn_layer(self.dropout(new_feats))


class DTIPredictorV4_V2_Test(nn.Module):
    def __init__(self, node_feat_size, edge_feat_size, num_layers, graph_feat_size, outdim_g3,
                 d_FC_layer, n_FC_layer, dropout, n_tasks):
        super(DTIPredictorV4_V2_Test, self).__init__()
        # graph layers for ligand and protein
        self.cov_graph = ModifiedAttentiveFPPredictorV2(node_feat_size, edge_feat_size, num_layers, graph_feat_size, dropout)

        # graph layers for ligand and protein interaction
        self.noncov_graph = DTIConvGraph3LayerTest(graph_feat_size+1, outdim_g3, dropout)

        # MLP predictor
        self.FC = FC(outdim_g3*2, d_FC_layer, n_FC_layer, dropout, n_tasks)
        # read out
        self.readout = EdgeWeightedSumAndMax(outdim_g3)

    def forward(self, bg, bg3):
        atom_feats = bg.ndata.pop('h')
        bond_feats = bg.edata.pop('e')
        atom_feats = self.cov_graph(bg, atom_feats, bond_feats)
        bond_feats3 = bg3.edata['e']
        bond_feats3 = self.noncov_graph(bg3, atom_feats, bond_feats3)
        readouts = self.readout(bg3, bond_feats3)
        return self.FC(readouts)


from dgllife.model import GCNPredictor