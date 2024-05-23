from torch import nn
import torch


device = torch.device("cuda")


def unsorted_segment_sum(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result


def unsorted_segment_mean(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1)).to(device)
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)


class E_GCL(nn.Module):
    """
    E(n) Equivariant Convolutional Layer
    re
    """

    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, act_fn=nn.SiLU(), residual=True, attention=False,
                 normalize=False, coords_agg='mean', tanh=False):
        super(E_GCL, self).__init__()
        input_edge = input_nf * 2
        self.residual = residual
        self.attention = attention
        self.normalize = True
        # self.normalize = normalize
        self.coords_agg = coords_agg
        self.tanh = tanh
        self.epsilon = 1e-8
        edge_coords_nf = 1

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edge_coords_nf + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf))

        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        coord_mlp = []
        coord_mlp.append(nn.Linear(hidden_nf, hidden_nf))
        coord_mlp.append(act_fn)
        coord_mlp.append(layer)
        if self.tanh:
            coord_mlp.append(nn.Tanh())
        self.coord_mlp = nn.Sequential(*coord_mlp)

        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1))

    def edge_model(self, source, target, radial, edge_attr, batch_size, k):
        # if edge_attr is None:  # Unused.
        #     out = torch.cat([source, target, radial], dim=1).to(device)
        # else:
        #     out = torch.cat([source, target, radial, edge_attr], dim=1).to(device)
        # computing m_{ij}
        out = torch.cat([source, target, radial], dim=1).to(device)
        out = self.edge_mlp(out.float())
        # need to use softmax to normalize
        # if self.attention:
        #     attn = self.att_mlp(out)
        #     att_val = torch.softmax(attn.view(batch_size, -1, k), dim=-1).view(-1, 1)
        #     out = out * att_val
        return out

    def node_model(self, x, edge_index, edge_attr, node_attr, batch_size, k):
        row, col = edge_index
        row = row.to(device)
        dim = edge_attr.size(-1)
        # edge_attr = edge_attr.view(batch_size, -1, k, dim)
        # agg = torch.sum(edge_attr, dim=2).view(-1, dim)
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        out = self.node_mlp(agg)
        if self.residual:
            out = x + out
        return out, agg

    def coord_model(self, coord, edge_index, coord_diff, edge_feat, batch_size, k):
        row, col = edge_index
        row = row.to(device)
        coord_diff = coord_diff.to(device)
        trans = coord_diff * self.coord_mlp(edge_feat)  # [B * L * 30, 3]
        # trans = trans.view(batch_size, -1, k, 3)
        if self.coords_agg == 'sum':
            # agg = torch.sum(trans, dim=2).view(-1, 3)
            agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0))
        elif self.coords_agg == 'mean':
            agg = torch.mean(trans, dim=2).view(-1, 3)
            # agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        else:
            raise Exception('Wrong coords_agg parameter' % self.coords_agg)
        coord = coord + agg
        return coord

    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum(coord_diff**2, 1).unsqueeze(1).to(device)  # [KNN]

        if self.normalize:
            norm = torch.sqrt(radial).detach() + self.epsilon
            coord_diff = coord_diff / norm

        return radial, coord_diff

    def forward(self, h, edge_index, coord, edge_attr=None, node_attr=None, batch_size=1, k=30):
        """
        h: [batch * length, 320]
        edges: [B * L * L, B * L * L]
        coord: [batch * length, 3]
        """
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)   # [B * L * 30], [B * L * 30, 3]

        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr, batch_size, k)   # m_{ij}, [B * L * 30, dim]
        coord = self.coord_model(coord, edge_index, coord_diff, edge_feat, batch_size, k)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr, batch_size, k)

        return h, coord, edge_attr


class E_GCL_RM_Node(nn.Module):
    """
    E(n) Equivariant Convolutional Layer
    re
    """

    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, act_fn=nn.SiLU(), residual=True, attention=False,
                 normalize=False, coords_agg='mean', tanh=False):
        super(E_GCL_RM_Node, self).__init__()
        input_edge = input_nf * 2
        self.residual = residual
        self.attention = attention
        self.normalize = normalize
        self.coords_agg = coords_agg
        self.tanh = tanh
        self.epsilon = 1e-8
        edge_coords_nf = 1

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edge_coords_nf + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)

        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        coord_mlp = []
        coord_mlp.append(nn.Linear(hidden_nf, hidden_nf))
        coord_mlp.append(act_fn)
        coord_mlp.append(layer)
        if self.tanh:
            coord_mlp.append(nn.Tanh())
        self.coord_mlp = nn.Sequential(*coord_mlp)

        self.node_gate = nn.Sequential(
            nn.Linear(hidden_nf, hidden_nf),
            nn.ReLU(),
            nn.Linear(hidden_nf, hidden_nf),
            nn.Sigmoid()
        )
        # self.node_gate = nn.Sequential(
        #     nn.Linear(hidden_nf, hidden_nf),
        #     nn.ReLU(),
        #     nn.Linear(hidden_nf, hidden_nf),
        #     nn.ReLU(),
        #     nn.Linear(hidden_nf, hidden_nf),
        #     nn.Sigmoid()
        # )

        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1))

    def edge_model(self, source, target, radial, edge_attr, batch_size, k):
        # if edge_attr is None:  # Unused.
        #     out = torch.cat([source, target, radial], dim=1).to(device)
        # else:
        #     out = torch.cat([source, target, radial, edge_attr], dim=1).to(device)
        # computing m_{ij}
        out = torch.cat([source, target, radial], dim=1).to(device)
        out = self.edge_mlp(out.float())
        # need to use softmax to normalize
        if self.attention:
            attn = self.att_mlp(out)
            att_val = torch.softmax(attn.view(batch_size, -1, k), dim=-1).view(-1, 1)
            out = out * att_val
        return out

    def node_model(self, x, edge_index, edge_attr, node_attr, batch_size, k):
        row, col = edge_index
        # row = row.to(device)
        dim = edge_attr.size(-1)
        edge_attr = edge_attr.view(batch_size, -1, k, dim)
        agg = torch.sum(edge_attr, dim=2).view(-1, dim)  # gathered information from K neareast neighbors
        out = x
        if self.residual:
            out = x + self.node_gate(agg) * agg
        return out, agg

    def coord_model(self, coord, edge_index, coord_diff, edge_feat, batch_size, k):
        row, col = edge_index
        # row = row.to(device)
        coord_diff = coord_diff.to(device)
        trans = coord_diff * self.coord_mlp(edge_feat)  # [B * L * 30, 3]
        trans = trans.view(batch_size, -1, k, 3)
        if self.coords_agg == 'sum':
            agg = torch.sum(trans, dim=2).view(-1, 3)
            # agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0))
        elif self.coords_agg == 'mean':
            agg = torch.mean(trans, dim=2).view(-1, 3)
            # agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        else:
            raise Exception('Wrong coords_agg parameter' % self.coords_agg)
        coord = coord + agg
        return coord

    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum(coord_diff**2, 1).unsqueeze(1).to(device)  # [KNN]

        if self.normalize:
            norm = torch.sqrt(radial).detach() + self.epsilon
            coord_diff = coord_diff / norm

        return radial, coord_diff

    def forward(self, h, edge_index, coord, edge_attr=None, node_attr=None, batch_size=1, k=30):
        """
        h: [batch * length, 320]
        edges: [B * L * L, B * L * L]
        coord: [batch * length, 3]
        """
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)   # [B * L * 30], [B * L * 30, 3]

        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr, batch_size, k)   # m_{ij}, [B * L * 30, dim]
        coord = self.coord_model(coord, edge_index, coord_diff, edge_feat, batch_size, k)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr, batch_size, k)

        return h, coord, edge_attr


class E_GCL_RM_Edge(nn.Module):
    """
    E(n) Equivariant Convolutional Layer
    re
    """

    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, act_fn=nn.SiLU(), residual=True, attention=False,
                 normalize=False, coords_agg='mean', tanh=False):
        super(E_GCL_RM_Edge, self).__init__()
        input_edge = input_nf * 2
        self.residual = residual
        self.attention = attention
        self.normalize = normalize
        self.coords_agg = coords_agg
        self.tanh = tanh
        self.epsilon = 1e-8
        self.dist_epsilon = 1e-8
        self.dist_weight = 1.0

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf))

    def edge_model(self, source, target, radial, edge_attr, batch_size, k):
        # computing m_{ij}; [B * L * 30, dim]
        attn = torch.sum(source * target, dim=-1).reshape(batch_size, -1, k)
        dist_score = (1.0 / (radial + self.dist_epsilon)).reshape(batch_size, -1, k)
        attn_score = torch.softmax(attn + self.dist_weight * dist_score, dim=-1).unsqueeze(-1)
        out = (target.reshape(batch_size, -1, k, target.size()[-1]) * attn_score).view(-1, target.size()[-1])
        return out, attn_score.view(-1, 1)

    def node_model(self, x, edge_index, edge_attr, node_attr, batch_size, k):
        dim = edge_attr.size(-1)
        edge_attr = edge_attr.view(batch_size, -1, k, dim)
        agg = torch.sum(edge_attr, dim=2).view(-1, dim)
        # agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        out = self.node_mlp(agg)
        if self.residual:
            out = x + out
        return out, agg

    def coord_model(self, coord, edge_index, coord_diff, attn_score, batch_size, k):
        coord_diff = coord_diff.to(device)
        trans = coord_diff * attn_score  # [B * L * 30, 3]
        trans = trans.view(batch_size, -1, k, 3)
        if self.coords_agg == 'sum':
            agg = torch.sum(trans, dim=2).view(-1, 3)
            # agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0))
        elif self.coords_agg == 'mean':
            agg = torch.mean(trans, dim=2).view(-1, 3)
            # agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        else:
            raise Exception('Wrong coords_agg parameter' % self.coords_agg)
        coord = coord + agg
        return coord

    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum(coord_diff**2, 1).unsqueeze(1).to(device)  # [KNN]

        if self.normalize:
            norm = torch.sqrt(radial).detach() + self.epsilon
            coord_diff = coord_diff / norm

        return radial, coord_diff

    def forward(self, h, edge_index, coord, edge_attr=None, node_attr=None, batch_size=1, k=30):
        """
        h: [batch * length, 320]
        edges: [B * L * L, B * L * L]
        coord: [batch * length, 3]
        """
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)   # [B * L * 30], [B * L * 30, 3]

        edge_feat, attn_score = self.edge_model(h[row], h[col], radial, edge_attr, batch_size, k)   # m_{ij}, [B * L * 30, dim]
        coord = self.coord_model(coord, edge_index, coord_diff, attn_score, batch_size, k)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr, batch_size, k)

        return h, coord, edge_attr


class E_GCL_RM_All(nn.Module):
    """
    E(n) Equivariant Convolutional Layer
    re
    """

    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, act_fn=nn.SiLU(), residual=True, attention=False,
                 normalize=False, coords_agg='mean', tanh=False):
        super(E_GCL_RM_All, self).__init__()
        input_edge = input_nf * 2
        self.residual = residual
        self.attention = attention
        self.normalize = normalize
        self.coords_agg = coords_agg
        self.tanh = tanh
        self.epsilon = 1e-8
        self.dist_epsilon = 1e-8
        self.dist_weight = 1.0

    def edge_model(self, source, target, radial, edge_attr, batch_size, k):
        # computing m_{ij}; [B * L * 30, dim]
        attn = torch.sum(source * target, dim=-1).reshape(batch_size, -1, k)
        dist_score = (1.0 / (radial + self.dist_epsilon)).reshape(batch_size, -1, k)
        attn_score = torch.softmax(attn + self.dist_weight * dist_score, dim=-1).unsqueeze(-1)
        out = (target.reshape(batch_size, -1, k, target.size()[-1]) * attn_score).view(-1, target.size()[-1])
        return out, attn_score.view(-1, 1)

    def node_model(self, x, edge_index, edge_attr, node_attr, batch_size, k):
        dim = edge_attr.size(-1)
        edge_attr = edge_attr.view(batch_size, -1, k, dim)
        agg = torch.sum(edge_attr, dim=2).view(-1, dim)  # gathered information from K neareast neighbors
        out = x
        if self.residual:
            out = x + agg
        return out, agg

    def coord_model(self, coord, edge_index, coord_diff, attn_score, batch_size, k):
        coord_diff = coord_diff.to(device)
        trans = coord_diff * attn_score  # [B * L * 30, 3]
        trans = trans.view(batch_size, -1, k, 3)
        if self.coords_agg == 'sum':
            agg = torch.sum(trans, dim=2).view(-1, 3)
            # agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0))
        elif self.coords_agg == 'mean':
            agg = torch.mean(trans, dim=2).view(-1, 3)
            # agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        else:
            raise Exception('Wrong coords_agg parameter' % self.coords_agg)
        coord = coord + agg
        return coord

    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum(coord_diff**2, 1).unsqueeze(1).to(device)  # [KNN]

        if self.normalize:
            norm = torch.sqrt(radial).detach() + self.epsilon
            coord_diff = coord_diff / norm

        return radial, coord_diff

    def forward(self, h, edge_index, coord, edge_attr=None, node_attr=None, batch_size=1, k=30):
        """
        h: [batch * length, 320]
        edges: [B * L * L, B * L * L]
        coord: [batch * length, 3]
        """
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)   # [B * L * 30], [B * L * 30, 3]

        edge_feat, attn_score = self.edge_model(h[row], h[col], radial, edge_attr, batch_size, k)   # m_{ij}, [B * L * 30, dim]
        coord = self.coord_model(coord, edge_index, coord_diff, attn_score, batch_size, k)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr, batch_size, k)

        return h, coord, edge_attr


class EGNN(nn.Module):
    def __init__(self, in_node_nf, hidden_nf, out_node_nf, in_edge_nf=0, device='cpu', act_fn=nn.SiLU(), n_layers=4,
                 residual=True, attention=False, normalize=False, tanh=False, mode="full"):
        '''
        :param in_node_nf: Number of features for 'h' at the input
        :param hidden_nf: Number of hidden features
        :param out_node_nf: Number of features for 'h' at the output
        :param in_edge_nf: Number of features for the edge features
        :param device: Device (e.g. 'cpu', 'cuda:0',...)
        :param act_fn: Non-linearity
        :param n_layers: Number of layer for the EGNN
        :param residual: Use residual connections, we recommend not changing this one
        :param attention: Whether using attention or not
        :param normalize: Normalizes the coordinates messages such that:
                    instead of: x^{l+1}_i = x^{l}_i + Σ(x_i - x_j)phi_x(m_ij)
                    we get:     x^{l+1}_i = x^{l}_i + Σ(x_i - x_j)phi_x(m_ij)/||x_i - x_j||
                    We noticed it may help in the stability or generalization in some future works.
                    We didn't use it in our paper.
        :param tanh: Sets a tanh activation function at the output of phi_x(m_ij). I.e. it bounds the output of
                        phi_x(m_ij) which definitely improves in stability but it may decrease in accuracy.
                        We didn't use it in our paper.
        '''

        super(EGNN, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.mode = mode

        # for i in range(0, n_layers):
        #     self.add_module("gcl_%d" % i, E_GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf,
        #                                         act_fn=act_fn, residual=residual, attention=attention,
        #                                         normalize=normalize, tanh=tanh, coords_agg="sum"))

        if self.mode == "full":
            for i in range(0, 3):
                self.add_module("gcl_%d" % i, E_GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf,
                                                    act_fn=act_fn, residual=residual, attention=attention,
                                                    normalize=normalize, tanh=tanh, coords_agg="sum"))
        elif self.mode == "rm-node":
            for i in range(0, n_layers):
                self.add_module("gcl_%d" % i, E_GCL_RM_Node(self.hidden_nf, self.hidden_nf, self.hidden_nf,
                                                            edges_in_d=in_edge_nf, act_fn=act_fn, residual=residual,
                                                            attention=attention, normalize=normalize, tanh=tanh,
                                                            coords_agg="sum"))
        elif self.mode == "rm-edge":
            for i in range(0, n_layers):
                self.add_module("gcl_%d" % i, E_GCL_RM_Edge(self.hidden_nf, self.hidden_nf, self.hidden_nf,
                                                            edges_in_d=in_edge_nf, act_fn=act_fn, residual=residual,
                                                            attention=attention, normalize=normalize, tanh=tanh,
                                                            coords_agg="sum"))
        elif self.mode == "rm-all":
            for i in range(0, n_layers):
                self.add_module("gcl_%d" % i, E_GCL_RM_All(self.hidden_nf, self.hidden_nf, self.hidden_nf,
                                                           edges_in_d=in_edge_nf, act_fn=act_fn, residual=residual,
                                                           attention=attention, normalize=normalize, tanh=tanh,
                                                           coords_agg="sum"))
        else:
            raise Exception("No EGNN mode: {}!".format(self.mode))
        self.to(self.device)

    def forward(self, h, x, edges, edge_attr, k=30):
        # h = self.embedding_in(h)
        h = h.reshape(-1, h.size()[-1])   # [batch * length, hidden]
        # x = x.reshape(-1, x.size()[-1])    # [batch * length, 3]
        for i in range(0, self.n_layers):
            h, x, _ = self._modules["gcl_%d" % i](h, edges, x, edge_attr=edge_attr, k=k)
        #h = self.embedding_out(h)
        return h, x


class SubstrateEGNN(nn.Module):
    def __init__(self, in_node_nf, hidden_nf, out_node_nf, in_edge_nf=0, device='cpu', act_fn=nn.SiLU(), n_layers=4,
                 residual=True, attention=False, normalize=False, tanh=False, mode="full"):
        '''
        :param in_node_nf: Number of features for 'h' at the input
        :param hidden_nf: Number of hidden features
        :param out_node_nf: Number of features for 'h' at the output
        :param in_edge_nf: Number of features for the edge features
        :param device: Device (e.g. 'cpu', 'cuda:0',...)
        :param act_fn: Non-linearity
        :param n_layers: Number of layer for the EGNN
        :param residual: Use residual connections, we recommend not changing this one
        :param attention: Whether using attention or not
        :param normalize: Normalizes the coordinates messages such that:
                    instead of: x^{l+1}_i = x^{l}_i + Σ(x_i - x_j)phi_x(m_ij)
                    we get:     x^{l+1}_i = x^{l}_i + Σ(x_i - x_j)phi_x(m_ij)/||x_i - x_j||
                    We noticed it may help in the stability or generalization in some future works.
                    We didn't use it in our paper.
        :param tanh: Sets a tanh activation function at the output of phi_x(m_ij). I.e. it bounds the output of
                        phi_x(m_ij) which definitely improves in stability but it may decrease in accuracy.
                        We didn't use it in our paper.
        '''

        super(SubstrateEGNN, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.mode = mode
        self.embedding_in = nn.Linear(5, self.hidden_nf)

        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, E_GCL_RM_Node(self.hidden_nf, self.hidden_nf, self.hidden_nf,
                                                        edges_in_d=in_edge_nf, act_fn=act_fn, residual=residual,
                                                        attention=attention, normalize=normalize, tanh=tanh,
                                                        coords_agg="sum"))
        self.to(self.device)

    def forward(self, h, x, edges, edge_attr=None, k=30):
        h = self.embedding_in(h.float())
        h = h.reshape(-1, h.size()[-1])   # [batch * length, hidden]
        x = x.reshape(-1, x.size()[-1])    # [batch * length, 3]
        for i in range(0, self.n_layers):
            h, x, _ = self._modules["gcl_%d" % i](h, edges, x, edge_attr=edge_attr, k=k)
        #h = self.embedding_out(h)
        return h, x