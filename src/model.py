import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from statistics import mean
import torch_geometric
from torch_geometric.data import Data, DataLoader


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=None, num_layers=1, activation="relu"):
        super(MLP, self).__init__()
        self.input_dim = in_dim
        self.output_dim = out_dim

        if num_layers < 1:
            raise ValueError
        if not hidden_dim:
            hidden_dim = out_dim

        layers = []
        if num_layers == 1:
            out_layer = nn.Linear(in_dim, out_dim)
        else:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.extend([nn.Linear(hidden_dim, hidden_dim)] * (num_layers - 2))
            out_layer = nn.Linear(hidden_dim, out_dim)

        self.layers = nn.ModuleList(layers)
        self.out_layer = out_layer
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        else:
            raise ValueError

    def forward(self, x):
        for layer in self.layers:
            x = self.activation(layer(x))
        x = self.out_layer(x)
        return x


class EdgeEditor(nn.Module):
    def __init__(
        self,
        hidden_size=512,
        rel_emb_size=32,
        transformer_embedding_size=768,
        transformer="allenai/longformer-base-4096",
        num_edge_classes=2,
        num_node_classes=13,
        gcn_layers=1,
        label_emb_dim=16,
        dropout_out=0.0,
        feature="add",
        dropout_edit=0.0,
        fc_out_layer=1,
        add_adj_emb=False,
        with_sigmoid=False,
        gcn_module=torch_geometric.nn.GCNConv,
        d_max=1,
        bigcn=False,
        random_order=False,
        edge_emb_dim=0,
        dist_emb_dim=0,
        dist_emb_max=100,
        head_tail_layers=1,
    ):
        # TODO: default value
        super(EdgeEditor, self).__init__()
        self.emb_saved = None
        self.feature = feature
        self.transformer_name = transformer
        self.add_adj_emb = add_adj_emb
        self.edge_emb_dim = edge_emb_dim

        self.hidden_size = hidden_size
        self.label_emb_dim = label_emb_dim
        self.transformer_embedding_size = transformer_embedding_size
        self.transformer = AutoModel.from_pretrained(transformer)
        self.tokenizer = AutoTokenizer.from_pretrained(transformer)
        self.gcn_layers = gcn_layers
        self.graph_dim = hidden_size + label_emb_dim
        self.head_tail_layers = head_tail_layers

        self.dist_emb_dim = dist_emb_dim
        self.dist_emb_max = dist_emb_max

        self.with_sigmoid = with_sigmoid
        self.d_max = d_max

        self.bigcn = bigcn
        self.random_order = random_order

        self.fc_reduction = nn.Linear(self.transformer_embedding_size, self.hidden_size)
        self.num_edge_classes = num_edge_classes
        self.num_node_classes = num_node_classes

        self.fc_embrel = MLP(num_edge_classes, rel_emb_size)

        self.emb_label = nn.Embedding(num_node_classes + 2, label_emb_dim)
        self.gconvs = nn.ModuleList(
            [
                nn.ModuleList([gcn_module(self.graph_dim, self.graph_dim) for i in range(self.gcn_layers)])
                for _ in range(self.num_edge_classes + 1)
            ]
        )

        if bigcn:
            self.gconvs_inv = nn.ModuleList(
                [
                    nn.ModuleList(
                        [
                            gcn_module(self.graph_dim, self.graph_dim, flow="target_to_source")
                            for i in range(self.gcn_layers)
                        ]
                    )
                    for _ in range(self.num_edge_classes + 1)
                ]
            )

        self.dropout_out = nn.Dropout(dropout_out)
        self.dropout_edit = nn.Dropout(dropout_edit)

        self.dist_emb = nn.Embedding((self.dist_emb_max + 1) * 2, self.dist_emb_dim)

        self.fc_out = MLP(
            self.graph_dim
            + int(self.add_adj_emb) * (self.num_edge_classes + 1)
            + self.dist_emb_dim
            + self.edge_emb_dim,
            self.num_edge_classes + 1,
            hidden_dim=self.hidden_size,
            num_layers=fc_out_layer,
        )

        if self.feature == "bilinear":
            self.bilinear = nn.Bilinear(self.graph_dim // 2, self.graph_dim // 2, self.graph_dim)
            self.fc_head = MLP(
                self.graph_dim, self.graph_dim // 2, hidden_dim=self.graph_dim, num_layers=self.head_tail_layers
            )
            self.fc_tail = MLP(
                self.graph_dim, self.graph_dim // 2, hidden_dim=self.graph_dim, num_layers=self.head_tail_layers
            )
        else:
            self.fc_head = MLP(
                self.graph_dim, self.graph_dim, hidden_dim=self.graph_dim, num_layers=self.head_tail_layers
            )
            self.fc_tail = MLP(
                self.graph_dim, self.graph_dim, hidden_dim=self.graph_dim, num_layers=self.head_tail_layers
            )

        if self.edge_emb_dim:
            self.edge_emb = nn.Embedding(self.num_edge_classes + 1, self.edge_emb_dim)

    def forward(
        self,
        data,
        finetune_transformer=False,
    ):

        batch_size = len(data)

        # read from data
        ent_indices = list(map(lambda d: d.ent_indices, data))
        ent_labels = list(map(lambda d: d.ent_labels, data))
        text_ids = list(map(lambda d: d.ids, data))

        ent_indices = self.pad(ent_indices).long()
        ent_labels = self.pad(ent_labels).long()
        text_ids = self.pad(text_ids, padding=self.tokenizer.pad_token_id).long()
        device = text_ids.device

        num_ent = ent_indices.max(dim=-1).values
        max_num_ent = int(ent_indices.max().item())
        all_num_ent = int(ent_indices.max().item()) + 1

        # mask of adj
        mask = torch.ones(batch_size, 1, max_num_ent, max_num_ent, dtype=torch.bool, device=device)
        for i, ne in enumerate(num_ent):
            mask[i, :, ne:] = False
            mask[i, :, :, ne:] = False

        tag_order, ordered_tags = self.get_tag_order(data)

        # Text Encoding ############
        # get text embedding
        attn_mask = text_ids != self.tokenizer.pad_token_id
        if finetune_transformer:
            self.transformer.train()
            emb = self.transformer(text_ids, attention_mask=attn_mask)[0]
        else:
            with torch.no_grad():
                self.transformer.eval()
                emb = self.transformer(text_ids, attention_mask=attn_mask)[0]

        # reduce embedding dim
        emb = self.fc_reduction(emb)

        # get representation of each entity
        emb_mat = torch.zeros(
            batch_size,
            all_num_ent,
            self.hidden_size + self.label_emb_dim,
            device=device,
        )

        for tag in ent_indices.unique().long():
            if tag == 0:
                continue
            tmp = (ent_indices == tag).unsqueeze(-1) * emb
            tmsk = (ent_indices == tag).sum(-1) != 0
            emb_mat[:, tag, -self.hidden_size :] = tmp.max(1).values

            e = ent_labels[:, tag - 1]
            assert e.max() < self.emb_label.num_embeddings
            elbl = (e != 0).unsqueeze(-1) * self.emb_label(e.long())

            emb_mat[:, tag, : self.label_emb_dim] = elbl

        emb_mat[:, 0, : self.label_emb_dim] = self.emb_label(torch.zeros(batch_size, dtype=torch.long, device=device))

        ######################################

        # adj editing ##################
        lookup = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], device=device)

        edges = list(map(lambda x: x.edge_index.long(), data))
        attrs = list(map(lambda x: x.edge_attr.long(), data))

        dist = self.get_distance(tag_order, max_dist=self.d_max, random_order=self.random_order)
        dist_for_emb = self.get_distance(tag_order, max_dist=self.dist_emb_max)

        # apply mask to dist
        for i, d in enumerate(data):
            dist[i][d.mask.size(0) :, d.mask.size(1) :] = 0
            dist[i][: d.mask.size(0), : d.mask.size(1)] = dist[i][: d.mask.size(0), : d.mask.size(1)] * d.mask

        # apply mask to dist
        for i, dat in enumerate(data):
            msk = dat.mask.bool()
            dist[i, : msk.size(0), : msk.size(1)] = dist[i, : msk.size(0), : msk.size(1)] * msk
        adj = torch.zeros(
            batch_size,
            self.num_edge_classes + 1,
            max_num_ent + 1,
            max_num_ent + 1,
            device=device,
            dtype=torch.float,
        )
        marged_cand = torch.zeros_like(dist, dtype=torch.bool, device=device)
        for num, cands in enumerate(self.get_cand_from_dist(dist)):

            edges_keep, attrs_keep = self.get_edges_to_keep(edges, attrs, cands)
            dset = []
            index = []
            marged_cand += cands
            emat = emb_mat
            for c in range(self.num_edge_classes + 1):
                for i, (edge, attr) in enumerate(zip(edges, attrs)):
                    e = (edge.t()[attr == c]).t()
                    dset.append(Data(x=emat[i], edge_index=e))
                    index.append((i, c))
            loader = DataLoader(dset, batch_size=batch_size, shuffle=False)

            # Graph convolution
            result = torch.zeros_like(emat)
            for c, (dat, conv) in enumerate(zip(loader, self.gconvs)):
                x = dat.x
                edge = dat.edge_index
                for ngcn in range(self.gcn_layers):
                    x = conv[ngcn](x=x, edge_index=edge)
                # reshape and add result of GCN
                result += torch.stack([x[dat.batch == i] for i in range(batch_size)], dim=0)
                if self.bigcn:
                    for ngcn in range(self.gcn_layers):
                        x = self.gconvs_inv[c][ngcn](x=x, edge_index=edge)
                        result += torch.stack([x[dat.batch == i] for i in range(batch_size)], dim=0)
                assert torch.isnan(result).sum() == 0

            # normalize gcn result by edge classes
            result = result / (self.num_edge_classes + 1)
            x_head = self.fc_head(result)
            x_tail = self.fc_tail(result)
            del result

            feat = x_head[:, 0].unsqueeze(1) + x_tail[:, 1:]

            adj_pred = torch.zeros(
                batch_size,
                self.num_edge_classes + 1,
                max_num_ent + 1,
                max_num_ent + 1,
                device=device,
                dtype=torch.float,
            )
            heads = cands.nonzero()[:, 1:].unique(sorted=True)
            feats = torch.zeros(batch_size, x_head.size(1), x_head.size(1) - 1, self.fc_out.input_dim, device=device)
            hs = torch.stack([x_head] * (x_head.size(1) - 1), dim=2) * cands[:, :, 1:].unsqueeze(-1)
            ts = torch.stack([x_tail[:, 1:]] * (x_tail.size(1)), dim=1) * cands[:, :, 1:].unsqueeze(-1)
            if self.feature == "add":
                feats = hs + ts
            elif self.feature == "bilinear":
                feats = self.bilinear(hs, ts)
            elif self.feature == "mul":
                feats = hs * ts
            if self.edge_emb_dim:
                eemb_index = torch.zeros(batch_size, feats.size(1), feats.size(2), dtype=torch.long, device=device)
                for i, (edge, attr) in enumerate(zip(edges, attrs)):
                    for h in heads:
                        target_edge = edge[1][edge[0] == h]
                        target_attr = attr[edge[0] == h]
                        for j in target_edge - 1:
                            eemb_index[i, h, j] = target_attr[(target_edge - 1) == j]
                assert eemb_index.max() < self.edge_emb.num_embeddings
                eemb = self.edge_emb(eemb_index)
                feats = torch.cat([feats, eemb], dim=-1)
            if self.dist_emb_dim:
                idx = dist_for_emb[:, :, 1:] + self.dist_emb_max
                demb = self.dist_emb(idx)
                feats = torch.cat([feats, demb], dim=-1)

            feats = self.dropout_out(feats)
            assert torch.isnan(feat).sum() == 0
            feats = self.fc_out(feats)

            for h in heads:
                feat = feats[:, h]
                adj_pred[:, :, h, 1:] = feat.transpose(-1, -2)
            del feat, x_head, x_tail, feats

            edges_next = edges_keep
            attrs_next = attrs_keep

            for i, (dat, a, cand) in enumerate(zip(data, adj_pred, cands)):
                cnd = cand[: num_ent[i] + 1, : num_ent[i] + 1]
                cmsk = torch.stack([cnd] * a.size(0), dim=0).bool()
                a = a[:, : num_ent[i] + 1, : num_ent[i] + 1] * dat.mask.bool() * cmsk
                msk = adj[i, :, : num_ent[i] + 1, : num_ent[i] + 1] == 0
                adj[i, :, : num_ent[i] + 1, : num_ent[i] + 1][msk] = a[msk]

                arg = a.argmax(dim=0) * dat.mask.bool() * cnd
                for c in range(1, self.num_edge_classes + 1):
                    nz = (arg == c).nonzero().transpose(0, 1)

                    edges_next[i] = torch.cat([edges_next[i], nz], dim=-1)
                    attrs_next[i] = torch.cat((attrs_next[i], torch.tensor([c] * nz.size(-1)).to(attrs_next[i])), dim=0)
                    assert len(attrs_next[i]) == edges_next[i].size(-1)

            edges = edges_next
            attrs = attrs_next

            del adj_pred, edges_next, attrs_next

            assert all([e.size(-1) == a.size(0) for e, a in zip(edges, attrs)])
            assert all([(e > num).sum() == 0 for e, num in zip(edges, num_ent)])
        ####################

        out_data = [
            Data(
                x=None,
                edge_index=edge.long(),
                edge_attr=attr,
                ent_indices=dat.ent_indices,
                ent_labels=dat.ent_labels,
                name=dat.name,
                text=dat.text,
                entity=dat.entity,
                gold_edge_index=dat.gold_edge_index,
                gold_edge_attr=dat.gold_edge_attr,
            )
            for edge, attr, dat in zip(edges, attrs, data)
        ]

        return out_data, adj

    @staticmethod
    def pad(vec, padding=0):
        max_len = max(map(len, vec))
        return torch.stack([torch.cat([v, torch.zeros(max_len - len(v)).to(v).fill_(padding)]) for v in vec], dim=0)

    @staticmethod
    def get_distance(tag_order, max_dist=1000, random_order=False):
        dist = torch.zeros(
            tag_order.size(0),
            tag_order.size(1),
            tag_order.size(1),
            device=tag_order.device,
            dtype=torch.long,
        )
        for e, num in enumerate(tag_order.transpose(0, 1)):
            dist[:, e] = tag_order - num.unsqueeze(-1)
            dist[:, :, e] = tag_order - num.unsqueeze(-1)
        # mask
        dist = dist * torch.stack([(tag_order != 0)] * tag_order.size(-1), dim=-1)

        # cut max
        dist[dist.abs() >= max_dist] = min(dist.max(), max_dist)

        if random_order:
            tmp = dist[dist != 0]
            dist[dist != 0] = torch.randint_like(tmp, 1, max(2, dist.max()))
            for i, d in enumerate(dist):
                tri = d.triu(diagonal=1)
                dist[i] = tri + tri.transpose(0, 1)
        return dist

    @staticmethod
    def get_cand_from_dist(dist):
        for num in dist.abs().unique(sorted=True):
            if num != 0:
                yield (dist.abs() == num)

    @staticmethod
    def get_edges_to_keep(edges, attrs, cands):
        device = cands.device
        batch_size = len(cands)
        ekeep = []
        akeep = []
        for i, (edge, attr, cand) in enumerate(zip(edges, attrs, cands)):
            keep = ~cand[edge[0], edge[1]]
            ekeep.append(edge[:, keep])
            akeep.append(attr[keep])
        return ekeep, akeep

    def get_tag_order(self, data):
        ent_indices = list(map(lambda d: d.ent_indices, data))
        ent_indices = self.pad(ent_indices).long()
        batch_size = len(ent_indices)
        device = ent_indices[0].device
        all_num_ent = int(ent_indices.max().item()) + 1

        ordered_tags = []
        tag_order = torch.zeros(
            batch_size,
            all_num_ent,
            dtype=torch.long,
            device=device,
        )
        m_len = 0
        for i, ind in enumerate(ent_indices):
            ind = ind[ind != 0]
            p_idx = None
            ordered = []
            for idx in ind:
                if idx != p_idx:
                    ordered.append(idx.item())
                p_idx = idx
            ordered_tags.append(ordered)
            for j, o in enumerate(ordered, start=1):
                tag_order[i, int(o)] = j
            m_len = max(len(ordered), m_len)
        for i in range(batch_size):
            ordered_tags[i].extend([0 for i in range(m_len - len(ordered_tags[i]))])
        ordered_tags = torch.tensor(ordered_tags, dtype=torch.long, device=device)

        return tag_order, ordered_tags


class MyLossFunc(nn.Module):
    def __init__(self, reduction="mean"):
        super(MyLossFunc, self).__init__()
        self.reduction = reduction

    @staticmethod
    def indices2mask(ent_indices):
        mask = torch.zeros(
            ent_indices.max().long() + 1,
            ent_indices.max().long() + 1,
            dtype=torch.bool,
            device=ent_indices.device,
        )
        ents = ent_indices.unique().long()
        ents = ents[ents != 0]
        for e1, e2 in torch.cartesian_prod(ents, ents):
            mask[e1, e2] = True
        return mask

    def forward(self, data, adj):
        loss = 0

        func = nn.CrossEntropyLoss(reduction=self.reduction)

        entropy = 0
        for i, dat in enumerate(data):
            num_ent = dat.ent_indices.max().long()
            ans = torch.zeros(dat.mask.size(0), dat.mask.size(1), dtype=torch.long, device=adj.device)
            mask = self.indices2mask(dat.ent_indices)
            edge = dat.gold_edge_index.transpose(0, 1).long()
            attr = dat.gold_edge_attr
            for e, att in zip(edge, attr):
                ans[e[0], e[1]] = att
            ans[~dat.mask.bool()] = -100
            entropy += func(adj[i, :, : dat.mask.size(0), : dat.mask.size(1)].unsqueeze(0), ans.unsqueeze(0))

        loss += entropy / adj.size(0)

        return loss