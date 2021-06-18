from difflib import Differ
import torch
import numpy as np
import random
from copy import deepcopy
import json
from pathlib import Path


def get_similar_adj(gold_adj, num_ent, num_edit=10):
    dist = torch.distributions.Bernoulli(0.5)
    shape = gold_adj.size()
    view_size = np.prod(gold_adj.size())
    mask = torch.zeros_like(gold_adj)
    mask[:, :, :num_ent, :num_ent] = 1
    gold_viewed = gold_adj.view(view_size)
    mask_viewed = mask.view(view_size)
    out_viewed = gold_viewed.clone()
    for i in range(num_edit):
        prob = dist.sample()
        add_adj = torch.zeros_like(out_viewed)
        if prob:
            # add edge
            cand = ((out_viewed == 0) * mask_viewed).nonzero()
            edit_pos = random.choice(cand)
            if len(cand) <= 0:
                continue
            add_adj[edit_pos] = 1
        else:
            # delete edge
            cand = ((out_viewed != 0) * mask_viewed).nonzero()
            if len(cand) <= 0:
                continue
            edit_pos = random.choice(cand)
            add_adj[edit_pos] = -1
        out_viewed += add_adj
    return out_viewed.view(*shape)


def in_func(tensor, list_of_tensor):
    return any(map(lambda x: (tensor != x).sum() == 0, list_of_tensor))


def diff_string(s1, s2):
    """
    return: index to convert s1 -> s2
    """
    differ = Differ()
    diff = list(differ.compare(s1, s2))
    diff_idx = [0] * len(s1)
    count = -1
    add = 0
    sub = 0
    for d in diff:
        mode = d[0]
        ch = d[2]
        if mode == "+":
            add += 1
        elif mode == "-":
            sub += 1
            # count += 1
            # diff_idx[count] = diff_idx[count - 1]
        else:
            count += 1
            for i in range(sub):
                if count == 0:
                    diff_idx[count] = 0
                else:
                    diff_idx[count] = diff_idx[count - 1] + 1
                count += 1
            if count == 0:
                diff_idx[count] = max(0, add - sub)
            else:
                diff_idx[count] = diff_idx[count - 1] + add + 1 - sub
            add = 0
            sub = 0
    # assert ''.join([s2[d] for d in diff_idx if d>0]) == s1
    for i in range(count, len(s1)):
        diff_idx[i] = diff_idx[i - 1] + 1
    return diff_idx


def count_tokens(tokens, exclude=[], max_len=0):
    start = [0] * max(len(tokens), max_len)
    end = [0] * max(len(tokens), max_len)
    for i, tok in enumerate(tokens):
        for ex in exclude:
            tok = tok.replace(ex, "")
        if i != len(tokens) - 1:
            start[i + 1] = start[i] + len(tok)
        if i != 0:
            end[i] = end[i - 1] + len(tok)
        else:
            end[i] = len(tok)

    return start, end


def calc_score_adj(pred, label):
    pred = pred != 0
    label = label != 0
    tp = (pred * label).sum().to(torch.float)
    fp = (pred * ~label).sum().to(torch.float)
    # tn = (~pred * ~label).sum()
    fn = (~pred * label).sum().to(torch.float)

    if tp == 0:
        recall = torch.tensor(0).to(pred).to(torch.float)
        precision = torch.tensor(0).to(pred).to(torch.float)
    else:
        recall = tp / (tp + fn)
        precision = tp / (tp + fp)

    if recall == 0 and precision == 0:
        f1 = torch.tensor(0).to(pred).to(torch.float)
    else:
        f1 = 2 * recall * precision / (recall + precision)

    return precision, recall, f1


def calc_score(data):
    tp = fn = fp = 0.0
    for dat in data:
        p_edge = dat.edge_index.transpose(0, 1)
        p_attr = dat.edge_attr
        g_edge = dat.gold_edge_index.transpose(0, 1)
        g_attr = dat.gold_edge_attr
        for idx, (pe, pa) in enumerate(zip(p_edge, p_attr)):
            flag = False
            for e_both in ((pe == g_edge).sum(-1) == 2).nonzero():
                if g_attr[e_both[0]] == pa:
                    flag = True
                    break
            if flag:
                tp += 1
            else:
                fp += 1
        for idx, (ge, ga) in enumerate(zip(g_edge, g_attr)):
            flag = False
            for e_both in ((ge == p_edge).sum(-1) == 2).nonzero():
                if p_attr[e_both[0]] == ga:
                    flag = True
                    break
            if flag:
                # tp += 1
                pass
            else:
                fn += 1
    precision = tp / (tp + fp + 1e-20)
    recall = tp / (tp + fn + 1e-20)
    f_measure = 2 * precision * recall / (precision + recall + 1e-20)
    return precision, recall, f_measure


class EarlyStopper:
    def __init__(self, model, patience=10, threshold=1e-3, startup=0, mode="min"):
        assert mode in ["min", "max"]
        self.patience = patience
        self.threshold = threshold
        self.startup = startup
        self.values = []
        self.mode = mode
        if self.mode == "min":
            self.sign = 1.0
        elif self.mode == "max":
            self.sign = -1.0
        self.count = 0
        self.model = model
        self.best_param = deepcopy(model.state_dict())
        self.best_epoch = 0

    def step(self, val):
        if len(self.values) == 0:
            self.values.append(val)
            return False
        val_past = self.values[-1]
        sub = val - val_past
        self.values.append(val)
        if self.sign * sub > 0 or abs(sub) < self.threshold:
            self.count += 1
        else:
            self.count = 0
            self.best_param = deepcopy(self.model.state_dict())
            self.best_epoch = len(self.values)

        return self.count > self.patience and len(self.values) > self.startup


def generate_ann_adj(data, adjs, class_list):
    assert adjs.size(1) == len(class_list) + 1
    adjs = adjs[:, :-1]
    files = []
    for (basename, datum), adj in zip(data.items(), adjs):
        txt = datum["text"]
        ann_lines = []

        # entity
        for tag, ent in datum["entity"].items():
            start = ent["start"]
            end = ent["end"]
            e = ent["entity"]
            label = ent["label"]
            line = "\t".join([tag, " ".join([label, str(start), str(end)]), e])
            ann_lines.append(line)

        # relation
        count = 1
        for cl, mat in zip(class_list, adj):
            for e1, e2 in mat.nonzero():
                line = "\t".join(
                    [
                        "R{}".format(count),
                        " ".join([cl, "Arg1:T{}".format(e1 + 1), "Arg2:T{}".format(e2 + 1)]),
                    ]
                )
                ann_lines.append(line)
                count += 1

        ann = "\n".join(ann_lines)
        files.append((basename, txt, ann))

    return files


def generate_ann(data, class_list):
    files = []
    for dat in data:
        txt = dat.text
        ann_lines = []

        # entity
        for tag, ent in dat.entity.items():
            start = ent["start"]
            end = ent["end"]
            e = ent["entity"]
            label = ent["label"]
            line = "\t".join([tag, " ".join([label, str(start), str(end)]), e])
            ann_lines.append(line)

        # relation
        count = 1
        for edge, lab in zip(dat.edge_index.transpose(0, 1), dat.edge_attr):
            e1 = edge[0]
            e2 = edge[1]

            line = "\t".join(
                [
                    "R{}".format(count),
                    " ".join(
                        [
                            class_list[lab.long().item()],
                            "Arg1:T{}".format(e1.long().item()),
                            "Arg2:T{}".format(e2.long().item()),
                        ]
                    ),
                ]
            )
            ann_lines.append(line)
            count += 1

        ann = "\n".join(ann_lines)
        files.append((dat.name, txt, ann))

    return files


legal_type = [int, str, float, bool, list, tuple, dict, set]


class Namespace:
    def __init__(self, **kwargs) -> None:
        self.__dict__.update(kwargs)

    def json(self):
        dic = self.__dict__
        for key in dic.keys():
            if not any([isinstance(dic[key], t) for t in legal_type]):
                dic[key] = str(dic[key])
        return json.dumps(self.__dict__)

    def __getitem__(self, item):
        return self.__dict__[item]

    def __setitem__(self, key, val):
        self.__dict__[key] = val

    def load(self, json_file: Path):
        self.__dict__.update(json.loads(json_file.read_text()))
        return self

    def __repr__(self) -> str:
        return self.json()

    def dict(self):
        return self.__dict__