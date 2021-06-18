import torch
from datautil import AnnDataLoader
import os
import pickle as pkl
from collections import OrderedDict
from transformers import AutoTokenizer
import utils
from configparser import ConfigParser
import contextlib
import argparse
import spacy
import torch_geometric
from torch_geometric.data import Data, Dataset
from pathlib import Path
import numpy as np

# make path list
def get_pathes(gold_path, pred_path):
    pathes = []
    txt_pathes = gold_path.glob("*.txt")
    gold_ann_pathes = gold_path.glob("*.ann")
    pred_ann_pathes = pred_path.glob("*.ann")

    for tp in txt_pathes:
        tag = tp.stem
        gold_ap = gold_path / "{}.ann".format(tag)
        pred_ap = pred_path / "{}.ann".format(tag)

        if gold_ap.exists() and pred_ap.exists():
            pathes.append((tp, gold_ap, pred_ap))
        else:
            print(tp, "do not have pair ann.")
    return pathes


# load data
def load(pathes):
    loader = AnnDataLoader()
    gold_data = OrderedDict()
    pred_data = OrderedDict()
    for p in pathes:
        txt_path, ann_gold, ann_pred = p
        key = txt_path.stem
        text, ents, rels, corefs = loader(txt_path, ann_gold)
        gold_data[key] = {
            "text": text,
            "entity": ents,
            "relation": rels,
            "coref": corefs,
        }
        text, ents, rels, corefs = loader(txt_path, ann_pred)
        pred_data[key] = {
            "text": text,
            "entity": ents,
            "relation": rels,
            "coref": corefs,
        }

    return gold_data, pred_data


# adjacency matrix
# |V| x |V|
def get_edges(rels, relation_classes):
    edges = []
    attr = []
    for key, rel in rels.items():
        arg1 = int(rel["arg1"][1:])
        arg2 = int(rel["arg2"][1:])
        label = rel["label"]
        lbl_idx = relation_classes.index(label)
        attr.append(lbl_idx)
        edges.append((arg1, arg2))
    return edges, attr


def get_relation_data(data, relation_classes):
    edges = []
    attrs = []
    for key, val in data.items():
        rels = val["relation"]
        edge, attr = get_edges(rels, relation_classes)
        edges.append(torch.tensor(edge, dtype=torch.long).t())
        attrs.append(torch.tensor(attr, dtype=torch.long))

    return edges, attrs


def encode_sentences(
    data,
    span_join=False,
    special=False,
    nlp=spacy.load("en_core_sci_sm"),
    tokenizer=AutoTokenizer.from_pretrained("allenai/longformer-base-4096"),
):
    ids_all = []
    max_len = 0
    for key, val in data.items():
        if special:
            text = val["text"]
            doc = nlp(val["text"])

            # join sentences if span over the sentence
            if span_join:
                sent_starts = [s[0].idx for s in doc.sents]
                to_join = []
                for ent in val["entity"].values():
                    for idx, start in enumerate(sent_starts):
                        if ent["start"] < start <= ent["end"]:
                            flag = False
                            for i in range(len(to_join)):
                                if idx in to_join[i] or idx - 1 in to_join[i]:
                                    to_join[i].append(idx)
                                    flag = True
                            if not flag:
                                to_join.append([idx, idx - 1])
                sents = list(map(lambda x: x.text, doc.sents))

                joined_sents = []
                buf = []
                seps = []
                for i, s in enumerate(doc.sents):
                    flag = False
                    for idxs in to_join:
                        if i in idxs:
                            flag = True
                            break
                    if not flag:
                        if buf:
                            joined_sents.append(" ".join(buf))
                            buf = []
                        joined_sents.append(s.text)
                    else:
                        buf.append(s.text)
                        if buf:
                            seps.append(text[s.start_char - 1])
                if buf:
                    joined_sents.append(" ".join(buf))
                sents = joined_sents

            else:
                sents = list(map(lambda x: x.text, doc.sents))

            ids_sent = tokenizer(sents, add_special_tokens=True)["input_ids"]
            # join sents
            ids = ids_sent[0]
            for s in ids_sent[1:]:
                ids.extend([w for w in s if w != tokenizer.cls_token_id])
            ids_all.append(torch.tensor(ids, dtype=torch.long))
            max_len = max(max_len, len(ids))

        else:
            ids_all.append(tokenizer(val["text"], add_special_tokens=True)["input_ids"])

    return ids_all


def get_ent_indices(data, text_ids, tokenizer=AutoTokenizer.from_pretrained("allenai/longformer-base-4096")):
    entity_tags = []
    for _, ((key, dat), ids) in enumerate(zip(data.items(), text_ids)):
        text = dat["text"]
        # map: tokens -> converted string
        tokens = tokenizer.convert_ids_to_tokens(ids)
        mask = [s in [tokenizer.sep_token_id, tokenizer.cls_token_id, tokenizer.pad_token_id] for s in ids]
        string = "".join(tokens)

        tokens_string_start = []
        tokens_string_end = []
        pos = 0
        for i, tok in enumerate(tokens):
            word = tok
            add = 1
            while not word in string[pos : pos + add]:
                add += 1
            tokens_string_start.append(pos)
            tokens_string_end.append(pos + add)
            pos = pos + add
        assert "".join(tokens) == string

        # map: converted_string -> tokens
        string_tokens = torch.zeros(len(string), dtype=torch.long)
        for i, (st, en) in enumerate(zip(tokens_string_start, tokens_string_end)):
            assert (string_tokens[st:en] != 0).sum() == 0
            assert string[st:en] == tokens[i]
            string_tokens[st:en] = i

        # check string_tokens
        pre = -1
        cnt = 0
        for i, st in enumerate(string_tokens):
            ch = string[i]
            if st == pre:
                cnt += 1
            else:
                cnt = 0
            assert ch == tokens[st][cnt]
            pre = st

        # map: text -> converted string
        text_string = utils.diff_string(text, string)

        ent_tag = torch.zeros(len(tokens), dtype=torch.long)
        for tag, ent in dat["entity"].items():
            start = ent["start"]
            end = ent["end"]

            start_st = text_string[start]
            end_st = text_string[end] if end < len(text_string) else -1

            start_tok = string_tokens[start_st]
            end_tok = string_tokens[end_st]
            assert (ent_tag[start_tok:end_tok] != 0).sum() == 0
            if start_tok == end_tok:
                ent_tag[start_tok] = int(tag[1:])
            else:
                ent_tag[start_tok:end_tok] = int(tag[1:])

            if not string[start_st:end_st][-1] == tokens[end_tok - 1][-1]:
                end_tok += 1
            # if not ent["entity"] in tokenizer.convert_tokens_to_string(tokens[start_tok:end_tok]):
            #     print(
            #         ent["entity"],
            #         tokenizer.convert_tokens_to_string(tokens[start_tok:end_tok]),
            #         start_tok.item(),
            #         end_tok.item(),
            #         sep="\t",
            #     )

        entity_tags.append(ent_tag)
    return entity_tags


def check_ent_indices(
    data, text_ids, ent_indices, tokenizer=AutoTokenizer.from_pretrained("allenai/longformer-base-4096")
):
    for _, ((key, dat), ids, ind) in enumerate(zip(data.items(), text_ids, ent_indices)):
        for key, ent in dat["entity"]:
            spn = (ind == int(key[1:])).nonzero()

            print(
                ent["entity"],
                tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(ids[spn.min() : spn.max() + 1])),
                sep="\t",
            )


# entity label embedding
def get_ent_labels(data, entity_classes):
    entity_labels = []
    for i, val in enumerate(data.values()):
        elbl = torch.zeros(max(list(map(lambda x: int(x[1:]), val["entity"].keys()))))
        for tag, ent in val["entity"].items():
            label = ent["label"]
            idx = int(tag[1:]) - 1
            assert label in entity_classes
            elbl[idx] = entity_classes.index(label)
        entity_labels.append(elbl)
        assert elbl.max() < len(entity_classes)
    return entity_labels


def get_mask(ent_indices):
    ents = ent_indices.unique()
    ents = ents[ents != 0]
    mx = ents.max() + 1
    mask = torch.zeros(mx, mx, dtype=torch.bool, device=ent_indices.device)
    for ind1 in ents:
        for ind2 in ents:
            mask[ind1, ind2] = True

    return mask


def get_dataset(
    data,
    entity_classes,
    relation_classes,
    gold_data=None,
    nlp=spacy.load("en_core_sci_sm"),
    tokenizer=AutoTokenizer.from_pretrained("allenai/longformer-base-4096"),
):
    edges, attrs = get_relation_data(data, relation_classes)
    ids = encode_sentences(data, span_join=True, nlp=nlp, tokenizer=tokenizer)
    ent_indices = get_ent_indices(data, ids, tokenizer=tokenizer)
    ent_labels = get_ent_labels(data, entity_classes)

    # check_ent_indices(data, ids, ent_indices)

    dataset = []
    for i, (key, edge, attr, idx, ent_ind, ent_lbl) in enumerate(
        zip(data.keys(), edges, attrs, ids, ent_indices, ent_labels)
    ):
        dat = data[key]
        raw_txt = dat["text"]
        ent = dat["entity"]
        rel = dat["relation"]
        coref = dat["coref"]
        g_edge = gold_data[i].edge_index if gold_data else None
        g_attr = gold_data[i].edge_attr if gold_data else None
        d = Data(
            x=None,
            edge_index=edge,
            edge_attr=attr,
            ent_indices=ent_ind,
            ent_labels=ent_lbl,
            ids=torch.tensor(idx, dtype=torch.long),
            mask=get_mask(ent_ind),
            name=key,
            text=raw_txt,
            entity=ent,
            relation=rel,
            coref=coref,
            gold_edge_index=g_edge,
            gold_edge_attr=g_attr,
        )
        if g_edge is not None:
            assert g_edge.max() <= ent_ind.max()
        assert edge.max() <= ent_ind.max()

        dataset.append(d)
    return dataset


@contextlib.contextmanager
def preprocess(args):
    gold_prefix = args.gold_prefix
    pred_prefix = args.pred_prefix
    pkl_path = args.pkl_path
    transformer = args.transformer
    config_file = args.config
    spacy_name = args.spacy

    if not pkl_path.exists():
        os.makedirs(pkl_path)

    config = ConfigParser()
    config.read(config_file)

    relation_classes = [
        "ROOT",
        *config["relation"]["classes"].replace(" ", "").split(","),
    ]
    entity_classes = ["ROOT", *config["entity"]["classes"].replace(" ", "").split(",")]

    tokenizer = AutoTokenizer.from_pretrained(transformer)
    nlp = spacy.load(spacy_name)

    train_gold_path = gold_prefix / "train"
    devel_gold_path = gold_prefix / "dev"
    test_gold_path = gold_prefix / "test"

    train_pred_path = pred_prefix / "train"
    test_pred_path = pred_prefix / "test"
    devel_pred_path = pred_prefix / "dev"

    print("Loading ann and txt...")

    train_pathes = get_pathes(train_gold_path, train_pred_path)
    devel_pathes = get_pathes(devel_gold_path, devel_pred_path)
    test_pathes = get_pathes(test_gold_path, test_pred_path)

    train_gold_data, train_pred_data = load(train_pathes)
    devel_gold_data, devel_pred_data = load(devel_pathes)
    test_gold_data, test_pred_data = load(test_pathes)
    print("finished")

    print("Processing...")
    train_gold = get_dataset(train_gold_data, entity_classes, relation_classes, nlp=nlp, tokenizer=tokenizer)
    devel_gold = get_dataset(devel_gold_data, entity_classes, relation_classes, nlp=nlp, tokenizer=tokenizer)
    test_gold = get_dataset(test_gold_data, entity_classes, relation_classes, nlp=nlp, tokenizer=tokenizer)

    train_dataset = get_dataset(
        train_pred_data, entity_classes, relation_classes, nlp=nlp, tokenizer=tokenizer, gold_data=train_gold
    )
    devel_dataset = get_dataset(
        devel_pred_data, entity_classes, relation_classes, nlp=nlp, tokenizer=tokenizer, gold_data=devel_gold
    )
    test_dataset = get_dataset(
        test_pred_data, entity_classes, relation_classes, nlp=nlp, tokenizer=tokenizer, gold_data=test_gold
    )
    print("finished")

    print("Saving...")
    torch.save(
        train_dataset,
        pkl_path / "train.pkl",
        pickle_protocol=pkl.HIGHEST_PROTOCOL,
    )
    torch.save(
        devel_dataset,
        pkl_path / "dev.pkl",
        pickle_protocol=pkl.HIGHEST_PROTOCOL,
    )
    torch.save(
        test_dataset,
        pkl_path / "test.pkl",
        pickle_protocol=pkl.HIGHEST_PROTOCOL,
    )
    print("finished")


if __name__ == "__main__":
    # TODO: default value
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl_path", type=Path, default="preprocess/pkl/", help='Output path to save preprocessed data')
    parser.add_argument("--gold_prefix", type=Path, default="preprocess/data_rel/", help='Path to gold data')
    parser.add_argument("--pred_prefix", type=Path, default="preprocess/rule_rel",help='Path to input of the model e.g. output of the rule based model')
    parser.add_argument("--config", type=Path, default="olivetti.conf", help='Path to config file')
    parser.add_argument("--transformer", type=str, default="allenai/longformer-base-4096", help='Transformer model name to load tokenizer')
    parser.add_argument("--spacy", type=str, default="en_core_sci_sm", help='Spacy model')

    args = parser.parse_args()

    preprocess(args)