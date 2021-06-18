import argparse
import os
from glob import glob
import numpy as np
from copy import deepcopy
from datautil import AnnDataLoader, EntityConverter
from tqdm import tqdm
from statistics import mean


def cal_score(conf):
    if conf[0][0] + conf[0][1] == 0:
        recall = float("inf")
    else:
        recall = float(conf[0][0] / (conf[0][0] + conf[0][1]))
    if conf[0][0] + conf[1][0] == 0:
        precision = float("inf")
    else:
        precision = float(conf[0][0] / (conf[0][0] + conf[1][0]))
    if recall + precision == 0:
        f1 = float("inf")
    else:
        f1 = float(2 * recall * precision / (recall + precision))
    return recall, precision, f1


def main(args):
    gold_prefix = args.gold_prefix
    pred_prefix = args.pred_prefix

    gold_path = gold_prefix
    pred_path = pred_prefix
    is_tex = args.tex

    if not os.path.exists(gold_path):
        raise ValueError
    if not os.path.exists(pred_path):
        raise ValueError

    gold_ann_files = glob(os.path.join(gold_path, "*.ann"))
    pred_ann_files = glob(os.path.join(pred_path, "*.ann"))

    all_conf_mats = {}

    dataloader = AnnDataLoader()

    for gold_ann_file in tqdm(gold_ann_files):
        fname = os.path.basename(gold_ann_file)
        pred_ann_file = os.path.join(pred_path, fname)
        gold_txt_file = os.path.join(gold_path, os.path.splitext(fname)[0] + ".txt")
        pred_txt_file = os.path.join(pred_path, os.path.splitext(fname)[0] + ".txt")
        if not pred_ann_file in pred_ann_files:
            continue
        if not os.path.exists(gold_txt_file):
            continue
        if not os.path.exists(pred_txt_file):
            continue

        gold_text, gold_entity, gold_relation, gold_event = dataloader(gold_txt_file, gold_ann_file)
        pred_text, pred_entity, pred_relation, pred_event = dataloader(pred_txt_file, pred_ann_file)

        converter = EntityConverter(gold_entity, gold_event, pred_entity, pred_event)

        conf_mats = {}
        # relation
        for tag, pr in pred_relation.items():
            label = pr["label"]
            arg1 = pr["arg1"]
            arg2 = pr["arg2"]

            if "E" in arg1:
                head_label = "Operation"
            else:
                head_label = pred_entity[arg1]["label"]
            if "E" in arg2:
                tail_label = "Operation"
            else:
                tail_label = pred_entity[arg2]["label"]

            label_s = "/".join((label, head_label, tail_label))

            if not label in conf_mats.keys():
                conf_mats[label] = np.zeros((2, 2))
            if not label_s in conf_mats.keys():
                conf_mats[label_s] = np.zeros((2, 2))

            arg1_conv = converter.p2g(arg1)
            arg2_conv = converter.p2g(arg2)
            assert len(arg1_conv) == 1
            assert len(arg2_conv) == 1

            tp = False
            for tag_gold, gld in gold_relation.items():
                a1_gld = gld["arg1"]
                a2_gld = gld["arg2"]
                label_gld = gld["label"]

                if (a1_gld in arg1_conv) and (a2_gld in arg2_conv) and label == label_gld:
                    tp = True
                    break

            if tp:
                conf_mats[label][0, 0] += 1
                conf_mats[label_s][0, 0] += 1
            else:
                conf_mats[label][1, 0] += 1
                conf_mats[label_s][1, 0] += 1

        for tag, pr in gold_relation.items():
            label = pr["label"]
            arg1 = pr["arg1"]
            arg2 = pr["arg2"]

            if "E" in arg1:
                head_label = "Operation"
            else:
                head_label = gold_entity[arg1]["label"]
            if "E" in arg2:
                tail_label = "Operation"
            else:
                tail_label = gold_entity[arg2]["label"]

            label_s = "/".join((label, head_label, tail_label))
            if not label in conf_mats.keys():
                conf_mats[label] = np.zeros((2, 2))
            if not label_s in conf_mats.keys():
                conf_mats[label_s] = np.zeros((2, 2))

            arg1_conv = converter.g2p(arg1)
            arg2_conv = converter.g2p(arg2)
            assert len(arg1_conv) == 1
            assert len(arg2_conv) == 1

            tp = False
            for tag_pred, prd in pred_relation.items():
                a1_prd = prd["arg1"]
                a2_prd = prd["arg2"]
                label_prd = prd["label"]

                if (a1_prd in arg1_conv) and (a2_prd in arg2_conv) and label == label_prd:
                    tp = True
                    break

            if tp:
                pass
            else:
                conf_mats[label][0, 1] += 1
                conf_mats[label_s][0, 1] += 1

        # event
        for tag, pr in pred_event.items():
            tag_glds = converter.p2g(tag)
            for p in pr:
                label = p[0]
                if label == "Operation":
                    continue
                label_s = "/".join([p[0], "Operation", pred_entity[p[1]]["label"]])
                tp = False
                p_conv = (p[0], converter.p2g(p[1]))

                if not label in conf_mats.keys():
                    conf_mats[label] = np.zeros((2, 2))
                if not label_s in conf_mats.keys():
                    conf_mats[label_s] = np.zeros((2, 2))

                for t in tag_glds:
                    judge = [gld[0] == p_conv[0] and gld[1] in p_conv[1] for gld in gold_event[t]]
                    if any(judge):
                        tp = True

                if tp:
                    conf_mats[label][0, 0] += 1
                    conf_mats[label_s][0, 0] += 1
                else:
                    conf_mats[label][1, 0] += 1
                    conf_mats[label_s][1, 0] += 1

                # event
        for tag, gld in gold_event.items():
            tag_prd = converter.g2p(tag)
            for g in gld:
                label = g[0]
                if label == "Operation":
                    continue
                tp = False
                g_conv = (g[0], converter.g2p(g[1]))
                label_s = "/".join([g[0], "Operation", pred_entity[g[1]]["label"]])

                if not label in conf_mats.keys():
                    conf_mats[label] = np.zeros((2, 2))
                if not label_s in conf_mats.keys():
                    conf_mats[label_s] = np.zeros((2, 2))

                for t in tag_prd:
                    judge = [prd[0] == g_conv[0] and prd[1] in g_conv[1] for prd in pred_event[t]]
                    if any(judge):
                        tp = True

                if tp:
                    # conf_mats[label][0, 0] += 1
                    # conf_mats[label_s][0, 0] += 1
                    pass
                else:
                    conf_mats[label][0, 1] += 1
                    conf_mats[label_s][0, 1] += 1

        for key, val in conf_mats.items():
            if not key in all_conf_mats:
                all_conf_mats[key] = val
            else:
                all_conf_mats[key] += val

    f1s = {}
    ps = {}
    rs = {}
    if is_tex:
        print('name & Precision & Recall & F-score \\\\')
    for key in reversed(sorted(all_conf_mats.keys(), key=lambda x: x.count("/"))):
        recall, precision, f1 = cal_score(all_conf_mats[key])
        f1s[key] = f1
        ps[key] = precision
        rs[key] = recall
        if is_tex:
            print('{} & {:.3} & {:.3} & {:.3} \\\\'.format(key,precision,recall,f1))
        else:
            print(key)
            print(all_conf_mats[key])
            print("P: {:.5f} R: {:.5f} F1: {:.5f}".format(precision, recall, f1))


    cls_conf = [val for key, val in all_conf_mats.items() if key.count("/") == 0]
    conf_sum = sum(cls_conf)
    recall_all, precision_all, micro_f = cal_score(conf_sum)
    if is_tex:
        print('{} & {:.3} & {:.3} & {:.3} \\\\'.format('Micro',precision_all,recall_all,micro_f))
    else:
        print()
        print("Micro-F")
        print(conf_sum)
        print("P: {:.5f} R: {:.5f} F1: {:.5f}".format(precision_all, recall_all, micro_f))

    cls_f1 = [f for key, f in f1s.items() if key.count("/") == 0]
    cls_p = [f for key, f in ps.items() if key.count("/") == 0]
    cls_r = [f for key, f in rs.items() if key.count("/") == 0]
    for i, f in enumerate(cls_f1):
        if np.isnan(f) or np.isinf(f):
            cls_f1[i] = 0.0
    for i, f in enumerate(cls_p):
        if np.isnan(f) or np.isinf(f):
            cls_p[i] = 0.0
    for i, f in enumerate(cls_r):
        if np.isnan(f) or np.isinf(f):
            cls_r[i] = 0.0
    if is_tex:
        print('{} & {:.3} & {:.3} & {:.3} \\\\'.format('Macro',mean(cls_p), mean(cls_r), mean(cls_f1)))
    else:
        # print("{:.5f}".format(mean(cls_f1)))
        print("Macro")
        print("P: {:.5f} R: {:.5f} F1: {:.5f}".format(mean(cls_p), mean(cls_r), mean(cls_f1)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("gold_prefix", type=str, help="path to ground truth")
    parser.add_argument("pred_prefix", type=str, help="path to prediction")
    parser.add_argument("--tex", action='store_true')

    args = parser.parse_args()

    main(args)