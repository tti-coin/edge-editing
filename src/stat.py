from mylib import RelationData
from pathlib import Path
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument("input", type=Path)
parser.add_argument("--tex", action="store_true")
parser.add_argument("--all", action="store_true")
args = parser.parse_args()


def get_stat(data):
    stat_ent = {}
    stat_rel = {}
    for dat in data.values():
        for ent in dat["entity"].values():
            label = ent["label"]
            if not label in stat_ent:
                stat_ent[label] = 0
            stat_ent[label] += 1

        for rel in dat["relation"].values():
            label = rel["label"]
            if not label in stat_rel:
                stat_rel[label] = 0
            stat_rel[label] += 1

    std_ent = dict(sorted(stat_ent.items(), key=lambda x: x[1], reverse=True))
    std_rel = dict(sorted(stat_rel.items(), key=lambda x: x[1], reverse=True))
    return std_ent,std_rel




if args.all:
    edics = {}
    rdics = {}
    modes=  ['train','devel','test']
    for m in modes:
        d = args.input/m
        if d.is_dir():
            data = RelationData(d, pattern="*.ann")
            e,r = get_stat(data)
            e['all'] = sum(e.values())
            edics[m]=e
            r['all'] = sum(r.values())
            rdics[m] = r
    std_ent = []
    std_rel = []
    for k in edics['train'].keys():
        lst = [k]
        lst.extend([edics[m][k] if k in edics[m] else 0 for m in modes])
        std_ent.append(lst)
    for k in rdics['train'].keys():
        lst = [k]
        lst.extend([rdics[m][k] if k in rdics[m] else 0 for m in modes])
        std_rel.append(lst)

else:
    data = RelationData(args.input, pattern="*.ann")
    std_ent,std_rel = get_stat(data)
if args.tex:
    txt_ent = " \\\\ \n".join(map(lambda x: " & ".join(map(str, x)), std_ent))
    txt_rel = "\\\\ \n".join(map(lambda x: " & ".join(map(str, x)), std_rel))
else:
    txt_ent = "\n".join(map(lambda x: "\t".join(map(str, x)), std_ent))
    txt_rel = "\n".join(map(lambda x: "\t".join(map(str, x)), std_rel))

print("Entity:")
print(txt_ent)
print()
print("Relation:")
print(txt_rel)