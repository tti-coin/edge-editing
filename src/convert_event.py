from argparse import ArgumentParser
import os
from glob import glob
from tqdm import tqdm
from datautil import AnnDataLoader


class ConverterEventToRelation():
    def __init__(self):
        self.dataloader = AnnDataLoader()

    def convert(self, txt_path, ann_path):
        text, entities, relations, events = self.dataloader(txt_path, ann_path)

        lines = []
        # entity
        for tag, ent in entities.items():
            start = ent['start']
            end = ent['end']
            label = ent['label']
            word = ent['entity']
            line = '{}\t{} {} {}\t{}'.format(tag, label, start, end, word)
            lines.append(line)

        # relation
        r_num = 1
        for rtag, relation in relations.items():
            tag = 'R{}'.format(r_num)
            arg1 = relation['arg1']
            label = relation['label']
            arg2  = relation['arg2']

            if 'E' in arg1:
                arg1 = self.get_operation(arg1, events)
            if 'E' in arg2:
                arg2 = self.get_operation(arg2, events)

            line = '{}\t{} Arg1:{} Arg2:{}'.format(tag, label, arg1, arg2)
            lines.append(line)
            r_num += 1

        # events
        for tag, event in events.items():
            ope = self.get_operation(tag, events)
            for e in event:
                if e[0] == 'Operation':
                    continue
                t = 'R{}'.format(r_num)
                label = e[0]
                arg2 = e[1]
                line = '{}\t{} Arg1:{} Arg2:{}'.format(t, label, ope, arg2)
                lines.append(line)
                r_num += 1

        ann = '\n'.join(lines)

        return text,ann

    @staticmethod
    def get_operation(tag, events):
        event = events[tag]
        for e in event:
            if e[0] == 'Operation':
                return e[1]


def main(args):
    input_dir = args.input_dir
    output_dir = args.output_dir

    ann_files = glob(os.path.join(input_dir, '*.ann'))

    converter = ConverterEventToRelation()

    file_pairs = []
    for f in ann_files:
        tfile = os.path.splitext(f)[0] + '.txt'
        if os.path.exists(tfile):
            file_pairs.append((tfile, f))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for tfile, afile in tqdm(file_pairs):
        txt, ann = converter.convert(tfile, afile)
        with open(os.path.join(output_dir, os.path.basename(afile)), 'w') as f:
            f.write(ann)
        with open(os.path.join(output_dir, os.path.basename(tfile)), 'w') as f:
            f.write(txt)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument('--input_dir', default='out/', type=str)
    parser.add_argument('--output_dir', default='rel/', type=str)

    args = parser.parse_args()
    main(args)