from argparse import ArgumentParser
import os
from rule import RuleBasedRelationExtractor
from tqdm import tqdm
from pathlib import Path


def main(args):
    input_dir = args.input_dir
    output_dir = args.output_dir

    ann_files = input_dir.glob("*.ann")

    extractor = RuleBasedRelationExtractor()

    file_pairs = []
    for f in ann_files:
        tfile = f.parent/ (f.stem + ".txt")
        if tfile.exists():
            file_pairs.append((tfile, f))

    if not output_dir.exists():
        os.makedirs(output_dir)

    for tfile, afile in tqdm(file_pairs):
        ann = extractor(tfile, afile)
        (output_dir / afile.name).write_text(ann)
        txt = tfile.read_text()
        (output_dir / tfile.name).write_text(txt)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--input_dir", default="data/", type=Path)
    parser.add_argument("--output_dir", default="out/", type=Path)

    args = parser.parse_args()
    main(args)