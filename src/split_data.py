from glob import glob
from argparse import ArgumentParser
import os
import shutil
from pathlib import Path


def preprocess(args):
    output_dir = args.output

    train_list_file = args.train_files
    dev_list_file = args.dev_files
    test_list_file = args.test_files

    train_output_dir = output_dir / "train"
    dev_output_dir = output_dir / "dev"
    test_output_dir = output_dir / "test"

    for o in (train_output_dir, dev_output_dir, test_output_dir):
        if not o.exists():
            os.makedirs(o)

    for lst_file, output_dir in (
        (train_list_file, train_output_dir),
        (dev_list_file, dev_output_dir),
        (test_list_file, test_output_dir),
    ):
        file_list = lst_file.read_text().strip().split()
        for basename in file_list:
            ann_base = "{}.ann".format(basename)
            txt_base = "{}.txt".format(basename)
            inann = lst_file.parent / "data" / ann_base
            intxt = lst_file.parent / "data" / txt_base
            if not os.path.exists(inann) or not os.path.exists(intxt):
                raise ValueError

            # exception case in train data
            if "101039b910641c" in str(basename):
                continue

            outann = output_dir / ann_base
            outtxt = output_dir / txt_base
            shutil.copy(inann, outann)
            shutil.copy(intxt, outtxt)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--output", type=Path, default="data_split")
    parser.add_argument("--train_files", type=Path, default="ner-train-fnames.txt")
    parser.add_argument("--dev_files", type=Path, default="ner-dev-fnames.txt")
    parser.add_argument("--test_files", type=Path, default="ner-test-fnames.txt")

    args = parser.parse_args()

    preprocess(args)