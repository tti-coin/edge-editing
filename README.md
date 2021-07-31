# edge-editing
The source code for "A Neural Edge-Editing Approach for Document-Level Relation Graph Extraction" in Findings of ACL-IJCNLP 2021

Please refer to our [paper](https://arxiv.org/abs/2106.09900) for details.

## Citation

When you utilize our code, cite our paper.

```bibtex
@inproceedings{makino-etal-2021-neural,
    title = "A Neural Edge-Editing Approach for Document-Level Relation Graph Extraction",
    author = "Makino, Kohei  and
      Miwa, Makoto  and
      Sasaki, Yutaka",
    booktitle = "Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.findings-acl.234",
    doi = "10.18653/v1/2021.findings-acl.234",
    pages = "2653--2662",
}
```

## Dependencies

- pytorch (tested on 1.8.0)
- pytorch-geometric
- spacy==2.3.7
- scispacy
- tqdm
- optuna

You can prepare the environment easily with docker.
```bash
docker build . -t name_of_image --build-arg UID=`id -u`
docker run -t -d --name edge-edit -v `pwd`:/workspace --gpus all name_of_image
docker exec -it edge-edit bash
```

## Usage

- Preprocess
```bash
sh scripts/preprocess.sh
```

- Run our script you like
  - We have prepared scripts for each of the experiments in the paper.
```bash
sh train_wo?rule_.*\.sh
```

- Tuning using optuna
  - (optional) Prepare SQL server at first e.g. MySQL.
  - Run our script.
  - If you want to change search space, change source code directly.
```bash
python src/train.py --optuna study_name --optuna_storage SQL_server --optuna_n_trials number +(other arguments)
```
