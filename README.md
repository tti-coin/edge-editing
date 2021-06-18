# edge-editing
The source code for "A Neural Edge-Editing Approach for Document-Level Relation Graph Extraction" in Findings of ACL-IJCNLP 2021

Please refer to our [paper]() for details.

## Citation

When you utilize our code, cite our paper.

```bibtex
@inproceedings{makino-2021-edge-editing,
    title = "A Neural Edge-Editing Approach for Document-Level Relation Graph Extraction",
    author = "Makino, Kohei and Miwa, Makoto and Sasaki, Yutaka",
    booktitle = "Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    abstract = "In this paper, we propose a novel edge-editing approach to extract relation information from a document. We treat the relations in a document as a relation graph among entities in this approach. The relation graph is iteratively constructed by editing edges of an initial graph, which might be a graph extracted by another system or an empty graph. The way to edit edges is to classify them in a close-first manner using the document and temporally-constructed graph information; each edge is represented with a document context information by a pretrained transformer model and a graph context information by a graph convolutional neural network model. We evaluate our approach on the task to extract material synthesis procedures from materials science texts. The experimental results show the effectiveness of our approach in editing the graphs initialized by our in-house rule-based system and empty graphs.",
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
