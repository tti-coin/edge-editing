import torch
from tqdm import tqdm
import os
from model import EdgeEditor, MyLossFunc
from torch.utils.tensorboard import SummaryWriter
from glob import glob
from collections import OrderedDict
from statistics import mean
import utils
from utils import Namespace
import argparse
import random
from configparser import ConfigParser
from pathlib import Path
import torch_geometric
import optuna
import itertools


def _train(args):
    # setting path
    pkl_path = Path(args.pkl_path) if args.pkl_path else None
    output_path = args.output_path / args.job_tag
    if (output_path / "{:04}".format(0)).exists():
        trial_num = (
            max(
                list(
                    map(
                        lambda x: int(x.name),
                        output_path.glob("*"),
                    )
                )
            )
            + 1
        )
    else:
        trial_num = 0
    output_path = output_path / "{:04}".format(trial_num)
    ann_save_path = output_path / "out"
    model_save_path = output_path / "model"
    tbpath = output_path
    model_path = Path(args.model_path) if args.model_path else None
    config_file = args.config

    # setting model
    hidden_dim = args.hidden_dim
    num_gcn = args.num_gcn
    bigcn = args.bigcn
    gcn_layer = args.gcn_layer
    transformer = args.transformer
    finetune_transformer = args.finetune_transformer
    finetune_transformer_start = args.finetune_transformer_start
    head_tail_layers = args.head_tail_layers
    edge_emb_dim = args.edge_emb_dim
    feature = args.feature
    dropout_out = args.dropout_out
    fc_out_layer = args.fc_out_layer
    d_max = args.d_max
    dist_emb_dim = args.dist_emb_dim
    dist_emb_max = args.dist_emb_max
    from_empty = args.from_empty
    from_random = args.from_random
    random_order = args.random_order

    # setting training
    epochs = args.epochs
    batch_size = args.batch_size
    max_batch_size = args.max_batch_size
    device = torch.device(args.device)
    eval_test = args.eval_test
    debug = args.debug
    optimizer_name = args.optimizer
    lr = args.lr
    score_name = args.score

    # other params
    comment = args.comment

    if "cuda" in args.device:
        torch.cuda.set_device(device)

    # read config file
    config = ConfigParser()
    config.read(config_file)
    relation_classes = [
        "ROOT",
        *config["relation"]["classes"].replace(" ", "").split(","),
    ]
    entity_classes = ["ROOT", *config["entity"]["classes"].replace(" ", "").split(",")]

    # save options
    options = Namespace(**vars(args))
    options["relation_classes"] = relation_classes
    options["entity_classes"] = entity_classes

    writer = SummaryWriter(tbpath)

    if not output_path.exists():
        os.makedirs(output)

    # model save path
    if not model_save_path.exists():
        os.makedirs(model_save_path)
    (model_save_path / "config.json").write_text(options.json())

    if not ann_save_path.exists():
        os.makedirs(ann_save_path)

    # load data
    print("Loading data...")
    train_dataset = torch.load(pkl_path / "train.pkl")
    devel_dataset = torch.load(pkl_path / "dev.pkl")
    test_dataset = torch.load(pkl_path / "test.pkl")
    print("Completed")

    # upload to device
    def upload_device(dataset, device):
        for i in range(len(dataset)):
            dataset[i].mask = dataset[i].mask.bool()
            dataset[i] = dataset[i].to(device)
        return dataset

    train_dataset = upload_device(train_dataset, device)
    devel_dataset = upload_device(devel_dataset, device)
    test_dataset = upload_device(test_dataset, device)

    # cacluate score for rule-based model
    train_rule_precision, train_rule_recall, train_rule_f1 = utils.calc_score(train_dataset)
    devel_rule_precision, devel_rule_recall, devel_rule_f1 = utils.calc_score(devel_dataset)
    test_rule_precision, test_rule_recall, test_rule_f1 = utils.calc_score(test_dataset)

    # from empty, i.e., without rule base model
    if from_empty:

        def del_edge(data):
            for i in range(len(data)):
                data[i].edge_index = torch.empty(2, 0, device=device)
                data[i].edge_attr = torch.empty(0, device=device)
            return data

        train_dataset = del_edge(train_dataset)
        devel_dataset = del_edge(devel_dataset)
        test_dataset = del_edge(test_dataset)

    # from random graph, i.e., edges are not reliable
    if from_random:

        def randomize_edge(data):
            for i in range(len(data)):
                nodes = data[i].ent_indices.unique()
                nodes = nodes[nodes != 0]
                # random edge
                n_edge = data[i].edge_index.size(1)
                all_perm = list(itertools.permutations(list(range(0, len(nodes))), 2))
                index = torch.tensor(random.sample(all_perm, n_edge), dtype=torch.long, device=device).transpose(0, 1)
                r_edges = nodes[index]
                attribs = torch.randint(1, len(relation_classes), (n_edge,), device=device)
                data[i].edge_index = r_edges
                data[i].edge_attr = attribs
            return data

        train_dataset = randomize_edge(train_dataset)
        devel_dataset = randomize_edge(devel_dataset)
        test_dataset = randomize_edge(test_dataset)

    # create instance of model
    net = EdgeEditor(
        transformer=transformer,
        hidden_size=hidden_dim,
        gcn_layers=num_gcn,
        feature=feature,
        dropout_out=dropout_out,
        fc_out_layer=fc_out_layer,
        num_edge_classes=len(relation_classes) - 1,
        num_node_classes=len(entity_classes) - 1,
        gcn_module=torch_geometric.nn.__dict__[gcn_layer],
        d_max=d_max,
        bigcn=bigcn,
        random_order=random_order,
        edge_emb_dim=edge_emb_dim,
        dist_emb_max=dist_emb_max,
        dist_emb_dim=dist_emb_dim,
        head_tail_layers=head_tail_layers,
    )
    net = net.to(device)

    writer.add_text("comment", comment)

    # load weight if provided
    if model_path:
        if os.path.exists(model_path):
            net.load_state_dict(torch.load(model_path))

    # loss function
    loss_func = MyLossFunc()
    # optimizer
    optimizer = torch.optim.__dict__[optimizer_name](net.parameters(), lr=lr)

    iteration = 0
    max_score = -float("inf")

    train_min_loss = float("inf")
    devel_min_loss = float("inf")
    devel_max_f1 = -float("inf")
    score_max_epoch = 0
    best_test_score = 0
    for epoch in range(epochs):
        losses = []
        recalls = []
        precisions = []
        f1_epoch = []
        train_itr = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda x: x,
        )

        with tqdm(train_itr) as pbar:
            pbar.set_description("{:04}/[Epoch {:3}]".format(trial_num, epoch))
            for _, dat in enumerate(pbar):
                net.train()
                optimizer.zero_grad()

                # forward
                dat_pred, adj = net(
                    dat,
                    finetune_transformer=finetune_transformer and epoch >= finetune_transformer_start,
                )

                loss = loss_func(dat, adj)

                # backward
                loss.backward()

                losses.append(loss.item())

                optimizer.step()

                # record to the tensorboard
                writer.add_scalar("train/loss_itr", loss, iteration)
                pbar.set_postfix(OrderedDict(loss="{:.4}".format(losses[-1])))
                precision_itr, recall_itr, f1_itr = utils.calc_score(dat_pred)

                recalls.append(recall_itr)
                precisions.append(precision_itr)
                f1_epoch.append(f1_itr)

                writer.add_scalar("train/f1_itr", f1_itr, iteration)
                writer.add_scalar("train/recall_itr", recall_itr, iteration)
                writer.add_scalar("train/precision_itr", precision_itr, iteration)

                iteration += 1
                if debug:
                    break
        train_loss = mean(losses)
        train_min_loss = min(train_loss, train_min_loss)

        writer.add_scalar("train/loss", train_loss, epoch)

        f1_epoch = torch.tensor(f1_epoch)
        recalls = torch.tensor(recalls)
        precisions = torch.tensor(precisions)

        writer.add_scalar("train/f1", f1_epoch.mean(), epoch)
        writer.add_scalar("train/recall", recalls.mean(), epoch)
        writer.add_scalar("train/precision", precisions.mean(), epoch)
        writer.add_scalar("train/f1_sub", f1_epoch.mean() - train_rule_f1, epoch)
        writer.add_scalar("train/recall_sub", recalls.mean() - train_rule_recall, epoch)
        writer.add_scalar("train/precision_sub", precisions.mean() - train_rule_precision, epoch)

        # evaluation ####################
        devel_itr = torch.utils.data.DataLoader(
            devel_dataset,
            batch_size=max_batch_size,
            shuffle=False,
            collate_fn=lambda x: x,
        )
        with torch.no_grad():
            # dev
            net.eval()
            pred_scores = []
            f1s = []
            with tqdm(devel_itr) as pbar:
                pbar.set_description("{:04}/Devel".format(trial_num, epoch))
                devel_data_pred = []
                losses = []
                for _, dat in enumerate(pbar):
                    dat_pred, adj = net(
                        dat,
                    )

                    devel_data_pred.extend(dat_pred)
                    loss = loss_func(dat, adj)
                    losses.append(loss)

            devel_loss = torch.stack(losses).mean()
            devel_min_loss = min(devel_loss.item(), devel_min_loss)

            devel_precision, devel_recall, devel_f1 = utils.calc_score(devel_data_pred)
            devel_max_f1 = max(devel_max_f1, devel_f1)

            writer.add_scalar("devel/loss", devel_loss, epoch)

            writer.add_scalar("devel/f1", devel_f1, epoch)
            writer.add_scalar("devel/recall", devel_recall, epoch)
            writer.add_scalar("devel/precision", devel_precision, epoch)
            writer.add_scalar("devel/f1_sub", devel_f1 - devel_rule_f1, epoch)
            writer.add_scalar("devel/recall_sub", devel_recall - devel_rule_recall, epoch)
            writer.add_scalar("devel/precision_sub", devel_precision - devel_rule_precision, epoch)

            # save ann
            files = utils.generate_ann(devel_data_pred, relation_classes)
            dpth = ann_save_path / "dev" / "{:04}".format(epoch)
            if not dpth.exists():
                os.makedirs(dpth)
            for basename, txt, ann in files:
                fname = os.path.join(ann_save_path, "{:04}".format(epoch), basename)
                (dpth / "{}.txt".format(basename)).write_text(txt)
                (dpth / "{}.ann".format(basename)).write_text(ann)

        # check scores
        if score_name == "train_loss":
            score = train_loss
        if "devel" in score_name:
            if "loss" in score_name:
                score = -devel_loss
            elif "f1" in score_name:
                score = devel_f1

        if epoch % 5 == 4:
            print("Saving checkpoint model (epoch {})...".format(epoch))
            torch.save(
                net.state_dict(),
                os.path.join(model_save_path, "checkpoint_{:04}.pth".format(epoch)),
            )
            print("Completed")

        # save best param
        is_best = False
        if score > max_score:
            max_score = score
            score_max_epoch = epoch
            torch.save(net.state_dict(), os.path.join(model_save_path, "best_model.pth"))
            is_best = True
        if epoch != epochs - 1:
            yield score

        if eval_test:
            with torch.no_grad():
                test_itr = torch.utils.data.DataLoader(
                    test_dataset,
                    batch_size=max_batch_size,
                    shuffle=False,
                    collate_fn=lambda x: x,
                )
                with tqdm(test_itr) as pbar:
                    pbar.set_description("{:04}/test".format(trial_num, epoch))
                    test_data_pred = []
                    losses = []
                    for _, dat in enumerate(pbar):
                        dat_pred, adj = net(
                            dat,
                        )

                        test_data_pred.extend(dat_pred)
                        loss = loss_func(dat, adj)
                        losses.append(loss)

                test_loss = torch.stack(losses).mean()

                test_precision, test_recall, test_f1 = utils.calc_score(test_data_pred)

                writer.add_scalar("test/loss", test_loss, epoch)

                writer.add_scalar("test/f1", test_f1, epoch)
                writer.add_scalar("test/recall", test_recall, epoch)
                writer.add_scalar("test/precision", test_precision, epoch)
                writer.add_scalar("test/f1_sub", test_f1 - test_rule_f1, epoch)
                writer.add_scalar("test/recall_sub", test_recall - test_rule_recall, epoch)
                writer.add_scalar("test/precision_sub", test_precision - test_rule_precision, epoch)

                # save ann
                files = utils.generate_ann(test_data_pred, relation_classes)
                dpth = ann_save_path / "test" / "{:04}".format(epoch)
                if not dpth.exists():
                    os.makedirs(dpth)
                for basename, txt, ann in files:
                    fname = os.path.join(ann_save_path, "{:04}".format(epoch), basename)
                    (dpth / "{}.txt".format(basename)).write_text(txt)
                    (dpth / "{}.ann".format(basename)).write_text(ann)

                if is_best:
                    best_test_score = test_f1

    valid_types = [str, int, float, bool]
    dic = dict([(k, v) for k, v in options.dict().items() if any([isinstance(v, t) for t in valid_types])])
    writer.add_hparams(
        hparam_dict=dic,
        metric_dict={
            "best_devel_f1": devel_max_f1,
            "best_devel_loss": devel_min_loss,
            "best_train_loss": train_min_loss,
            "best_epoch": score_max_epoch,
            "best_test_score": best_test_score,  # test score is recored when the score is best
        },
    )
    yield score


def train(args):
    return list(_train(args))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # settings for path
    parser.add_argument("--pkl_path", type=Path, default="preprocess/pkl/", help="The path to the preprocessed data")
    parser.add_argument("--output_path", type=Path, default="output/", help="The path for the output")
    parser.add_argument("--model_path", type=Path, help="Path to the weight to load")
    parser.add_argument("--job_tag", type=str, default="sample", help="The name of experiment")
    parser.add_argument("--config", default="olivetti.conf", type=str, help="The config file for corpus")

    # settings for training
    parser.add_argument("--epochs", type=int, default=1, help="The iteration for training")
    parser.add_argument("--batch_size", type=int, default=1, help="The batch size for learning")
    parser.add_argument("--max_batch_size", type=int, default=1, help="The batch size for development")
    parser.add_argument(
        "--device", type=str, default="cpu", help="The device for computing, e.g., cpu, cuda, cuda:0, and cuda:1"
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="Adam",
        choices=torch.optim.__dict__.keys(),
        help="Optimizer name in torch.optim",
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--eval_test",
        action="store_true",
        default=False,
        help="Toggle whether evaluation for test set is performed for all epochs",
    )
    parser.add_argument("--finetune_transformer", action="store_true", default=False, help="Finetune transformer model")
    parser.add_argument(
        "--finetune_transformer_start", type=int, default=0, help="An epoch to start finetuning for transformer"
    )
    parser.add_argument(
        "--score",
        choices=["train_loss", "devel_loss", "devel_f1"],
        default="devel_f1",
        help="The name of score to optimize",
    )

    # settings for model
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dim for all layers")
    parser.add_argument("--num_gcn", type=int, default=1, help="A number of the GCN layer")
    parser.add_argument("--bigcn", action="store_true", default=False, help="Switch GCN to bidirectional")
    parser.add_argument(
        "--gcn_layer", type=str, default="GCNConv", choices=torch_geometric.nn.__dict__.keys(), help="Type of GCN layer"
    )
    parser.add_argument(
        "--transformer",
        type=str,
        default="allenai/longformer-base-4096",
        help="The transformer model in haggingface/transformers",
    )
    parser.add_argument(
        "--feature",
        type=str,
        choices=["add", "bilinear", "mul"],
        default="bilinear",
        help="The way to calculate to obtain \\bar E in the paper",
    )
    parser.add_argument("--head_tail_layers", type=int, default=1, help="A number of layers of FC^h and FC^t")
    parser.add_argument("--fc_out_layer", type=int, default=1, help="A number of output fc layers (FC_out)")
    parser.add_argument("--dropout_out", type=float, default=0.0, help="Dropout ratio applied before FC_out")
    parser.add_argument(
        "--edge_emb_dim",
        type=int,
        default=0,
        help="Dimension of the embedding of edge class before editing, denoted as e^old in the paper",
    )
    parser.add_argument(
        "--dist_emb_max",
        type=int,
        default=1000,
        help="Maximum distance of the embedding for distance of entity pair, which is denoted as b in the paper",
    )
    parser.add_argument(
        "--dist_emb_dim",
        type=int,
        default=0,
        help="Dimension of the embedding for distance of entity pair, which is denoted as b in the paper",
    )
    parser.add_argument("--d_max", type=int, default=1000, help="d_max, the maximum distance to edit individually")

    parser.add_argument(
        "--from_empty",
        action="store_true",
        default=False,
        help="The input of the model will be empty graph, i.e. without rule-based model",
    )
    parser.add_argument(
        "--from_random", action="store_true", default=False, help="The input of the model will be random graph"
    )
    parser.add_argument(
        "--random_order",
        action="store_true",
        default=False,
        help="The order of editing will be random, and not following close-first manner",
    )

    # Other arguments
    parser.add_argument("--debug", action="store_true", default=False, help="Debug mode to stop training short")
    parser.add_argument("--comment", type=str, default="", help="Commet added to the tensorboard")

    # parameters for optuna
    parser.add_argument("--optuna", type=str, help="study_name of optuna")
    parser.add_argument("--optuna_storage", type=str, help="Storage for optuna")
    parser.add_argument("--optuna_n_trials", type=int, default=1, help="n_trials for optuna")

    args = parser.parse_args()

    if args.optuna:

        def objective(trial):
            args.lr = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
            args.num_gcn = trial.suggest_int("num_gcn", 0, 4)
            args.d_max = trial.suggest_int("d_max", 1, 10)
            args.hidden_dim = trial.suggest_int("hidden_dim", 32, 128, log=True)
            args.fc_out_layer = trial.suggest_int("fc_out_layer", 1, 5)
            args.dropout_out = trial.suggest_float("dropout_out", 0.0, 1.0)
            args.edge_emb_dim = trial.suggest_int("edge_emb_dim", 1, 32, log=True)
            args.dist_emb_dim = trial.suggest_int("dist_emb_dim", 1, 32, log=True)
            args.dist_emb_max = trial.suggest_int("dist_emb_max", 1, 100, log=True)
            args.bigcn = trial.suggest_categorical("bigcn", [True, False])

            max_score = -float("inf")
            for step, score in enumerate(_train(args)):
                max_score = max(max_score, score)
                trial.report(score, step)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            return max_score

        pruner = optuna.pruners.SuccessiveHalvingPruner(
            min_resource="auto", reduction_factor=4, min_early_stopping_rate=0
        )
        study = optuna.create_study(
            study_name=args.optuna,
            storage=args.optuna_storage,
            load_if_exists=True,
            direction="maximize",
            pruner=pruner,
        )
        study.optimize(objective, n_trials=args.optuna_n_trials)

        # if without SQL, output tsv
        if not args.optuna_storage:
            study.trials_dataframe().to_csv(args.optuna + ".tsv", sep="\t")

    else:
        train(args)