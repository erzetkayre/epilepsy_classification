"""Experiment runner for the FORTEI-ICEE 2026 revision.

Every methodological choice that was wrong in the original study is a flag here,
so the ablation can report what each one is worth in accuracy points.

Protocol (corrected defaults):
  * StratifiedGroupKFold, grouped on source recording segment, so augmented
    copies and windows of the same segment never straddle the split.
  * Augmentation applied inside the training fold only, after the split.
  * Each training fold is further split into train/validation. Validation drives
    early stopping and the decision threshold. The test fold is touched once.
  * Early stopping on validation loss with weight restoration, not a fixed
    20 epochs halted by an accuracy trigger.
  * Multiple seeds; mean +/- std reported.
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)
from tensorflow import keras

import prep
from models import BUILDERS

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)


def make_folds(x, y, groups, n_splits, grouped, seed):
    """Grouped CV keeps same-segment samples together. Ungrouped reproduces the
    original bug, where an augmented copy can sit opposite its original."""
    if grouped:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(splitter.split(x, y, groups))
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(splitter.split(x, y))


def balance_by_augment(x, y, groups, rng):
    """Augment minority classes up to the majority count. Called on training data
    only. New samples inherit the group id of the segment they came from."""
    counts = np.bincount(y)
    target = counts.max()
    xs, ys, gs = [x], [y], [groups]
    for cls in range(len(counts)):
        need = target - counts[cls]
        if need <= 0:
            continue
        idx = np.where(y == cls)[0]
        pick = rng.choice(idx, size=need, replace=need > len(idx))
        xs.append(prep.augment(x[pick, :, 0], rng)[..., None].astype(np.float32))
        ys.append(np.full(need, cls, dtype=y.dtype))
        gs.append(groups[pick])
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(gs)


def pick_threshold(y_true, prob):
    """Choose the operating point on validation data by Youden's J. The original
    study left this at 0.5, which cost it recall on the epileptic class."""
    best, best_j = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 91):
        pred = (prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if tp + fn else 0.0
        spec = tn / (tn + fp) if tn + fp else 0.0
        if sens + spec - 1 > best_j:
            best_j, best = sens + spec - 1, t
    return best


def evaluate(y_true, prob, n_classes, thr):
    if n_classes == 2:
        pred = (prob >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        return {
            "accuracy": accuracy_score(y_true, pred),
            "precision": precision_score(y_true, pred, zero_division=0),
            "sensitivity": tp / (tp + fn) if tp + fn else 0.0,
            "specificity": tn / (tn + fp) if tn + fp else 0.0,
            "f1": f1_score(y_true, pred, zero_division=0),
            "auc": roc_auc_score(y_true, prob),
            "threshold": thr,
        }
    pred = prob.argmax(axis=1)
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, average="macro", zero_division=0),
        "sensitivity": recall_score(y_true, pred, average="macro", zero_division=0),
        "specificity": np.nan,
        "f1": f1_score(y_true, pred, average="macro", zero_division=0),
        "auc": roc_auc_score(y_true, prob, multi_class="ovr", average="macro"),
        "threshold": np.nan,
    }


def run(cfg):
    n_classes = 2 if cfg["task"] == "binary" else 3
    data = prep.build(task=cfg["task"], filt=cfg["filt"], decim=cfg["decim"],
                      win=cfg["win"], norm=cfg["norm"])
    x, y, groups = data["x"], data["y"], data["groups"]

    rows = []
    best_acc = -1.0
    best_path = os.path.join(OUT, "%s_best.keras" % cfg["tag"])
    for seed in cfg["seeds"]:
        xs, ys, gs = x, y, groups
        if cfg["aug"] == "presplit":
            # Reproduces the original defect: copies enter the pool before the split.
            xs, ys, gs = balance_by_augment(x, y, groups, np.random.default_rng(seed))

        for fold, (tr, te) in enumerate(make_folds(xs, ys, gs, cfg["folds"],
                                                   cfg["grouped"], seed), 1):
            if cfg.get("fold_only") and fold != cfg["fold_only"]:
                continue
            keras.utils.set_random_seed(seed * 100 + fold)
            rng = np.random.default_rng(seed * 1000 + fold)

            x_tr, y_tr, g_tr = xs[tr], ys[tr], gs[tr]
            x_te, y_te = xs[te], ys[te]

            # inner split for early stopping and threshold selection
            inner = make_folds(x_tr, y_tr, g_tr, 5, cfg["grouped"], seed)[0]
            itr, iva = inner
            x_va, y_va = x_tr[iva], y_tr[iva]
            x_fit, y_fit, g_fit = x_tr[itr], y_tr[itr], g_tr[itr]

            if cfg["aug"] == "infold":
                x_fit, y_fit, g_fit = balance_by_augment(x_fit, y_fit, g_fit, rng)

            model = BUILDERS[cfg["model"]](x.shape[1:], n_classes,
                                           **({"head": cfg["head"]} if cfg["model"] == "cnn" else {}))
            model.compile(
                optimizer=keras.optimizers.Adam(cfg["lr"], clipnorm=cfg.get("clipnorm") or None),
                loss="binary_crossentropy" if n_classes == 2 else "sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )

            callbacks = []
            if cfg["early_stop"]:
                callbacks.append(keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=cfg["patience"],
                    restore_best_weights=True, verbose=0))

            t0 = time.time()
            hist = model.fit(x_fit, y_fit, validation_data=(x_va, y_va),
                             epochs=cfg["epochs"], batch_size=cfg["batch"],
                             callbacks=callbacks, verbose=0)
            train_time = time.time() - t0

            if cfg.get("dump_history"):
                hist_path = os.path.join(OUT, "%s_seed%d_fold%d_history.json" % (cfg["tag"], seed, fold))
                with open(hist_path, "w") as f:
                    json.dump({k: [float(v) for v in vs] for k, vs in hist.history.items()}, f, indent=2)

            if n_classes == 2:
                pv = model.predict(x_va, verbose=0).ravel()
                thr = pick_threshold(y_va, pv) if cfg["tune_thr"] else 0.5
                pt = model.predict(x_te, verbose=0).ravel()
            else:
                thr = np.nan
                pt = model.predict(x_te, verbose=0)

            m = evaluate(y_te, pt, n_classes, thr)
            m.update(seed=seed, fold=fold, train_time=train_time,
                     epochs=len(hist.history["loss"]),
                     train_acc=hist.history["accuracy"][-1],
                     params=model.count_params())
            rows.append(m)
            print("  seed %d fold %2d  acc %.4f  auc %.4f  ep %3d  %5.1fs"
                  % (seed, fold, m["accuracy"], m["auc"], m["epochs"], train_time), flush=True)

            if cfg["save_best"] and m["accuracy"] > best_acc:
                best_acc = m["accuracy"]
                model.save(best_path)
                with open(os.path.join(OUT, "%s_best_meta.json" % cfg["tag"]), "w") as f:
                    json.dump({"seed": seed, "fold": fold, "accuracy": m["accuracy"],
                               "threshold": m["threshold"]}, f, indent=2, default=str)

            keras.backend.clear_session()

    df = pd.DataFrame(rows)
    tag = cfg["tag"]
    df.to_csv(os.path.join(OUT, "%s_folds.csv" % tag), index=False)
    with open(os.path.join(OUT, "%s_config.json" % tag), "w") as f:
        json.dump(cfg, f, indent=2, default=str)
    print("%s -> acc %.4f +/- %.4f | auc %.4f | sens %.4f | spec %.4f"
          % (tag, df.accuracy.mean(), df.accuracy.std(), df.auc.mean(),
             df.sensitivity.mean(), df.specificity.mean()), flush=True)
    return df


DEFAULTS = dict(
    task="binary", model="cnn", seeds=[0, 1, 2], folds=10,
    filt=True, decim=2, win=None, norm=True,
    grouped=True, aug="infold", tune_thr=True,
    early_stop=True, patience=20, epochs=100, batch=32, lr=1e-3,
    head="gap", tag="run", save_best=False,
    clipnorm=1.0, dump_history=False, fold_only=None,
)


def parse():
    p = argparse.ArgumentParser()
    for k, v in DEFAULTS.items():
        if k == "seeds":
            p.add_argument("--seeds", type=int, nargs="+", default=v)
        elif k == "clipnorm":
            p.add_argument("--clipnorm", type=float, default=v)
        elif isinstance(v, bool):
            p.add_argument("--" + k, type=lambda s: s.lower() in ("1", "true", "yes"), default=v)
        elif v is None:
            p.add_argument("--" + k, type=int, default=None)
        else:
            p.add_argument("--" + k, type=type(v), default=v)
    return vars(p.parse_args())


if __name__ == "__main__":
    cfg = parse()
    print("CONFIG:", cfg, flush=True)
    run(cfg)
