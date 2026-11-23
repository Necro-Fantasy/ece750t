"""Step 1: environment, seed separation, and ACSIncome download check.

Run from the project folder:
    python step1_check.py

It does three things and writes step1_report.json:
  A. Prints library versions.
  B. On a small synthetic dataset, checks that the three sources of
     randomness are controlled separately:
       - same (init, order, bootstrap) seeds twice  -> identical predictions
       - change only init_seed                      -> predictions change
       - change only order_seed                     -> predictions change
       - change only boot_seed                      -> predictions change
     and that logistic regression (deterministic solver) gives the same
     predictions when only the row order changes.
  C. Downloads ACSIncome for California 2018 (about a few hundred MB,
     first run only) and prints its size, label rate, and sex counts.
"""
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

REPORT = {}


def train_mlp(X, y, init_seed, order_seed, boot_seed=None,
              epochs=5, batch_size=256, lr=1e-3, hidden=64):
    """Train a one-hidden-layer MLP with Adam.

    Each source of randomness has its own seed:
      init_seed  -> initial weights (global torch RNG, used only at init)
      order_seed -> minibatch order (its own torch.Generator)
      boot_seed  -> bootstrap resample of the training rows (numpy);
                    None means use the training set as is
    """
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)

    if boot_seed is not None:
        rows = np.random.default_rng(boot_seed).integers(0, len(y), len(y))
        X, y = X[rows], y[rows]

    Xt = torch.from_numpy(X.astype(np.float32))
    yt = torch.from_numpy(y.astype(np.float32))

    torch.manual_seed(init_seed)  # only affects the weight initialization below
    model = torch.nn.Sequential(
        torch.nn.Linear(X.shape[1], hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, 1))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    order_gen = torch.Generator().manual_seed(order_seed)

    for _ in range(epochs):
        model.train()
        for idx in torch.randperm(len(yt), generator=order_gen).split(batch_size):
            logits = model(Xt[idx]).squeeze(1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, yt[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model


def predict_proba(model, X):
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.from_numpy(X.astype(np.float32))).squeeze(1)).numpy()


def part_a_versions():
    versions = dict(python=platform.python_version(), numpy=np.__version__,
                    pandas=pd.__version__, sklearn=sklearn.__version__,
                    torch=torch.__version__, platform=platform.platform())
    try:
        import folktables
        versions["folktables"] = getattr(folktables, "__version__", "installed")
    except ImportError:
        versions["folktables"] = "MISSING"
    print("A. Versions:", versions)
    REPORT["versions"] = versions


def part_b_seed_checks():
    X, y = make_classification(n_samples=5000, n_features=20, n_informative=8,
                               flip_y=0.1, random_state=0)
    X = (X - X.mean(0)) / X.std(0)
    X_train, y_train, X_test = X[:4000], y[:4000], X[4000:]

    base = predict_proba(train_mlp(X_train, y_train, 1, 1, 1), X_test)
    cases = {
        "same seeds again": dict(init_seed=1, order_seed=1, boot_seed=1),
        "only init_seed changed": dict(init_seed=2, order_seed=1, boot_seed=1),
        "only order_seed changed": dict(init_seed=1, order_seed=2, boot_seed=1),
        "only boot_seed changed": dict(init_seed=1, order_seed=1, boot_seed=2),
    }
    results = {}
    for name, seeds in cases.items():
        p = predict_proba(train_mlp(X_train, y_train, **seeds), X_test)
        results[name] = dict(
            max_prob_diff=float(np.abs(p - base).max()),
            label_flips=int(((p >= 0.5) != (base >= 0.5)).sum()))

    # Logistic regression: deterministic solver, only the row order changes.
    lr_a = LogisticRegression(max_iter=2000).fit(X_train, y_train)
    perm = np.random.default_rng(7).permutation(len(y_train))
    lr_b = LogisticRegression(max_iter=2000).fit(X_train[perm], y_train[perm])
    pa, pb = lr_a.predict_proba(X_test)[:, 1], lr_b.predict_proba(X_test)[:, 1]
    results["logistic regression, row order changed"] = dict(
        max_prob_diff=float(np.abs(pa - pb).max()),
        label_flips=int(((pa >= 0.5) != (pb >= 0.5)).sum()))

    checks = {
        "same seeds give identical predictions": results["same seeds again"]["max_prob_diff"] == 0.0,
        "init seed changes predictions": results["only init_seed changed"]["max_prob_diff"] > 0,
        "order seed changes predictions": results["only order_seed changed"]["max_prob_diff"] > 0,
        "bootstrap seed changes predictions": results["only boot_seed changed"]["max_prob_diff"] > 0,
        "logistic regression has no label flips": results["logistic regression, row order changed"]["label_flips"] == 0,
    }
    print("\nB. Seed checks (1000 synthetic test points):")
    for name, r in results.items():
        print(f"   {name:42s} max prob diff = {r['max_prob_diff']:.2e}, label flips = {r['label_flips']}")
    for name, ok in checks.items():
        print(f"   [{'PASS' if ok else 'FAIL'}] {name}")
    REPORT["seed_checks"] = dict(results=results, checks=checks)


def part_c_acs_download():
    from folktables import ACSDataSource, ACSIncome
    root = Path("data") / "acs"
    root.mkdir(parents=True, exist_ok=True)
    print("\nC. Downloading ACSIncome (California, 2018). The first run can take several minutes...")
    t0 = time.perf_counter()
    source = ACSDataSource(survey_year="2018", horizon="1-Year", survey="person", root_dir=str(root))
    raw = source.get_data(states=["CA"], download=True)
    features, label, _ = ACSIncome.df_to_pandas(raw)
    seconds = time.perf_counter() - t0
    y = label.iloc[:, 0].astype(int).to_numpy()
    sex = features["SEX"].astype(int).to_numpy()  # ACS coding: 1 = male, 2 = female
    summary = dict(
        rows=int(len(y)), feature_columns=list(features.columns),
        positive_rate=float(y.mean()),
        male=int((sex == 1).sum()), female=int((sex == 2).sum()),
        positive_rate_male=float(y[sex == 1].mean()),
        positive_rate_female=float(y[sex == 2].mean()),
        load_seconds=round(seconds, 1))
    for k, v in summary.items():
        print(f"   {k}: {v}")
    REPORT["acs_income_ca_2018"] = summary


if __name__ == "__main__":
    part_a_versions()
    part_b_seed_checks()
    try:
        part_c_acs_download()
    except Exception as exc:  # report the problem instead of stopping silently
        print(f"\nC. FAILED: {type(exc).__name__}: {exc}")
        REPORT["acs_income_ca_2018"] = dict(error=f"{type(exc).__name__}: {exc}")
    Path("step1_report.json").write_text(json.dumps(REPORT, indent=2))
    print("\nSaved step1_report.json")
