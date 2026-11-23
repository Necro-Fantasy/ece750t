"""Shared data loading, splitting, preprocessing, and training code.

Group coding used everywhere: group = 0 for male, 1 for female
(ACS SEX: 1 = male, 2 = female). SEX is NOT used as a model input;
it is kept only for group evaluation.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch

NUM_COLS = ["AGEP", "WKHP"]  # age, usual hours worked per week
CAT_COLS = ["COW", "SCHL", "MAR", "RELP", "RAC1P", "OCCP_GROUP", "POBP_GROUP"]


def load_acs_income(root="data/acs"):
    """Return (features DataFrame without SEX, labels y, group) for CA 2018."""
    from folktables import ACSDataSource, ACSIncome
    Path(root).mkdir(parents=True, exist_ok=True)
    source = ACSDataSource(survey_year="2018", horizon="1-Year", survey="person", root_dir=str(root))
    raw = source.get_data(states=["CA"], download=True)
    features, label, _ = ACSIncome.df_to_pandas(raw)
    y = label.iloc[:, 0].astype(int).to_numpy()
    group = (features["SEX"].astype(int).to_numpy() == 2).astype(int)
    df = features.drop(columns=["SEX"]).copy()
    # Occupation (about 500 codes) and place of birth (about 200 codes) are
    # merged into coarse groups by their leading digits to keep the input small.
    df["OCCP_GROUP"] = (df["OCCP"].fillna(-100) // 100).astype(int)
    df["POBP_GROUP"] = (df["POBP"].fillna(-100) // 100).astype(int)
    df = df.drop(columns=["OCCP", "POBP"])
    return df, y, group


def stratified_split(y, group, seed, test_frac=0.2):
    """Split row indices, keeping the label x group proportions in both parts."""
    rng = np.random.default_rng(seed)
    strata = 2 * np.asarray(y) + np.asarray(group)
    is_test = np.zeros(len(strata), dtype=bool)
    for s in np.unique(strata):
        idx = np.flatnonzero(strata == s)
        rng.shuffle(idx)
        is_test[idx[: int(round(len(idx) * test_frac))]] = True
    return np.flatnonzero(~is_test), np.flatnonzero(is_test)


def preprocess(df, fit_idx):
    """Standardize numeric columns and one-hot encode categories.

    All statistics and category lists come from the rows in fit_idx only,
    so no information from validation or test rows leaks into the inputs.
    A category that does not appear in fit_idx becomes an all-zero row.
    """
    fit = df.iloc[fit_idx]
    parts, names = [], []
    for c in NUM_COLS:
        mu, sd = float(fit[c].mean()), float(fit[c].std())
        parts.append(((df[c].to_numpy(dtype=float) - mu) / (sd if sd > 0 else 1.0))[:, None])
        names.append(c)
    for c in CAT_COLS:
        values = df[c].fillna(-1).astype(int).to_numpy()
        cats = np.sort(np.unique(values[fit_idx]))
        parts.append((values[:, None] == cats[None, :]).astype(np.float32))
        names += [f"{c}={v}" for v in cats]
    return np.hstack(parts).astype(np.float32), names


def make_mlp(n_features, hidden=64):
    return torch.nn.Sequential(
        torch.nn.Linear(n_features, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, 1))


def train_mlp(X, y, init_seed, order_seed, boot_seed=None, epochs=20,
              batch_size=256, lr=1e-3, hidden=64, on_epoch=None):
    """Train the one-hidden-layer MLP with Adam.

    Each source of randomness has its own seed:
      init_seed  -> initial weights (global torch RNG, used only at init)
      order_seed -> minibatch order (its own torch.Generator)
      boot_seed  -> bootstrap resample of the training rows; None = no resampling
    on_epoch(epoch, model, mean_loss) is called after every epoch if given.
    """
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)

    if boot_seed is not None:
        rows = np.random.default_rng(boot_seed).integers(0, len(y), len(y))
        X, y = X[rows], y[rows]
    Xt = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
    yt = torch.from_numpy(np.asarray(y, dtype=np.float32))

    torch.manual_seed(init_seed)
    model = make_mlp(X.shape[1], hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    order_gen = torch.Generator().manual_seed(order_seed)

    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for idx in torch.randperm(len(yt), generator=order_gen).split(batch_size):
            logits = model(Xt[idx]).squeeze(1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, yt[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(idx)
        if on_epoch is not None:
            on_epoch(epoch, model, total / len(yt))
    return model


def predict_proba(model, X, chunk=65536):
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(X), chunk):
            xb = torch.from_numpy(np.ascontiguousarray(X[start:start + chunk], dtype=np.float32))
            out.append(torch.sigmoid(model(xb).squeeze(1)).numpy())
    return np.concatenate(out)
