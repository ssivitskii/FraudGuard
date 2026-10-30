"""Elliptic transaction data and completed-snapshot graph features."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix

LOCAL_COLUMNS = [f"local_{i}" for i in range(1, 94)]
STRUCTURE_COLUMNS = ["in_degree", "out_degree"]
NEIGHBOR_COLUMNS = [
    f"{direction}_mean_local_{i}" for direction in ("incoming", "outgoing") for i in range(1, 94)
]


def _ids(values: pd.Series) -> pd.Series:
    ids = pd.to_numeric(values, errors="coerce")
    if ids.isna().any() or not np.isfinite(ids).all() or (ids % 1 != 0).any():
        raise ValueError("transaction IDs must be finite integers")
    return ids


def _validate_edges(frame: pd.DataFrame, edges: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    if frame["txId"].duplicated().any():
        raise ValueError("transaction IDs must be unique")
    if edges.duplicated(["txId1", "txId2"]).any():
        raise ValueError("duplicate edges")
    if (edges["txId1"] == edges["txId2"]).any():
        raise ValueError("self-loop edges")
    ids = pd.Index(frame["txId"])
    source = ids.get_indexer(edges["txId1"])
    target = ids.get_indexer(edges["txId2"])
    if (source < 0).any() or (target < 0).any():
        raise ValueError("edge endpoint is missing")
    times = frame["time_step"].to_numpy()
    if (times[source] != times[target]).any():
        raise ValueError("cross-step edges are not supported")
    return source, target


def load_elliptic(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the original three CSVs, preserving unknown labels as -1."""
    raw = pd.read_csv(data_dir / "elliptic_txs_features.csv", header=None)
    if raw.shape[1] != 167:
        raise ValueError("expected txId and 166 features")
    raw.columns = ["txId", "time_step"] + LOCAL_COLUMNS + [f"aggregate_{i}" for i in range(1, 73)]
    raw["txId"] = _ids(raw["txId"])
    if raw["txId"].duplicated().any():
        raise ValueError("duplicate transaction IDs")
    numeric = raw.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("features must be finite numbers")
    times = numeric["time_step"]
    if (times <= 0).any() or (times % 1 != 0).any():
        raise ValueError("time steps must be positive integers")
    raw.iloc[:, 1:] = numeric
    classes = pd.read_csv(data_dir / "elliptic_txs_classes.csv", dtype={"class": str})
    if set(classes.columns) != {"txId", "class"}:
        raise ValueError("invalid class schema")
    classes["txId"] = _ids(classes["txId"])
    if classes["txId"].duplicated().any() or not classes["class"].isin(["1", "2", "unknown"]).all():
        raise ValueError("invalid class IDs or labels")
    if len(classes) != len(raw) or not classes["txId"].isin(raw["txId"]).all():
        raise ValueError("class and feature IDs must match")
    classes["is_illicit"] = classes["class"].map({"1": 1, "2": 0, "unknown": -1})
    frame = raw[["txId", "time_step"] + LOCAL_COLUMNS].merge(
        classes[["txId", "is_illicit"]], on="txId", validate="one_to_one"
    )
    frame = (
        frame[["txId", "time_step", "is_illicit"] + LOCAL_COLUMNS]
        .sort_values(["time_step", "txId"], kind="stable")
        .reset_index(drop=True)
    )
    edges = pd.read_csv(data_dir / "elliptic_txs_edgelist.csv")
    if set(edges.columns) != {"txId1", "txId2"}:
        raise ValueError("invalid edge schema")
    for col in ("txId1", "txId2"):
        edges[col] = _ids(edges[col])
    _validate_edges(frame, edges)
    return frame, edges


def build_graph_features(frame: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    """Use all nodes in each completed snapshot, without accessing labels."""
    source, target = _validate_edges(frame, edges)
    size = len(frame)
    adjacency = coo_matrix((np.ones(len(source)), (source, target)), shape=(size, size)).tocsr()
    local = frame[LOCAL_COLUMNS].to_numpy(dtype=float)
    outgoing_degree = np.asarray(adjacency.sum(axis=1)).ravel()
    incoming_degree = np.asarray(adjacency.sum(axis=0)).ravel()
    incoming_sum = adjacency.T @ local
    outgoing_sum = adjacency @ local
    incoming_mean = np.divide(
        incoming_sum,
        incoming_degree[:, None],
        out=np.zeros_like(incoming_sum),
        where=incoming_degree[:, None] != 0,
    )
    outgoing_mean = np.divide(
        outgoing_sum,
        outgoing_degree[:, None],
        out=np.zeros_like(outgoing_sum),
        where=outgoing_degree[:, None] != 0,
    )
    values = np.column_stack([incoming_degree, outgoing_degree, incoming_mean, outgoing_mean])
    return pd.DataFrame(values, index=frame.index, columns=STRUCTURE_COLUMNS + NEIGHBOR_COLUMNS)
