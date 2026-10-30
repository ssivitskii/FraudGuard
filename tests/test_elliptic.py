import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from fraudguard.elliptic import (
    NEIGHBOR_COLUMNS,
    STRUCTURE_COLUMNS,
    build_graph_features,
    load_elliptic,
)


def write_data(path):
    # Feature and label files intentionally have different row orders.
    rows = [
        [3, 2] + [30] * 93 + [300] * 72,
        [1, 1] + [10] * 93 + [100] * 72,
        [2, 1] + [20] * 93 + [200] * 72,
    ]
    pd.DataFrame(rows).to_csv(path / "elliptic_txs_features.csv", header=False, index=False)
    pd.DataFrame({"txId": [2, 3, 1], "class": ["unknown", "2", "1"]}).to_csv(
        path / "elliptic_txs_classes.csv", index=False
    )
    pd.DataFrame({"txId1": [1], "txId2": [2]}).to_csv(
        path / "elliptic_txs_edgelist.csv", index=False
    )


def test_load_and_graph_with_unknown_neighbor(tmp_path):
    write_data(tmp_path)
    frame, edges = load_elliptic(tmp_path)
    assert frame.txId.tolist() == [1, 2, 3]
    assert frame.is_illicit.tolist() == [1, -1, 0]
    assert frame.local_1.tolist() == [10, 20, 30]
    assert len(frame.columns) == 96
    graph = build_graph_features(frame, edges)
    assert graph.columns.tolist() == STRUCTURE_COLUMNS + NEIGHBOR_COLUMNS
    assert graph.loc[0, STRUCTURE_COLUMNS].tolist() == [0, 1]
    assert graph.loc[1, STRUCTURE_COLUMNS].tolist() == [1, 0]
    assert graph.loc[0, "outgoing_mean_local_1"] == 20
    assert graph.loc[1, "incoming_mean_local_93"] == 10
    assert graph.loc[2].eq(0).all()


def test_snapshot_independence_alignment_and_labels(tmp_path):
    write_data(tmp_path)
    frame, edges = load_elliptic(tmp_path)
    expected = build_graph_features(frame.iloc[:2], edges)
    changed = frame.copy()
    changed["is_illicit"] = [0, 1, -1]
    changed = changed.iloc[[2, 1, 0]]
    changed.index = [17, 19, 23]
    result = build_graph_features(changed, edges)
    assert_frame_equal(result.loc[[23, 19]].reset_index(drop=True), expected)


def test_mean_multiple_neighbors(tmp_path):
    write_data(tmp_path)
    frame, _ = load_elliptic(tmp_path)
    frame.loc[2, "time_step"] = 1
    edges = pd.DataFrame({"txId1": [1, 3], "txId2": [2, 2]})
    result = build_graph_features(frame, edges)
    assert result.loc[1, "in_degree"] == 2
    assert result.loc[1, "incoming_mean_local_1"] == 20


@pytest.mark.parametrize(
    "kind",
    [
        "schema",
        "duplicate_feature",
        "bad_feature",
        "bad_time",
        "missing_class",
        "duplicate_class",
        "bad_class",
        "noninteger_id",
        "unknown_endpoint",
        "duplicate_edge",
        "self_loop",
        "cross_step",
    ],
)
def test_reject_invalid_data(tmp_path, kind):
    write_data(tmp_path)
    feature_path = tmp_path / "elliptic_txs_features.csv"
    class_path = tmp_path / "elliptic_txs_classes.csv"
    edge_path = tmp_path / "elliptic_txs_edgelist.csv"
    features = pd.read_csv(feature_path, header=None)
    classes = pd.read_csv(class_path, dtype={"class": str})
    edges = pd.read_csv(edge_path)
    if kind == "schema":
        features = features.iloc[:, :-1]
    elif kind == "duplicate_feature":
        features.loc[1, 0] = features.loc[0, 0]
    elif kind == "bad_feature":
        features[10] = features[10].astype(float)
        features.loc[0, 10] = np.inf
    elif kind == "bad_time":
        features.loc[0, 1] = 0
    elif kind == "missing_class":
        classes = classes.iloc[:-1]
    elif kind == "duplicate_class":
        classes.loc[1, "txId"] = classes.loc[0, "txId"]
    elif kind == "bad_class":
        classes.loc[0, "class"] = "other"
    elif kind == "noninteger_id":
        features[0] = features[0].astype(float)
        features.loc[0, 0] = 1.5
    elif kind == "unknown_endpoint":
        edges.loc[0, "txId2"] = 99
    elif kind == "duplicate_edge":
        edges = pd.concat([edges, edges], ignore_index=True)
    elif kind == "self_loop":
        edges.loc[0, "txId2"] = 1
    elif kind == "cross_step":
        edges.loc[0, "txId2"] = 3
    features.to_csv(feature_path, header=False, index=False)
    classes.to_csv(class_path, index=False)
    edges.to_csv(edge_path, index=False)
    with pytest.raises(ValueError):
        load_elliptic(tmp_path)
