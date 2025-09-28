import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_blobs

from src.clustering import (
    ClusteringResult,
    DimensionalityReductionResult,
    dbscan_cluster,
    hierarchical_cluster,
    kmeans_cluster,
    run_pca,
    run_tsne,
    run_umap,
)


def test_dimensionality_reduction_shapes():
    X, _ = make_blobs(n_samples=40, centers=3, n_features=5, random_state=42)

    pca_result = run_pca(X, n_components=3)
    assert isinstance(pca_result, DimensionalityReductionResult)
    assert pca_result.embedding.shape == (40, 3)
    assert not pca_result.embedding.isna().any().any()

    tsne_result = run_tsne(X[:25], n_components=2, perplexity=10)
    assert tsne_result.embedding.shape == (25, 2)

    pytest.importorskip("umap")
    umap_result = run_umap(X, n_components=2, n_neighbors=10)
    assert umap_result.embedding.shape == (40, 2)


def test_clustering_algorithms():
    X, _ = make_blobs(n_samples=60, centers=3, n_features=4, cluster_std=0.60, random_state=21)
    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(4)])

    kmeans_result = kmeans_cluster(X_df, n_clusters=3)
    assert isinstance(kmeans_result, ClusteringResult)
    assert len(np.unique(kmeans_result.labels)) == 3
    assert kmeans_result.inertia is not None

    dbscan_result = dbscan_cluster(X_df, eps=0.7, min_samples=3)
    assert isinstance(dbscan_result, ClusteringResult)
    labels = dbscan_result.labels
    unique_labels = set(labels)
    assert -1 in unique_labels or len(unique_labels) > 1

    hierarchical_result = hierarchical_cluster(X_df, n_clusters=3)
    assert isinstance(hierarchical_result, ClusteringResult)
    assert hierarchical_result.linkage_matrix is not None
    assert hierarchical_result.linkage_matrix.shape[0] == len(X_df) - 1
    assert len(np.unique(hierarchical_result.labels)) == 3