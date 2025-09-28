"""Clustering utilities for customer behavior profiling.

The module provides:

* Dimensionality reduction helpers (PCA, t-SNE, UMAP) to project high-dimensional
  behavioural features onto lower-dimensional manifolds.
* Multiple clustering algorithms (K-Means, DBSCAN, Agglomerative/Hierarchical)
  with convenience wrappers that return structured results and common metrics.

The implementations are lightweight wrappers around scikit-learn / umap-learn to
standardise configuration and outputs across the project.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

try:  # umap-learn is an optional dependency but included in requirements.txt
	import umap
except ImportError:  # pragma: no cover - optional during docs builds
	umap = None  # type: ignore


@dataclass
class DimensionalityReductionResult:
	embedding: pd.DataFrame
	transformer: Any
	explained_variance: Optional[np.ndarray] = None


@dataclass
class ClusteringResult:
	labels: np.ndarray
	model: Any
	silhouette: Optional[float] = None
	inertia: Optional[float] = None
	linkage_matrix: Optional[np.ndarray] = None


def _ensure_array(data: pd.DataFrame | np.ndarray) -> np.ndarray:
	if isinstance(data, pd.DataFrame):
		return data.values
	return np.asarray(data)


def run_pca(
	data: pd.DataFrame | np.ndarray,
	*,
	n_components: int = 2,
	scale: bool = True,
	random_state: int = 42,
) -> DimensionalityReductionResult:
	"""Project ``data`` to ``n_components`` dimensions using PCA."""

	array = _ensure_array(data)
	transformer = PCA(n_components=n_components, random_state=random_state)
	if scale:
		array = StandardScaler().fit_transform(array)
	embedding = transformer.fit_transform(array)
	embedding = np.asarray(embedding)
	columns = [f"pca_{i+1}" for i in range(embedding.shape[1])]
	return DimensionalityReductionResult(
		embedding=pd.DataFrame(embedding, columns=columns),
		transformer=transformer,
		explained_variance=transformer.explained_variance_ratio_,
	)


def run_tsne(
	data: pd.DataFrame | np.ndarray,
	*,
	n_components: int = 2,
	perplexity: float = 30.0,
	random_state: int = 42,
) -> DimensionalityReductionResult:
	"""Compute a t-SNE manifold embedding."""

	array = _ensure_array(data)
	transformer = TSNE(
		n_components=n_components,
		perplexity=min(perplexity, max(5, array.shape[0] - 1)),
		random_state=random_state,
		init="pca",
		learning_rate="auto",
	)
	embedding = transformer.fit_transform(array)
	embedding = np.asarray(embedding)
	columns = [f"tsne_{i+1}" for i in range(embedding.shape[1])]
	return DimensionalityReductionResult(
		embedding=pd.DataFrame(embedding, columns=columns),
		transformer=transformer,
	)


def run_umap(
	data: pd.DataFrame | np.ndarray,
	*,
	n_components: int = 2,
	random_state: int = 42,
	min_dist: float = 0.1,
	n_neighbors: int = 15,
) -> DimensionalityReductionResult:
	"""Generate a UMAP embedding; raises if umap-learn is unavailable."""

	if umap is None:  # pragma: no cover - dependency missing
		raise ImportError("umap-learn is not installed; install it to use UMAP embeddings.")

	array = _ensure_array(data)
	transformer = umap.UMAP(
		n_components=n_components,
		random_state=random_state,
		min_dist=min_dist,
		n_neighbors=min(n_neighbors, array.shape[0] - 1),
	)
	embedding = transformer.fit_transform(array)
	embedding = np.asarray(embedding)
	columns = [f"umap_{i+1}" for i in range(embedding.shape[1])]
	return DimensionalityReductionResult(
		embedding=pd.DataFrame(embedding, columns=columns),
		transformer=transformer,
	)


def kmeans_cluster(
	data: pd.DataFrame | np.ndarray,
	*,
	n_clusters: int = 3,
	random_state: int = 42,
	compute_silhouette: bool = True,
) -> ClusteringResult:
	"""Run K-Means clustering returning labels and inertia/silhouette metrics."""

	array = _ensure_array(data)
	model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
	labels = model.fit_predict(array)

	silhouette: Optional[float] = None
	if compute_silhouette and len(set(labels)) > 1:
		silhouette = silhouette_score(array, labels)

	return ClusteringResult(labels=labels, model=model, silhouette=silhouette, inertia=model.inertia_)


def dbscan_cluster(
	data: pd.DataFrame | np.ndarray,
	*,
	eps: float = 0.5,
	min_samples: int = 5,
) -> ClusteringResult:
	"""Run DBSCAN to detect dense regions and noise points."""

	array = _ensure_array(data)
	model = DBSCAN(eps=eps, min_samples=min_samples)
	labels = model.fit_predict(array)

	silhouette: Optional[float] = None
	unique_labels = set(labels)
	if -1 in unique_labels:
		unique_labels.remove(-1)
	if unique_labels:
		mask = labels != -1
		if mask.any() and len(set(labels[mask])) > 1:
			silhouette = silhouette_score(array[mask], labels[mask])

	return ClusteringResult(labels=labels, model=model, silhouette=silhouette)


def hierarchical_cluster(
	data: pd.DataFrame | np.ndarray,
	*,
	n_clusters: int = 3,
	metric: str = "euclidean",
	linkage_method: str = "ward",
) -> ClusteringResult:
	"""Perform agglomerative clustering and return cluster labels with dendrogram data."""

	array = _ensure_array(data)
	if linkage_method == "ward":
		model = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
	else:
		model = AgglomerativeClustering(
			n_clusters=n_clusters, metric=metric, linkage=linkage_method
		)
	labels = model.fit_predict(array)

	# Compute linkage matrix for dendrogram visualisation.
	linkage_metric = metric if linkage_method != "ward" else "euclidean"
	linkage_matrix = linkage(array, method=linkage_method, metric=linkage_metric)

	silhouette: Optional[float] = None
	if len(set(labels)) > 1:
		silhouette = silhouette_score(array, labels)

	return ClusteringResult(
		labels=labels,
		model=model,
		silhouette=silhouette,
		linkage_matrix=linkage_matrix,
	)


__all__ = [
	"DimensionalityReductionResult",
	"ClusteringResult",
	"run_pca",
	"run_tsne",
	"run_umap",
	"kmeans_cluster",
	"dbscan_cluster",
	"hierarchical_cluster",
]
