# adapted from squidpy.datasets._dataset.pyi
from typing import Any, Protocol, Union

from anndata import AnnData

from squidpy.datasets._utils import PathLike


class Dataset(Protocol):
    def __call__(self, path: PathLike | None = ..., **kwargs: Any) -> AnnData: ...


mibitof_breast_cancer: Dataset
visium_heart: Dataset
stereoseq_axolotl_development: Dataset
stereoseq_axolotl_regeneration: Dataset
stereoseq_axolotl_subset: Dataset
