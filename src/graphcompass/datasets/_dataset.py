# adapted from squidpy.datasets._dataset.py
from __future__ import annotations
from copy import copy

from squidpy.datasets._utils import AMetadata

_mibitof_breast_cancer = AMetadata(
    name="mibitof_breast_cancer",
    doc_header="MIBI-TOF breast cancer dataset fro `Risom et al. <https://doi.org/10.17632/d87vg86zd8.3>`",
    url="https://figshare.com/ndownloader/files/44696053",
    shape=(69672, 59),
)

_visium_heart = AMetadata(
    name="visium_heart",
    doc_header="Visium Myocardial tissue (heart) data from `Kuppe et al. <https://doi.org/10.5281/zenodo.6578047>`",
    url="https://figshare.com/ndownloader/files/44715052",
    shape=(88456, 11669),
)

_stereoseq_axolotl_development = AMetadata(
    name="stereoseq_axolotl_development",
    doc_header="Stereo-seq Axolotl Brain Development dataset from `Wei et al. <https://db.cngb.org/stomics/artista/>`",
    url="https://figshare.com/ndownloader/files/44714629",
    shape=(36198, 12704),
)

_stereoseq_axolotl_regeneration = AMetadata(
    name="stereoseq_axolotl_regeneration",
    doc_header="Stereo-seq Axolotl Brain Regeneration dataset from `Wei et al. <https://db.cngb.org/stomics/artista/>`",
    url="https://figshare.com/ndownloader/files/44715166",
    shape=(182142, 16176),
)

_stereoseq_axolotl_subset = AMetadata(
    name="stereoseq_axolotl_subset",
    doc_header="Stereo-seq Axolotl Brain subset (30DPI, 60DPI, Adult) dataset from `Wei et al. <https://db.cngb.org/stomics/artista/>`",
    url="https://figshare.com/ndownloader/files/44714335",
    shape=(28459, 18611),
)

for name, var in copy(locals()).items():
    if isinstance(var, AMetadata):
        var._create_function(name, globals())

__all__ = [  # noqa: F822
    "mibitof_breast_cancer",
    "visium_heart",
    "stereoseq_axolotl_development",
    "stereoseq_axolotl_regeneration",
    "stereoseq_axolotl_subset",
]
