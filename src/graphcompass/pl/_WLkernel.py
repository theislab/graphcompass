import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path
from anndata import AnnData
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from graphcompass.tl._distance import compare_groups
from typing import Any, Sequence, Tuple, Union


def compare_conditions(
    adata: AnnData,
    library_key="sample_id",
    condition_key="status",
    control_group="normal",
    metric_key="wasserstein_distance", # kernel_matrix or wasserstein_distance
    method="wl_kernel",
    fig: Union[Figure, None] = None,
    ax: Union[Axes, Sequence[Axes], None] = None,
    return_ax: bool = False,
    figsize: Union[Tuple[float, float], None] = (20,10),
    dpi: Union[int, None] = 300,
    color: Union[str, list] = "grey",
    palette: str = "Set2",
    add_sign: bool = False,
    save: Union[str, Path, None] = None,
    **kwargs: Any,
) -> Union[Axes, Sequence[Axes], None]:
    """
    Plot group comparison for full samples.

    Parameters
    ----------
    adata
        Annotated data matrix.
    library_key
        Key in `adata.obs` where the library information is stored.
    condition_key
        Key in `adata.obs` where the condition information is stored.
    control_group
        Name of the control group.
    metric_key
        Key in `adata.uns` where the metric of interest is stored.
    method 
        Method used to calculate the comparison, also a parent key for metric_key in `adata.uns`
    fig
        Figure object to be used for plotting.
    ax
        Axes object to be used for plotting.
    return_ax
        If True, then return the axes object.
    figsize
        Figure size.
    dpi
        Figure resolution.
    color
        Color(s) for the bars in the plot (monocolor).
    palette
        Palette for the bar plot (multicolor).
    add_sign
        Significance between pairs of contrasts.
    save
        Filename under which to save the plot.
    **kwargs
        Keyword arguments to be passed to plotting functions.
    """
    pairwise_similarities = adata.uns[method][metric_key]
    
    dict_sample_to_status = {}
    for sample in adata.obs[library_key].unique():
        dict_sample_to_status[sample] = adata[adata.obs[library_key] == sample].obs[condition_key].values.unique()[0]

    df_for_plot = None
    
    sample_ids = dict_sample_to_status.keys()
    status = dict_sample_to_status.values()

    sample_to_status = pd.DataFrame({"sample_id": sample_ids, "contrast": status})
    disease_status = list(set(sample_to_status.contrast))
    disease_status.remove(control_group)
    contrasts = [(control_group, c) for c in disease_status]
    
    df_for_plot = compare_groups(
        pairwise_similarities=pairwise_similarities,
        sample_to_contrasts=sample_to_status,
        contrasts=contrasts,
        output_format="tidy"
    )
    
    if metric_key == "kernel_matrix":
        xlabel = "Kernel matrix values"
    elif metric_key == "wasserstein_distance":
        xlabel = "Wasserstein distance"
    else:
        ValueError(
            "Parameter 'metric_key' must be of type either kernel_matrix or wasserstein_distance."
        )
        
    # plot
    plt.rcParams["font.size"] = 12 
    contrasts = df_for_plot["contrast"].unique()
    num_contrasts = len(contrasts)
    if color:
        edgecolor = color
    else:
        edgecolor = sns.color_palette(palette)[:num_contrasts]

    plt.figure(figsize=figsize, dpi=dpi)
    
    one_sample_per_contrast = num_contrasts == len(df_for_plot)
    if one_sample_per_contrast:
        xlabel = xlabel
        ylabel = ""
        sns.barplot(
                data=df_for_plot,
                y="contrast",
                x="vals",
                facecolor='none', edgecolor=edgecolor,
                linewidth=3,
                color=color,
                palette=palette,
            )
    else:
        ylabel = xlabel
        xlabel = ""
        ax = sns.boxplot(
                data=df_for_plot,
                x="contrast",
                y="vals",
                color='white', width=.5,
            )
        if add_sign:
            pairs = []
            # defining contrast pairs
            for i in range(len(contrasts)):
                for j in range(i+1, len(contrasts)):
                    # Create a tuple and append it to the list
                    pairs.append((contrasts[i], contrasts[j]))

            from statannot import add_stat_annotation
            add_stat_annotation(data=df_for_plot, x="contrast", y="vals",
                                ax=ax,
                                box_pairs=pairs,
                                test='t-test_ind', text_format='star', loc='outside', verbose=2, comparisons_correction=None)
        plt.xticks(rotation=90)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(False)
    sns.despine()
    plt.tight_layout()

    if save:
        plt.savefig(save, dpi=dpi)

    if return_ax:
        return ax
    else:
        plt.show()
