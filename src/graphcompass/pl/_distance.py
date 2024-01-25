import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path

from anndata import AnnData
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from tqdm import tqdm
from graphcompass.tl._distance import compare_groups

from typing import Any, Sequence, Tuple, Union


def convert_dataframe(df_ct, samples):
    num_samples = len(samples)

    df = pd.DataFrame(
     data=np.full((num_samples, num_samples), fill_value=np.nan),
     columns=samples,
     index=samples
    )

    for sample_a, sample_b in itertools.product(samples, samples):

        if sample_a == sample_b:
            df[sample_a][sample_b] = 0.0
            df[sample_b][sample_a] = 0.0

        if sample_a < sample_b:
            df_ct_a = df_ct[df_ct["sample_a"] == sample_a]
            df_ct_ab = df_ct_a[df_ct_a["sample_b"] == sample_b]
            if len(df_ct_ab) > 0:
                value = df_ct_ab["similarity_score"].values[0]
                df[sample_a][sample_b] = value
                df[sample_b][sample_a] = value
            else:
                df_ct_a = df_ct[df_ct["sample_b"] == sample_a]
                df_ct_ab = df_ct_a[df_ct_a["sample_a"] == sample_b]
                if len(df_ct_ab) > 0:
                    value = df_ct_ab["similarity_score"].values[0]
                    df[sample_a][sample_b] = value
                    df[sample_b][sample_a] = value

    return df


def compare_conditions(
    adata: AnnData,
    library_key: str = "sample_id",
    condition_key: str = "status",
    control_group: str = " normal",
    metric_key: str = "pairwise_similarities",
    add_ncells_and_density_plots: bool = False,
    plot_groups_separately: bool = False,  # if False then all in one plot
    fig: Union[Figure, None] = None,
    ax: Union[Axes, Sequence[Axes], None] = None,
    return_ax: bool = False,
    figsize: Union[Tuple[float, float], None] = (20, 10),
    palette="Greys",
    dpi: Union[int, None] = 300,
    save: Union[str, Path, None] = None,
    vertical: bool = True,
    **kwargs: Any,
) -> Union[Axes, Sequence[Axes], None]:
    """
    Plot group comparison for each cell type.

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
    add_ncells_and_density_plots
    plot_groups_separately
        If True, then each group is plotted separately. If False, then all
        groups are plotted in one plot.
    fig
        Figure object to be used for plotting.
    ax
        Axes object to be used for plotting.
    return_ax
        If True, then return the axes object.
    figsize
        Figure size.
    palette
        Color palette.
    dpi
        Figure resolution.
    save
        Filename under which to save the plot.
    vertical
    **kwargs
        Keyword arguments to be passed to plotting functions.
    """
    pairwise_similarities = adata.uns[metric_key]

    dict_sample_to_status = {}
    for sample in adata.obs[library_key].unique():
        dict_sample_to_status[sample] = (
            adata[adata.obs[library_key] == sample].obs[condition_key].values.unique()[0]
        )

    df_for_plot = None
    for i, ct in tqdm(enumerate(pairwise_similarities.cell_type.unique())):
        samples = np.unique(
            np.append(
                list(pairwise_similarities.sample_a.values),
                list(pairwise_similarities.sample_b.values),
            )
        )
        df_ct = pairwise_similarities[pairwise_similarities.cell_type == ct]
        dataframe = convert_dataframe(df_ct, samples)
        sample_ids = dict_sample_to_status.keys()
        status = dict_sample_to_status.values()

        sample_to_status = pd.DataFrame(
            {"sample_id": sample_ids, "contrast": status}
        )

        disease_status = list(set(sample_to_status.contrast))
        disease_status.remove(control_group)
        contrasts = [(control_group, c) for c in disease_status]

        if i > 0:
            df = compare_groups(
                pairwise_similarities=dataframe,
                sample_to_contrasts=sample_to_status,
                contrasts=contrasts,
                output_format="tidy"
            )
            df["cell_type"] = np.full(df.shape[0], ct)
            df_for_plot = pd.concat([df_for_plot, df])
        else:
            df_for_plot = compare_groups(
                pairwise_similarities=dataframe,
                sample_to_contrasts=sample_to_status,
                contrasts=contrasts,
                output_format="tidy"
            )
            df_for_plot["cell_type"] = np.full(df_for_plot.shape[0], ct) 

    # set figure parameters
    # if fig is None and ax is None:
    #     fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    # elif fig is None and ax is not None:
    #     fig = ax[0].figure
    # elif fig is not None and ax is None:
    #     ax = fig.gca()

    # plot
    plt.rcParams["font.size"] = 12

    # Function to assign color based on value
    def get_bar_color(value, max_value, min_value, color):
        """ Return color shade based on value. """
        normalized = (value - min_value) / (max_value - min_value)
        return plt.cm.get_cmap(color)(normalized)

    width, height = figsize
    # return df_for_plot
    if plot_groups_separately:
        # plot each contrast in a plot

        num_contrasts = len(df_for_plot["contrast"].unique())
        num_rows = int(np.ceil(num_contrasts/2))
        num_cols = 2

        if vertical:
            x = "cell_type"
            y = "vals"
        else:
            x = "vals"
            y = "cell_type"

        if add_ncells_and_density_plots:
            # get ncells and density from adata.uns[metric_key]
            df = pd.DataFrame()
            properties = pd.DataFrame(
                columns=["condition", "sample", "cell_type", "ncells", "density"]
            )

            for i, row in adata.uns[metric_key].iterrows():
                ct = row["cell_type"]
                sample_a = row["sample_a"]
                condition_a = adata[adata.obs[library_key] == sample_a].obs[condition_key].values[0]
                sample_b = row["sample_b"]
                condition_b = adata[adata.obs[library_key] == sample_b].obs[condition_key].values[0]
                p = properties[properties["cell_type"] == ct]
                if len(p) == 0 or len(p[p["sample"] == sample_a]) == 0:
                    df["condition"] = [condition_a]
                    df["sample"] = [sample_a]
                    df["cell_type"] = [ct]
                    df["ncells"] = [row["ncells_a"]]
                    df["density"] = [row["density_a"]]
                    properties = pd.concat([properties, df])
                if len(p) == 0 or len(p[p["sample"] == sample_b]) == 0:
                    df["condition"] = [condition_b]
                    df["sample"] = [sample_b]
                    df["cell_type"] = [ct]
                    df["ncells"] = [row["ncells_b"]]
                    df["density"] = [row["density_b"]]
                    properties = pd.concat([properties, df])

            fig, axes = plt.subplots(
                num_contrasts,
                3,
                figsize=(width, height*num_contrasts),
                dpi=dpi
            )  # Adjust the figsize as needed
            y = "cell_type"
            x = "vals"
            # Adjust this value as needed for padding
            fig.subplots_adjust(hspace=0.5)

            for i, contrast in enumerate(df_for_plot["contrast"].unique()):
                df_contrast = df_for_plot[df_for_plot["contrast"] == contrast]
                # return df_contrast
                ax_title = fig.add_subplot(num_contrasts, 1, i + 1)
                ax_title.set_title(contrast, fontsize=16, pad=50)
                ax_title.axis('off')

                contrast_properties = properties[properties["condition"].isin(contrast.split("_vs_"))]
                mean_vals = df_contrast.groupby('cell_type')['vals'].mean()
                palette = {ct: get_bar_color(mean_vals[ct], mean_vals.max(), mean_vals.min(), "Greens") for ct in mean_vals.index}

                sns.barplot(
                    data=df_contrast,
                    x=x,
                    y=y,
                    # errorbar=("pi", 50), capsize=.4, errcolor=".5",
                    linewidth=1, edgecolor=".5",
                    palette=palette,
                    ax=axes[i, 0],
                )
                axes[i, 0].set_xticks([0, 1])
                axes[i, 1].set_xlabel('Similarity score')

                axes[i, 0].set_xticklabels(["Identical graphs", "Maximally different"], rotation=90)
                axes[i, 0].set_title('Pairwise similarity')
                sns.despine()
                
                # Plot 2
                sns.barplot(
                    data=contrast_properties,
                    y=y,
                    x="ncells",
                    hue="condition",
                    palette="tab20",
                    ax=axes[i, 1]
                )
                axes[i, 1].set_ylabel('')
                axes[i, 1].legend().remove()
                axes[i, 1].set_title('Number of cells')
                sns.despine()

                # Plot 3
                sns_plot = sns.barplot(
                    data=contrast_properties,
                    y=y,
                    x="density",
                    hue="condition",
                    palette="tab20",
                    ax=axes[i, 2]
                )
                # axes[i, 2].set_xticklabels(axes[i, 2].get_xticklabels(), rotation=90)
                axes[i, 2].set_ylabel('')
                axes[i, 2].set_title('Graph density')
                sns_plot.legend(loc='upper center', bbox_to_anchor=(0.0, -0.07), ncol=2)

                sns.despine()
            plt.tight_layout()  # rect=[0, 0.03, 1, 0.95])

        else:
            fig, ax = plt.subplots(num_rows, num_cols, figsize=figsize, dpi=dpi)
            ax = ax.flatten()
            # plot
            for i, contrast in enumerate(df_for_plot["contrast"].unique()):
                df_contrast = df_for_plot[df_for_plot["contrast"] == contrast]
                mean_vals = df_contrast.groupby('cell_type')['vals'].mean()
                palette = {ct: get_bar_color(mean_vals[ct], mean_vals.max(), mean_vals.min(), "Greens") for ct in mean_vals.index}

                sns.barplot(
                    data=df_contrast,
                    x=x,
                    y=y,
                    errorbar=("pi", 50), capsize=.4, errcolor=".5",
                    linewidth=1, edgecolor=".5",  # facecolor=(0, 0, 0, 0),
                    palette=palette,
                    ax=ax[i]
                )
                ax[i].set_yticks([0, 1])
                ax[i].set_yticklabels(["Identical graphs", "Maximally different"])
                ax[i].set_ylabel("Similarity score")
                ax[i].set_xticklabels(ax[i].get_xticklabels(), rotation=90)
                sns.despine()

                ax[i].set_title(' '.join(contrast.split("_")))

            plt.tight_layout()

        if return_ax:
            return ax
        else:
            plt.show()

        if save:
            plt.savefig(save, dpi=dpi)
    else:
        # plot all contrasts in one plot
        result = df_for_plot.groupby(['contrast', 'cell_type'])['vals'].agg(['median', 'var']).reset_index()

        # Rename columns
        result.columns = ['contrast', 'cell_type', 'median', 'variance']
        result["median"] = result["median"].fillna(1)
        result["variance"] = result["variance"].fillna(0)

        # Function to calculate binned_variance
        def calculate_binned_variance(row):
            tmp = (1 - row['variance']) * 100
            if tmp > 95:
                return 20 ** 2
            elif tmp > 90:
                return 15 ** 2
            elif tmp > 80:
                return 10 ** 2
            else:
                return 5

        # Add the 'binned_variance' column
        result['binned_variance'] = result.apply(calculate_binned_variance, axis=1)
        result["contrast"] = [' '.join(contrast.split("_")) for contrast in result["contrast"].values]

        if add_ncells_and_density_plots:
            df = pd.DataFrame()
            properties = pd.DataFrame(columns=["condition", "sample", "cell_type", "ncells", "density"])

            for i, row in adata.uns[metric_key].iterrows():
                ct = row["cell_type"]
                sample_a = row["sample_a"]
                condition_a = adata[adata.obs[library_key] == sample_a].obs[condition_key].values[0]
                sample_b = row["sample_b"]
                condition_b = adata[adata.obs[library_key] == sample_b].obs[condition_key].values[0]
                p = properties[properties["cell_type"] == ct]
                if len(p) == 0 or len(p[p["sample"] == sample_a]) == 0:
                    df["condition"] = [condition_a]
                    df["sample"] = [sample_a]
                    df["cell_type"] = [ct]
                    df["ncells"] = [row["ncells_a"]]
                    df["density"] = [row["density_a"]]
                    properties = pd.concat([properties, df])
                if len(p) == 0 or len(p[p["sample"] == sample_b]) == 0:
                    df["condition"] = [condition_b]
                    df["sample"] = [sample_b]
                    df["cell_type"] = [ct]
                    df["ncells"] = [row["ncells_b"]]
                    df["density"] = [row["density_b"]]
                    properties = pd.concat([properties, df])

            result_ncells = properties.groupby(['condition', 'cell_type'])['ncells'].agg(['median', 'var']).reset_index()
            result_ncells.columns = ['condition', 'cell_type', 'median', 'variance']
            result_density = properties.groupby(['condition', 'cell_type'])['density'].agg(['median', 'var']).reset_index()
            result_density.columns = ['condition', 'cell_type', 'median', 'variance']

            result_ncells["median"] = result_ncells["median"].fillna(1)
            result_ncells["variance"] = result_ncells["variance"].fillna(0)
            result_density["median"] = result_density["median"].fillna(1)
            result_density["variance"] = result_density["variance"].fillna(0)

            result_ncells['binned_variance'] = result_ncells.apply(calculate_binned_variance, axis=1)
            result_density['binned_variance'] = result_density.apply(calculate_binned_variance, axis=1)

        # Plotting
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        if add_ncells_and_density_plots:
            n_subplots = 3
            fig, axes = plt.subplots(
                n_subplots, 1, figsize=(width, height * n_subplots), dpi=dpi
            )
            ax = axes[0]
        else:
            n_subplots = 1
            fig, ax = plt.subplots(
                n_subplots, 1, figsize=(width, height * n_subplots), dpi=dpi
            )

        # Plot 1
        im1 = ax.scatter(
            x=result["cell_type"],
            y=result["contrast"],
            c=result["median"],
            s=result["binned_variance"],
            cmap=palette
        )

        ax.tick_params(axis='x', labelrotation=90)
        ax.set_axisbelow(True)
        ax.grid(color='gray', linestyle='dashed', which='both', axis='both')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        ax.set_title("Pairwise similarity")

        # Adding colorbar to the right of the first plot
        divider1 = make_axes_locatable(ax)
        cax1 = divider1.append_axes('right', size='3%', pad=0.05)
        cbar1 = fig.colorbar(im1, cax=cax1, orientation='vertical')
        cbar1.set_ticks([0.0, 1.0])
        cbar1.set_ticklabels(['Identical Graphs', 'Maximally Different'])

        if add_ncells_and_density_plots:
            # Plot 2
            im2 = axes[1].scatter(
                x=result_density["cell_type"],
                y=result_density["condition"],
                s=result_density["median"]*1e4,
                c=result_density["binned_variance"],  # should be variance
                cmap=palette,
                vmin=5, vmax=20**2
            )

            axes[1].tick_params(axis='x', labelrotation=90)
            axes[1].set_axisbelow(True)
            axes[1].grid(color='gray', linestyle='dashed', which='both', axis='both')
            axes[1].spines['top'].set_visible(False)
            axes[1].spines['right'].set_visible(False)
            axes[1].spines['bottom'].set_visible(True)
            axes[1].spines['left'].set_visible(True)
            axes[1].set_title("Graph density")

            divider2 = make_axes_locatable(axes[1])
            cax2 = divider2.append_axes('right', size='3%', pad=0.05)
            cbar2 = fig.colorbar(im2, cax=cax2, orientation='vertical')
            cbar2.set_ticks([5, 20**2])  # should be set correctly
            cbar2.set_ticklabels(['High variance', 'Low variance'])

            # Plot 3
            im3 = axes[2].scatter(
                x=result_ncells["cell_type"],
                y=result_ncells["condition"],
                s=result_ncells["median"],
                c=result_ncells["binned_variance"],
                cmap=palette,
                vmin=5, vmax=20**2
            )

            axes[2].tick_params(axis='x', labelrotation=90)
            axes[2].set_axisbelow(True)
            axes[2].grid(color='gray', linestyle='dashed', which='both', axis='both')
            axes[2].spines['top'].set_visible(False)
            axes[2].spines['right'].set_visible(False)
            axes[2].spines['bottom'].set_visible(True)
            axes[2].spines['left'].set_visible(True)
            axes[2].set_title("Number of cells")

            # Adding a shared colorbar for the second and third plots
            # Since the colorbar settings are the same for both, we can create
            # one common colorbar
            divider23 = make_axes_locatable(axes[2])
            cax23 = divider23.append_axes('right', size='3%', pad=0.05)
            cbar23 = fig.colorbar(im3, cax=cax23, orientation='vertical')
            cbar23.set_ticks([5, 20**2])
            cbar23.set_ticklabels(['High variance', 'Low variance'])

        plt.tight_layout()  # Adjust layout to fit everything neatly

        if save:
            plt.savefig(save, dpi=dpi)

        if return_ax:
            return ax
        else:
            plt.show()
