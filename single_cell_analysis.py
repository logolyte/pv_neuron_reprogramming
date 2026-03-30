import scanpy
import anndata
import matplotlib
from matplotlib import pyplot
import hdf5plugin
import numpy
import scvelo
import seaborn
import pandas
import warnings
import cellrank
import gseapy
import decoupler
import IPython
import pygam
import textwrap
import scipy
import mpl_toolkits.axes_grid1

def plot_group_composition(adata, group_key, sample_key="sample", normalize=False):
    if f"{sample_key}_colors" in adata.uns.keys():
        sample_color_map = {}
        for sample, color in zip(adata.obs[sample_key].cat.categories, adata.uns[f"{sample_key}_colors"]):
            sample_color_map[sample] = color
    else:
        sample_color_map = None

    # Calculate the frequency of each sample within each cluster
    composition = pandas.crosstab(adata.obs[group_key], adata.obs[sample_key])
    if normalize:
        composition = composition.divide(composition.sum(axis=1), axis=0)

    # Plot the stacked bar chart
    composition.plot(kind="bar", stacked=True, figsize=(10, 6), color=sample_color_map)
    axis = pyplot.gca()
    axis.legend(title="Sample", bbox_to_anchor=(1.05, 1), loc="upper left")
    axis.grid(False)
    axis.set_xlabel(group_key.capitalize())
    if normalize:
        ylabel = "Proportion"
    else:
        ylabel = "Cells"
    axis.set_ylabel(ylabel)
    figure = pyplot.gcf()
    figure.tight_layout()
    pyplot.show()

# Function for plotting trends independent of trajectory

def plot_grouped_gene_trend(adata, genes, group_key="group", time_key="latent_time", layer=None, groups="all", columns=4, figure_width=14, plot_height=4, n_splines=10,
    obsm=False, show=True):
    """
    Plots smoothed gene expression using generalized additive models (GAMs).
    This provides true smooth local regression with confidence intervals.
    """

    number_of_genes = len(genes)
    number_of_columns = min(columns, number_of_genes)
    
    # Calculate rows
    rows = (number_of_genes + number_of_columns - 1) // number_of_columns
    
    figure, axes = pyplot.subplots(rows, number_of_columns)
    figure.set_figwidth(figure_width)
    figure.set_figheight(plot_height * rows)

    # Handle single gene case (axes is not an array) and 1D array case
    if number_of_genes == 1:
        axes_flat = [axes]
    else:
        axes_flat = axes.flatten()

    # Define colors for groups
    unique_groups = adata.obs[group_key].cat.categories
    if f"{group_key}_colors" in adata.uns_keys():
        palette = adata.uns[f"{group_key}_colors"]
        color_map = dict(zip(unique_groups, palette))
    else:
        palette = seaborn.color_palette("tab10", n_colors=len(unique_groups))
        color_map = dict(zip(unique_groups, palette))


    for gene_index, gene in enumerate(genes):
        axis = axes_flat[gene_index]
        
        # Fetch expression data
        if obsm:
            expr_data = adata.obsm[layer][gene]
        else:
            if (layer == None and gene in adata.obs_keys()) or layer == "obs":
                expr_data = adata.obs[gene]
            elif layer == "raw":
                expr_data = adata.raw.to_adata()[:, gene].X
            elif layer == "X" or layer == None:
                expr_data = adata[:, gene].X
            else:
                expr_data = adata[:, gene].layers[layer]
        if hasattr(expr_data, "toarray"):
            expr_data = expr_data.toarray()
        if hasattr(expr_data, "flatten"):
            expr_data = expr_data.flatten()
        
        # Create DataFrame
        df = pandas.DataFrame({
            "Latent Time": adata.obs[time_key],
            "Expression": expr_data,
            "Group": adata.obs[group_key]
        })
        
        # Determine which groups to plot
        if groups is None:
            # If no specific list, plot all groups found in the data
            plot_groups = df["Group"].unique()
        else:
            plot_groups = groups

        # Loop through each group and fit a GAM
        for group in plot_groups:
            group_data = df[df["Group"] == group]
            color = color_map[group]
            
            # Skip empty groups
            if len(group_data) < 10:
                continue

            # Prepare X and y for pygam
            X = group_data["Latent Time"].values.reshape(-1, 1)
            y = group_data["Expression"].values
            
            # Fit LinearGAM using a spline term s(0)
            gam = pygam.LinearGAM(pygam.s(0, n_splines=n_splines)).fit(X, y)
            
            # Greate grid
            X_grid = numpy.linspace(X.min(), X.max(), 500)
            
            # Predict values and confidence intervals
            y_pred = gam.predict(X_grid)
            confidence_intervals = gam.confidence_intervals(X_grid, width=0.95)
            
            # Plot line
            axis.plot(X_grid, y_pred, label=group, color=color, linewidth=3)
            
            # Plot confidence interval
            axis.fill_between(
                X_grid, 
                confidence_intervals[:, 0], 
                confidence_intervals[:, 1], 
                color=color, 
                alpha=0.2
            )

        axis.set_title(f"{gene} expression")
        axis.set_xlabel("Latent Time")
        
        # Add legend to the first plot
        if gene_index == 0:
            axis.legend()

    # Clean up empty subplots
    for i in range(number_of_genes, len(axes_flat)):
        axes_flat[i].axis('off')

    pyplot.tight_layout()
    if show:
        pyplot.show()
    else:
        return axes_flat

def compare_gsea(data, gene_set, group_key="macrostate", layer=None, n_top_terms=5, significance_cutoff=0.05, sort_key="NES", remove_parenthesis=False,
                 wrap_width=30, font_size=10, min_dot_size=5, top_from="all", significance_metric="FDR q-val", enrichment_cutoff=0, reverse = False,
                 terms="all"):

    if type(data) == anndata.AnnData:
        adata = data
        # extract unique groups from the categorical column
        groups = adata.obs[group_key].cat.categories

        scanpy.tl.rank_genes_groups(
            adata,
            groupby=group_key,
            method="wilcoxon",
            key_added=f"{group_key}_rank_genes",
            layer=layer,
            use_raw=False
        )
    elif type(data) == dict:
        gsea_results = []
        groups = data.keys()

    # compute gene set enrichment for each individual group
    gsea_results = []
    for group in groups:
        if type(data) == anndata.AnnData:
            rank_df = scanpy.get.rank_genes_groups_df(
                adata,
                key=f"{group_key}_rank_genes",
                group=group
            )
        elif type(data) == dict:
            rank_df = data[group]
        if "names" in rank_df.columns:
            rank_df.index = rank_df["names"]
            rank_df = rank_df.drop(columns="names")

        pre_res = gseapy.prerank(
            rnk=rank_df[["scores"]], 
            gene_sets=gene_set,
            threads=32,
            min_size=5,
            max_size=1000,
        )
        results_dataframe = pre_res.res2d.copy()
        results_dataframe["group"] = group
        gsea_results.append(results_dataframe)

    # concatenate all enrichment results into a single long-form dataframe
    gsea_dataframe = pandas.concat(gsea_results, ignore_index=True)

    if not terms == "all":
        gsea_dataframe = gsea_dataframe[gsea_dataframe["Term"].isin(terms)]


    # identify the top significant gene sets across all groups or one group
    top_gene_sets = []
    if top_from == "all":
        group_set = groups
    else:
        group_set = [top_from]
    for group in group_set:
            if reverse:
                significant_results = gsea_dataframe[(gsea_dataframe["group"] == group) & (gsea_dataframe[significance_metric] <= significance_cutoff)\
                                                     & (gsea_dataframe["NES"] < enrichment_cutoff)]
            else:
                significant_results = gsea_dataframe[(gsea_dataframe["group"] == group) & (gsea_dataframe[significance_metric] <= significance_cutoff)\
                                                     & (gsea_dataframe["NES"] > enrichment_cutoff)]
            top_terms = significant_results.sort_values(by=sort_key, ascending=reverse).head(n_top_terms)["Term"].tolist()
            top_gene_sets.extend(top_terms)

    # retain unique gene sets while preserving their initial order
    unique_top_gene_sets = list(dict.fromkeys(top_gene_sets))

    # filter the main dataframe to include only the selected top gene sets
    plot_dataframe = gsea_dataframe[gsea_dataframe["Term"].isin(unique_top_gene_sets)].copy()

    # sort dataframe
    plot_dataframe.index = plot_dataframe["Term"]
    plot_dataframe = plot_dataframe.loc[unique_top_gene_sets]

    # remove parentheses using vectorized string operations
    if remove_parenthesis:
        plot_dataframe["Term"] = plot_dataframe["Term"].str.split("(").str[0].str.strip()

    # wrap long text strings to prevent horizontal axes compression
    plot_dataframe["Term"] = plot_dataframe["Term"].apply(lambda text: textwrap.fill(text, width=wrap_width))

    # convert false discovery rate to negative log10 scale for dot size scaling
    plot_dataframe[significance_metric] = plot_dataframe[significance_metric].replace(0.0, 1e-10)
    
    # strictly cast the column to float to prevent numpy object array errors
    plot_dataframe[significance_metric] = plot_dataframe[significance_metric].astype(float)
    
    plot_dataframe["-log10(FDR)"] = -numpy.log10(plot_dataframe[significance_metric])
    
    # convert -0 to +0
    plot_dataframe["-log10(FDR)"] = numpy.abs(plot_dataframe["-log10(FDR)"])

    # determine the maximum absolute enrichment score to strictly center the color map at zero
    max_absolute_score = plot_dataframe["NES"].abs().max()

    # dynamically adjust figure size based on the number of terms and groups
    number_of_terms = len(plot_dataframe["Term"].unique())
    number_of_groups = len(groups)
    figure_height = max(6.0, number_of_terms * 0.6)
    figure_width = max(8.0, number_of_groups * 1.2)

    # generate the dotplot visualization
    matplotlib.pyplot.figure(figsize=(figure_width, figure_height))
    axis = seaborn.scatterplot(
        data=plot_dataframe,
        x="group",
        y="Term",
        hue="NES",
        size="-log10(FDR)",
        sizes=(min_dot_size, 300),
        palette="coolwarm",
        hue_norm=(-max_absolute_score, max_absolute_score)
    )

    # format the axes and reposition the legend outside the plot area
    axis.set_xlabel(group_key.capitalize(), fontsize=font_size)
    axis.set_xticklabels(axis.get_xticklabels(), rotation=45, rotation_mode="anchor", ha="right")
    axis.set_ylabel("Gene set", fontsize=font_size)
    axis.grid(False)
    axis.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)
    axis.tick_params(axis="both", labelsize=font_size)

    figure = axis.get_figure()
    figure.tight_layout()
    pyplot.show()

# Score TF activity in all macrostates

def plot_tf_activity(adata, network, group_key="macrostate", layer=None, n_top=5, top_from="all", significance_cutoff=0.05, min_dot_size=5, font_size=12,
    activity_cutoff=-numpy.inf, reverse = False, pathway_key="TF"):

    groups = adata.obs[group_key].cat.categories

    scanpy.tl.rank_genes_groups(
        adata,
        groupby=group_key,
        method="wilcoxon",
        key_added="macrostate_rank_genes",
        layer=layer,
        use_raw=False
    )

    tf_activity_dfs = []
    for group in groups:
        rank_df = scanpy.get.rank_genes_groups_df(
            adata,
            key=f"{group_key}_rank_genes",
            group=group
        )
        rank_df.index = rank_df["names"]
        rank_df = rank_df.drop(columns="names")

        data = rank_df[["scores"]].T.dropna(axis=1)
        tf_acts, tf_padj = decoupler.mt.ulm(data=data, net=network)
        activities_df = pandas.DataFrame(data={pathway_key: tf_acts.columns, "Activity": tf_acts.transpose()["scores"], "Adjusted p-value": tf_padj.transpose()["scores"],
                                               "group": group})
        tf_activity_dfs.append(activities_df)
    tf_activities = pandas.concat(tf_activity_dfs, ignore_index=True)

    top_gene_sets = []
    if top_from == "all":
        for group in groups:
            if reverse:
                significant_results = tf_activities[(tf_activities["group"] == group) & (tf_activities["Adjusted p-value"] <= significance_cutoff) &\
                                                (tf_activities["Activity"] < activity_cutoff)]
            else:
                significant_results = tf_activities[(tf_activities["group"] == group) & (tf_activities["Adjusted p-value"] <= significance_cutoff) &\
                                                (tf_activities["Activity"] > activity_cutoff)]
            top_terms = significant_results.sort_values(by="Activity", ascending=reverse).head(n_top)[pathway_key].tolist()
            top_gene_sets.extend(top_terms)
    else:
        group = top_from
        if reverse:
                significant_results = tf_activities[(tf_activities["group"] == group) & (tf_activities["Adjusted p-value"] <= significance_cutoff) &\
                                            (tf_activities["Activity"] < activity_cutoff)]
        else:
            significant_results = tf_activities[(tf_activities["group"] == group) & (tf_activities["Adjusted p-value"] <= significance_cutoff) &\
                                            (tf_activities["Activity"] > activity_cutoff)]
        top_terms = significant_results.sort_values(by="Activity", ascending=reverse).head(n_top)[pathway_key].tolist()
        top_gene_sets.extend(top_terms)

    # retain unique gene sets while preserving their initial order
    unique_top_gene_sets = list(dict.fromkeys(top_gene_sets))

    # filter the main dataframe to include only the selected top gene sets
    plot_dataframe = tf_activities[tf_activities[pathway_key].isin(unique_top_gene_sets)].copy()

    # sort dataframe
    plot_dataframe.index = plot_dataframe[pathway_key]
    plot_dataframe = plot_dataframe.loc[unique_top_gene_sets]

    # convert false discovery rate to negative log10 scale for dot size scaling
    plot_dataframe["Adjusted p-value"] = plot_dataframe["Adjusted p-value"].replace(0.0, 1e-10)
    
    # strictly cast the column to float to prevent numpy object array errors
    plot_dataframe["Adjusted p-value"] = plot_dataframe["Adjusted p-value"].astype(float)
    
    plot_dataframe["-log10(padj)"] = -numpy.log10(plot_dataframe["Adjusted p-value"])
    
    # convert -0 to +0
    plot_dataframe["-log10(padj)"] = numpy.abs(plot_dataframe["-log10(padj)"])

    # determine the maximum absolute enrichment score to strictly center the color map at zero
    max_absolute_score = plot_dataframe["Activity"].abs().max()

    # dynamically adjust figure size based on the number of terms and groups
    number_of_terms = len(plot_dataframe[pathway_key].unique())
    number_of_groups = len(groups)
    figure_height = max(6.0, number_of_terms * 0.6)
    figure_width = max(8.0, number_of_groups * 1.2)

    # generate the dotplot visualization
    matplotlib.pyplot.figure(figsize=(figure_width, figure_height))
    axis = seaborn.scatterplot(
        data=plot_dataframe,
        x="group",
        y=pathway_key,
        hue="Activity",
        size="-log10(padj)",
        sizes=(min_dot_size, 300),
        palette="coolwarm",
        hue_norm=(-max_absolute_score, max_absolute_score)
    )

    # format the axes and reposition the legend outside the plot area
    axis.set_xlabel(group_key.capitalize(), fontsize=font_size)
    axis.set_xticklabels(axis.get_xticklabels(), rotation=45, rotation_mode="anchor", ha="right")
    axis.set_ylabel(pathway_key, fontsize=font_size)
    axis.grid(False)
    axis.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)
    axis.tick_params(axis="both", labelsize=font_size)

    figure = axis.get_figure()
    figure.tight_layout()
    pyplot.show()

def calculate_correlation(adata, variable_1, variable_2, group_key="group", groups="all", layers=["Ms", "Ms"], print_results=True, method="spearman"):
    if print_results:
        print(f"Correlation between {variable_1} and {variable_2}")
    if method == "spearman":
        correlation_method = scipy.stats.spearmanr
    elif method == "pearson":
        correlation_method = scipy.stats.pearsonr
    results_dict = {}
    if groups == "all":
        groups = adata.obs[group_key].cat.categories
    for group in groups:
        indices = adata.obs[group_key] == group
        arrays = []
        for variable, layer in zip([variable_1, variable_2], layers):
            if variable in adata.obs.columns:
                array = adata[indices].obs[variable].to_numpy()
            else:
                if layer == "raw":
                    array = adata.raw.to_adata()[indices, variable].X
                    if hasattr(array, "toarray"):
                        array = array.toarray()
                    else:
                        array = numpy.array(array)
                elif layer == "X":
                    array = adata[indices, variable].X
                    if hasattr(array, "toarray"):
                        array = array.toarray()
                    else:
                        array = numpy.array(array)
                else:
                    array = numpy.array(adata[indices, variable].layers[layer])
            array = array.flatten()
            arrays.append(array)
        correlation_result = correlation_method(arrays[0], arrays[1])
        if isinstance(correlation_result.statistic, numpy.ndarray):
            statistic = correlation_result.statistic[0]
        else:
            statistic = correlation_result.statistic
        if isinstance(correlation_result.pvalue, numpy.ndarray):
            pvalue = correlation_result.pvalue[0]
        else:
            pvalue = correlation_result.pvalue
        if print_results:
            print(f"{group}: {statistic:.3g}, p-value: {pvalue:.3g}")
        else:
            results_dict[group] = {"statistic": statistic, "pvalue": pvalue}
    
    # Total correlation
    if len(groups) > 1:
        indices = adata.obs[group_key].isin(groups)
        arrays = []
        for variable, layer in zip([variable_1, variable_2], layers):
            if variable in adata.obs.columns:
                array = adata[indices].obs[variable].to_numpy()
            else:
                if layer == "raw":
                    array = adata.raw.to_adata()[indices, variable].X
                    if hasattr(array, "toarray"):
                        array = array.toarray()
                    else:
                        array = numpy.array(array)
                elif layer == "X":
                    array = adata[indices, variable].X
                    if hasattr(array, "toarray"):
                        array = array.toarray()
                    else:
                        array = numpy.array(array)
                else:
                    array = numpy.array(adata[indices, variable].layers[layer])
            array = array.flatten()
            arrays.append(array)
        correlation_result = correlation_method(arrays[0], arrays[1])
        if isinstance(correlation_result.statistic, numpy.ndarray):
            statistic = correlation_result.statistic[0]
        else:
            statistic = correlation_result.statistic
        if isinstance(correlation_result.pvalue, numpy.ndarray):
            pvalue = correlation_result.pvalue[0]
        else:
            pvalue = correlation_result.pvalue
        if print_results:
            print(f"Total: {statistic:.3g}, p-value: {pvalue:.3g}")
        else:
            results_dict["total"] = {"statistic": statistic, "pvalue": pvalue}
    if not print_results:
        return results_dict

def plot_gene_correlation(adata, x_genes, y_genes, layer="Ms", annotation_style="stars", figsize=(10, 8), title="", group_key=None, groups=None):
    
    # subset the anndata object if specific groups are requested
    if group_key is not None and groups is not None:
        if isinstance(groups, str):
            groups = [groups]
        adata_subset = adata[adata.obs[group_key].isin(groups)]
    else:
        adata_subset = adata
    
    # ensure requested genes are present in the dataset to prevent key errors
    valid_x_genes = [gene for gene in x_genes if gene in adata_subset.var_names]
    valid_y_genes = [gene for gene in y_genes if gene in adata_subset.var_names]
    
    # extract the expression data into dataframes using the subsetted data
    x_data = pandas.DataFrame(
        data=adata_subset[:, valid_x_genes].layers[layer], 
        index=adata_subset.obs.index, 
        columns=valid_x_genes
    )
    y_data = pandas.DataFrame(
        data=adata_subset[:, valid_y_genes].layers[layer], 
        index=adata_subset.obs.index, 
        columns=valid_y_genes
    )
    
    # initialize empty dataframes for the correlation coefficients and p-values
    correlation_matrix = pandas.DataFrame(index=valid_y_genes, columns=valid_x_genes, dtype=float)
    pvalue_matrix = pandas.DataFrame(index=valid_y_genes, columns=valid_x_genes, dtype=float)
    
    # calculate the spearman correlation and p-value for each gene pair
    for y_gene in valid_y_genes:
        for x_gene in valid_x_genes:
            correlation, pvalue = scipy.stats.spearmanr(y_data[y_gene], x_data[x_gene])
            correlation_matrix.loc[y_gene, x_gene] = correlation
            pvalue_matrix.loc[y_gene, x_gene] = pvalue
            
    # create the text annotation matrix based on the specified style
    annotation_matrix = pandas.DataFrame(index=valid_y_genes, columns=valid_x_genes, dtype=str)
    
    for y_gene in valid_y_genes:
        for x_gene in valid_x_genes:
            p = pvalue_matrix.loc[y_gene, x_gene]
            if annotation_style == "stars":
                if p < 0.001:
                    annotation = "***"
                elif p < 0.01:
                    annotation = "**"
                elif p < 0.05:
                    annotation = "*"
                else:
                    annotation = ""
            elif annotation_style == "p-value":
                annotation = f"{p:.1e}"
            else:
                annotation = ""
                
            annotation_matrix.loc[y_gene, x_gene] = annotation
            
    # initialize the figure canvas and axis
    figure, axis = matplotlib.pyplot.subplots(figsize=figsize)

    # create an axes divider mapped to the main axis to ensure the colorbar matches the height
    divider = mpl_toolkits.axes_grid1.make_axes_locatable(axis)
    colorbar_axis = divider.append_axes("right", size="5%", pad=0.1)

    # plot the heatmap mapping the colorbar to the newly appended axis and applying the annotations
    seaborn.heatmap(
        data=correlation_matrix,
        annot=annotation_matrix,
        fmt="",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        xticklabels=True,
        yticklabels=True,
        ax=axis,
        cbar_ax=colorbar_axis,
        cbar_kws={"label": "Spearman Correlation"}
    )

    # apply axis labels
    axis.set_xlabel("Target Genes")
    axis.set_ylabel("Source Genes")

    # rotate the x-axis tick labels to prevent text overlap
    axis.set_xticklabels(axis.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # ensure y-axis labels are horizontal
    axis.set_yticklabels(axis.get_yticklabels(), rotation=0)

    axis.set_title(title)

    matplotlib.pyplot.tight_layout()
    matplotlib.pyplot.show()


def plot_trends_by_trajectory(
    adata, 
    tfs, 
    time_key="latent_time", 
    model=None, 
    group_key=None, 
    groups="all",
    lineages="all",
    layer=None, 
    n_knots=10, 
    smoothing_penalty=10.0,
    n_columns=4
):
    """
    Plots decoupler transcription factor activity scores using cellrank gene trends.
    """

    if group_key is None or groups == "all":
        adata_sub = adata
    else:
        group_mask = adata.obs[group_key].isin(groups)
        adata_sub = adata[group_mask].copy()

    term_state_colors = adata.uns["term_states_fwd_colors"].copy()

    if layer == None:
        data = {}
        index = adata_sub.obs_names
        for var in tfs:
            if var in adata.obs_keys():
                data[var] = adata_sub.obs[var]
            else:
                data[var] = adata_sub[:, var].X
                if hasattr(data[var], "toarray"):
                    data[var] = data[var].toarray()
                if hasattr(data[var], "flatten"):
                    data[var] = data[var].flatten()
        activity_matrix = pandas.DataFrame(data = data, index = index)

    elif layer == "obsm":
        activity_matrix = pandas.DataFrame(
            adata_sub.obsm["score_ulm"], 
            index=adata_sub.obs_names
        )
    elif layer == "obs":
        activity_matrix = pandas.DataFrame(
            data={var: adata_sub.obs[var] for var in tfs}, 
            index=adata_sub.obs_names
        )
    elif layer == "X":
        X_data = adata_sub[:, tfs].X
        if hasattr(X_data, "toarray"):
            X_data = X_data.toarray()
        activity_matrix = pandas.DataFrame(
            X_data, 
            columns=tfs, 
            index=adata_sub.obs_names
        )
    elif layer == "raw":
        X_data = adata_sub.raw.to_adata()[:, tfs].X
        if hasattr(X_data, "toarray"):
            X_data = X_data.toarray()
        activity_matrix = pandas.DataFrame(
            X_data, 
            columns=tfs, 
            index=adata_sub.obs_names
        )
    else:
        layer_data = adata_sub[:, tfs].layers[layer]
        if hasattr(layer_data, "toarray"):
            layer_data = layer_data.toarray()
        activity_matrix = pandas.DataFrame(
            layer_data, 
            columns=tfs, 
            index=adata_sub.obs_names
        )
    
    tf_adata = scanpy.AnnData(X=activity_matrix)
    tf_adata.obs = adata_sub.obs.copy()
    tf_adata.obsm = adata_sub.obsm.copy()
    tf_adata.uns = adata_sub.uns.copy()
    tf_adata.uns["term_states_fwd_colors"] = term_state_colors
    
    # Restore original categories to prevent missing lineages when subsetting drops states
    tf_adata.obs["term_states_fwd"] = pandas.Categorical(
        adata_sub.obs["term_states_fwd"],
        categories=adata.obs["term_states_fwd"].cat.categories
    )
    
    # Reconstruct the lineage object using the correct forward fate probabilities key
    if "lineages_fwd" in tf_adata.obsm:
        tf_adata.obsm["lineages_fwd"] = cellrank.Lineage.from_adata(
            tf_adata, 
            kind="fate_probs"
        )

    if model is None:
        model = cellrank.models.GAMR(
            tf_adata, 
            n_knots=n_knots, 
            smoothing_penalty=smoothing_penalty
        )

    if lineages == "all":
        lineages = adata.obs["term_states_fwd"].cat.categories
        
    cellrank.pl.gene_trends(
        tf_adata,
        model=model,
        genes=tfs,
        data_key="X",
        same_plot=True,
        lineages=lineages,
        ncols=n_columns,
        time_key=time_key,
        hide_cells=True,
        weight_threshold=(1e-3, 1e-3),
    )

def plot_differential_expression(differential_expression_dataframe, genes, groups="all"):

    if groups == "all":
        groups = differential_expression_dataframe.columns[differential_expression_dataframe.columns.str.contains("LFC")].str.split(" ").str[:-1].str.join(sep=" ")

    # assign the filtered and sorted dataframe slice to a variable
    plot_data = differential_expression_dataframe.loc[genes].sort_values("Reprogramming LFC")

    # separate the log fold change columns and the adjusted p-value columns
    lfc_columns = [f"{group} LFC" for group in groups]
    padj_columns = [f"{group} p adjusted" for group in groups]

    lfc_matrix = plot_data[lfc_columns]
    padj_matrix = plot_data[padj_columns]

    # clean the column names for the x-axis labels on the final plot
    lfc_matrix.columns = groups

    # convert the adjusted p-values to a float array
    padj_array = padj_matrix.to_numpy(dtype=float)
    # Replace NaN with 1
    numpy.nan_to_num(padj_array, nan=1, copy=False)

    # generate an annotation matrix where significant values receive an asterisk
    # the condition ignores nan values to prevent runtime warnings
    significance_annotations = numpy.full(padj_array.shape, "", dtype=numpy.dtypes.StrDType)
    significance_annotations[padj_array < 0.05] += "*"
    significance_annotations[padj_array < 0.01] += " *"
    significance_annotations[padj_array < 0.001] += " *"

    # initialize the figure canvas
    pyplot.figure(figsize=(8, 6))

    # plot the heatmap using a divergent colormap centered at zero
    seaborn.heatmap(
        data=lfc_matrix,
        annot=significance_annotations,
        fmt="s",
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
        cbar_kws={"label": "Log2 Fold Change"}
    )

    # apply axis labels and adjust the layout to prevent clipping
    pyplot.xlabel("Condition")
    pyplot.ylabel("Gene")
    pyplot.tight_layout()
    pyplot.show()