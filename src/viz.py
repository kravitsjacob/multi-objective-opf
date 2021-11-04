"""Source code for visualization"""


# Import Modules
import pandas as pd
import numpy as np
from matplotlib import ticker
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import matplotlib as mpl
from matplotlib import cm
from itertools import chain
import seaborn as sns
sns.set()


def k_means(df, n_clusters, cluster_columns):
    # K Means Clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=1008).fit(df[cluster_columns].values)
    df['Color Index'] = ['cluster_' + i for i in kmeans.labels_.astype(str)]
    return df


def set_color_cluster(df, color_idx, colors, labels):
    # Set Columns
    df['Color'] = df.replace(dict(zip(color_idx, colors)))['Color Index']
    df['Color Label'] = df.replace(dict(zip(color_idx, labels)))['Color Index']
    # Export
    return df


def set_color_gradient(df, colormap, label):
    # Compute Proportion of Each Line
    mixes = (df['Color Index'] - df['Color Index'].max()) / (df['Color Index'].min() - df['Color Index'].max())
    # Get Colors Corresponding to Proportions
    df['Color'] = [mpl.colors.rgb2hex(cm.get_cmap(colormap)(i)[:3]) for i in mixes]
    df['Color Label'] = label
    return df


def format_ticks(
        i,
        ax,
        n_ticks,
        limits,
        cols,
        flip_idx,
        x,
        tick_precision,
        explicit_ticks,
        label_fontsize,
        df,
        rotate_x_labels=False
):
    # Format X axis
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    # Format X axis
    if i == len(cols)-1:
        ax.xaxis.set_major_locator(ticker.FixedLocator([x[-2], x[-1]]))
        ax.set_xticklabels([cols[-2], cols[-1]])
    else:
        ax.xaxis.set_major_locator(ticker.FixedLocator([i]))
        ax.set_xticklabels([cols[i]])
    # Format Y axis
    if explicit_ticks is None:
        step = (limits[i][1] - limits[i][0]) / (n_ticks[i] - 1)
        tick_labels = [round(limits[i][0] + step * j, tick_precision[i]) for j in range(n_ticks[i])]
        norm_step = 1 / float(n_ticks[i] - 1)
        ticks = [round(0 + norm_step * i, 3) for i in range(n_ticks[i])]
        ax.yaxis.set_ticks(ticks)
        if i in flip_idx:
            tick_labels.reverse()
        ax.set_yticklabels(tick_labels)
    else:
        lower = 0 + (df[cols[i]].min() - limits[i][0]) / (limits[i][1] - limits[i][0])
        upper = 0.00001 + (df[cols[i]].max() - limits[i][0]) / (limits[i][1] - limits[i][0])
        scaler = MinMaxScaler(feature_range=(lower, upper))
        scaler.fit_transform(df[cols[i]].values.reshape((-1, 1)))
        ticks = scaler.transform(np.array(explicit_ticks[i]).reshape((-1, 1)))
        if i in flip_idx:
            ticks = 0.5 + (0.5 - ticks)
        tick_labels = explicit_ticks[i]
        lims_temp = ax.get_ylim()
        ax.yaxis.set_ticks(list(chain.from_iterable(ticks.tolist())))
        ax.set_yticklabels(tick_labels.astype(str))
        ax.set_ylim(lims_temp)
    if rotate_x_labels:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.tick_params(axis='x', which='major', labelsize=label_fontsize)
    return 0


def static_parallel(
        df,
        columns,
        limits=None,
        columns_to_flip=None,
        n_ticks=6,
        tick_precision=None,
        explicit_ticks=None,
        label_fontsize=12,
        plot_legend=False,
        plot_colorbar=False,
        colorbar_colormap='viridis',
        figure_size=(11, 3),
        colorbar_adjust_args=(0.90, 0.2, 0.02, 0.70),
        subplots_adjust_args={'left': 0.05, 'bottom': 0.20, 'right': 0.85, 'top': 0.95, 'wspace': 0.0, 'hspace': 0.0},
        rotate_x_labels=False):
    """
    https://benalexkeen.com/parallel-coordinates-in-matplotlib/
    Coming soon, this will be turned into a proper module

    Parameters
    ----------
    df
    columns
    limits
    columns_to_flip
    n_ticks
    tick_precision
    explicit_ticks
    label_fontsize
    plot_legend
    plot_colorbar
    colorbar_colormap
    figure_size
    colorbar_adjust_args
    subplots_adjust_args
    rotate_x_labels

    Returns
    -------

    """
    # Set automatic Values
    if limits is None:
        limits = list(zip(df[columns].min().values, df[columns].max().values))
    if tick_precision is None:
        tick_precision = [2]*len(columns)
    if isinstance(n_ticks, int):
        n_ticks = [n_ticks]*len(columns)
    if 'Linestyle' not in df.columns:
        df['Linestyle'] = '-'
    # Compute Numeric List of Columns
    x = [i for i, _ in enumerate(columns)]
    # Compute Indices of Columns to Flip
    try:
        flip_idx = [i for i, item in enumerate(columns) if item in columns_to_flip]
    except TypeError:
        flip_idx = [len(x) + 1]
    # Initialize Plots
    fig, axes = plt.subplots(1, len(columns) - 1, sharey=False, figsize=figure_size)
    if len(columns) == 2:
        axes = [axes]
    # Create Scaled DataFrame
    df_scaled = df.copy()
    for i, lim in enumerate(limits):
        lower = 0 + (df[columns[i]].min() - lim[0]) / (lim[1] - lim[0])
        upper = 0.0000001 + (df[columns[i]].max() - lim[0]) / (lim[1] - lim[0])
        scaler = MinMaxScaler(feature_range=(lower, upper))
        scaled_data = scaler.fit_transform(df[columns[i]].values.reshape((-1, 1)))
        if i in flip_idx:
            df_scaled[columns[i]] = 0.5 + (0.5 - scaled_data)
        else:
            df_scaled[columns[i]] = scaled_data
    # Plot each row
    for i, ax in enumerate(axes):
        for idx in df_scaled.index:
            ax.plot(x, df_scaled.loc[idx, columns], df_scaled.loc[idx, 'Color'],  linestyle=df.loc[idx, 'Linestyle'])
        ax.set_xlim([x[i], x[i + 1]])
        ax.set_ylim([0, 1])
    # Format Last Axis
    axes = np.append(axes, plt.twinx(axes[-1]))
    axes[-1].set_ylim(axes[-2].get_ylim())
    if rotate_x_labels:
        axes[-1].set_xticklabels(axes[-1].get_xticklabels(), rotation=90)
        axes[-2].set_xticklabels(axes[-2].get_xticklabels(), rotation=90)
    # Format All Axes
    for i, ax in enumerate(axes):
        format_ticks(i, ax, n_ticks=n_ticks, limits=limits, cols=columns, flip_idx=flip_idx, x=x,
                     tick_precision=tick_precision, explicit_ticks=explicit_ticks, label_fontsize=label_fontsize, df=df,
                     rotate_x_labels=rotate_x_labels)
    # Remove space between subplots
    plt.subplots_adjust(**subplots_adjust_args)
    # Add legend to plot
    if plot_legend:
        plt.legend(
            [plt.Line2D((0, 1), (0, 0), color=i) for i in df['Color'][df['Color Label'].drop_duplicates().index].values],
            df['Color Label'][df['Color Label'].drop_duplicates().index],
            bbox_to_anchor=(1.2, 1), loc=2, borderaxespad=0.0
        )
    if plot_colorbar:
        ax = plt.axes(colorbar_adjust_args)
        cmap = cm.get_cmap(colorbar_colormap)
        norm = mpl.colors.Normalize(vmin=df['Color Index'].min(), vmax=df['Color Index'].max())
        mpl.colorbar.ColorbarBase(
            ax,
            cmap=cmap.reversed(),
            norm=norm,
            orientation='vertical',
            label=df.iloc[0]['Color Label']
        )
    plt.show()
    return fig


def main():
    # Example Data
    obj_labs = ['Objective 1', 'Objective 2', 'Objective 3', 'Objective 4']
    df = pd.DataFrame({'Objective 1': [0, 0.5, 1], 'Objective 2': [0, 0.5, 1], 'Objective 3': [1, 0.5, 0],
                       'Objective 4': [100, 50, 10]})  # Example Data
    df['Color Index'] = df['Objective 1']  # Specify Color
    df = set_color_gradient(df, colormap='viridis', label='Objective 1')

    # Example Quantitative Plotting
    static_parallel(df=df, columns=obj_labs)  # default plot
    static_parallel(df=df, columns=obj_labs, plot_colorbar=True)  # with colorbar
    static_parallel(df=df, columns=obj_labs, n_ticks=[10, 20, 10, 10])  # with user-specified number of ticks
    static_parallel(
        df=df,
        columns=obj_labs,
        tick_precision=[4, 2, 1, -1]
    )  # with user-specified number of ticks and precision
    static_parallel(df=df, columns=obj_labs, columns_to_flip=['Objective 1'])  # Flipping columns
    static_parallel(
        df=df,
        columns=obj_labs,
        limits=[[0, 0.2], [0, 0.6], [0, 0.6], [10, 50]]
    )  # Setting user-specific column limits
    ticks = [np.array([0, 0.1, 0.9]), np.array([0, 0.6, 0.9]), np.array([0, 0.5]), np.array([10, 50])]
    static_parallel(df=df, columns=obj_labs, explicit_ticks=ticks)  # with explicit ticks

    # Example Qualitative Plotting
    df['Color Label'] = ['a', 'b', 'c']
    static_parallel(df=df, columns=obj_labs, plot_legend=True)
    return 0


def correlation_heatmap(df):
    """
    Get correlation heatmap of objectives

    Parameters
    ----------
    df: DataFrame
        DataFrame to visualize

    Returns
    -------
    fig: matplotlib.figure.Figure
        Correlation heatmap figure
    """
    # Compute correlation
    df_corr = df.corr()

    # Get mask for lower triangle
    mask = np.triu(np.ones_like(df_corr, dtype=np.bool))
    np.fill_diagonal(mask, False)

    # Plot
    g = sns.heatmap(df_corr, mask=mask, vmin=-1, vmax=1, annot=True, cmap='BrBG')
    plt.tight_layout()
    fig = g.figure

    # Show Plot
    plt.show()

    return fig


if __name__ == '__main__':
    main()
