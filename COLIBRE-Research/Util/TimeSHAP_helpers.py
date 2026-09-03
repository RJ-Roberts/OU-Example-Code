import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import timeshap

plt.style.use('/users/arirrobe/MachineLearning/MBH-MorphoKinem-COLIBRE/mnras_mpl.stylesheet')


def plot_global_event_timeshap(
    event_data: pd.DataFrame,
    time_col='time',
    shap_col='Shapley Value',
    figsize=(7, 4),
    point_alpha=0.08,
    point_size=10,
    mean_color='tab:red',
    scatter_color='tab:blue',
    percentile_clip=(1, 99),
    absolute_vals=False,
):
    """
    Matplotlib clone of TimeSHAP global event-level plot.

    Parameters
    ----------
    event_data : pd.DataFrame
        Must contain columns [time_col, shap_col]

    percentile_clip : tuple
        Percentiles for y-axis clipping
    """

    df = event_data.copy()
    df = df[df[shap_col].notna()]
    
    if absolute_vals:
        df[shap_col] = df[shap_col].abs()

    # Mean SHAP per timestep
    mean_df = (
        df.groupby(time_col)[shap_col]
        .mean()
        .reset_index()
        .sort_values(time_col)
    )

    if percentile_clip is not None:
        # Y-axis limits (robust)
        v = df[shap_col].values
        lo, hi = np.percentile(v, percentile_clip)
        lim = max(abs(lo), abs(hi))
    else:
        lim = max(abs(df[shap_col].min()), abs(df[shap_col].max()))

    fig, ax = plt.subplots(figsize=figsize)

    # Scatter all SHAP values
    ax.scatter(
        df[time_col],
        df[shap_col],
        s=point_size,
        alpha=point_alpha,
        color=scatter_color,
        rasterized=True
    )

    # Mean SHAP line
    ax.scatter(
        mean_df[time_col],
        mean_df[shap_col],
        color=mean_color,
        s=5*point_size,
        label='Mean'
    )

    print(len(df[time_col]))
    # Formatting
    ax.axhline(0, color='k', lw=1, alpha=0.6)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('SHAP value')
    ax.set_ylim(-lim, lim)
    ax.legend(frameon=False)

    return fig, ax



def plot_global_feat_matplotlib(
    feat_data: pd.DataFrame,
    feature_order=None,
    axis_lim=None,
    figsize=(6, 4),
    point_alpha=0.3,
    point_size=7,
    mean_size=80,
):
    """
    Matplotlib version of TimeSHAP global feature plot
    """

    # Split raw SHAP values
    df = feat_data.copy()
    df = df[df['Shapley Value'].notna()]

    # Compute mean per feature
    mean_df = (
        df.groupby('Feature')['Shapley Value']
        .mean()
        .reset_index()
    )

    # Sort features by |mean|
    mean_df['abs_mean'] = mean_df['Shapley Value'].abs()
    mean_df = mean_df.sort_values('abs_mean', ascending=True)

    features = mean_df['Feature'].values
    y_pos = np.arange(len(features))

    fig, ax = plt.subplots(figsize=figsize)

    # Scatter raw SHAP values
    for i, feat in enumerate(features):
        vals = df.loc[df['Feature'] == feat, 'Shapley Value']
        ax.scatter(
            vals,
            np.full_like(vals, i),
            s=point_size,
            alpha=point_alpha,
            color='tab:blue'
        )

    # Plot mean SHAP values
    ax.scatter(
        mean_df['Shapley Value'],
        y_pos,
        s=mean_size,
        color='tab:red',
        zorder=3,
        label='Mean'
    )

    # Formatting
    ax.axvline(0, color='k', lw=1, alpha=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.set_xlabel('SHAP value')

    if axis_lim is not None:
        ax.set_xlim(axis_lim)

    ax.legend(frameon=False)
    return fig, ax





def plot_global_feat_violin(
    feat_data: pd.DataFrame,
    axis_percentiles=(1, 99),
    figsize=(7, 4),
    violin_width=0.8,
    mean_size=70,
    violin_color='tab:blue',
    mean_color='tab:red'
):
    """
    Matplotlib TimeSHAP global feature violin plot.

    Parameters
    ----------
    feat_data : pd.DataFrame
        Must contain columns:
        ['Feature', 'Shapley Value']
        (optionally also 'time', 'sequence_id')

    axis_percentiles : tuple
        Percentiles used to clip x-axis (default: 1–99)

    figsize : tuple
        Figure size

    violin_width : float
        Width of each violin

    mean_size : int
        Marker size for mean points
    """

    # Keep only raw SHAP values
    df = feat_data.copy()
    df = df[df['Shapley Value'].notna()]

    # Compute mean SHAP per feature
    mean_df = (
        df.groupby('Feature')['Shapley Value']
        .mean()
        .reset_index()
    )

    # Sort features by |mean SHAP|
    mean_df['abs_mean'] = mean_df['Shapley Value'].abs()
    mean_df = mean_df.sort_values('abs_mean', ascending=True)

    features = mean_df['Feature'].values
    y_pos = np.arange(len(features))

    # Axis limits via percentile clipping
    v = df['Shapley Value'].values
    lo, hi = np.percentile(v, axis_percentiles)
    lim = max(abs(lo), abs(hi))

    fig, ax = plt.subplots(figsize=figsize)

    # Draw violins
    for i, feat in enumerate(features):
        vals = df.loc[df['Feature'] == feat, 'Shapley Value'].values
        if len(vals) == 0:
            continue

        vp = ax.violinplot(
            vals,
            positions=[i],
            vert=False,
            widths=violin_width,
            showmeans=False,
            showextrema=False,
            showmedians=False
        )

        for body in vp['bodies']:
            body.set_facecolor(violin_color)
            body.set_edgecolor('black')
            body.set_alpha(0.35)

    # Plot mean SHAP values
    ax.scatter(
        mean_df['Shapley Value'],
        y_pos,
        s=mean_size,
        color=mean_color,
        zorder=3,
        label='Mean'
    )

    # Formatting
    ax.axvline(0, color='k', lw=1, alpha=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.set_xlim(-lim, lim)
    ax.set_xlabel('SHAP value')

    ax.legend(frameon=False)

    return fig, ax