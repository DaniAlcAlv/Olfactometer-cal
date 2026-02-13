import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import AutoDateLocator, AutoDateFormatter
import seaborn as sns
from utils.analysis import PulseGroup


# -----------------------------
# Plots
# -----------------------------

def draw_general_graph(
    register_dict: dict[str, pd.Series | pd.DataFrame],
    start_pct: float | None = None,
    end_pct: float | None = None,
    *,
    # Names of keys and columns in register_dict
    flow_df_key: str = 'ChannelsTargetFlow',
    endvalve_df_key: str = 'EndValveState',
    endvalve_col: str = 'EndValve0',
    analog_key: str = 'Analog',
    # Plot appearance
    figsize: tuple[int, int] = (18, 5),
    grid_alpha: float = 0.3,
    add_legend: bool = True,
    legend_ncol: int = 2,
    title: str | None = None,
    xlabel: str = 'Timestamp',
    ylabel: str = 'Signal',
    # Saving
    save_path: str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Draw the 'Analog' signal and background shading:
      • End-valve state shading (on/off).
      • ChannelXTarget (X=0,1,2) shading when the channel target is non-zero (mutually exclusive).

    If start_pct and end_pct are provided, the x-axis is zoomed to that window of the 'Analog' timeline.

    Parameters
    ----------
    register_dict : dict[str, Series | DataFrame]
        Must include keys:
          - analog_key (default 'Analog'): Pd.Series with a sortable, monotonic index.
          - flow_df_key (default 'ChannelsTargetFlow'): DataFrame with channel columns (e.g., 'Channel0','Channel1','Channel2').
          - endvalve_df_key (default 'EndValveState'): DataFrame with boolean-like column endvalve_col (default 'EndValve0').
    Plot appearance parameters (figsize, grid_alpha, add_legend, legend_ncol, title, xlabel, ylabel).
    save_path : str, optional
        If provided, saves the figure to this path.
    """
    # ---- Validate presence of keys ----
    missing = [k for k in (analog_key, flow_df_key, endvalve_df_key) if k not in register_dict]
    if missing:
        raise KeyError(f"register_dict missing required keys: {missing}")

    analog = register_dict[analog_key]
    flow_df = register_dict[flow_df_key]
    endvalve_df = register_dict[endvalve_df_key]

    # ---- Normalize Analog to a DataFrame ----
    if isinstance(analog, pd.Series):
        analog = analog.to_frame('Analog')

    # # ---- Basic checks and sorting ----
    # if analog.empty:
    #     raise ValueError("Empty 'Analog' data; nothing to plot.")
    # analog = analog.sort_index()
    # flow_df = flow_df.sort_index()
    # endvalve_df = endvalve_df.sort_index()


    # ---- Create figure/axes ----
    fig, ax = plt.subplots(figsize=figsize)

    # ---- Plot Analog lines ----
    for col in analog.columns:
        ax.plot(analog.index, analog[col].values,
                label=f"Analog{'' if col=='Analog' else f'/{col}'}",
                alpha=1, color = "#000C2E", linewidth= 0.7, zorder=10)

    # ---- Build/align EndValve state to Analog range ----
    if endvalve_col not in endvalve_df.columns:
        raise KeyError(f"'{endvalve_col}' not found in {endvalve_df_key} columns: {list(endvalve_df.columns)}")

    ev = endvalve_df[[endvalve_col]].copy()
    if not ev.index.equals(analog.index):
        union_idx = analog.index.union(ev.index).unique().sort_values()
        ev = ev.reindex(union_idx).ffill()

    ev_state = ev[endvalve_col].astype(bool)

    # ---- Compute EndValve segments ----
    x_start_full = analog.index[0]
    x_end_full = analog.index[-1]
    ev_changes = ev_state.ne(ev_state.shift(1).infer_objects())
    ev_change_idx = ev.index[ev_changes].to_list()

    ev_segments = []
    current_ev_state = ev_state.loc[ev.index[0]]
    cur_start = x_start_full
    for ix in ev_change_idx:
        if ix <= x_start_full:
            current_ev_state = ev_state.loc[ix]
            cur_start = x_start_full
            continue
        if ix > x_end_full:
            break
        ev_segments.append((cur_start, ix, current_ev_state))
        current_ev_state = ev_state.loc[ix]
        cur_start = ix
    ev_segments.append((cur_start, x_end_full, current_ev_state))

    # ---- Channel background shading (non-zero -> open) ----
    # Use the original (unscaled) numeric columns to decide openness
    flow_numeric_raw = flow_df.select_dtypes(include=[np.number]).copy()
    # Keep only the conventional Channel0/1/2 if present, but support any numeric channels
    channel_cols = [c for c in flow_numeric_raw.columns if c.startswith('Channel')]
    if not channel_cols:
        channel_cols = list(flow_numeric_raw.columns)

    # Align to Analog range (forward-fill so state holds between updates)
    if not flow_numeric_raw.index.equals(analog.index):
        union_idx = analog.index.union(flow_numeric_raw.index).unique().sort_values()
        flow_numeric_raw = flow_numeric_raw.reindex(union_idx).ffill()

    # Determine which channel is active at each timestamp.
    # pick the first >threshold if multiple (shouldn't happen).
    threshold = 0.0
    def pick_active_channel(row) -> str | None:
        for ch in channel_cols:
            val = row.get(ch, 0.0)
            if pd.notna(val) and (val > threshold):
                return ch
        return None
    active_channel = flow_numeric_raw.apply(pick_active_channel, axis=1)

    # Find transitions in active_channel (categorical changes including None)
    ch_changes = active_channel.ne(active_channel.shift(1))
    ch_change_idx = active_channel.index[ch_changes].to_list()

    ch_segments = []
    cur_ch = active_channel.iloc[0]
    cur_start = x_start_full
    for ix in ch_change_idx:
        if ix <= x_start_full:
            cur_ch = active_channel.loc[ix]
            cur_start = x_start_full
            continue
        if ix > x_end_full:
            break
        ch_segments.append((cur_start, ix, cur_ch))
        cur_ch = active_channel.loc[ix]
        cur_start = ix
    ch_segments.append((cur_start, x_end_full, cur_ch))

    # Draw channel spans first (behind everything else)

    channel_bg_colors = {
            'Channel0': "#00b1f7", 
            'Channel1': "#ffa200",  
            'Channel2': "#09ff00",  
        }
    for s, e, ch in ch_segments:
        if ch is None:
            continue
        color = channel_bg_colors.get(ch, "#999999")
        ax.axvspan(s, e, color=color, alpha=0.5, zorder=0, ymin=0.1, ymax=0.9)

    # Draw EndValve spans above channel backgrounds but under lines
    for s, e, st in ev_segments:
        ax.axvspan(s, e, color=( "#ffffff00" if st else "#ff5050ff"), alpha=0.3, zorder=5)


    # ---- Software events overlay (points at vertical center) ----
    if "Software" in register_dict:
        sw_df = register_dict["Software"]
        # Expect a DataFrame with 'timestamp' and 'state_index' columns
        ts = pd.to_numeric(sw_df["timestamp"], errors="coerce")
        si = pd.to_numeric(sw_df["state_index"], errors="coerce")

        # Sort by timestamp for consistent order
        # order = np.argsort(ts.values.astype("datetime64[ns]"))
        # ts = ts.iloc[order]
        # si = si.iloc[order]

        # Compute the current vertical center and keep y-lims intact
        y_min, y_max = ax.get_ylim()
        y_center = 0.5 * (y_min + y_max)

        # Map states to colors (0,1,2)
        software_cmap = {
            2: "#00b1f7",  # Channel0
            0: "#ffa200",  # Channel1
            1: "#09ff00",  # Channel2
        }
        colors = [software_cmap.get(int(v), "k") for v in si.values]


        # Scatter points at center line
        ax.scatter(
            ts.values,
            np.full(len(ts), y_center, dtype=float),
            c=colors,
            s=24,
            marker="o",
            zorder=15,
            edgecolors="none",
            linewidths=0.0,
        )


    # ---- Apply percent window if provided ----
    if start_pct is not None and end_pct is not None:
        sp = max(-5.0, min(100.0, float(start_pct)))
        ep = max(0.0, min(105.0, float(end_pct)))
        if sp == ep:
            raise ValueError("start_pct and end_pct must be different.")
        if sp > ep:
            sp, ep = ep, sp

        x_min = analog.index[0]
        x_max = analog.index[-1]
        x0 = x_min + (x_max - x_min) * (sp / 100.0)
        x1 = x_min + (x_max - x_min) * (ep / 100.0)
        ax.set_xlim(left=x0, right=x1)


    # ---- Formatting ----
    if pd.api.types.is_datetime64_any_dtype(analog.index):
        locator = AutoDateLocator()
        formatter = AutoDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    # Optional legend (only Analog is in legend now)
    if add_legend:
        ax.legend(loc='upper right', ncol=legend_ncol, frameon=False)

    ax.grid(True, alpha=grid_alpha)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')

    return fig, ax

def draw_groups_of_pulses(groups_of_pulses: dict[str, PulseGroup]) -> None:
    """Plot each group of pulses in a shared 3x2 grid, robust to empty groups and attributes."""
    # Create a fixed 3x2 grid like your original code
    fig, axes = plt.subplots(3, 2, figsize=(16, 12), sharex=True)
    axes = axes.flatten()

    # Prepare a color palette once
    base_colors = sns.color_palette("husl", 30)
    base_colors = [(r, g, b, 0.7) for r, g, b in base_colors]
    n_colors = len(base_colors)

    for ax, pulse_group in zip(axes, groups_of_pulses.values()):
        ax.set_xlim(-1.0, 3.0)

        # --- Safely obtain a descriptive tag for pulses_created_by_envalve ---
        created_attr = getattr(pulse_group, "pulses_created_by_envalve", None)
        # Create a compact, non-crashing description for the title
        if created_attr is None:
            created_descr = "None"
        else:
            # If it’s a collection (e.g., list/array), show length; if scalar/bool, show value
            try:
                created_descr = f"len={len(created_attr)}"
            except TypeError:
                created_descr = str(created_attr)

        any_line_plotted = False

        # If pulse_group itself is iterable but might be empty, guard the loop
        # Many custom containers implement __len__, so try to check length if possible
        group_is_empty = False
        try:
            group_is_empty = (len(pulse_group) == 0)
        except TypeError:
            # If len is not supported, we’ll infer emptiness after attempting to iterate
            pass

        if group_is_empty:
            ax.set_title(f"Channel: {getattr(pulse_group, 'channel', '?')} | Created by EndValve: {created_descr}")
            ax.set_ylabel("Signal (baseline corrected)")
            ax.text(0.5, 0.5, "No pulses in this group", ha='center', va='center', transform=ax.transAxes, alpha=0.6)
            ax.grid(True, linestyle=':', alpha=0.5)
            continue

        for j, pulse in enumerate(pulse_group):
            # In case the object is iterable but empty, this loop just won't execute
            # Use modulo to avoid IndexError when there are > n_colors pulses
            color = base_colors[j % n_colors]

            # Baseline-corrected signal & time alignment
            y = pulse.slice_df.iloc[:, 0] - pulse.pre_pulse.q16
            aligned_time = pulse.slice_df.index.to_numpy() - pulse.pulse_start

            if getattr(pulse, "created_by_envalve", False):
                trigger_time = pulse.t_active_endvalve - pulse.pulse_start
                label = (
                    f"{pulse.idx+1} | {pulse.squareness*100:.1f}% | "
                    f"{pulse.plateau.mean - pulse.pre_pulse.q16:.0f} | "
                    f"{pulse.dt_active_endvalve*1000:.0f}ms"
                )
            else:
                trigger_time = pulse.t_active_channel - pulse.pulse_start
                label = (
                    f"{pulse.idx+1} | {pulse.squareness*100:.1f}% | "
                    f"{pulse.plateau.mean - pulse.pre_pulse.q16:.0f} | "
                    f"{pulse.dt_active_channel*1000:.0f}ms"
                )

            ax.plot(aligned_time, y, label=label, color=color)
            ax.axvline(trigger_time, color=color, linestyle='--')
            any_line_plotted = True

        ax.set_title(f"Channel: {getattr(pulse_group, 'channel', '?')} | Created by EndValve: {created_descr}")
        ax.set_ylabel("Signal (baseline corrected)")

        # Only add a legend if we actually plotted something
        if any_line_plotted:
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(loc='upper right', fontsize='small')

        ax.grid(True, linestyle=':', alpha=0.5)

    # Label bottom subplot's x-axis
    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.show()

def plot_combined_group_dashboard(
        ax:plt.Axes, 
        groups_of_pulses:dict, 
        metrics_to_plot:list[str], 
        channels:list[str]|None = None, 
        title:str|None = None
    ) -> None:
    """
    Plot grouped bar charts of aggregated metrics for multiple groups on a
    single Matplotlib axis.

    Each group (channel) is represented on the x-axis, and each metric is
    plotted as a separate bar within each group. Bars display the mean value
    of the metric, with asymmetric error bars if provided.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis object on which the plot will be drawn.
    groups_of_pulses : dict[str, Group]
        A dictionary: The keys are the name of the group (like Channel0-Endvalve:True) and the values are Group objects containing the pulses and their metrics.
    metrics : list[str]
        List of metric names to plot. Metrics missing from a group's data will be skipped or plotted as zero.
    channels : list[str], optional
        Subset and/or order of groups (like Channel0-Endvalve:True) to display. If None, all keys from `groups_of_pulses` are used in their insertion order.
        Channel names not found in the data will be ignored and will raise a warning.
    title : str, optional
        Title for the plot. If None, no title is set.

    Returns
    -------
    None
        The function modifies the provided axis in place.
    """
    from matplotlib.ticker import MaxNLocator
    data = {key: group.aggregate_metrics(metrics_to_plot) for key, group in groups_of_pulses.items()}
    channels = list(data.keys()) if channels is None else channels

    valid_channels = [ch for ch in channels if ch in data]
    missing = set(channels) - set(valid_channels)

    for ch in missing:
        print(f"Warning: Impossible to plot '{ch}' not found in Pulse group data ({list(data.keys())}). Skipping.")

    channels = valid_channels

    x = np.arange(len(channels))
    width = 0.8 / len(metrics_to_plot)

    for i, metric in enumerate(metrics_to_plot):
        means = []
        err_minus = []
        err_plus = []

        for ch in channels:
            metric_data = data[ch].get(metric)
            if metric_data is None:
                print(f"Warning: Metric '{metric}' not found for channel '{ch}'. Using 0.")
                means.append(0)
                err_minus.append(0)
                err_plus.append(0)
            else:
                means.append(metric_data['mean'] or 0)
                err_minus.append(metric_data['err_minus'] or 0)
                err_plus.append(metric_data['err_plus'] or 0)

        ax.bar(
            x + i * width,
            means,
            width=width,
            yerr=[err_minus, err_plus],
            capsize=4,
            label=metric,
        )

    bar_label = [ch.replace("-", "\n") for ch in channels]
    if title:
        ax.set_title(title)

    ax.set_xticks(x + width * (len(metrics_to_plot) - 1) / 2)
    ax.set_xticklabels(bar_label)
    ax.set_ylabel("Value")
    ax.legend()
    ax.yaxis.set_major_locator(MaxNLocator(nbins=10))
    ax.grid(True, axis='y', linestyle='-', alpha=0.5)


