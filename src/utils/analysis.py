import pandas as pd 
import numpy as np
import numbers
import matplotlib.pyplot as plt
from collections import defaultdict

verbose = True

class RegionMetrics:
    """
    Compute summary stats and a linear slope of y vs index (x).
    - Drops first and last 2% of rows.
    - Converts non-numeric indexes to numeric (supports DatetimeIndex).
    - Drops rows with non-finite x/y before computing stats and fitting.
    """
    def __init__(self, df: pd.DataFrame):
        # Drops the first and last 2% of rows
        n = len(df)
        k = int(n * 0.02)  # number of rows to drop at each end
        df = df.iloc[k : n - k]
    
        x = df.index.values
        y = df.iloc[:, 0].values
        self.mean = y.mean()
        self.std = y.std()
        self.median, self.q16, self.q84 = np.percentile(y, [50, 16, 84])
        self.err_minus = max(0, self.mean - self.q16)
        self.err_plus  = max(0, self.q84 - self.mean)
        self.corrected_slope, _, _ = linear_regression(x.tolist(), y.tolist())

class Pulse:
    def __init__(self, signal: pd.DataFrame, registers: dict, approx_start: float, approx_end: float, next_pulse_start: float|None, previous_pulse_end: float|None):
        self.register_dict = registers
        
        if previous_pulse_end is None:
            self.data_start = signal.index[0]
        else:
            self.data_start = previous_pulse_end + (approx_start - previous_pulse_end) / 3
        
        if next_pulse_start is None:
            self.data_end = signal.index[-1]
        else:
            self.data_end = next_pulse_start - (next_pulse_start - approx_end) / 3

        self._slice_data(signal, approx_start, approx_end)
        self._compute_region_metrics()
        self._get_channel_status()

    def __repr__(self) -> str:
        """"Style representation showing key info about the pulse."""
        return (f"<Data start={self.data_start:.2f}s end={self.data_end:.2f}s "
                f"Pulse start={self.pulse_start:.2f}s end={self.pulse_end:.2f}s "
                f"channel={self.active_channel} envalve={self.created_by_envalve} "
                f"squareness={self.squareness:.2f}>")

    def _slice_data(self, signal, approx_start, approx_end) -> None:
        """"Extracts the pulse region from the signal based on approximate start/end and refines it using quantiles."""
        self.slice_df = signal.loc[(signal.index >= self.data_start) & (signal.index <= self.data_end)]

        self.pulse_center = approx_start + 0.5 * (approx_end - approx_start)

        first_half = self.slice_df.loc[(self.slice_df.index >= self.data_start) & (self.slice_df.index <= self.pulse_center)]
        second_half = self.slice_df.loc[(self.slice_df.index >= self.pulse_center) & (self.slice_df.index <= self.data_end)]

        q5_1, q95_1 = np.percentile(first_half.iloc[:, 0], [5, 95])
        q5_2, q95_2 = np.percentile(second_half.iloc[:, 0], [5, 95])

        trigger_start = q5_1 + (q95_1 - q5_1) / 5
        trigger_end = q5_2 + (q95_2 - q5_2) * 4 / 5

        #Recalculating real pulse start and end
        self.pulse_start = (first_half.iloc[:, 0] >= trigger_start).iloc[::-1].idxmin()
        self.pulse_end = (second_half.iloc[:, 0] <= trigger_end).idxmax()
        self.pulse_duration = self.pulse_end - self.pulse_start

        self.plateau_region = self.slice_df.loc[(self.slice_df.index >= self.pulse_start) & (self.slice_df.index <= self.pulse_end)]
        self.pre_pulse_region = self.slice_df.loc[self.slice_df.index <= self.pulse_start]
        self.post_pulse_region = self.slice_df.loc[self.slice_df.index >= self.pulse_end]

    def _compute_region_metrics(self) -> None:
        """"Computes metrics for each region, including peak, mean values and squareness."""
        # Peak        
        self.peak_value = self.slice_df.iloc[:, 0].max()
        self.peak_time = self.slice_df.iloc[:, 0].idxmax()
        
        # Regions
        self.plateau = RegionMetrics(self.plateau_region)
        self.pre_pulse = RegionMetrics(self.pre_pulse_region)
        self.post_pulse = RegionMetrics(self.post_pulse_region)

        # Pulse Squareness
        pulse_auc = np.trapezoid(self.plateau_region.iloc[:, 0], self.plateau_region.index) - self.pre_pulse.q16 * self.pulse_duration
        above_pulse = self.plateau_region.iloc[:, 0] - self.plateau.q84
        above_pulse[above_pulse < 0] = 0
        above_pulse_auc = np.trapezoid(above_pulse, self.plateau_region.index)
        ideal_auc = (self.plateau.q84 - self.pre_pulse.q16) * self.pulse_duration
        self.pulse_squareness = (pulse_auc - above_pulse_auc) / ideal_auc if ideal_auc else 0

        # Baseline squareness
        duration_prepulse = self.pulse_start - self.data_start
        duration_postpulse = self.data_end - self.pulse_end
        before_auc = max(np.trapezoid(self.pre_pulse_region.iloc[:, 0], self.pre_pulse_region.index) - self.pre_pulse.q16 * duration_prepulse, 0)
        after_auc = max(np.trapezoid(self.post_pulse_region.iloc[:, 0], self.post_pulse_region.index) - self.post_pulse.q16 * duration_postpulse, 0)
        before_empty = duration_prepulse * (self.plateau.mean - self.pre_pulse.mean)
        after_empty = duration_postpulse * (self.plateau.mean - self.post_pulse.q16)
        self.baselines_squareness = (1 - (before_auc + after_auc) / (before_empty + after_empty))

        # Total Squareness
        self.squareness = self.pulse_squareness * self.baselines_squareness


    def _get_channel_status(self) -> None:
        """ Determines which channel/valve is active at the pulse center time, its value, and how long it has been active."""

        def get_open_status(df: pd.DataFrame, query_time: float, columns: list[str] | None = None):
            """
            Given a DataFrame with time-indexed or time-column data, returns which channel/valve is 
            active (non-zero or True), its value, and how long it has been active.

            Returns: ['channel', 'value', 'open_time'] or [None, None, -1] if none active.
            """

            # 1) Ensure we have a 'Time' column
            if 'Time' not in df.columns:
                # If index is named 'Time', bring it out. Otherwise this will NOT create 'Time'!
                df = df.reset_index()
                if 'Time' not in df.columns:
                    raise KeyError(
                        "No 'Time' column found. If your time is in the index, name the index 'Time' before calling, "
                        "or ensure the DataFrame has a 'Time' column."
                    )

            # 2) Sort by time
            df = df.sort_values('Time').reset_index(drop=True)

            # 3) Default columns to all except 'Time'
            if columns is None:
                columns = [c for c in df.columns if c != 'Time']

            # 4) Take all rows up to query_time
            valid_rows = df[df['Time'] <= query_time]
            if valid_rows.empty:
                print(f"Warning! No data available before {query_time}s")
                return [None, None, -1]

            current_row = valid_rows.iloc[-1]

            # 5) Determine active columns (True or non-zero numeric)
            active_cols = []
            # print(f"Checking channels at {query_time}s: ", end="")
            for c in columns:
                val = current_row[c]
                # Treat NaN as inactive
                if pd.isna(val):
                    continue

                if isinstance(val, (bool, np.bool_)):
                    if bool(val):
                        active_cols.append(c)
                # Accept both Python and NumPy numeric scalars
                elif isinstance(val, (numbers.Number, np.number)):
                    if float(val) != 0.0: # type: ignore
                        active_cols.append(c)
                # else: non-numeric/non-bool → treat as inactive and ignore

            if len(active_cols) > 1:
                print(f"Warning! At {query_time:.3f}s there is more than one channel open: {active_cols}")

            if not active_cols:
                print(f"Warning! At {query_time:.3f}s there is no channel open")
                return [None, None, -1]

            # 6) Find open duration for the first active column
            col = active_cols[0]
            current_time = current_row['Time']

            df_slice = df[df['Time'] <= current_time]

            # Walk backward to find the last transition from inactive → active
            open_time = df_slice.iloc[-1]['Time']  # default to current_time
            for i in range(len(df_slice) - 2, -1, -1):
                prev_val = df_slice.iloc[i][col]
                # NaN → treat as inactive boundary
                if pd.isna(prev_val):
                    open_time = df_slice.iloc[i + 1]['Time']
                    break
                if isinstance(prev_val, (bool, np.bool_)):
                    if not bool(prev_val):
                        open_time = df_slice.iloc[i + 1]['Time']
                        break
                elif isinstance(prev_val, (numbers.Number, np.number)):
                    if float(prev_val) == 0.0:
                        open_time = df_slice.iloc[i + 1]['Time']
                        break
                elif i == 0:
                    open_time = df_slice.iloc[0]['Time']

            # Keep original value type (don’t force int)
            val = current_row[col]
            return [col, val, open_time]

        self.active_channel, _, self.t_active_channel = get_open_status(pd.DataFrame(self.register_dict['ChannelsTargetFlow']), self.pulse_center, columns=['Channel0', 'Channel1', 'Channel2'])
        self.dt_active_channel = self.pulse_start - self.t_active_channel if self.t_active_channel != -1 else -1
        
        endvalve = get_open_status(pd.DataFrame(self.register_dict['EndValveState']), self.pulse_center)
        self.endvalve_state = endvalve[1]
        self.t_active_endvalve = endvalve[2]
        self.dt_active_endvalve = self.pulse_start - self.t_active_endvalve if self.t_active_endvalve != -1 else -1

        self.created_by_envalve = self.dt_active_endvalve < self.dt_active_channel if self.active_channel else False


    def is_valid(self) -> bool:
        """Returns True if the pulse is valid based on several criteria. Criteria include:  
        - Plateau mean must be >5% above both pre-pulse and post-pulse means.
        - End valve must be active at pulse center.
        - An active channel must be detected at pulse center.
        """
        return (
            self.plateau.mean > self.pre_pulse.mean * 1.05 and
            self.plateau.mean > self.post_pulse.mean * 1.05 and
            self.endvalve_state and
            self.active_channel is not None
        )
    
    def plot(self, ax=None, color='blue', draw_regression=True) -> None:
        """
        Plot the pulse with baseline-corrected signal.
        
        Parameters
        ----------
        ax : matplotlib axis, optional
        color : str, blue by default
        draw_regression : bool
            If True, also draw the linear regression fits for:
            - pre_pulse region
            - plateau region
            - post_pulse region
            using the slopes computed in _compute_region_metrics().
        """

        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 5))

        slice_df = self.slice_df
        y = slice_df.iloc[:, 0] - self.pre_pulse.mean
        aligned_time = slice_df.index.to_numpy() - self.pulse_start

        # Trigger time
        trigger_time = (
            self.pulse_start -
            (self.t_active_endvalve if self.created_by_envalve else self.t_active_channel)
            
        )
        

        # Main signal
        ax.plot(aligned_time, y, label='Analog Signal', color=color)

        # Markers
        ax.axvline(0, color='0', linestyle='--', label='PulseStart')
        ax.axvline(self.pulse_end - self.pulse_start, color='0', linestyle='--', label='PulseEnd')
        ax.axhline(self.plateau.q84 - self.pre_pulse.mean,   color='0.4', linestyle=':', label='Plateau')
        ax.axhline(self.pre_pulse.q16 - self.pre_pulse.mean, color='0.2', linestyle=':', label='Pre-pulse')
        ax.axhline(self.post_pulse.q16 - self.pre_pulse.mean, color='0.6', linestyle=':', label='Post-Pulse')
        ax.axvline(-trigger_time, color=color, linestyle='--', label=f'Trigger (Delay={trigger_time*1000:.0f}ms)')

        # ------------------------------------------------------------
        # OPTIONAL: DRAW REGRESSION LINES
        # ------------------------------------------------------------  
        if draw_regression:

            # mapping region names → attributes
            region_map = {
                "Pre-pulse": ("pre_pulse",  "pre_pulse_region",  "tab:orange"),
                "Plateau":   ("plateau",    "plateau_region",    "tab:green"),
                "Post-pulse":("post_pulse", "post_pulse_region", "tab:red"),
            }

            for region_name, (metrics_attr, df_attr, color_reg) in region_map.items():
                reg = getattr(self, metrics_attr)
                df = getattr(self, df_attr)
                x = df.index.values
                intercept = (reg.mean - self.pre_pulse.mean) - reg.corrected_slope * x.mean()
                
                y_fit = reg.corrected_slope * x + intercept # fitted line
                ax.plot(
                    x - self.pulse_start,
                    y_fit,
                    linestyle='--',
                    linewidth=2,
                    color=color_reg,
                    label=f"{region_name} fit (Slope={reg.corrected_slope:.2f})"
                )

        # Labels, style
        ax.set_title(f"Channel: {self.active_channel} | Envalve: {self.created_by_envalve} | Sq={self.squareness:.2f}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Signal (baseline corrected)")
        ax.grid(True, linestyle=':', alpha=0.5)
        ax.legend()

    def to_dict(self) -> dict:
        """Used to convert the Pulse object to a dictionary for easier analysis and grouping.""" 
        return self.__dict__

class PulseGroup:
    def __init__(self, 
                 pulses: list, 
                 channel:str, 
                 pulses_created_by_envalve:bool,
                 channel_cfg:dict, 
                 rig_info:dict, 
                 metrics:list[str]|None = None
                 ):
        
        self.pulses = pulses
        self.channel = channel
        self.rig = rig_info["computer_name"]
        self.computer = rig_info["rig_name"]
        self.odorant = channel_cfg[channel]['odorant'] if channel_cfg else 'Unknown'
        self.flow = channel_cfg[channel]['flow_rate'] if channel_cfg else np.nan
        self.dilution = channel_cfg[channel]['odorant_dilution'] if channel_cfg else np.nan
        self.pulses_created_by_envalve = pulses_created_by_envalve
        self.metrics = {}
        self.aggregate_metrics(['plateau.median', 'pre_pulse.mean'])
        if metrics: self.aggregate_metrics(metrics)
        self.baseline = self.metrics['pre_pulse.mean']['mean']
        self.plateau_value = self.metrics['plateau.median']['mean']

    def __iter__(self) -> iter:
        return iter(self.pulses)

    def __len__(self) -> int:
        return len(self.pulses)

    def __repr__(self) -> str:
        return f"<PulseGroup. {len(self)} pulses from {(self.channel)}-Endvalve:{self.pulses_created_by_envalve}. Plateau={self.plateau_value}>"

    def aggregate_metrics(self, metrics_list: list[str]):
        """
        Computes mean and std of a nested attribute across all pulses in this group.
        Updates self.metrics and returns a dictionary keyed by each metric.
        """
        def get_nested_attr(obj, path) -> object:
            for part in path.split('.'):
                obj = getattr(obj, part)
            return obj

        metrics_by_group = {}
        for metric in metrics_list:
            values = []
            for pulse in self.pulses:
                try:
                    val = get_nested_attr(pulse, metric)
                    values.append(val)
                except AttributeError:
                    print(f"Attribute {metric} not found in pulse")
                    continue

            if values:
                arr = np.array(values)
                mean = float(np.mean(arr))
                std = round(float(np.std(arr)), 2)
                p16, p84 = np.percentile(values, [16, 84])
                err_minus =  max(0, float(mean - p16))
                err_plus  = max(0,float(p84 - mean))
                metrics_by_group[metric] = {'mean': mean, 'std': std, 
                                            'err_minus': err_minus, 'err_plus': err_plus}
            else:
                metrics_by_group[metric] = {'mean': None, 'std': None, 'err_minus': None, 'err_plus': None}

        # Merge into self.metrics
        self.metrics.update(metrics_by_group)
        return metrics_by_group

# -----------------------------
# Processing data
# -----------------------------

def normalize_timestamp(series: pd.Series, anchor:float, offset: float = 0.0) -> pd.Series:
    """Normalize the timestamp to start at a specific value by shifting the index based on an anchor point."""  
    normalized = series.copy()
    normalized.index = (normalized.index - anchor) + offset
    return normalized

def preprocess_signal(signal: pd.Series, rate: int = 250) -> tuple[np.ndarray, np.ndarray]:
    """
    Resample and smooth a time-indexed signal.
    The signal is interpolated to a uniform time grid at the specified sampling rate (in Hz, default is 250) 
    and then smoothed using a Gaussian filter.

    """

    from scipy.ndimage import gaussian_filter1d

    timestamp: pd.Index = signal.index
    analog_signal: np.ndarray = signal.values

    new_time = np.arange(timestamp[0], timestamp[-1], 1 / rate)# Resample to new rate ---
    resampled = np.interp(new_time, timestamp, analog_signal) # Interpolate signal to new time grid
    smoothed = gaussian_filter1d(resampled, sigma=100) # Apply filter
    return new_time, smoothed

def detect_pulses(first_der: np.ndarray, t: np.ndarray, prominence: float = 10.0) -> pd.DataFrame:
    """
    Detect pulse start and end points from a signal derivative.
    Local maxima → pulse starts, local minima → pulse ends, using a prominence threshold.

    Parameters
    ----------
    first_der : np.ndarray
        First derivative of the signal.
    t : np.ndarray
        Time or index values corresponding to `first_der`.
    prominence : float, optional
        Prominence threshold passed to `scipy.signal.find_peaks`.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by `t` with boolean columns indicating `PulseStart` and `PulseEnd`.
    """
    from scipy.signal import find_peaks
    max_idx, _ = find_peaks(first_der, prominence=prominence)
    min_idx, _ = find_peaks(-first_der, prominence=prominence)

    df = pd.DataFrame({"PulseStart": False, "PulseEnd": False}, index=t)
    df.loc[t[max_idx], "PulseStart"] = True
    df.loc[t[min_idx], "PulseEnd"] = True

    if verbose:
        # --- Print summary ---
        print(f"Found {len(max_idx)} maxima and {len(min_idx)} minima")

        # --- Plot results ---
        plt.figure(figsize=(10, 5))
        plt.plot(t, first_der, color='blue')
        plt.scatter(t[max_idx], first_der[max_idx], color='red', label='Local Maxima', zorder=5)
        plt.scatter(t[min_idx], first_der[min_idx], color='green', label='Local Minima', zorder=5)

        plt.title(f'Local Maxima & Minima')
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return df

def slice_pulses(signal: pd.Series, register_dict: dict, pulses_df: pd.DataFrame) -> list[Pulse]:
    """
    Extract Pulse objects from a signal using pulse start/end markers. Invalid pulses are skipped.

    Parameters
    ----------
    signal : pd.Series
        Time-indexed signal data from which pulses are sliced.
    register_dict : dict
        Register values associated with the signal, passed to each Pulse.
    pulses_df : pd.DataFrame
        DataFrame indexed like `signal`, containing boolean columns
        `PulseStart` and `PulseEnd`.

    Returns
    -------
    list[Pulse]
        List of valid Pulse objects extracted from the signal.
    """
    pulses = []
    invalid_pulses = []

    for pulse_start in pulses_df.index[pulses_df['PulseStart']]:
        next_pulse_end = pulses_df.index[(pulses_df.index > pulse_start) & pulses_df['PulseEnd']]
        if len(next_pulse_end) == 0:
            continue
        pulse_end = next_pulse_end[0]

        previous_pulse_end = pulses_df.index[(pulses_df.index < pulse_start) & pulses_df['PulseEnd']].max() if len(pulses_df.index[(pulses_df.index < pulse_start) & pulses_df['PulseEnd']]) > 0 else None
        next_pulse_start = pulses_df.index[(pulses_df.index > pulse_end) & pulses_df['PulseStart']].min() if len(pulses_df.index[(pulses_df.index > pulse_end) & pulses_df['PulseStart']]) > 0 else None

        pulse = Pulse(signal, register_dict, pulse_start, pulse_end, next_pulse_start, previous_pulse_end)
        if pulse.is_valid():
            pulses.append(pulse)
        else:
            invalid_pulses.append(pulse)

    if verbose: 
        print(f"Number of valid pulses: {len(pulses)}/{len(pulses)+len(invalid_pulses)}")
        if len(pulses):
            n_pulse = min(42, len(pulses))
            print(f"Example of pulse number {n_pulse}: {pulses[n_pulse-1]}")
            pulses[n_pulse-1].plot()
    
    return pulses

def group_pulses(pulses: list[Pulse], metrics: list[str], channel_cfg: dict, rig_info: dict
                 ) -> dict[str, PulseGroup]:
    """ 
    Group pulses by (channel, envalve) in a dict { "ChannelX-Endvalve:Bool": PulseGroup, ...} and compute aggregate metrics for each group

    Parameters
    ----------
    pulses : list[Pulse]
        Pulses to group.
    metrics : list[str]
        Metrics to compute for each group.
    channel_cfg : dict
        Channel configuration passed to each PulseGroup.
    rig_info : dict
        Rig information passed to each PulseGroup.

    Returns
    -------
    dict[str, PulseGroup]
        Mapping from group key to PulseGroup.
    """
    grouped = defaultdict(list)
    for pulse in pulses:
        grouped[(pulse.active_channel, pulse.created_by_envalve)].append(pulse)

    # Create PulseGroup instances and assign index to each pulse
    groups = {}
    for (channel, pulses_created_by_envalve), pulses_in_group in grouped.items():
        key = f"{str(channel)}-Endvalve:{str(pulses_created_by_envalve)}"
        for idx, pulse in enumerate(pulses_in_group):
            pulse.idx = idx
        groups[key]=PulseGroup(pulses_in_group, channel, pulses_created_by_envalve, channel_cfg, rig_info, metrics=metrics)
    return groups

def linear_regression(x: list[float], y: list[float]) -> tuple[float, float, float]:
    """
    Compute simple linear regression for y = m*x + n.

    Returns:
        m (float): slope
        n (float): intercept
        r2 (float): coefficient of determination
    """
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    n_x = len(x)
    if n_x < 2:
        raise ValueError("At least two data points are required for linear regression")

    mean_x = sum(x) / n_x
    mean_y = sum(y) / n_x

    ss_xy = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    ss_xx = sum((xi - mean_x) ** 2 for xi in x)
    
    if ss_xx == 0:
        raise ValueError("Variance of x is zero (cannot compute slope)")
    m = ss_xy / ss_xx
    n = mean_y - m * mean_x

    ss_res = sum((yi - (m * xi + n)) ** 2 for xi, yi in zip(x, y))
    ss_tot = sum((yi - mean_y) ** 2 for yi in y)

    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else (1.0 if ss_res == 0 else 0.0)
    return m, n, r2

# -----------------------------
# Saving data
# -----------------------------
def save_to_csv(groups_of_pulses: dict, csv_name: str) -> pd.DataFrame:
    rows = []

    for group in groups_of_pulses.values():
        # Base metadata
        all_info_dict = {
            "rig": group.rig,
            "computer": group.computer,
            "channel": group.channel,
            "odorant": group.odorant,
            "flow": group.flow,
            "dilution": group.dilution,
            "envalve": group.pulses_created_by_envalve,
        }

        # Add metrics
        for metric, summary in group.metrics.items():
            all_info_dict[metric] = summary["mean"]
            all_info_dict[f"{metric}_std"] = summary["std"]

        rows.append(all_info_dict)

    df = pd.DataFrame(rows)
    df.to_csv(f"../saved/{csv_name}.csv", index=False)
    return df
