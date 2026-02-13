import pandas as pd 

verbose = True

# -----------------------------
# Saving data
# -----------------------------
def save_pulses_to_csv(groups_of_pulses: dict, csv_name: str) -> pd.DataFrame:
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
