import harp
import os
import pandas as pd 
import json
from pathlib import Path

verbose = True

# -----------------------------
# Loading data from json files
# -----------------------------

def _read_json(path: str) -> dict:
    """Read a JSON file and return its contents as a dict."""
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Required file not found: {path}") from e
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format in file: {path}") from e

def load_channel_info(base_path: str) -> tuple[dict[str], bool] :
    """  Read task logic and/or rig input JSON files to extract channel configuration and rig info and returns them as a dict. 
    It also detects whether the task is a calibration task and returns it as a bool.   """
    
    tsk_lgc = _read_json(f"{base_path}/behavior/Logs/tasklogic_input.json")
 
    channel_cfg = {}
    if tsk_lgc.get("name") == "OlfactometerCalibration" and "channel_config" in tsk_lgc.get("task_parameters", {}):
        # before v1, the channel config was in the tasklogic json on calibration tasks. We want to keep supporting this format for old sessions, but for new ones we moved the channel config to the rig input json, so now we need to check both places.
        is_calibration = True
        for num in range(0, 4):
            try:
                channel_cfg["Channel"+str(num)] = {}  # initialize as dict
                channel_cfg["Channel"+str(num)]["odorant"] = tsk_lgc["task_parameters"]["channel_config"][str(num)]["odorant"]
                channel_cfg["Channel"+str(num)]["flow_rate"] = tsk_lgc["task_parameters"]["channel_config"][str(num)]["flow_rate"]
                channel_cfg["Channel"+str(num)]["odorant_dilution"] = tsk_lgc["task_parameters"]["channel_config"][str(num)]["odorant_dilution"]
            except:
                print(f"Error when reading channel{num} in the tasklogic json")

    elif tsk_lgc.get("name") == "OlfactometerCalibration":
        is_calibration = True
        rig_data = _read_json(f"{base_path}/behavior/Logs/rig_input.json")
        for num in range(0, 4):
            try:
                channel_cfg["Channel"+str(num)] = {}
                channel_cfg["Channel"+str(num)]["odorant"] = rig_data["harp_olfactometer"]["calibration"]["channel_config"][str(num)]["odorant"]
                channel_cfg["Channel"+str(num)]["flow_rate"] = rig_data["harp_olfactometer"]["calibration"]["channel_config"][str(num)]["flow_rate"]
                channel_cfg["Channel"+str(num)]["odorant_dilution"] = rig_data["harp_olfactometer"]["calibration"]["channel_config"][str(num)]["odorant_dilution"]
            except:
                print(f"Error when reading channel{num} in the rig input json for a calibration session")
        
        if "harp_olfactometer_extension" in rig_data:
            for i, multiplexed_olfactometer in enumerate(rig_data["harp_olfactometer_extension"]):
                if "calibration" in multiplexed_olfactometer and "channel_config" in multiplexed_olfactometer["calibration"]:
                    if verbose: print("Found multiplexed olfactometer. Loading extra channels")
                    for num in range(0, 4):
                        try:
                            channel_cfg["Channel"+str(num+4*(i+1))] = {}
                            channel_cfg["Channel"+str(num+4*(i+1))]["odorant"] = multiplexed_olfactometer["calibration"]["channel_config"][str(num)]["odorant"]
                            channel_cfg["Channel"+str(num+4*(i+1))]["flow_rate"] = multiplexed_olfactometer["calibration"]["channel_config"][str(num)]["flow_rate"]
                            channel_cfg["Channel"+str(num+4*(i+1))]["odorant_dilution"] = multiplexed_olfactometer["calibration"]["channel_config"][str(num)]["odorant_dilution"]
                        except:
                            print(f"Error when reading channel{num} in the harp_olfactometer_extension calibration config in the rig input json for a calibration session")

    else:
        is_calibration = False
        rig_data = _read_json(f"{base_path}/behavior/Logs/rig_input.json")
        for num in range(0, 4):
            print("aaaa", rig_data["harp_olfactometer"]["calibration"]["input"]["channel_config"])
            try:
                channel_cfg["Channel"+str(num)] = {}  
                channel_cfg["Channel"+str(num)]["odorant"] = rig_data["harp_olfactometer"]["calibration"]["input"]["channel_config"][str(num)]["odorant"]
                channel_cfg["Channel"+str(num)]["flow_rate"] = rig_data["harp_olfactometer"]["calibration"]["input"]["channel_config"][str(num)]["flow_rate"]
                channel_cfg["Channel"+str(num)]["odorant_dilution"] = rig_data["harp_olfactometer"]["calibration"]["input"]["channel_config"][str(num)]["odorant_dilution"]
            except:
                print(f"Error when reading channel{num} in the rig input json for a real session")

    # --- PRINT INFO ---
    if verbose:   print(f"Channels - {channel_cfg}")

    return channel_cfg, is_calibration

def load_rig_info(base_path: str) -> dict[str, str] :
    """  Read rig input JSON file to extract rig info and returns it as a dict. 
    Returns
    -------
    dict
        {"computer_name": "DTXXXXXXX", "rig_name": "YY"}
    """
    # LOAD RIG INFO (COMPUTER AND RIG NAME) 
    rig_data = _read_json(f"{base_path}/behavior/Logs/rig_input.json")
    
    rig_info = {}  # initialize as dict
    rig_info["computer_name"] = rig_data["computer_name"]
    rig_info["rig_name"] = rig_data["rig_name"]

    # --- PRINT INFO ---
    if verbose:
        print(f"Loaded rig info: Computer - {rig_info['computer_name']}, Rig - {rig_info['rig_name']}")

    return  rig_info

def load_sw_register(base_path: str, register: str) -> pd.DataFrame:
    """
    Reads a register, such as ActivePatch, ActiveSite, or any of the others located in the <base_path>/behavior/SoftwareEvents directory. 
    Returns a dataframe with these columns: 'timestamp' and the flattened keys from 'data' 
    """ 

    rows = []
    path = Path(base_path) / "behavior" / "SoftwareEvents" / f"{register}.json"
    with path.open("r") as f:
        for line in f:
            obj = json.loads(line)
            row = {"timestamp": obj["timestamp"]}
            row.update(obj["data"])
            rows.append(row)
    return pd.DataFrame(rows)

# -----------------------------
# Loading data from harp devices
# -----------------------------

def load_olf_registers(base_path: str) -> dict[str, pd.DataFrame|pd.Series]:
    """
    Reads the registers from the harp files and returns a dictionary, where the keys correspond to olfactometer registers and are pd.DataFrames.
    """ 
    register_dict = {}
    OlfReader = harp.create_reader(base_path + "/behavior/Olfactometer.harp", include_common_registers=False)

    for name, reg in OlfReader.registers.items():
        register_dict[name] = reg.read()

    dirs = [entry.name for entry in os.scandir(base_path + "/behavior") if entry.is_dir()]
    for i in range(1, 10):
        if f"OlfactometerExtension{i}.harp" in dirs:
            if verbose: print(f"Found OlfactometerExtension{i}. Loading its registers and saving them as *_m{i}")
            MultiplexReader = harp.create_reader(base_path + f"/behavior/OlfactometerExtension{i}.harp", include_common_registers=False)
            for name, reg in MultiplexReader.registers.items():
                register_dict[name+f"_m{i}"] = reg.read()

    return register_dict

def load_olf_and_alog(base_path: str, alog_channel: str = 'Channel0') -> dict[str, pd.DataFrame|pd.Series]:
    """
    Reads the registers from the harp files and returns a dictionary, where:
        -The 'Analog' key contains a pd.Series of the PID signal recorded from <channel>, 
        -The other keys correspond to olfactometer registers and are pd.DataFrames.
    """ 
    register_dict = load_olf_registers(base_path)
    AlogReader = harp.create_reader(base_path+ "/behavior/AnalogInput.harp", include_common_registers=False)
    register_dict['Analog'] = AlogReader.AnalogData.read()[alog_channel]

    return register_dict

def available_devices(base_path: str) -> list[str]:
    """Reads the behavior folder and returns a list of possible harp device files to read registers from. 
    This is useful to detect which registers are available in a given folder, as some sessions may not have all the registers""" 
    devices = []
    for file in os.listdir(base_path + "/behavior"):
        if file.endswith(".harp"):
            devices.append(file)
    
    return devices

def possible_registers(base_path: str, harp_name: str) -> list[str]:
    """Returns a list of possible registers to read from a given HARP device folder. Useful to detect which registers are available in a given device."""
    registers = read_harp_register(base_path, harp_name, register_name=None)  # This will raise an error if the device doesn't exist
    return sorted(registers.keys())

def read_harp_register(
    base_path: str,
    harp_name: str,
    register_name: str | list[str] | None = None,
    include_common_registers: bool = False, # Optional: whether to include common registers in the output 
    always_dict: bool = False,  # optional: force dict return even for a single register
    ) -> pd.DataFrame | dict[str, pd.DataFrame]:
    """
    Read one, many, or all registers from a HARP device folder.

    register_name:
      - None  -> read all -> dict[str, DataFrame]
      - str   -> read one -> DataFrame (unless always_dict=True -> dict)
      - list  -> read many -> dict[str, DataFrame]
    """

    device_path = os.path.join(base_path, "behavior", harp_name)
    reader = harp.create_reader(device_path, include_common_registers=include_common_registers)

    def _read_one(name: str) -> pd.DataFrame:
        try:
            return reader.registers[name].read()
        except KeyError as e:
            available = ", ".join(sorted(reader.registers.keys()))
            raise KeyError(f"Register '{name}' not found. Available: [{available}]") from e

    if register_name is None:
        out = {name: reg.read() for name, reg in reader.registers.items()}
        return out

    if isinstance(register_name, str):
        df = _read_one(register_name)
        return {register_name: df} if always_dict else df

    if isinstance(register_name, (list, tuple)):
        missing = [n for n in register_name if n not in reader.registers]
        if missing:
            available = ", ".join(sorted(reader.registers.keys()))
            raise KeyError(f"Missing registers: {missing}. Available: [{available}]")
        return {name: _read_one(name) for name in register_name}

    raise TypeError("register_name must be None, a string, or a list/tuple of strings.")


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



base_path = r"C:\Data\Multiplex\First try"
load_channel_info(base_path)