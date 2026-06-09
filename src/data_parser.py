import os
import pandas as pd
from pyulog import ULog

def extract_telemetry(ulog_path):
    """
    Parses a PX4 .ulog file and extracts core flight dynamics 
    into a time-synchronized Pandas DataFrame.
    """
    print(f"Loading {ulog_path}...")
    ulog = ULog(ulog_path)
    
    data_dict = {}
    
    # 1. Extract Attitude (Pitch, Roll, Yaw)
    attitude_data = ulog.get_dataset('vehicle_attitude')
    data_dict['timestamp'] = attitude_data.data['timestamp']
    data_dict['roll'] = attitude_data.data['q[0]'] # Note: You'll convert quaternions to Euler later
    data_dict['pitch'] = attitude_data.data['q[1]']
    
    # 2. Extract Battery Status (Voltage drops can indicate motor failure)
    try:
        battery_data = ulog.get_dataset('battery_status')
        # ULogs can have different sampling rates; pandas merge_asof will handle syncing later
        df_batt = pd.DataFrame({
            'timestamp': battery_data.data['timestamp'],
            'voltage_v': battery_data.data['voltage_v']
        })
    except KeyError:
        print("Warning: No battery data found in this log.")
        df_batt = pd.DataFrame()

    df_attitude = pd.DataFrame(data_dict)
    
    return df_attitude, df_batt

if __name__ == "__main__":
    # Test the parser when you get a sample log
    sample_log = "../data/raw/sample_flight.ulg"
    
    if os.path.exists(sample_log):
        df_att, df_batt = extract_telemetry(sample_log)
        print(df_att.head())
    else:
        print(f"Waiting for raw log file at {sample_log}")
