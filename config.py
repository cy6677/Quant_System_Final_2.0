import json
import os

def ensuredirs(path_or_obj):
    if isinstance(path_or_obj, str):
        # 只建立父目錄，唔建立檔案本身
        dirname = os.path.dirname(path_or_obj)
        if dirname:  # 如果有目錄部分
            os.makedirs(dirname, exist_ok=True)
        return
    if isinstance(path_or_obj, dict):
        for v in path_or_obj.values():
            ensuredirs(v)
        return
    if isinstance(path_or_obj, (list, tuple)):
        for v in path_or_obj:
            ensuredirs(v)

def load_config(config_path="config.json"):
    # 預設設定（作為 fallback）
    default_config = {
        "data": {
            "price": {
                "start_date": "2015-01-01",
                "end_date": None,
            }
        },
        "backtest": {
            "initial_capital": 100000.0,
            "calendar_ticker": "SPY",
            "allow_fractional": True,
            "commission_rate": 0.001,
            "slippage": 0.001,
            "min_commission": 1.0,
        },
        "universe": {
            "extraetfs": ["SPY"],
            "usehistorical": False,
        },
        "dataquality": {
            "usecleanuniverse": False,
        },
        "paths": {
            "rawdata": "data/raw",
            "pricesdir": "data/prices",
            "prices_dir": "data/prices",
            "cleanuniversefile": "data/clean/universe.csv",
            "log_file": "logs/system.log",   # 補上預設 log_file
        },
        "cost_model": {}
    }
    
    # 如果 config.json 存在，讀取並合併
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                file_config = json.load(f)
            # 簡單合併（只 update 第一層）
            default_config.update(file_config)
            # 但對於 paths 呢啲 nested dict，需要深度合併
            # 以下做簡單遞迴合併（你可以根據需要加強）
            for key, value in file_config.items():
                if key in default_config and isinstance(default_config[key], dict) and isinstance(value, dict):
                    default_config[key].update(value)
                else:
                    default_config[key] = value
        except Exception as e:
            print(f"Warning: 無法讀取 {config_path}: {e}")
    
    return default_config

def loadconfig(_config_path=None):
    return load_config()
