import json
import os
import copy


def ensuredirs(path_or_obj):
    """遞迴建立所有路徑嘅父目錄"""
    if isinstance(path_or_obj, str):
        dirname = os.path.dirname(path_or_obj)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        return
    if isinstance(path_or_obj, dict):
        for v in path_or_obj.values():
            ensuredirs(v)
        return
    if isinstance(path_or_obj, (list, tuple)):
        for v in path_or_obj:
            ensuredirs(v)


def _deep_merge(base: dict, override: dict) -> dict:
    """
    遞迴深度合併兩個 dict。
    override 嘅值會覆蓋 base，但 nested dict 會遞迴合併而唔係直接替換。
    """
    result = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _resolve_env_keys(config: dict) -> dict:
    """
    將 config 入面以 _env 結尾嘅 key 自動從環境變量讀取。
    例如 "api_key_env": "SIMFIN_API_KEY" → "api_key": os.environ["SIMFIN_API_KEY"]
    """
    # 嘗試載入 .env file（如果有 python-dotenv）
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    resolved = {}
    for key, value in config.items():
        if isinstance(value, dict):
            resolved[key] = _resolve_env_keys(value)
        elif isinstance(key, str) and key.endswith("_env"):
            real_key = key[:-4]  # 去掉 _env 後綴
            env_val = os.environ.get(value, "")
            if not env_val:
                print(f"⚠️  環境變量 {value} 未設定，{real_key} 將為空字串")
            resolved[real_key] = env_val
        else:
            resolved[key] = value
    return resolved


def load_config(config_path="config.json"):
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
            "extra_etfs": ["SPY"],
            "use_historical": False,
        },
        "data_quality": {
            "use_clean_universe": False,
            "max_missing_pct": 0.2,
            "max_spike_count": 5,
        },
        "paths": {
            "raw_data": "data/raw",
            "processed_data": "data/processed",
            "prices_dir": "data/prices",
            "universe_file": "data/universe.csv",
            "clean_universe_file": "data/clean_universe.csv",
            "log_file": "logs/system.log",
        },
        "cost_model": {},
    }

    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                file_config = json.load(f)
            # ✅ 修正：用遞迴深度合併取代 shallow update
            default_config = _deep_merge(default_config, file_config)
        except Exception as e:
            print(f"Warning: 無法讀取 {config_path}: {e}")

    # ✅ 修正：自動解析環境變量
    default_config = _resolve_env_keys(default_config)

    return default_config


def loadconfig(_config_path=None):
    return load_config()
