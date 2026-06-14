"""Export utilities for timescore metric results.

Convert metric results to various formats for analysis and reporting.
"""

import json
from typing import Dict, Union


def to_dict(results: Dict[str, "torch.Tensor"]) -> Dict[str, float]:
    """Convert metric results to a plain Python dict with float values.

    Args:
        results: dict mapping metric name -> scalar tensor

    Returns:
        dict mapping metric name -> float
    """
    import torch
    out = {}
    for key, val in results.items():
        if isinstance(val, torch.Tensor):
            out[key] = val.item()
        elif isinstance(val, dict):
            out[key] = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in val.items()}
        else:
            out[key] = val
    return out


def to_dataframe(results: Dict[str, "torch.Tensor"]):
    """Convert metric results to a pandas DataFrame.

    Args:
        results: dict mapping metric name -> scalar tensor or dict

    Returns:
        pandas DataFrame with metrics as rows
    """
    import pandas as pd

    rows = []
    for key, val in results.items():
        import torch
        if isinstance(val, torch.Tensor):
            if val.ndim == 0:
                rows.append({"metric": key, "value": val.item()})
            else:
                for i, v in enumerate(val.flatten()):
                    rows.append({"metric": f"{key}_{i}", "value": v.item()})
        elif isinstance(val, dict):
            for sub_key, sub_val in val.items():
                v = sub_val.item() if isinstance(sub_val, torch.Tensor) else sub_val
                rows.append({"metric": f"{key}[{sub_key}]", "value": v})
        else:
            rows.append({"metric": key, "value": val})

    return pd.DataFrame(rows).set_index("metric")


def to_json(results: Dict[str, "torch.Tensor"], path: str = None, indent: int = 2) -> Union[str, None]:
    """Export metric results to JSON.

    Args:
        results: dict mapping metric name -> scalar tensor or dict
        path: file path to save JSON (if None, returns JSON string)
        indent: JSON indentation

    Returns:
        JSON string if path is None, else None (writes to file)
    """
    data = to_dict(results)

    json_str = json.dumps(data, indent=indent, ensure_ascii=False)

    if path is not None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(json_str)
        return None
    return json_str


def to_csv(results: Dict[str, "torch.Tensor"], path: str = None) -> Union[str, None]:
    """Export metric results to CSV.

    Args:
        results: dict mapping metric name -> scalar tensor or dict
        path: file path to save CSV (if None, returns CSV string)

    Returns:
        CSV string if path is None, else None (writes to file)
    """
    df = to_dataframe(results)

    if path is not None:
        df.to_csv(path)
        return None
    return df.to_csv()
