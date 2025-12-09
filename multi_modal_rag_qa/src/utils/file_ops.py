"""
File Operations Utility
Helper functions for file and directory operations.
"""

import json
import pickle
from pathlib import Path
from typing import Any, Dict
import shutil


def ensure_dir(directory: str) -> Path:
    """
    Ensure a directory exists, create if it doesn't.

    Args:
        directory: Directory path

    Returns:
        Path object
    """
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: Dict[str, Any], filepath: str):
    """
    Save data to JSON file.

    Args:
        data: Data to save
        filepath: Path to save file
    """
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


def load_json(filepath: str) -> Dict[str, Any]:
    """
    Load data from JSON file.

    Args:
        filepath: Path to JSON file

    Returns:
        Loaded data
    """
    with open(filepath, "r") as f:
        return json.load(f)


def save_pickle(data: Any, filepath: str):
    """
    Save data to pickle file.

    Args:
        data: Data to save
        filepath: Path to save file
    """
    with open(filepath, "wb") as f:
        pickle.dump(data, f)


def load_pickle(filepath: str) -> Any:
    """
    Load data from pickle file.

    Args:
        filepath: Path to pickle file

    Returns:
        Loaded data
    """
    with open(filepath, "rb") as f:
        return pickle.load(f)


def clear_directory(directory: str, keep_gitkeep: bool = True):
    """
    Clear all files in a directory.

    Args:
        directory: Directory to clear
        keep_gitkeep: Whether to keep .gitkeep files
    """
    path = Path(directory)
    if not path.exists():
        return

    for item in path.iterdir():
        if keep_gitkeep and item.name == ".gitkeep":
            continue

        if item.is_file():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)


def get_file_size(filepath: str) -> int:
    """
    Get file size in bytes.

    Args:
        filepath: Path to file

    Returns:
        File size in bytes
    """
    return Path(filepath).stat().st_size
