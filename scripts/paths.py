#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Helper module for managing file paths in the reorganized project structure
"""

import os
from pathlib import Path

# Get the project root directory (parent of scripts/)
PROJECT_ROOT = Path(__file__).parent.parent

# Define paths for different resource types
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
CONFIG_DIR = PROJECT_ROOT / "config"
SRC_DIR = PROJECT_ROOT / "src"

def get_model_path(filename: str) -> str:
    """Get path to a model file"""
    return str(MODELS_DIR / filename)

def get_data_path(filename: str) -> str:
    """Get path to a data file"""
    return str(DATA_DIR / filename)

def get_config_path(filename: str = "config.yaml") -> str:
    """Get path to a config file"""
    return str(CONFIG_DIR / filename)

def get_db_path(filename: str = "turnover_data.db") -> str:
    """Get path to database file"""
    return str(DATA_DIR / filename)



