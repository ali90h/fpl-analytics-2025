"""
FPL Prediction Module
Modular architecture for FPL analytics and predictions
"""

from .data_processor import DataProcessor
from .core_predictor import CorePredictor

__version__ = "1.0.0"
__all__ = ["DataProcessor", "CorePredictor"]
