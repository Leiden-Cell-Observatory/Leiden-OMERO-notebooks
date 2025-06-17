"""OMERO Annotate AI: Integration of AI annotation tools with OMERO for automated image segmentation."""

from .core.config import AnnotationConfig, load_config, create_default_config
from .core.pipeline import create_pipeline, AnnotationPipeline
from .widgets.config_widget import create_config_widget

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

__all__ = [
    "AnnotationConfig",
    "load_config", 
    "create_default_config",
    "create_pipeline",
    "AnnotationPipeline",
    "create_config_widget",
]