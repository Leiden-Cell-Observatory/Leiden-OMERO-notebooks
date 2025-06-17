"""Test package installation and basic imports."""

import pytest


def test_package_import():
    """Test that the package can be imported."""
    import omero_microsam
    assert omero_microsam.__version__ == "0.1.0"


def test_core_imports():
    """Test core module imports."""
    from omero_microsam import MicroSAMConfig, create_default_config, load_config
    from omero_microsam import create_pipeline, create_config_widget
    
    # Test that we can create instances
    config = create_default_config()
    assert isinstance(config, MicroSAMConfig)


def test_widget_import():
    """Test widget imports (might fail in headless environment)."""
    try:
        from omero_microsam.widgets import ConfigWidget
        from omero_microsam import create_config_widget
        # Just test import, don't create widget (requires ipywidgets/jupyter)
    except ImportError:
        pytest.skip("ipywidgets not available in test environment")


def test_config_functionality():
    """Test basic config functionality."""
    from omero_microsam import create_default_config
    
    config = create_default_config()
    
    # Test basic properties
    assert config.batch_processing.batch_size == 3
    assert config.omero.container_type == "dataset"
    assert config.image_processing.model_type == "vit_l"
    
    # Test YAML conversion
    yaml_str = config.to_yaml()
    assert "batch_processing:" in yaml_str
    
    # Test dictionary conversion
    config_dict = config.to_dict()
    assert "batch_processing" in config_dict
    
    # Test legacy params
    legacy_params = config.get_legacy_params()
    assert "batch_size" in legacy_params
    assert legacy_params["batch_size"] == 3