"""Tests for configuration management."""

import pytest
import yaml
from omero_microsam.core.config import (
    MicroSAMConfig,
    create_default_config,
    load_config,
    get_config_template
)


class TestMicroSAMConfig:
    """Test the MicroSAMConfig class."""
    
    def test_default_config_creation(self):
        """Test creating a default configuration."""
        config = create_default_config()
        assert isinstance(config, MicroSAMConfig)
        assert config.batch_processing.batch_size == 3
        assert config.omero.container_type == "dataset"
        assert config.image_processing.model_type == "vit_l"
    
    def test_config_to_dict(self):
        """Test converting configuration to dictionary."""
        config = create_default_config()
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert "batch_processing" in config_dict
        assert "omero" in config_dict
        assert "image_processing" in config_dict
        assert "patches" in config_dict
        assert "training" in config_dict
        assert "workflow" in config_dict
    
    def test_config_to_yaml(self):
        """Test converting configuration to YAML."""
        config = create_default_config()
        yaml_str = config.to_yaml()
        
        assert isinstance(yaml_str, str)
        assert "batch_processing:" in yaml_str
        assert "omero:" in yaml_str
        
        # Test that it's valid YAML
        parsed = yaml.safe_load(yaml_str)
        assert isinstance(parsed, dict)
    
    def test_config_from_dict(self):
        """Test creating configuration from dictionary."""
        config_dict = {
            "batch_processing": {"batch_size": 5},
            "omero": {"container_type": "plate", "container_id": 123},
            "image_processing": {"model_type": "vit_h", "three_d": True}
        }
        
        config = MicroSAMConfig.from_dict(config_dict)
        
        assert config.batch_processing.batch_size == 5
        assert config.omero.container_type == "plate"
        assert config.omero.container_id == 123
        assert config.image_processing.model_type == "vit_h"
        assert config.image_processing.three_d is True
    
    def test_config_from_yaml_string(self):
        """Test creating configuration from YAML string."""
        yaml_str = """
        batch_processing:
          batch_size: 8
        omero:
          container_type: project
          container_id: 456
        """
        
        config = MicroSAMConfig.from_yaml(yaml_str)
        
        assert config.batch_processing.batch_size == 8
        assert config.omero.container_type == "project"
        assert config.omero.container_id == 456
    
    def test_config_validation_success(self):
        """Test successful configuration validation."""
        config = create_default_config()
        config.omero.container_id = 123  # Set required field
        
        # Should not raise exception
        config.validate()
    
    def test_config_validation_failures(self):
        """Test configuration validation failures."""
        config = create_default_config()
        
        # Test invalid batch size
        config.batch_processing.batch_size = -1
        with pytest.raises(ValueError, match="batch_size must be positive"):
            config.validate()
        
        # Reset and test invalid container type
        config = create_default_config()
        config.omero.container_type = "invalid"
        with pytest.raises(ValueError, match="container_type must be one of"):
            config.validate()
    
    def test_get_legacy_params(self):
        """Test conversion to legacy parameters."""
        config = create_default_config()
        config.omero.container_id = 123
        
        legacy_params = config.get_legacy_params()
        
        assert isinstance(legacy_params, dict)
        assert legacy_params["batch_size"] == 3
        assert legacy_params["container_type"] == "dataset"
        assert legacy_params["container_id"] == 123
        assert legacy_params["model_type"] == "vit_l"
        assert legacy_params["use_patches"] is False
    
    def test_load_config_from_dict(self):
        """Test load_config function with dictionary."""
        config_dict = {"omero": {"container_id": 999}}
        config = load_config(config_dict)
        
        assert isinstance(config, MicroSAMConfig)
        assert config.omero.container_id == 999
    
    def test_get_config_template(self):
        """Test getting configuration template."""
        template = get_config_template()
        
        assert isinstance(template, str)
        assert "batch_processing:" in template
        assert "# SAM model:" in template  # Check for comments
        
        # Test that template is valid YAML
        parsed = yaml.safe_load(template)
        assert isinstance(parsed, dict)


class TestConfigEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_patch_size_conversion(self):
        """Test patch size conversion from list to tuple."""
        config_dict = {
            "patches": {
                "patch_size": [256, 256]  # List instead of tuple
            }
        }
        
        config = MicroSAMConfig.from_dict(config_dict)
        assert config.patches.patch_size == (256, 256)
        assert isinstance(config.patches.patch_size, tuple)
    
    def test_optional_trainingset_name(self):
        """Test handling of optional trainingset_name."""
        # Test with None value
        config_dict = {"training": {"trainingset_name": None}}
        config = MicroSAMConfig.from_dict(config_dict)
        assert config.training.trainingset_name is None
        
        # Test with string value
        config_dict = {"training": {"trainingset_name": "my_set"}}
        config = MicroSAMConfig.from_dict(config_dict)
        assert config.training.trainingset_name == "my_set"
    
    def test_invalid_config_source(self):
        """Test load_config with invalid source."""
        with pytest.raises(ValueError, match="config_source must be"):
            load_config(123)  # Invalid type