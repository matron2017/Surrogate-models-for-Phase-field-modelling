import pytest
import torch
import numpy as np
from pathlib import Path
import sys

# Add parent directories to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from models.unet import UNet
from models.afno import AFNO
from models.deterministic import DeterministicModel


class TestModelForwardPass:
    """Test forward pass of different model architectures."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup common test parameters."""
        self.batch_size = 2
        self.input_channels = 1
        self.output_channels = 1
        self.height = 64
        self.width = 64
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def test_unet_forward_pass(self):
        """Test UNet model forward pass with small data."""
        model = UNet(
            in_channels=self.input_channels,
            out_channels=self.output_channels,
            features=[64, 128, 256]
        ).to(self.device)
        
        # Create small test data
        x = torch.randn(self.batch_size, self.input_channels, self.height, self.width).to(self.device)
        
        # Forward pass
        with torch.no_grad():
            output = model(x)
        
        # Assertions
        assert output.shape == (self.batch_size, self.output_channels, self.height, self.width)
        assert not torch.isnan(output).any(), "Output contains NaN values"
        assert not torch.isinf(output).any(), "Output contains Inf values"
        print(f"✓ UNet forward pass successful. Input shape: {x.shape}, Output shape: {output.shape}")
    
    def test_afno_forward_pass(self):
        """Test AFNO model forward pass with small data."""
        model = AFNO(
            in_channels=self.input_channels,
            out_channels=self.output_channels,
            hidden_size=64,
            num_blocks=4,
            patch_size=8
        ).to(self.device)
        
        # Create small test data
        x = torch.randn(self.batch_size, self.input_channels, self.height, self.width).to(self.device)
        
        # Forward pass
        with torch.no_grad():
            output = model(x)
        
        # Assertions
        assert output.shape == (self.batch_size, self.output_channels, self.height, self.width)
        assert not torch.isnan(output).any(), "Output contains NaN values"
        assert not torch.isinf(output).any(), "Output contains Inf values"
        print(f"✓ AFNO forward pass successful. Input shape: {x.shape}, Output shape: {output.shape}")
    
    def test_deterministic_forward_pass(self):
        """Test Deterministic model forward pass with small data."""
        model = DeterministicModel(
            in_channels=self.input_channels,
            out_channels=self.output_channels,
            hidden_dim=128
        ).to(self.device)
        
        # Create small test data
        x = torch.randn(self.batch_size, self.input_channels, self.height, self.width).to(self.device)
        
        # Forward pass
        with torch.no_grad():
            output = model(x)
        
        # Assertions
        assert output.shape == (self.batch_size, self.output_channels, self.height, self.width)
        assert not torch.isnan(output).any(), "Output contains NaN values"
        assert not torch.isinf(output).any(), "Output contains Inf values"
        print(f"✓ Deterministic forward pass successful. Input shape: {x.shape}, Output shape: {output.shape}")
    
    def test_models_batch_consistency(self):
        """Test that models handle different batch sizes correctly."""
        batch_sizes = [1, 2, 4]
        x_base = torch.randn(1, self.input_channels, self.height, self.width).to(self.device)
        
        models = {
            "UNet": UNet(self.input_channels, self.output_channels).to(self.device),
            "AFNO": AFNO(self.input_channels, self.output_channels).to(self.device),
            "Deterministic": DeterministicModel(self.input_channels, self.output_channels).to(self.device),
        }
        
        for model_name, model in models.items():
            for batch_size in batch_sizes:
                x = x_base.repeat(batch_size, 1, 1, 1)
                with torch.no_grad():
                    output = model(x)
                assert output.shape[0] == batch_size, f"{model_name} batch size mismatch"
            print(f"✓ {model_name} batch consistency test passed")
    
    def test_models_gradient_flow(self):
        """Test that gradients flow through models correctly."""
        models = {
            "UNet": UNet(self.input_channels, self.output_channels).to(self.device),
            "AFNO": AFNO(self.input_channels, self.output_channels).to(self.device),
            "Deterministic": DeterministicModel(self.input_channels, self.output_channels).to(self.device),
        }
        
        x = torch.randn(self.batch_size, self.input_channels, self.height, self.width, requires_grad=True).to(self.device)
        
        for model_name, model in models.items():
            output = model(x)
            loss = output.mean()
            loss.backward()
            
            assert x.grad is not None, f"{model_name}: Input gradient is None"
            assert x.grad.abs().sum() > 0, f"{model_name}: Input gradient is zero"
            
            for name, param in model.named_parameters():
                if param.requires_grad:
                    assert param.grad is not None, f"{model_name}: {name} gradient is None"
            
            print(f"✓ {model_name} gradient flow test passed")
            x.grad.zero_()


@pytest.mark.parametrize("model_class,model_kwargs", [
    (UNet, {"in_channels": 1, "out_channels": 1, "features": [32, 64]}),
    (AFNO, {"in_channels": 1, "out_channels": 1, "hidden_size": 32, "num_blocks": 2}),
    (DeterministicModel, {"in_channels": 1, "out_channels": 1, "hidden_dim": 64}),
])
def test_model_output_shape(model_class, model_kwargs):
    """Parameterized test for output shape verification."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_class(**model_kwargs).to(device)
    
    x = torch.randn(1, 1, 64, 64).to(device)
    output = model(x)
    
    assert output.shape == (1, 1, 64, 64)
    print(f"✓ {model_class.__name__} output shape test passed")


if __name__ == "__main__":
    # Run tests directly
    test = TestModelForwardPass()
    test.setup()
    
    print("\n=== Running Model Forward Pass Tests ===\n")
    test.test_unet_forward_pass()
    test.test_afno_forward_pass()
    test.test_deterministic_forward_pass()
    test.test_models_batch_consistency()
    test.test_models_gradient_flow()
    
    print("\n=== All tests passed! ===\n")
