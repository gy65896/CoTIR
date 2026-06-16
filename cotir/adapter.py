import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor
from .flux.modules.layers import QKNorm, Modulation, LastLayer
from .flux.math import attention

class CoTIRBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0, qk_scale: float | None = None):
        super().__init__()
        self.hidden_dim = hidden_size
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        # qkv and mlp_in
        self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim)
        # proj and mlp_out
        self.linear2 = nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size)

        self.norm = QKNorm(head_dim)

        self.hidden_size = hidden_size
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.mlp_act = nn.GELU(approximate="tanh")
        self.modulation = Modulation(hidden_size, double=False)

    def forward(self, x: Tensor, img: Tensor, vec: Tensor, pe: Tensor) -> Tensor:
        N_visual = x.shape[1]
        x_hat = torch.cat([x,img],dim=1)
        mod, _ = self.modulation(vec)
        x_mod = (1 + mod.scale) * self.pre_norm(x_hat) + mod.shift
        qkv, mlp = torch.split(self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1)

        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)

        # compute attention
        attn = attention(q, k, v, pe=pe)
        mlp = mlp[:,0:N_visual,:]
        attn = attn[:,0:N_visual,:]
        # compute activation in mlp stream, cat again and run second linear layer
        output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))

        return x + mod.gate * output

class CoTAdapter(nn.Module):
    def __init__(self, 
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qk_scale: float | None = None,
        out_channels: int = 3072,
        depth: int = 19):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.decoder = nn.ModuleList(
            [
                CoTIRBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, qk_scale=qk_scale)
                for _ in range(depth)
            ]
        )
        self.degradation_layer = nn.Linear(hidden_size, hidden_size)
        self.sharpness_layer = nn.Linear(hidden_size, hidden_size)
        self.planning_layer = nn.Linear(hidden_size * 2, hidden_size)
        self.final_degradation_layer = LastLayer(hidden_size, 1, out_channels)
        self.final_sharpness_layer = LastLayer(hidden_size, 1, out_channels)
        self.final_planning_layer = LastLayer(hidden_size, 1, out_channels)

        self.mlp_act = nn.GELU(approximate="tanh")
        
        # Zero-convolution like mechanism: learnable scale parameter initialized to 0
        # This ensures the adapter output is zero at the beginning of training
        self.scale_degradation = nn.Parameter(torch.zeros(1))
        self.scale_sharpness = nn.Parameter(torch.zeros(1))
        self.scale_planning = nn.Parameter(torch.zeros(1))
    
    def forward(self, x: Tensor, img: Tensor, vec: Tensor, pe: Tensor) -> tuple[Tensor, Tensor]:
        """
        Forward pass with zero-convolution scaling.
        
        Returns:
            x_out: Final output (scaled by scale_txt_out, starts at ~0)
            x_hat: Residual addition to txt (scaled by scale_txt_hat, starts at ~0)
        """
        for block in self.decoder:
            x = block(x, img, vec=vec, pe=pe)
        
        x_degradation = self.degradation_layer(x)
        x_sharpness = self.sharpness_layer(x)
        x_planning = self.planning_layer(torch.cat([x_degradation, x_sharpness], dim=-1))
        x_out_degradation = self.final_degradation_layer(self.mlp_act(x_degradation), vec)
        x_out_sharpness = self.final_sharpness_layer(self.mlp_act(x_sharpness), vec)
        x_out_planning = self.final_planning_layer(self.mlp_act(x_planning), vec)

        x_out = x_degradation * self.scale_degradation + x_sharpness * self.scale_sharpness + x_planning * self.scale_planning

        return x_out, x_out_sharpness, x_out_degradation, x_out_planning

if __name__ == "__main__":
    print("Testing CoTAdapter initialization and forward pass...")
    
    # Create adapter
    adapter = CoTAdapter(hidden_size=3072, num_heads=12, mlp_ratio=4.0, qk_scale=None, out_channels=512, depth=2)
    
    # Check zero-convolution scales are initialized to 0
    print(f"Initial scale_txt_hat: {adapter.scale_txt_hat.item()}")
    print(f"Initial scale_txt_out: {adapter.scale_txt_out.item()}")
    assert adapter.scale_txt_hat.item() == 0.0, "scale_txt_hat should be initialized to 0"
    assert adapter.scale_txt_out.item() == 0.0, "scale_txt_out should be initialized to 0"
    
    # Create test inputs
    x = torch.randn(1, 1024, 3072)
    img = torch.randn(1, 1024, 3072)
    vec = torch.randn(1, 3072)
    from .flux.modules.layers import EmbedND
    embedder = EmbedND(dim=256, theta=10000, axes_dim=[256])
    ids = torch.arange(2048).view(1, 2048, 1)
    pe = embedder(ids)
    
    # Forward pass
    with torch.no_grad():
        x_out, x_hat = adapter(x, img, vec, pe)
    
    print(f"Output shapes: x_out={x_out.shape}, x_hat={x_hat.shape}")
    print(f"x_out max abs value: {x_out.abs().max().item():.6f} (should be ~0 at init)")
    print(f"x_hat max abs value: {x_hat.abs().max().item():.6f} (should be ~0 at init)")
    
    # Verify outputs are approximately zero due to zero-convolution scaling
    assert x_out.abs().max().item() < 1e-6, "x_out should be ~0 at initialization"
    assert x_hat.abs().max().item() < 1e-6, "x_hat should be ~0 at initialization"
    
    print("✓ All tests passed! CoTAdapter initialized correctly.")