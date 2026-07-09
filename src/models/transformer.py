import torch
import torch.nn as nn

class SpatialCrossAttentionTransformer(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, grid_size=(100, 100, 10)):
        super().__init__()
        self.embed_dim = embed_dim
        self.grid_size = grid_size # X, Y, Z spatial resolution around the car
        
        # Define the virtual 3D grid spatial embedding queries (The Virtual World)
        self.volume_queries = nn.Embedding(grid_size[0] * grid_size[1] * grid_size[2], embed_dim)
        
        # Multi-Head Cross Attention layer to map 2D camera tokens to 3D spatial queries
        self.cross_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        
        self.layer_norm = nn.LayerNorm(embed_dim)
        
    def forward(self, camera_features):
        """
        Args:
            camera_features (Tensor): Shape (B, Total_Cam_Tokens, embed_dim) 
                                      Flattened visual feature maps from all 8 cameras.
        Returns:
            Tensor: Shape (B, X, Y, Z, embed_dim) Unified 3D Vector Space
        """
        batch_size = camera_features.shape[0]
        
        # 1. Fetch our un-allocated 3D world grid queries
        queries = self.volume_queries.weight.unsqueeze(0).repeat(batch_size, 1, 1) # (B, Grid_Vol, embed_dim)
        
        # 2. Execute cross attention: Queries look across Keys/Values (Camera Pixels)
        # Q = Spatial Grid Map, K & V = 2D Visual Camera Features
        spatial_tokens, _ = self.cross_attention(
            query=queries,
            key=camera_features,
            value=camera_features
        )
        
        # 3. Residual connection and normalisation
        spatial_tokens = self.layer_norm(spatial_tokens + queries)
        
        # 4. Reshape flat spatial tokens back into a physical 3D tensor volume map
        vector_space_3d = spatial_tokens.view(batch_size, self.grid_size[0], self.grid_size[1], self.grid_size[2], self.embed_dim)
        
        return vector_space_3d
