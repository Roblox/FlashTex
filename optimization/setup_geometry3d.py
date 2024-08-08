import os
import sys

from threestudio.models.geometry.implicit_sdf import ImplicitSDF
from threestudio.models.geometry.tetrahedra_sdf_grid import TetrahedraSDFGrid
from threestudio.models.geometry.custom_mesh import CustomMesh


def setup_custom_mesh(mesh_file, n_feature_dims, centering='minmax', scaling='max', device='cuda'):
    mesh_config = dict(
        shape_init=f"mesh:{mesh_file}",
        shape_init_params=1.0,
        shape_init_mesh_up="+z",
        shape_init_mesh_front="-x",
        shape_init_scaling=scaling,
        shape_init_center=centering,
        radius=1.0, # consistent with coarse
        pos_encoding_config=dict(
            otype="HashGrid",
            n_levels=16,
            n_features_per_level=2,
            log2_hashmap_size=19,
            base_resolution=16,
            per_level_scale=1.447269237440378,
        ),
        n_feature_dims=n_feature_dims,
    )
    mesh = CustomMesh(mesh_config).to(device)
    return mesh

def setup_geometry3d(mesh_file:str, geometry:str='custom_mesh', material:str='no_material', fix_geometry:bool=True, centering:str='minmax', scaling='max', device:str='cuda'):

    if material == 'no_material':
        n_feature_dims = 3
    elif material == 'pbr':
        n_feature_dims = 8

    if geometry == 'custom_mesh':
        geom3d = setup_custom_mesh(mesh_file, n_feature_dims=n_feature_dims, centering=centering, scaling=scaling, device=device)
    else:
        sys.exit(f"Unknown geometry type {geometry}")

    return geom3d
