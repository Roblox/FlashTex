from typing import Optional, List
import torch
import numpy as np

from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    DirectionalLights,
    TexturesUV,
    TexturesVertex,
)
from pytorch3d.structures import Meshes

from mesh.render import (
    draw_meshes, 
    normalize_depth_01, 
    blur_depth,
)

from mesh.util import (
    load_mesh, 
    load_mesh_hdf5,
    load_mesh_glb,
    rotate_mesh,
)


class MeshDataset(object):
    def __init__(self,
                 input_mesh:str,
                 device:str='cuda',
                 head_only:Optional[bool]=None,
                 texture_tile_size:Optional[int]=None,
                 bbox_size:Optional[float]=None,
                 base_rotation_y:float=0.0,
                 rotation_x:float=0.0,
                 rotation_y:float=0.0,
                 uv_unwrap:Optional[bool]=None,
                 uv_rescale:Optional[bool]=None,
                 use_existing_textures:Optional[bool]=None):

        self.device = device
        
        if str(input_mesh).endswith('.hdf5'):
            self.mesh, self.landmarks, self.mesh_center, self.mesh_scale = load_mesh_hdf5(
                mesh_filename=input_mesh, uv_unwrap=uv_unwrap, uv_rescale=uv_rescale
            )
            self.texture_infos = {}
        elif str(input_mesh).endswith('.glb'):
            self.mesh, self.texture_infos, self.mesh_center, self.mesh_scale = load_mesh_glb(
                mesh_filename=input_mesh
            )
            self.landmarks = None
        else:
            self.mesh, self.texture_infos, self.mesh_center, self.mesh_scale = load_mesh(
                mesh_filename=input_mesh, 
                texture_tile_size=texture_tile_size, 
                uv_unwrap=uv_unwrap,
                uv_rescale=uv_rescale,
                use_existing_textures=use_existing_textures,
                head_only=head_only
            )
            self.landmarks = None
        
        if rotation_x != 0.0 or rotation_y != 0.0:
            self.mesh = rotate_mesh(mesh=self.mesh, 
                                    rotation_x=rotation_x,
                                    rotation_y=rotation_y)
        
        
    @torch.no_grad()
    def render_images(self, 
                      image_resolution:int,
                      dist:torch.tensor,
                      elev:torch.tensor,
                      azim:torch.tensor,
                      fov:torch.tensor,
                      light_dirs:List[List[float]]=None,
                      use_textures=True):

        R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim)
        cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T, fov=fov)
        num_views = len(cameras)

        lights = None if light_dirs is None else DirectionalLights(device=self.device, direction=light_dirs)

        if not use_textures:
            saved_textures = self.mesh.textures
            if isinstance(saved_textures, TexturesUV):
                self.mesh.textures = TexturesUV(verts_uvs=saved_textures.verts_uvs_padded(), faces_uvs=saved_textures.faces_uvs_padded(), maps=torch.ones_like(saved_textures.maps_padded()))
            else:
                self.mesh.textures = TexturesVertex(verts_features=saved_textures.verts_features_padded())

        rendered_data = draw_meshes(meshes=self.mesh.extend(num_views), cameras=cameras, lights=lights, image_size=image_resolution, draw_flat=(lights is None))
        
        if not use_textures:
            self.mesh.textures = saved_textures

        mesh_depths = blur_depth(normalize_depth_01(rendered_data['depths']))
        
        return {
            'mesh': self.mesh,
            'landmarks': self.landmarks,
            'texture_infos': self.texture_infos,
            'mesh_images': rendered_data['images'],
            'mesh_fragments': rendered_data['fragments'],
            'mesh_masks': rendered_data['masks'],
            'cameras': cameras,
            'mesh_depths': mesh_depths,
            'dist': dist,
            'elev': elev,
            'azim': azim,
            'fov': fov,
            'light_dirs': light_dirs,
            'center': self.mesh_center,
            'scale': self.mesh_scale,
        }
