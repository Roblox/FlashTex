import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torchvision

from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesUV, TexturesVertex
from pytorch3d.transforms import axis_angle_to_matrix

import imageio
from PIL import Image
import xatlas
import h5py
from pathlib import PosixPath
from typing import List

DEFAULT_TEXTURE_NAME = "default"

@torch.no_grad()
def rotate_mesh(mesh, rotation_x, rotation_y):
    verts = mesh.verts_padded().squeeze(0)
    verts_mean = verts.mean(dim=0)
    verts = verts - verts_mean
    rotation_matrix_x = axis_angle_to_matrix(torch.tensor([float(np.deg2rad(rotation_x)), 0.0, 0.0]))
    rotation_matrix_y = axis_angle_to_matrix(torch.tensor([0.0, float(np.deg2rad(rotation_y)), 0.0]))
    rotation_matrix = torch.matmul(rotation_matrix_y, rotation_matrix_x)
    verts = torch.matmul(rotation_matrix, verts.unsqueeze(2)).squeeze(2).unsqueeze(0)
    normals = mesh.verts_normals_padded() if mesh.has_verts_normals() else None
    mesh = Meshes(verts=verts, faces=mesh.faces_padded(), textures=mesh.textures, verts_normals=normals)
    return mesh
      
        
def create_texture_image(texture_infos:dict, texture_tile_size:int):
    num_textures = len(texture_infos)
    if num_textures == 1:
        W = texture_tile_size
        H = texture_tile_size
    elif num_textures == 2:
        W = texture_tile_size * 2
        H = texture_tile_size
    elif num_textures == 3 or num_textures == 4:
        W = texture_tile_size * 2
        H = texture_tile_size * 2
    elif num_textures == 5 or num_textures == 6:
        W = texture_tile_size * 2
        # fixme can't make this 3x taller because out of memory
        H = texture_tile_size * 2
    elif num_textures == 7 or num_textures == 8:
        # fixme can't make this 2x taller because out of memory
        W = texture_tile_size * 2
        H = texture_tile_size * 2
    else:
        raise NotImplementedError(f'Unsupported number of textures; mesh has {num_textures} textures')

    return 0.5*torch.ones((1, H, W, 3), dtype=torch.float32)


def load_texture_for_atlas(texture_info:dict, H:int, W:int):
    texture_base_dir = texture_info['base_dir']
    texture_name = texture_info['name']
    texture_filename = os.path.join(texture_base_dir, texture_name)        
    try:
        image = np.array(Image.open(texture_filename).resize((W, H), resample=Image.BILINEAR))
        image = image.astype(np.float32) * (1.0 / 255.0)
        if image.shape[2] > 3:
            image_alpha = image[:,:,3:4]
            image = image[:,:,:3]
            base_color_rgb = np.array([235.0 / 255.0, 186.0 / 255.0, 150.0 / 255.0])
            base_color_image = np.full(shape=image.shape, fill_value=base_color_rgb, dtype=np.float32)
            image = (image * image_alpha) + (base_color_image * (1.0 - image_alpha))
        image = torch.from_numpy(image).unsqueeze(0)
    except:
        print(f'Could not load texture {texture_filename}: {sys.exc_info()}')
        image = 0.5*torch.ones((1, H, W, 3), dtype=torch.float32)
    return image

        
def get_atlas_tile_rect(texture_maps:torch.tensor, texture_info:dict):
    # Get the height/width of a tile in the atlas
    _, H, W, C = texture_maps.size()
    uv_offset = texture_info['uv_offset']
    uv_scale = texture_info['uv_scale']
    tile_w = int(W * uv_scale[0])
    tile_h = int(H * uv_scale[1])
    
    # Get the used/filled area of the texture atlas.  
    num_cols = int(1.0 / uv_scale[0])
    num_rows = int(1.0 / uv_scale[1])
    filled_W = tile_w * num_cols
    filled_H = tile_h * num_rows
    
    # Convert uv_offset to pixels within the filled atlas area.
    tile_x = int(filled_W * uv_offset[0])
    tile_y = filled_H - int(filled_H * uv_offset[1]) - tile_h

    return tile_x, tile_y, tile_w, tile_h

    
def load_textures_into_atlas(texture_maps:torch.tensor, texture_infos:dict):
    for texture_name, texture_info in texture_infos.items():
        tile_x, tile_y, tile_w, tile_h = get_atlas_tile_rect(texture_maps=texture_maps, texture_info=texture_info)
        tile = load_texture_for_atlas(texture_info, H=tile_h, W=tile_w)
        texture_maps[:,tile_y:tile_y+tile_h,tile_x:tile_x+tile_w,:] = tile
        
    
def load_mesh(mesh_filename:str, 
              texture_tile_size:int=1024,
              uv_unwrap:bool=False,
              uv_rescale:bool=False,
              use_existing_textures:bool=True,
              head_only:bool=False,
              centering:str='minmax')->Meshes:
    
    if not os.path.isfile(mesh_filename):
        raise FileNotFoundError(f"File not found: {mesh_filename}")
    
    print(f'Loading mesh: {mesh_filename}')
    
    return load_mesh_obj(mesh_filename=mesh_filename, uv_unwrap=uv_unwrap, uv_rescale=uv_rescale)
    

def load_mesh_obj(mesh_filename, uv_unwrap:bool=False, uv_rescale:bool=False, centering:str='minmax'):
    vertices, faces_info, aux = load_obj(mesh_filename)
    vertices = vertices.unsqueeze(0)
    faces = faces_info.verts_idx.unsqueeze(0)
    uvs = aux.verts_uvs.unsqueeze(0)
    faces_uvs = faces_info.textures_idx.unsqueeze(0)

    # Rescale UV
    if uv_rescale:
        uvs_min = uvs.min(dim=1, keepdims=True).values
        uvs_max = uvs.max(dim=1, keepdims=True).values
        uvs = (uvs - uvs_min) / (uvs_max - uvs_min)

    # Center verts
    if centering == 'minmax':
        all_verts_min = vertices.min(dim=1, keepdims=True).values
        all_verts_max = vertices.max(dim=1, keepdims=True).values
        all_verts_center = all_verts_min + ((all_verts_max - all_verts_min) * 0.5)
        vertices = vertices - all_verts_center
    elif centering == 'mean':
        all_verts_center = vertices.mean(dim=1, keepdims=True)
        vertices = vertices - all_verts_center

    # Scale so max extent=1.0
    scale = 1.0 / vertices.abs().max()
    vertices = vertices * scale

    texture_tile_size = 1024 # fixme
    # textures = TexturesVertex(verts_features=vertex_colors)
    texture_maps = torch.ones((1, texture_tile_size, texture_tile_size, 3), dtype=torch.float32)
    textures = TexturesUV(verts_uvs=uvs, faces_uvs=faces_uvs, maps=texture_maps)
    mesh = Meshes(verts=vertices, faces=faces, textures=textures)

    texture_infos = {
        DEFAULT_TEXTURE_NAME: {
            'name': DEFAULT_TEXTURE_NAME,
            'base_dir': PosixPath(mesh_filename).parent,
            'pack_idx': 0,
            'uv_offset': torch.tensor([0.0, 0.0]).float(),
            'uv_scale': torch.tensor([1.0, 1.0]).float()
        }
    }

    return mesh, texture_infos, all_verts_center.squeeze(0), torch.tensor([scale]*3)

@torch.no_grad()
def generate_atlas(mesh, texture_tile_size:int):
    atlas = xatlas.Atlas()

    vertices = mesh.verts_padded()[0].numpy()
    faces = mesh.faces_padded()[0].numpy()
    atlas.add_mesh(vertices, faces)        

    pack_options = xatlas.PackOptions()
    pack_options.bruteForce = True

    atlas.generate(pack_options=pack_options)

    vmapping, new_faces, new_uvs = atlas[0]
    new_faces = torch.from_numpy(new_faces.astype(np.int64)).long()
    new_uvs = torch.from_numpy(new_uvs)
    new_vertices = torch.from_numpy(vertices[vmapping])
    new_mesh = Meshes(verts=new_vertices.unsqueeze(0), faces=new_faces.unsqueeze(0))

    texture_maps = 0.5*torch.ones((1,texture_tile_size,texture_tile_size,3), dtype=torch.float32)

    new_textures = TexturesUV(verts_uvs=new_uvs.unsqueeze(0), faces_uvs=new_faces.unsqueeze(0), maps=texture_maps)
    new_mesh.textures = new_textures
    
    return new_mesh


# From: https://github.com/YadiraF/face3d/blob/master/face3d/mesh/io.py
def write_obj_with_texture(obj_name, texture_name, mesh):
    mesh = mesh.cpu()
    vertices = mesh.verts_padded().squeeze(0).numpy()
    triangles = mesh.faces_padded().squeeze(0).numpy()
    triangles = triangles.copy() + 1 # mesh lab start with 1

    if isinstance(mesh.textures, TexturesUV):
        uv_coords = mesh.textures.verts_uvs_padded().squeeze(0).detach().cpu().numpy()
        uv_triangles = mesh.textures.faces_uvs_padded().squeeze(0).detach().cpu().numpy()
        uv_triangles = uv_triangles.copy() + 1 # mesh lab start with 1
    else:
        uv_coords, uv_triangles = None, None

    if mesh.has_verts_normals():
        normals = mesh.verts_normals_padded().squeeze(0).detach().cpu().numpy()
    else:
        normals = None
        
    mtl_name = obj_name.replace('.obj', '.mtl')
    
    # write obj
    with open(obj_name, 'w') as f:
        # first line: write mtlib(material library)
        s = "mtllib {}\n".format(mtl_name.split('/')[-1])
        f.write(s)

        # write vertices
        for i in range(vertices.shape[0]):
            s = 'v {} {} {}\n'.format(vertices[i, 0], vertices[i, 1], vertices[i, 2])
            f.write(s)
        
        # write uv coords
        if uv_coords is not None:
            for i in range(uv_coords.shape[0]):
                s = 'vt {} {}\n'.format(uv_coords[i,0], uv_coords[i,1])
                f.write(s)

        # write normals
        if normals is not None:
            for i in range(normals.shape[0]):
                s = 'vn {} {} {}\n'.format(normals[i,0], normals[i,1], normals[i,2])
                f.write(s)

        f.write("usemtl MainTexture\n")

        # write f: ver ind/ uv ind
        if uv_triangles is not None:
            for i in range(triangles.shape[0]):
                s = 'f {}/{} {}/{} {}/{}\n'.format(triangles[i,0], uv_triangles[i,0], triangles[i,1], uv_triangles[i,1], triangles[i,2], uv_triangles[i,2])
                f.write(s)
        else:
            for i in range(triangles.shape[0]):
                s = 'f {} {} {}\n'.format(triangles[i,0], triangles[i,1], triangles[i,2])
                f.write(s)

    # write mtl
    with open(mtl_name, 'w') as f:
        f.write("newmtl MainTexture\n")
        s = 'map_Kd {}\n'.format(texture_name) # map to image
        f.write(s)



def load_mesh_glb(mesh_filename:str, centering:str='minmax'):
    import trimesh

    mesh = trimesh.load(mesh_filename, force='mesh')
    
    all_verts = torch.tensor(mesh.vertices).unsqueeze(0).float()
    all_faces = torch.tensor(mesh.faces).unsqueeze(0)
    
    # Center verts
    if centering == 'minmax':
        all_verts_min = all_verts.min(dim=1, keepdims=True).values
        all_verts_max = all_verts.max(dim=1, keepdims=True).values
        all_verts_center = all_verts_min + ((all_verts_max - all_verts_min) * 0.5)
        all_verts = all_verts - all_verts_center
    elif centering == 'mean':
        all_verts_center = all_verts.mean(dim=1, keepdims=True)
        all_verts = all_verts - all_verts_center

    # Scale so max extent=1.0
    scale = 1.0 / all_verts.abs().max()
    all_verts = all_verts * scale
    
    uvs = torch.tensor(mesh.visual.uv).unsqueeze(0).float()
    
    texture_tile_size = 1024 # fixme
    # textures = TexturesVertex(verts_features=vertex_colors)
    texture_maps = torch.ones((1, texture_tile_size, texture_tile_size, 3), dtype=torch.float32)
    textures = TexturesUV(verts_uvs=uvs, faces_uvs=all_faces, maps=texture_maps)
    
    mesh = Meshes(verts=all_verts, faces=all_faces, textures=textures)
    
    # print("UV unwrapping...")
    # mesh = generate_atlas(mesh=mesh, texture_tile_size=1024)

    # Set reset texture_infos to one with a single texture (the atlas) and no uv transform applied.
    texture_infos = {
        DEFAULT_TEXTURE_NAME: {
            'name': DEFAULT_TEXTURE_NAME,
            'base_dir': PosixPath(mesh_filename).parent,
            'pack_idx': 0,
            'uv_offset': torch.tensor([0.0, 0.0]).float(),
            'uv_scale': torch.tensor([1.0, 1.0]).float()
        }
    }
    
    return mesh, texture_infos, all_verts_center.squeeze(0), torch.tensor([scale]*3)