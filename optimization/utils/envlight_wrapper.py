import envlight
import torch

class CustomEnvLight(envlight.EnvLight):
    
    def __call__(self, l, roughness=None, rotation=None):
        '''
        rotation: 3x3 torch tensor
        '''
        if rotation is not None:
            l_rotated = l.matmul(rotation.transpose(0,1))
            return super().__call__(l_rotated, roughness)
        return super().__call__(l, roughness)
    
    
if __name__ == '__main__':
    ref_light = envlight.EnvLight(
            '/home/jovyan/codebases/avatar-personalization/avatar/optimization/threestudio/load/lights/aerodynamics_workshop_2k.hdr', scale=2
        )
    custom_light = CustomEnvLight('/home/jovyan/codebases/avatar-personalization/avatar/optimization/threestudio/load/lights/aerodynamics_workshop_2k.hdr', scale=2)
    
    l = torch.tensor([0,0,1]).float().to('cuda')
    print(ref_light(l) == custom_light(l, rotation=torch.eye(3).to('cuda')))
    
    from scipy.spatial.transform import Rotation as R
    
    r = R.from_euler('zyx', [0, 180, 0], degrees=True)
    l1 = torch.tensor([1,0,0]).float().to('cuda')
    l2 = torch.tensor([-1,0,0]).float().to('cuda')
    
    print(ref_light(l1))
    print(custom_light(l2))
    print(custom_light(l2, rotation=torch.tensor(r.as_matrix()).float().to('cuda')))
    
    current_custom_light = lambda *args, **kwargs: custom_light(*args, **kwargs, rotation=torch.eye(3).to('cuda'))
#     import functools
#     def current_env_light(custom_light, rotation):
#         @functools.wraps(custom_light)
#         def wrapper(l, roughness=None):
#             return custom_light(l, roughness, rotation)
#         return wrapper
    
#     current_custom_light = current_env_light(custom_light, rotation=torch.eye(3).to('cuda'))

    roughness = torch.tensor(0.5).to('cuda')
    
    print(ref_light(l1, roughness=roughness), current_custom_light(l1, roughness=roughness))