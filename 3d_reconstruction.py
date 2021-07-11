import torch
from tqdm import tqdm, trange
import time
import os
import model
import numpy as np
from dataloader import SceneDataset
import imageio
import data.load_DTU as DTU
import mcubes
import trimesh

# creates a list of all points on a specified grid
def create_grid_points_from_xyz_bounds(min_x, max_x, min_y, max_y ,min_z, max_z, res):
    x = np.linspace(min_x, max_x, res)
    y = np.linspace(min_y, max_y, res)
    z = np.linspace(min_z, max_z, res)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij', sparse=False)
    X = X.reshape((np.prod(X.shape),))
    Y = Y.reshape((np.prod(Y.shape),))
    Z = Z.reshape((np.prod(Z.shape),))

    points_list = np.column_stack((X, Y, Z))
    del X, Y, Z, x
    return points_list

def generate_mesh(cfg, epoch, i4d, test_dataset, specific_obj):

    basedir = cfg.basedir
    expname = cfg.expname

    # setup directory to save data
    savedir = os.path.join(basedir, expname, '3d_mesh',
                               f'{specific_obj}_epoch_{epoch}')
    os.makedirs(savedir, exist_ok=True)

    # load the input data for the network
    test_dataloader = test_dataset.get_loader(num_workers=0)
    test_dataloader.dataset.load_specific_sample = specific_obj

    test_data = test_dataloader.__iter__().__next__()['complete']

    batch = test_data[0]

    rel_ref_cam_locs, target, idx, focal = batch[-4:]
    inputs = [tensor.reshape([-1] + list(tensor.shape[2:])) for tensor in batch[:-4]]
    focal = np.array(focal)
    rays_o, rays_d, viewdirs, pts, z_vals, ref_pts, ref_images, ref_poses = inputs

    # specify suitable grid resolution and grid bounds and create corresponding grid points
    res = 256
    if specific_obj == 'scan23':
        min_x, max_x, min_y, max_y, min_z, max_z = -20 * 11, 18 * 11, -16 * 11, 2 * 14 * 11, -22 * 11, 15 * 11
    else:
        min_x, max_x, min_y, max_y, min_z, max_z = -15 * 11, 18 * 11, -11 * 11, 2 * 11 * 11, -22 * 11, 15 * 11

    points = create_grid_points_from_xyz_bounds(min_x, max_x, min_y, max_y, min_z, max_z,res)

    # reshape and batch grid points in order to feed them to the network
    points = torch.Tensor(points).reshape((1,res**3,3))

    # use the same number of points for batching, that would be used when sampling along rays
    batch_points = cfg.N_rays_test * (cfg.N_importance + cfg.N_samples)
    points_split = torch.split(points, batch_points, dim=1)

    # grid points are fed into the network, in order to infer their density values
    print("Generating the mesh geometry...")
    all_sigma = []
    for grid_points in tqdm(points_split):
        with torch.no_grad():
            rgb, sigma = i4d.point_wise_3D_reconst( ref_images, ref_poses, grid_points, focal)
            all_sigma.extend(sigma.reshape((-1)))

    # gather all density values
    all_sigma = np.array(all_sigma).reshape((res,res,res))


    # padding to be able to retrieve object close to bounding box bondary
    all_sigma_padded = np.pad(all_sigma, ((1, 1), (1, 1), (1, 1)), 'constant', constant_values=-50)

    # create a mesh from the densities, by inferring the level set given by "threshold"
    threshold = -3.45
    vertices, triangles = mcubes.marching_cubes(
        all_sigma_padded, threshold)

    # remove translation due to padding
    vertices -= 1

    # rescale to original scale
    step = np.array([max_x - min_x, max_y - min_y, max_z - min_z]) / (res - 1)
    vertices = np.multiply(vertices, step)
    vertices += [min_x,min_y,min_z]

    # produce a trimesh object given vertices and triangles
    mesh = trimesh.Trimesh(vertices, triangles)

    # the mesh has no colors yet, this is what we will address next
    # for each vertex on the mesh we will query the network for a color and attach it to the mesh

    # important: use the vertices in order given by trimesh, trimesh seems to reorder!
    vertex_points = torch.Tensor(mesh.vertices).unsqueeze(0)

    # batch vertices in order to feed them to the network
    points_split = torch.split(vertex_points, batch_points, dim=1)

    print("Generating the mesh vertex colors...")
    all_rgb = []
    for grid_points in tqdm(points_split):
        with torch.no_grad():
            rgb, sigma = i4d.point_wise_3D_reconst( ref_images, ref_poses, grid_points, focal)
            all_rgb.extend(rgb.reshape((-1,3)))

    all_rgb_enc_a = np.array(all_rgb) * 255
    all_rgb_enc_b = all_rgb_enc_a.astype(np.int)
    all_rgb_enc_c = np.clip(all_rgb_enc_b, 0, 255)
    mesh.visual.vertex_colors[:,:3] = all_rgb_enc_c
    mesh.export(os.path.join(savedir, f'mesh_colored_{specific_obj}.obj'));


if __name__ == '__main__':
    import config_loader
    import model

    cfg = config_loader.get_config()
    cfg.video = True

    test_dataset = SceneDataset(cfg, 'test')

    i4d = model.Implicit4D(cfg, test_dataset.proj_pts_to_ref_torch)

    i4d.load_model()
    generate_mesh(cfg, i4d.start, i4d, test_dataset, cfg.generate_specific_samples[0])