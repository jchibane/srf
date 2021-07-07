from torch.utils.data import Dataset
import torch
import numpy as np
import os
import random
import data.load_DTU as DTU



class SceneDataset(Dataset):

    def __init__(self, cfg, mode, num_workers=30):
        self.cfg = cfg
        self.num_workers = num_workers
        self.mode = mode
        self.load_mode = mode
        self.data = None
        self.batch_size = cfg.batch_size
        self.num_reference_views = cfg.num_reference_views

        self.load_specific_sample = None
        self.load_specific_poses = None
        self.load_specific_target_pose = None
        self.load_fixed = True
        self.shuffle = False

        self.fine_tune  = cfg.fine_tune
        if self.fine_tune != "None":
            print(self.fine_tune, type(self.fine_tune))
            print('Dataloader set in fine-tune mode.')
            self.load_specific_sample = self.fine_tune
            self.shuffle = True
            self.load_mode = 'test'

        if self.cfg.video or cfg.eval:
            self.shuffle = False

        if cfg.dataset_type == 'DTU':
            self.data, self.H, self.W, self.focal, self.cc, self.camera_system = DTU.setup_DTU(self.load_mode, cfg)
            print(self.H, self.W, self.focal, self.cc, self.camera_system)

            self.near = cfg.near
            self.far = cfg.far
            self.multi_world2cam = DTU.multi_world2cam_grid_sample_mat
            self.multi_world2cam_torch = DTU.multi_world2cam_grid_sample_mat_torch




    def __len__(self):
        return len(self.data)


    def proj_pts_to_ref(self, pts, ref_poses):
        ref_pts = []
        if self.cfg.dataset_type == 'DTU':
            for ref_pose in ref_poses:
                ref_pts.append([self.multi_world2cam(p.numpy(), ref_pose) for p in pts])
        else:
            for ref_pose in ref_poses:
                ref_pts.append([self.multi_world2cam(p.numpy(), self.H, self.W, self.focal[0], ref_pose) for p in pts])
        return torch.Tensor(ref_pts)  # (num_ref_views, rays, num_samples, 2)

    def proj_pts_to_ref_torch(self, pts, ref_poses, device, focal = None):
        # print((len(ref_poses), pts.shape[0],pts.shape[1],2))
        ref_pts = torch.zeros([2, 2]).to(device)
        ref_pts = torch.zeros([10, 250, 192, 2]).to(device)
        ref_pts = torch.zeros((len(ref_poses), pts.shape[0],pts.shape[1],2)).to(device)

        if self.cfg.dataset_type == 'DTU':
            for i, ref_pose in enumerate(ref_poses):
                for j,p in enumerate(pts):
                    ref_pts[i,j] = self.multi_world2cam_torch(p, ref_pose,device)
        else:
            for i, ref_pose in enumerate(ref_poses):
                for j, p in enumerate(pts):
                    ref_pts[i,j] = self.multi_world2cam_torch(p, self.H, self.W, focal[0], ref_pose, device)
        return ref_pts  # (num_ref_views, rays, num_samples, 2)




    def __getitem__(self, idx):
        if not self.cfg.no_ndc:
            raise ValueError('Not implemented!')

        N_rand = self.cfg.N_rand
        N_rays_test = self.cfg.N_rays_test

        if self.cfg.dataset_type == 'DTU':

            # for comparison of models we implement to load specific input/output data
            if self.load_specific_sample is None:
                sample = self.data[idx]
            else:
                sample = self.load_specific_sample

            print('DATALOADER load ', sample)
            imgs, poses, poses_idx = DTU.load_scan_data(sample, self.load_mode, self.num_reference_views + 1, self.cfg, self.load_specific_poses, self.load_fixed, self.shuffle)
            ref_images = imgs[:self.cfg.num_reference_views] # (num_ref_views, H, W, 3) np.array, f32
            ref_poses_idx = poses_idx[:self.cfg.num_reference_views]  # (num_reference_views) list, str
            ref_poses = poses[:self.cfg.num_reference_views] # (num_ref_views, 4, 4) np.array, f32


            if self.load_specific_target_pose is None:
                target = imgs[-1]  # (H, W, 3) np.array, f32
                target_pose = poses[-1] # (4,4) np.array, f32
            else:
                target_pose = self.load_specific_target_pose


        else:
            raise



        ref_cam_locs = np.array([ref_pose[:3, 3] for ref_pose in ref_poses])  # (num_ref_views, 3)
        rel_ref_cam_locs = ref_cam_locs - target_pose[:3,3]  # (num_ref_views, 3)


        rays_o, rays_d = get_rays(self.H, self.W, self.focal, self.cc, torch.Tensor(target_pose), self.camera_system)  # (H, W, 3), (H, W, 3)
        output = {}


        def compute_cos(pts, target_viewdirs):
            all_cos = torch.zeros((pts.shape[0], self.num_reference_views, pts.shape[1]))
            for target_ray, ray_pts in enumerate(pts):
                for ref_cam_idx, ref_cam_loc in enumerate(ref_cam_locs):
                    ref_ray_dirs = ray_pts - ref_cam_loc  # (num_samples, 3)
                    ref_ray_dirs = ref_ray_dirs / torch.norm(ref_ray_dirs, dim=-1, keepdim=True)
                    all_cos[target_ray, ref_cam_idx] = torch.sum(ref_ray_dirs * target_viewdirs[target_ray], dim=-1)
            return all_cos  # (rays, num_ref_views, num_samples)





        # create relative reference view features
        self.ref_pose_features = [ref_pose[:3,3] - target_pose[:3,3] for ref_pose in ref_poses]


        if self.mode == 'test':
            rays_o = torch.reshape(rays_o[::self.cfg.render_factor, ::self.cfg.render_factor], (-1, 3))
            rays_d = torch.reshape(rays_d[::self.cfg.render_factor, ::self.cfg.render_factor], (-1, 3))
            pts, z_vals = self.sample_ray(rays_o,rays_d)  # pts: (rays, num_samples, 3), z_vals: (rays, num_samples)

            viewdirs = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)

            cosines = compute_cos(pts,viewdirs)
            ref_pts = self.proj_pts_to_ref(pts, ref_poses)

            if self.load_specific_target_pose is None:
                output['complete'] = [[rays_o[i:i+N_rays_test], rays_d[i:i+N_rays_test], viewdirs[i:i+N_rays_test],pts[i:i+N_rays_test],
                                       z_vals[i:i+N_rays_test], ref_pts[:,i:i+N_rays_test], cosines[i:i+N_rays_test],
                                       ref_images, ref_poses, rel_ref_cam_locs, target, sample, self.focal ] for i in range(0, rays_o.shape[0], N_rays_test)]
            else:
                output['complete'] = [[rays_o[i:i+N_rays_test], rays_d[i:i+N_rays_test], viewdirs[i:i+N_rays_test],pts[i:i+N_rays_test],
                                       z_vals[i:i+N_rays_test], ref_pts[:,i:i+N_rays_test], cosines[i:i+N_rays_test],
                                       ref_images, ref_poses, rel_ref_cam_locs, sample, self.focal] for i in range(0, rays_o.shape[0], N_rays_test)]

            return output

        else:

            dH = int(self.H // 2 * self.cfg.precrop_frac)
            dW = int(self.W // 2 * self.cfg.precrop_frac)
            coords_cropped = torch.stack(
                torch.meshgrid(
                    torch.linspace(self.H // 2 - dH, self.H // 2 + dH - 1, 2 * dH),
                    torch.linspace(self.W // 2 - dW, self.W // 2 + dW - 1, 2 * dW)
                ), -1)
            coords_full = torch.stack(torch.meshgrid(torch.linspace(0, self.H - 1, self.H),
                                                     torch.linspace(0, self.W - 1, self.W)), -1)  # (H, W, 2)

            for (name, coords) in [('cropped',coords_cropped), ('complete', coords_full)]:
                coords = torch.reshape(coords, [-1, 2])  # (H * W, 2)
                select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
                select_coords = coords[select_inds].long()  # (N_rand, 2)
                rays_o_selected = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                rays_d_selected = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)

                viewdirs = rays_d_selected / torch.norm(rays_d_selected, dim=-1, keepdim=True)

                # Sample points along a ray
                pts, z_vals = self.sample_ray(rays_o_selected, rays_d_selected)
                cosines = compute_cos(pts, viewdirs)
                ref_pts = self.proj_pts_to_ref(pts, ref_poses)

                output[name] = [rays_o_selected, rays_d_selected, viewdirs, target_s, pts, z_vals, cosines, ref_pts,
                                ref_images, rel_ref_cam_locs, ref_poses, self.focal]


            return output

    def get_loader(self, shuffle=True, num_workers = None):

        if num_workers is None:
            num_workers = self.num_workers

        if self.mode == 'test':
            self.batch_size = 1

        return torch.utils.data.DataLoader(
                self, batch_size=self.batch_size, num_workers= num_workers, shuffle=shuffle,
                worker_init_fn=self.worker_init_fn)

    # enforce randomness
    def worker_init_fn(self, worker_id):
        random_data = os.urandom(4)
        base_seed = int.from_bytes(random_data, byteorder="big")
        np.random.seed(base_seed + worker_id)
        random.seed(base_seed + worker_id)
        torch.random.manual_seed(base_seed + worker_id)

    def sample_ray(self, rays_o, rays_d):
        N_samples = self.cfg.N_samples
        N_rays = rays_o.shape[0]

        near, far = self.near * torch.ones_like(rays_d[..., :1]), self.far * torch.ones_like(rays_d[..., :1])

        t_vals = torch.linspace(0., 1., steps=N_samples)

        if not self.cfg.lindisp:
            z_vals = near * (1. - t_vals) + far * (t_vals)
        else:
            z_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * (t_vals))

        z_vals = z_vals.expand([N_rays, N_samples])

        if self.cfg.perturb > 0.:
            # get intervals between samples
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape)
            z_vals = lower + (upper - lower) * t_rand

        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [N_rays, N_samples, 3]
        return pts, z_vals

def get_rays(H, W, focal, cc, c2w, camera_system):
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    if camera_system == 'x_down_y_down_z_cam_dir':
        dirs = torch.stack([(i - cc[0]) / focal[0], (j - cc[1]) / focal[1], torch.ones_like(i)], -1)
    if camera_system == 'x_down_y_up_z_neg_cam_dir':
        dirs = torch.stack([(i-cc[0])/focal[0], -(j-cc[1])/focal[1], -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d
