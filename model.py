import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import os
from torchsearchsorted import searchsorted
import numpy as np
import itertools
import random
from tqdm import tqdm

to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)


class Implicit4D():

    def __init__(self, cfg, proj_pts_to_ref):
        self.proj_pts_to_ref = proj_pts_to_ref
        self.cfg = cfg
        self.device = torch.device( "cuda")
        models = {'model1': Implicit4DNN }
        self.model = models[cfg.model](cfg, self.device)
        self.grad_vars = list(self.model.parameters())

        self.model_fine = None
        if cfg.N_importance > 0:
            if cfg.fine_model_duplicate:
                self.model_fine = self.model
            else:
                raise ValueError('Not yet implemented / tested')
                self.model_fine = models[cfg.model](cfg, self.device)
                self.grad_vars += list(self.model_fine.parameters())

        self.start = 0
        self.val_min = None
        self.optimizer = torch.optim.Adam(params=self.grad_vars, lr=cfg.lrate, betas=(0.9, 0.999))


    def render_data(self, ref_images, ref_pts, rays_o, rays_d, viewdirs, z_vals, ref_poses, focal):

        def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False):
            raw2alpha = lambda raw, dists, act_fn=F.relu: 1. - torch.exp(-act_fn(raw) * dists)


            dists = z_vals[..., 1:] - z_vals[..., :-1]
            dists = torch.cat([dists, 1e10 * torch.ones_like(dists[..., :1])], -1)  # [N_rays, N_samples]

            dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

            rgb = torch.sigmoid(raw[..., :3])  # [N_rays, N_samples, 3]
            noise = 0.
            if raw_noise_std > 0.:
                noise = torch.randn(raw[..., 3].shape, device=self.device) * raw_noise_std

            if self.cfg.sigmoid:
                alpha = raw2alpha(raw[..., 3] + noise, dists, torch.sigmoid)
            else:
                alpha = raw2alpha(raw[..., 3] + noise, dists)  # [N_rays, N_samples]

            weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1), device=self.device), 1. - alpha + 1e-10], -1), -1)[:,
                              :-1]
            rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]

            depth_map = torch.sum(weights * z_vals, -1)
            disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
            acc_map = torch.sum(weights, -1)

            if white_bkgd:
                rgb_map = rgb_map + (1. - acc_map[..., None])

            return rgb_map, disp_map, acc_map, weights, depth_map

        ref_images = ref_images.to(self.device) # (batch_size x num_ref_views, H, W, 3)
        ref_pts = ref_pts.to(self.device) # (batch_size x num_ref_views, rays, num_samples, 2)
        viewdirs = viewdirs.to(self.device) # (batch_size x num_ref_views, rays, 3)
        z_vals = z_vals.to(self.device)  # (batch_size x rays, num_samples)
        rays_d = rays_d.to(self.device)  # (batch_size x  rays, 3)
        rays_o = rays_o.to(self.device)  # (batch_size x  rays, 3)
        ref_poses = ref_poses.to(self.device) # (batch_size x num_ref_views, 4, 4) np.array, f32

        if self.cfg.N_importance > 0:
            with torch.no_grad():
                raw = self.model(ref_images.float(), ref_pts.float())
                rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d,
                                                                             self.cfg.raw_noise_std,
                                                                             self.cfg.white_bkgd)
        else:
            raw = self.model(ref_images.float(), ref_pts.float())
            rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, self.cfg.raw_noise_std,
                                                                         self.cfg.white_bkgd)



        if self.cfg.N_importance > 0:
            rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map

            z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])

            z_samples = sample_pdf(z_vals_mid, weights[..., 1:-1], self.cfg.N_importance, det=(self.cfg.perturb==0.))
            z_samples = z_samples.detach()

            z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
            pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :,
                                                                None]  # [N_rays, N_samples + N_importance, 3]


            if self.cfg.batch_size != 1:
                raise ValueError('Not yet implemented. Next line accepts only single batch')

            ref_pts = self.proj_pts_to_ref(pts, ref_poses, self.device, focal)

            raw = self.model_fine(ref_images.float(), ref_pts.float())

            rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, self.cfg.raw_noise_std,
                                                                     self.cfg.white_bkgd)

        ret = {'rgb': rgb_map, 'disp': disp_map, 'acc': acc_map, 'raw': raw}

        if self.cfg.N_importance > 0:
            ret['rgb0'] = rgb_map_0
            ret['disp0'] = disp_map_0
            ret['acc0'] = acc_map_0
            ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]
        for k in ret:
            if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()):
                print(f"! [Numerical Error] {k} contains nan or inf.")

        return ret




    def render_img(self, data, H, W, specific_pose = False):
        all_ret = {}
        for batch in tqdm(data):
            # batch = [torch.Tensor(arr) for arr in batch]
            if specific_pose:
                rel_ref_cam_locs, idx, focal = batch[-3:]

                inputs = [tensor.reshape([-1] + list(tensor.shape[2:])) for tensor in batch[:-3]]
            else:
                rel_ref_cam_locs, target, idx, focal  = batch[-4:]
                inputs = [tensor.reshape([-1] + list(tensor.shape[2:])) for tensor in batch[:-4]]
            focal = np.array(focal)
            rays_o, rays_d, viewdirs, pts, z_vals, ref_pts, cosines, ref_images, ref_poses = inputs
            ret = self.render_data(ref_images, ref_pts, rays_o, rays_d, viewdirs, z_vals,  ref_poses, focal)
            # put all results into dictionary
            for k in ret:
                if k not in all_ret:
                    all_ret[k] = []
                all_ret[k].append(ret[k].cpu())

        # concat all results to single outputs
        all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}

        for k in all_ret:
            k_sh = [H//self.cfg.render_factor, W//self.cfg.render_factor] + list(all_ret[k].shape[1:])
            all_ret[k] = torch.reshape(all_ret[k], k_sh)


        if specific_pose:
            return all_ret['rgb'].numpy(), ref_images, idx[0]
        else:
            return all_ret['rgb'].numpy(), ref_images, target[0], idx[0]


    def load_model(self):
        basedir = self.cfg.basedir
        expname = self.cfg.expname

        # Load checkpoints
        if self.cfg.ft_path is not None and self.cfg.ft_path != 'None':
            ckpts = [os.path.join(basedir, expname, self.cfg.ft_path)]
        else:
            ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if
                     'tar' in f]

        print('Found ckpts', ckpts)
        if len(ckpts) > 0 and not self.cfg.no_reload:
            ckpt_path = ckpts[-1]
            print('Reloading from', ckpt_path)
            ckpt = torch.load(ckpt_path)

            self.start = ckpt['global_step']
            try:
                self.val_min = ckpt['val_min']
            except:
                self.val_min = None

            if self.cfg.fine_tune != "None":
                self.val_min = None
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])

            # Load model
            self.model.load_state_dict(ckpt['network_fn_state_dict'])
            if self.model_fine is not None and not self.cfg.fine_model_duplicate:
                self.model_fine.load_state_dict(ckpt['network_fine_state_dict'])

            if self.cfg.lrate_decay_off:
                print('Setting lr to fixed config value')
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.cfg.lrate

    def save_model(self, global_step):
        path = os.path.join(self.cfg.basedir, self.cfg.expname, '{:06d}.tar'.format(global_step))
        save_dict =         {
            'val_min' : self.val_min,
            'global_step': global_step + 1,
            'network_fn_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        if not self.model_fine is None and not self.cfg.fine_model_duplicate:
            save_dict['network_fine_state_dict'] = self.model_fine.state_dict()

        torch.save(save_dict, path)
        print('Saved checkpoints at', path)



class Implicit4DNN(nn.Module):


    def __init__(self, cfg, device):
        super(Implicit4DNN, self).__init__()

        # input should be (Scenes/Time instant x Views, img_channels, H, W)

        self.conv_in = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, dilation=1, padding=1,
                                 padding_mode='zeros')
        # after max pooling: (H/2, W/2)

        self.conv_0 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, dilation=1, padding=1,
                                padding_mode='zeros')
        self.conv_0_1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, dilation=1, padding=1,
                                padding_mode='zeros')
        # after max pooling: (H/4, W/4)

        self.conv_1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, dilation=1,
                                              padding=1, padding_mode='zeros')
        self.conv_1_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, dilation=1,
                                                padding=1, padding_mode='zeros')
        # after max pooling: (H/8, W/8)

        self.conv_2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, dilation=1,
                                              padding=1, padding_mode='zeros')
        self.conv_2_1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, dilation=1,
                                                padding=1, padding_mode='zeros')
        # after max pooling: (H/16, W/16)

        self.conv_3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, dilation=1,
                                              padding=1, padding_mode='zeros')
        self.conv_3_1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, dilation=1,
                                                padding=1, padding_mode='zeros')
        # after max pooling: (H/32, W/32)

        self.conv_4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, dilation=1,
                                              padding=1, padding_mode='zeros')
        self.conv_4_1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, dilation=1,
                                                padding=1, padding_mode='zeros')
        # after max pooling: (H/64, W/64)
        self.conv_5 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, dilation=1,
                                              padding=1, padding_mode='zeros')
        self.conv_5_1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, dilation=1,
                                                padding=1, padding_mode='zeros')

        self.feature_size = (3 +  16 + 32 + 64 + 128 + 128 + 128 + 128)


        num_merged_views = 2
        self.conv_sim_pair = torch.nn.Conv2d(in_channels=1, out_channels=128, padding=0,
                               kernel_size=(num_merged_views, self.feature_size), stride=num_merged_views, dilation=1,
                               padding_mode='circular')

        num_merged_pairs = 4
        self.conv_sim_merge_1 = torch.nn.Conv2d(in_channels=128, out_channels=128, padding=0,
                               kernel_size=(num_merged_pairs, 1), stride=1, dilation=1,
                               padding_mode='circular')

        self.conv_sim_merge_2 = torch.nn.Conv2d(in_channels=128, out_channels=128, padding=0,
                               kernel_size=(num_merged_pairs, 1), stride=1, dilation=1,
                               padding_mode='circular')

        self.max_over_views = nn.MaxPool2d(kernel_size=(-1, 1))

        self.fc_0 = nn.Linear(in_features=128, out_features=256)
        self.fc_1 = nn.Linear(in_features=256, out_features=128)
        self.fc_out = nn.Linear(in_features=128, out_features=4)
        self.actvn = nn.ReLU()

        self.maxpool = nn.MaxPool2d(2)

        self.conv_in_bn = nn.BatchNorm2d(16)
        self.conv0_1_bn = nn.BatchNorm2d(32)
        self.conv1_1_bn = nn.BatchNorm2d(64)
        self.conv2_1_bn = nn.BatchNorm2d(128)
        self.conv3_1_bn = nn.BatchNorm2d(128)
        self.conv4_1_bn = nn.BatchNorm2d(128)
        self.conv5_1_bn = nn.BatchNorm2d(128)

        self.to(device)
        self.device = device
        self.num_ref_views = cfg.num_reference_views
        self.batch_size = cfg.batch_size

    def forward(self, ref_images, ref_pts):


        rays, num_samples  = ref_pts.shape[1:-1]

        ref_images = ref_images.permute((0,3,1,2))

        feature_0 = F.grid_sample(ref_images, ref_pts, align_corners=True) # out (batch_size x num_ref_views, 3, rays, num_samples)


        net = self.actvn(self.conv_in(ref_images))
        net = self.conv_in_bn(net)
        feature_1 = F.grid_sample(net, ref_pts, align_corners=True) # out (batch_size x num_ref_views, 16, rays, num_samples)
        net = self.maxpool(net) # out (batch_size x num_ref_views, 16, H/2, W/2)

        net = self.actvn(self.conv_0(net))
        net = self.actvn(self.conv_0_1(net))
        net = self.conv0_1_bn(net)
        feature_2 = F.grid_sample(net, ref_pts, align_corners=True) # out (batch_size x num_ref_views, 32, rays, num_samples)
        net = self.maxpool(net) # out (batch_size x num_ref_views, 32, H/4, W/4)

        net = self.actvn(self.conv_1(net))
        net = self.actvn(self.conv_1_1(net))
        net = self.conv1_1_bn(net)
        feature_3 = F.grid_sample(net, ref_pts, align_corners=True) # out (batch_size x num_ref_views, 64, rays, num_samples)
        net = self.maxpool(net) # out (batch_size x num_ref_views, 64, H/8, W/8)

        net = self.actvn(self.conv_2(net))
        net = self.actvn(self.conv_2_1(net))
        net = self.conv2_1_bn(net)
        feature_4 = F.grid_sample(net, ref_pts, align_corners=True)  # out (batch_size x num_ref_views, 128, rays, num_samples)
        net = self.maxpool(net) # out (batch_size x num_ref_views, 128, H/16, W/16)

        net = self.actvn(self.conv_3(net))
        net = self.actvn(self.conv_3_1(net))
        net = self.conv3_1_bn(net)
        feature_5 = F.grid_sample(net, ref_pts, align_corners=True)  # out (batch_size x num_ref_views, 128, rays, num_samples)
        net = self.maxpool(net) # out (batch_size x num_ref_views, 128, H/32, W/32)

        net = self.actvn(self.conv_4(net))
        net = self.actvn(self.conv_4_1(net))
        net = self.conv4_1_bn(net)
        feature_6 = F.grid_sample(net, ref_pts, align_corners=True)  # out (batch_size x num_ref_views, 128, rays, num_samples)
        net = self.maxpool(net) # out (batch_size x num_ref_views, 128, H/64, W/64)

        net = self.actvn(self.conv_5(net))
        net = self.actvn(self.conv_5_1(net))
        net = self.conv5_1_bn(net)
        feature_7 = F.grid_sample(net, ref_pts, align_corners=True)  # out (batch_size x num_ref_views, 128, rays, num_samples)


        # here every channel corresponds to one feature.
        features = torch.cat((feature_0, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7),
                             dim=1)  # out (batch_size x num_ref_views, features, rays, num_samples), 300MB add


        features = features.reshape((self.batch_size, self.num_ref_views, self.feature_size, rays, num_samples)) # for free
        features = features.permute(0, 3, 4, 1, 2) # out (batch_size, rays, num_samples, num_ref_views, features) for free
        features = features.reshape((self.batch_size * rays * num_samples, 1, self.num_ref_views, self.feature_size)) # for free

        # similarity CNN
        def sim_combinations(num_views, num_merged_views):
            combis = list(itertools.combinations(range(num_views), num_merged_views))
            combis_list = []
            for combi in combis:
                combis_list += list(combi)
            return combis_list


        features = features[:,:,sim_combinations(self.num_ref_views, 2)] # out (batch_size * rays * num_samples, 1, num_ref_views pairs, features), 2.8 GB add
        features = self.actvn(self.conv_sim_pair(features)) # out (batch_size * rays * num_samples, channels, num_ref_views pairs /2, 1), 5.5 GB add


        features = self.actvn(self.conv_sim_merge_1(features)) # out (batch_size * rays * num_samples, 128, convoled views, 1), -2.5 GB add
        features = self.actvn(self.conv_sim_merge_2(features)) # out (batch_size * rays * num_samples, 128, convoled views, 1), unchanged

        self.max_over_views.stride = (features.shape[-2],1)
        self.max_over_views.kernel_size = (features.shape[-2],1)

        features = self.max_over_views(features).squeeze(-1).squeeze(-1) # out (batch_size * rays * num_samples, 128)

        features = self.actvn(self.fc_0(features))
        features = self.actvn(self.fc_1(features))
        features = self.fc_out(features) # out (batch_size * rays * num_samples, 4)

        features = features.reshape((self.batch_size * rays, num_samples,4))

        return features


# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_importance_samples, det=False):

    # Get pdf
    weights = weights + 1e-5 # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    # adding zero prob to start of cdf
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_importance_samples, device=cdf.device)
        # expand the sampling to all rays
        u = u.expand(list(cdf.shape[:-1]) + [N_importance_samples]) # u.expand( [[batch], N_sampes]) -> [batch, N_importance_samples]
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_importance_samples], device=cdf.device)

    # Invert CDF
    # for a val in 0-1 in cdf find where it came from along the ray
    inds = searchsorted(cdf.contiguous(), u.contiguous(), side='right')
    # use min and max coordinates for indices
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    # min(N_samples-1 * correct_Shape , inds)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds) #(batch, N_importance_samples)
    inds_g = torch.stack([below, above], -1)  # (batch, N_importance_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]] # [N_rand, N_samples, N_samples-1]

    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    #find fraction of how much we moved in between the bins, by interpolating and normalizing cdf vals
    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom

    # weights are computed on pts samples and bins are defined in between samples - does this make sense?
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])
    return samples