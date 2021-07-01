import torch
import os
import numpy as np
from dataloader import SceneDataset
import imageio
import data.load_DTU as DTU


def generate_video(cfg, i4d, test_dataset, epoch, specific_obj, specific_poses):

    # Create log dir and copy the config file
    basedir = cfg.basedir
    expname = cfg.expname

    test_dataloader = test_dataset.get_loader(num_workers=0)


    savedir = os.path.join(basedir, expname, 'video', f'epoch_{epoch}_{specific_obj}_batch{cfg.fixed_batch}'
                                                      f'_renderfactor{cfg.render_factor}')
    os.makedirs(savedir, exist_ok=True)


    for i, pose in enumerate(specific_poses):

        filename = os.path.join(savedir, f'pose_{pose[0]}.png')
        c2w = pose[1]

        if os.path.exists(filename):
            continue

        test_dataloader.dataset.load_specific_sample = specific_obj
        test_dataloader.dataset.load_specific_target_pose = c2w
        print(f'generating {test_dataloader.dataset.load_specific_sample, i}')
        test_data = test_dataloader.__iter__().__next__()['complete']

        with torch.no_grad():
            rgb8, ref_images, scan = i4d.render_img(test_data, test_dataset.H, test_dataset.W, True)


        imageio.imwrite(filename, rgb8)

    for i, ref_img in enumerate(ref_images):
        filename = os.path.join(savedir, f'ref_img_{i}.png')
        imageio.imwrite(filename, ref_img)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.figure(figsize=(50, 20), dpi=200)
    plt.xticks([]), plt.yticks([])
    for i in range(10):

        ax = plt.subplot(2, 5, i + 1)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

        ax.imshow(ref_images[i], interpolation='bicubic')

    plt.savefig(os.path.join(savedir, f'ref_images.png'))
    plt.close()

    test_dataloader.dataset.load_specific_sample = None
    test_dataloader.dataset.load_specific_target_pose = None


if __name__ == '__main__':
    import config_loader
    import model


    cfg = config_loader.get_config()

    set = 'test'
    test_dataset = SceneDataset(cfg, set)
    i4d = model.Implicit4D(cfg, test_dataset.proj_pts_to_ref_torch)

    i4d.load_model()

    if cfg.dataset_type == 'DTU':
        if cfg.video:
            for scan in cfg.generate_specific_samples:
                pose = DTU.load_cam_path()[cfg.gen_pose]
                generate_video(cfg, i4d, test_dataset, i4d.start, scan, [(cfg.gen_pose,pose)])