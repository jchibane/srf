import torch
from tqdm import tqdm, trange
import os
import numpy as np
from dataloader import SceneDataset
import generator
import math
from glob import glob
from torch.utils.tensorboard import SummaryWriter
import torchvision





img2mse = lambda x, y: torch.mean((x - y) ** 2)
mse2psnr = lambda x: -10. * torch.log(x) / torch.log(torch.Tensor([10.]).to(i4d.device))

def train(cfg):

    # Create log dir and copy the config file
    basedir = cfg.basedir
    expname = cfg.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(cfg)):
            attr = getattr(cfg, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if cfg.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(cfg.config, 'r').read())

    writer = SummaryWriter(os.path.join(basedir, expname, "tensorboard"))

    test_dataset = SceneDataset(cfg, 'test')
    train_dataset = SceneDataset(cfg, 'train')
    train_dataset_loader = train_dataset.get_loader()
    val_dataset_loader = SceneDataset(cfg, 'val').get_loader()

    train_dataset_iterator = train_dataset_loader.__iter__()
    val_dataset_iterator = val_dataset_loader.__iter__()


    global i4d
    i4d = model.Implicit4D(cfg, train_dataset.proj_pts_to_ref_torch)
    i4d.load_model()

    N_iters = 200000 + 1
    for global_step in trange(i4d.start, N_iters):
        batches_per_epoch = math.ceil(len(train_dataset) / cfg.batch_size)
        epoch = global_step // batches_per_epoch
        loss, psnr, train_dataset_iterator = compute_loss(train_dataset_iterator, train_dataset_loader, global_step, cfg)

        loss.backward()
        i4d.optimizer.step()

        if not cfg.lrate_decay_off:
            ###   update learning rate   ###
            decay_rate = 0.1
            decay_steps = cfg.lrate_decay * 1000
            new_lrate = cfg.lrate * (decay_rate ** (global_step / decay_steps))
            for param_group in i4d.optimizer.param_groups:
                param_group['lr'] = new_lrate
            ################################

        #####           end            #####

        # Rest is logging
        if global_step % cfg.i_weights == 0:
            i4d.save_model(global_step)

        if global_step % cfg.i_testset == 0:
            plot = generator.training_visualization(1, cfg, i4d, test_dataset, global_step)
            writer.add_figure('Visualization', plot, global_step)

        if global_step % cfg.i_print == 0:
            writer.add_scalar('Train PSNR', psnr.item(), global_step)
            writer.add_scalar('Train Loss(MSE)', loss.item(), global_step)
            tqdm.write(f"[TRAIN] Iter: {global_step} Loss: {loss.item()}  PSNR: {psnr.item()}")

        #fine tune validation steps
        fine_tune_val = False
        if cfg.fine_tune:
            fine_tune_val = global_step % cfg.i_val_fine_tune == 0
        # for every 10th epoch: if new epoch begins, compute validation loss
        prev_epoch = ((global_step - 1) // batches_per_epoch)
        if (epoch != prev_epoch and epoch % cfg.i_validation_loss == 0 and not cfg.i_no_val) or fine_tune_val :
            # clear cuda variables to enable releasing memory
            del loss, psnr
            val_batches = 20
            val_loss_sum = 0; val_psnr_sum = 0
            for i in range(val_batches):
                # torch.cuda.empty_cache()
                val_loss_batch, val_psnr_batch, val_dataset_iterator = compute_loss(val_dataset_iterator, val_dataset_loader, global_step, cfg)
                val_loss_sum += val_loss_batch.item(); val_psnr_sum += val_psnr_batch.item()
                tqdm.write(f"[VAL STEP] Iter: {global_step} Loss: {val_loss_batch.item()}  PSNR: {val_psnr_batch.item()}")
                writer.add_scalar('Validation Loss(MSE)/Per Batch', val_loss_batch.item(), global_step + i)
                writer.add_scalar('Validation PSNR/Per Batch', val_psnr_batch.item(), global_step + i)
                # clear cuda variables to enable releasing memory
                del val_loss_batch, val_psnr_batch
            val_loss = val_loss_sum / val_batches; val_psnr = val_psnr_sum/ val_batches
            tqdm.write(f"[VAL] Iter: {global_step} Loss: {val_loss}  PSNR: {val_psnr}")
            writer.add_scalar('Validation Loss(MSE)/AVG', val_loss, global_step)
            writer.add_scalar('Validation PSNR/AVG', val_psnr, global_step)

            if i4d.val_min is None:
                i4d.val_min = val_loss

            if val_loss < i4d.val_min:
                i4d.val_min = val_loss
                val_file_path = os.path.join(basedir, expname)
                for path in glob(val_file_path + '/val_min=*'):
                    os.remove(path)
                np.save( val_file_path + f'/val_min={global_step}', [epoch, val_loss, global_step])




def compute_loss(dataset_iterator, dataloader, global_step, cfg):
    try:
        data = dataset_iterator.next()
    except:
        # iterator not initialized or last element reached, python has no .hasNext
        dataset_iterator = dataloader.__iter__()
        data = dataset_iterator.next()

    if global_step < cfg.precrop_iters:
        # [rays_o, rays_d, viewdirs, target_s, pts, z_vals]
        # [batch_size, N_rand, 3] , ... , [batch_size, N_rand, 3] , [batch_size, N_rand, N_samples, 3], [batch_size, N_rand, N_samples]
        data = data['cropped']
    else:
        data = data['complete']

    # reshape batch_size dimension into data
    data_reshaped = [tensor.reshape([-1] + list(tensor.shape[2:])) for tensor in data[:-1]]

    rays_o, rays_d, viewdirs, target_s, pts, z_vals, ref_pts, ref_images, rel_ref_cam_locs, ref_poses  = data_reshaped
    focal = np.array(data[-1])

    i4d.model.train()
    i4d.optimizer.zero_grad()

    ret = i4d.render_data(ref_images, ref_pts, rays_o, rays_d, viewdirs, z_vals, ref_poses, focal)

    target_s = target_s.to(i4d.device)

    img_loss = img2mse(ret['rgb'], target_s)
    loss = img_loss
    psnr = mse2psnr(img_loss)

    if 'rgb0' in ret and not cfg.fine_model_duplicate:
        img_loss0 = img2mse(ret['rgb0'], target_s)
        loss = loss + img_loss0

    return loss, psnr, dataset_iterator

if __name__ == '__main__':
    import config_loader
    import model

    cfg = config_loader.get_config()

    train(cfg)