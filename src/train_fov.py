from util.base import *
from util.opt import Options
import util.utilities as utl
from util.utilities import cube_to_equirect
from dataset.dataset import Dataset
from model.Networks import Networks, Networks_Wnet, NetworksSegment
from model.Networks2 import Networks2
from model.ops import dstack
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    opt = Options(sys.argv[0])

    # Define Dataset
    # =================
    data_loader = Dataset('train',resize=opt.resize).load_data()
    # data_val = Dataset('val', resize=opt.resize).load_data()


    # net = NetworksSegment(device, num_class=8)
    net = Networks2(device, 'train')
    net.set_phase('train')
    net.print_structure()
    # net.load_model_G('model_overlap_medium_4000_latest2', False)
    # net.load_model('model_191017_medium_1000', True)
    # net.print_structure()
    # net.load_model('large_190723/model_190716_large_10000',True)
    # net.load_model_G('model_190712/model_n_medium_30000', gan=True, strict=False)
    # net.load_model('model_190716_large_10000', True)
    # net.load_model_G('model_rn_med_0', True)
    # net.load_model_G('segment_2_20000', False)
    # net.load_model_G('model_2_best/model_down_1_10000', False)
    # net.load_model_G('model_ul_large/model_ul_10000', False)

    forward_call = net.train_fov
    loss_call = net.compute_loss
    txt_summary_call = net.print_summary
    img_summary_call = net.write_img_summary


    # mask = utl.create_mask_portion()

    total_step = 0
    start = time.time()
    for epoch in range(opt.total_epochs):
        step = 0
        net.save_model(epoch, 'model_191105_' + opt.net)

        for item in data_loader:
            mask = utl.create_mask_ul()
            in_img, gt_img, gt_fov = item['input'], item['gt'], item['fov']

            # net.clear_gradient()

            if in_img.size()[0] != opt.train_batch:
                break

            # net.train_small(gt_img * mask, gt_img)
            # Zero all gradients after iteration

            # Load image to network
            net.load_input_batch(in_img, gt_img, gt_fov)
            # mask = utl.create_mask_overlap()
            # net.load_input_batch(gt_img * mask, gt_img, gt_fov)

            # Forward network
            forward_call() # net.forward_<type>()

            # Compute loss
            # loss_call() # net.compute_loss_<type>()

            # Update weights
            # net.optimize() # update D and G
            # if step % 3 == 0:
            #     net.optimize() # optimize both G and D
            # else:
            #     net.optimize_G() # optimize G only

            end = time.time()
            elapsed = end - start

            # Print network loss
            # # Add Tensorboard summary
            if step % 1 == 0:
                # net.write_scalars(step)
                txt_summary_call(epoch, step) # net.print_summary_<type>(epoch, step)
                print('Time elapsed', elapsed, 'seconds')
            # if step % 100 == 0:
            #     img_summary_call(step) # net.write_imgs_summary_<type>(step)
                # net.write_scalar_summary('train/L1', total_step)
            # if step % 1000 == 0:
            #     net.save_model(step, 'model_191105_' + opt.net)
                # validation
                # val_idx = random.randint(1,4999)
                # item = data_val[val_idx]
                # mask = utl.create_mask_ul()
            # in_img, gt_img, gt_fov = item['input'], item['gt'], item['fov']
                # in_img = torch.unsqueeze(in_img, 0)
                # gt_img = torch.unsqueeze(gt_img, 0)
                # net.clear_gradient()
                # net.load_input_batch(gt_img * mask, gt_img)
                # forward_call()
                # loss_call()
                # net.write_scalar_summary('val/L1',total_step)


            step += 1
            total_step += 1

    # Net = Networks(device, net_type=opt.net)
    # Net.set_phase('train')
    # Net.print_structure()
    # #Net.load_model('model_50000')
    #
    # print('Starting training loop')
    #
    # step = 0
        # for epoch in range(opt.total_epochs):
    #     for item in data_loader:
    #         in_img, gt_img = item['input'], item['gt']
    #         in_img = cube_to_equirect(in_img, opt.equi_coord, device)
    #         #gt_img = cube_to_equirect(gt_img, opt.equi_coord, device)
    #         in_img = in_img.to(device)
    #         gt_img = gt_img.to(device)
    #
    #         Net.load_input_batch(in_img, gt_img)
    #
    #         Net.clear_gradient() # zeros all gradient buffer
    #
    #         # Forward D real
    #         Net.d_out_real = Net.Discriminator(dstack(Net.in_img, Net.gt_img))
    #
    #         # Forward D fake
    #         Net.g_out = Net.Generator(in_img) # Forward G
    #         Net.d_out_fake = Net.Discriminator(dstack(Net.in_img, Net.g_out).detach())
    #
    #         # Gather loss and update
    #         Net.loss_d_total = Net.compute_loss_D() # total loss real and fake
    #         Net.loss_d_total.backward()
    #         Net.optim_d.step() # update
    #
    #         # Forward D adv
    #         Net.d_out_adv = Net.Discriminator(dstack(Net.in_img, Net.g_out))
    #         Net.loss_g_total = Net.compute_loss_G(Net.gt_img) # total loss adv + pix
    #         Net.loss_g_total.backward()
    #         Net.optim_g.step() # update
    #
    #         if step % 500 == 0:
    #             Net.print_summary(epoch, step)
    #             Net.write_imgs_summary(Net.in_img, Net.gt_img, step)
    #             # Net.write_img_summary(Net.g_out_sobel[0], 'out_sobelx', step)
    #             # Net.write_img_summary(Net.gt_sobel[0], 'gt_sobelx', step)
    #         if step % 10000 == 0:
    #             Net.save_model(step, model_name= Net.prefix + '_full')
    #         step += 1

if __name__ == '__main__':
    main()
