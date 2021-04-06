#!/home/juliussurya/miniconda2/envs/pytorch/bin/python
from util.base import *
from util.opt import Options
import util.utilities as utl
from util.utilities import min_visualize, calc_quanti, to_numpy, save_img, calc_l1
from dataset.dataset import Dataset
from model.Networks import Networks
from model.Networks2 import Networks2
from scipy.stats import gaussian_kde
import time
import glob
from model.ops import sobel_conv
from skimage import feature

opt = Options(sys.argv[0])

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

data_loader = Dataset('test',resize=opt.resize).load_data()


total_psnr = 0
total_mse = 0
count = 0

def test_fov_acc(model_name):
    net = Networks2(device, phase='test')
    net.set_phase('test')
    net.load_model(model_name)

    data_len = len(data_loader)
    true_count = 0
    total_error = 0
    for item in data_loader:
        in_img, gt_img, gt_fov = item['input'].to(device), item['gt'].to(device), item['fov'].to(device)
        net.load_input_batch(in_img, gt_img, gt_fov)
        net.forward()

        pred_fov = torch.argmax(net.fov)
        # print('Pred : %d || GT : %d' %(pred_fov.detach(), gt_fov.detach()))
        if pred_fov == gt_fov:
            true_count += 1

        curr_error = float(abs(pred_fov.detach() - gt_fov.detach()))/float(gt_fov.detach())
        print('error %.5f' %curr_error)
        total_error += curr_error

    avg_acc = float(true_count)/float(data_len)
    avg_err = float(total_error)/float(data_len)
    print('Total Accuracy : %.2f' %avg_acc )
    print('Total Error : %.2f' %avg_err )

def read_img_to_tensor2(im_path, sz=(256,256)):
    im_ = cv2.imread(im_path)
    im_ = cv2.resize(im_, (256,256))
    im = cv2.cvtColor(im_, cv2.COLOR_BGR2RGB)
    im = im / 127.5 - 1
    tsr = torch.from_numpy(im.transpose(2,0,1))
    tsr = tsr.type(torch.FloatTensor)
    return torch.unsqueeze(tsr, 0), im_

def read_img_to_tensor(im_path):
    im = cv2.imread(im_path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = im / 127.5 - 1
    tsr = torch.from_numpy(im.transpose(2,0,1))
    tsr = tsr.type(torch.FloatTensor)
    return tsr

def write_output_single(folder_path, im_name='trained_input.jpg', model_path=None, net_type='large'):
    in_img = read_img_to_tensor(folder_path + '/' + im_name)
    in_img = torch.unsqueeze(in_img, 0)
    net = Networks(device, net_type=net_type, log=False)
    net.set_phase('test')
    # net.print_structure()
    net.load_model(model_path, False)
    # net.load_model('model_small_3/segment_2_0', False)
    # net.load_model('model_large_1/segment_2_0', True)
    # net.load_model('segment_10000')
    start = time.time()
    net.load_input_batch(in_img, in_img, 0)
    # net.forward_small()
    if net_type == 'large':
        outs, outm, outl = net.Generator(net.in_img)
        end = time.time()
        print(end - start)
        im_name_splt = im_name.split('.')
        save_img(outs, folder_path + '/output'+ im_name_splt[0]+'_s.png')
        save_img(outm, folder_path + '/output'+ im_name_splt[0]+'_m.png')
        save_img(outl, folder_path + '/output'+ im_name_splt[0]+'_l.png')
        print('saving image')

    if net_type == 'medium':
        outs, outm = net.Generator(net.in_img_m)
        end = time.time()
        # print(end - start)
        im_name_splt = im_name.split('.')
        save_img(outs, folder_path + '/output'+ im_name_splt[0]+'_s.png')
        save_img(outm, folder_path + '/output'+ im_name_splt[0]+'_m.png')
        save_img(outm, '/var/www/html/output'+ im_name_splt[0]+'_m.png')
        print('saving image')


def write_output(output_path, model_path=None, net_type='small'):
    in_path = output_path[0]
    gt_path = output_path[1]
    out_path = output_path[2]
    net = Networks(device, net_type)
    # net.set_phase('test')
    net.Generator.decompose_layer()
    net.load_model_G(model_path, gan=False, strict=True)


    idx = 0
    for item in data_loader:
        mask = utl.create_mask_overlap2()
        mask = mask.type(torch.cuda.FloatTensor)
        in_img, gt_img = item['input'].to(device), item['gt'].to(device)
        # net.load_input_batch(gt_img * mask, gt_img, 0)
        net.load_input_batch(in_img, gt_img, 0)
        if net_type == 'small':
            # net.forward_small()
            g_out_s = net.Generator(net.in_img_s)
            print('saving image %d'% idx)
            save_img(net.in_img_s, os.path.join(in_path, 'input_' + str(idx) + '.png'))
            save_img(net.gt_img_s, os.path.join(gt_path, 'gt_' + str(idx) + '.png'))
            save_img(g_out_s, os.path.join(out_path, 'output_' + str(idx) + '.png'))
        elif net_type == 'medium':
            # net.forward_medium()
            g_out_s, g_out_m = net.Generator(net.in_img_m)
            print('saving image %d'% idx)
            # save_img(net.in_img_m, os.path.join(in_path, 'input_' + str(idx) + '.png'))
            # save_img(net.gt_img_m, os.path.join(gt_path, 'gt_' + str(idx) + '.png'))
            save_img(g_out_m, os.path.join(out_path, 'output_' + str(idx) + '.png'))
        elif net_type == 'large':
            g_out_s, g_out_m, g_out_l = net.Generator(net.in_img)
            # g_out_l = net.Generator(net.in_img)
            print('saving image %d'% idx)
            save_img(net.in_img, os.path.join(in_path, 'input_' + str(idx) + '.png'))
            # save_img(net.gt_img, os.path.join(gt_path, 'gt_' + str(idx) + '.png'))
            save_img(g_out_l, os.path.join(out_path, 'output_' + str(idx) + '.png'))
        idx += 1

def write_output_list(list_path):
    dir_idx = []
    fs = open(list_path, 'w')
    for item in data_loader:
        subdir = item['dir']
        dir_idx.append(subdir)
        fs.write(subdir[0] + '\n')
    fs.close()
    return dir_idx

def evaluate_output(folder_path, data_len):
    total_mse = []
    total_psnr = []
    total_ssim = []
    for i in range(0,data_len-2):
        print('Processing data %d/%d' %(i, data_len))
        gt_im_path = os.path.join(folder_path[1], 'gt_' + str(i) + '.png')
        out_im_path = os.path.join(folder_path[2], 'output_' + str(i) + '.png')
        # print(gt_im_path, out_im_path)
        gt_im = cv2.imread(gt_im_path)
        out_im = cv2.imread(out_im_path)
        mse, psnr, ssim = calc_quanti(out_im, gt_im)
        print(mse, psnr, ssim)
        total_mse.append(mse)
        total_psnr.append(psnr)
        total_ssim.append(ssim)
    return total_mse, total_psnr, total_ssim

def evaluate_output_pix(folder_path, data_len):
    total_mse = []
    total_psnr = []
    total_ssim = []
    for i in range(1,data_len-1):
        print('Processing data %d/%d' %(i, data_len))
        gt_im_path = os.path.join(folder_path[1], 'gt',  'gt_' + str(i) + '.png')
        out_im_path = os.path.join(folder_path[2], 'output', 'output_' + str(i) + '.png')
        gt_im = cv2.imread(gt_im_path)
        out_im = cv2.imread(out_im_path)
        mse, psnr, ssim = calc_quanti(out_im, gt_im)
        print(mse, psnr, ssim)
        total_mse.append(mse)
        total_psnr.append(psnr)
        total_ssim.append(ssim)
    return total_mse, total_psnr, total_ssim

def evaluate_l1(folder_path, data_len):
    total_l1 = []
    for i in range(0,data_len - 2):
        print('Processing data %d/%d' %(i, data_len))
        gt_im_path = os.path.join(folder_path[0], 'gt_' + str(i) + '.png')
        out_im_path = os.path.join(folder_path[1], 'output_' + str(i) + '.png')
        gt_im = cv2.imread(gt_im_path)
        out_im = cv2.imread(out_im_path)
        l1 = calc_l1(gt_im, out_im)
        print(l1)
        total_l1.append(l1)
    return total_l1

def evaluate_l1_pix(folder_path, data_len):
    total_l1 = []
    for i in range(1,data_len-1):
        print('Processing data %d/%d' %(i, data_len))
        gt_im_path = os.path.join(folder_path[0], 'gt',  'gt_' + str(i) + '.png')
        out_im_path = os.path.join(folder_path[1], 'output_' + str(i) + '.png')
        gt_im = cv2.imread(gt_im_path)
        out_im = cv2.imread(out_im_path)
        l1 = calc_l1(gt_im, out_im)
        print(l1)
        total_l1.append(l1)
    return total_l1

def predict_FOV(folder_path, model_path):
    imt = [None] * 4
    images = [None] * 4
    imt[0], images[0] = read_img_to_tensor2(folder_path + '/im0.jpg')
    imt[1], images[1] = read_img_to_tensor2(folder_path + '/im1.jpg')
    imt[2], images[2] = read_img_to_tensor2(folder_path + '/im2.jpg')
    imt[3], images[3] = read_img_to_tensor2(folder_path + '/im3.jpg')

    net = Networks2(device, 'test')
    net.load_model(model_path)
    img_cat = torch.cat((imt[0], imt[1], imt[2], imt[3]), 3)
    net.load_input_batch(img_cat, img_cat, torch.empty(1))

    fov_logits, out_mask = net.forward()
    fov = torch.argmax(fov_logits)
    fov = fov.detach() * 2
    print(fov)

    imh = []
    for im in images:
        imh.append(utl.imresize_pad(im, sz=fov, pad=int((256-fov)/2)))

    img_horiz = np.hstack((imh[0], imh[1], imh[2], imh[3]))
    img_full = np.vstack((np.zeros_like(img_horiz), img_horiz, np.zeros_like(img_horiz)))

    img_equi = utl.numpy_to_pano(img_full)

    cv2.imwrite(folder_path + '/trained_input_gt.jpg', img_equi)
    cv2.imwrite(folder_path + '/input_horiz.jpg', img_horiz)

    return fov


def generate_overlap(data_path='/home/juliussurya/work/360dataset/pano_data_val/', data_target=0, save_path='/home/juliussurya/workspace/360pano2/image_surveys/overlap1'):
    for i in range(1):
        impath = os.path.join(data_path, 'pano_' + str(data_target), 'pano_*.jpg.jpg')
        print(impath)
        im_list = glob.glob(impath)[0]
        im = cv2.imread(im_list)
        mask, im_ovl = utl.create_mask_non_horizontal(img=im)
        mask = np.expand_dims(mask, -1)
        cv2.imwrite(save_path + '/mask_img.jpg', (mask*im).astype(np.uint8))
        cv2.imwrite(save_path + '/mask.jpg', (mask * 255).astype(np.uint8))
        cv2.imwrite(save_path + '/gt.jpg', (im).astype(np.uint8))
        for idx, k in enumerate(im_ovl):
            print(save_path)
            cv2.imwrite(save_path + '/im' + str(idx) + '.jpg', k)
            # plt.imshow(k)
            # plt.show()



def write_to_file(in_list, out_file):
    fs = open(out_file, 'w')
    for item in in_list:
       fs.write(str(item) + '\n')
    fs.close()

def compute_kde(x, x_grid, bandwith=1.0, **kwargs):
    kde = gaussian_kde(x, bw_method=bandwith/x.std(ddof=1), **kwargs)
    return kde.evaluate(x_grid)

def read_list_from_file(filename):
    out = []
    fs = open(filename, 'r')
    for line in fs:
        line = line.strip()
        out.append(float(line))
    return out

# save_path='/home/juliussurya/workspace/360pano2/image_surveys/horiz11'
# os.system('mkdir ' + save_path)
#
# generate_overlap(data_target=1321, save_path=save_path)
# predict_FOV(save_path,'fov/acc/model_fov2_fov_2000')
# write_output_single(save_path, 'trained_input_gt.jpg',  model_path = 'large_190723/model_fov2_large_10000_2', net_type='large')
# input('.....End...')
output_path = ['../output2_best/in_med_n','../output2_best/gt_med_n','../output2_best/output_med_compressed3']
# output_path = ['../output2_best/in_rn_small','../output2_best/gt_rn_small','../output2_best/output_rn_small']
# output_path = ['../output2_best/in_med_n','../output2_best/gt_med_n','../output2_best/output_large_n']
# output_path = ['../output2_best/in_small_o','../output2_best/gt_med_o','../output2_best/output_small_o']
# output_path = ['../output_ul/in_ov4','../output_large_1/gt','../output_ul/output_ov4']
mse, psnr, ssim = evaluate_output(output_path, 1050)
print('PSNR :',np.average(psnr))
print('MSE :',np.average(mse))
print('SSIM :',np.average(ssim))
# write_to_file(psnr, 'out_psnr1.txt')
# write_to_file(psnr, 'out_ssim1.txt')
# output_path = ['../output2_best/in_dog','../output2_best/gt_dog','../output2_best/output_dog']

# write_output(output_path, model_path='ts_decompose_arch256_7000', net_type='medium')
# write_output(output_path, model_path='compressed/ts_decompose_256_17000', net_type='medium')
# write_output(output_path, model_path='teacher_student_1000', net_type='medium')
# write_output(output_path, model_path='model_2_best/model_down_1_10000')
# write_output(output_path, model_path='model_rn_small_0', net_type='small')
# write_output(output_path, model_path='model_rn_medium_0', net_type='medium')
# write_output(output_path, model_path='large_190723/model_fov2_large_10000', net_type='large')
#write_output(output_path, model_path='model_ul_large_final/model_ul_10000', net_type='large')
# write_output(output_path, model_path='model_small_outpaint/model_n_small_20000', net_type='small')
# test_fov_acc('model_fov_epoch_4')
# test_fov_acc('fov/model_fov2_fov_2000')

# im_path = '/home/juliussurya/Dropbox (IVCL)/lab_affair/ppt_images/pano_note8/01'
# fov = predict_FOV(im_path, 'fov/acc/model_fov2_fov_2000')
# print(fov)

# write_output_single(im_path, 'trained_input_gt.jpg',
#                     model_path='large_190723/model_fov2_large_20000_2', net_type='large')
                    # model_path='model_190712/model_n_medium_30000',  net_type='medium')
# write_output_list('../output/data_list_val.txt')

# mse, psnr, ssim = evaluate_output(['../output_large_1', '../output_large_1'],7888)
# l1 = evaluate_l1(['../output_large_1', '../output_large_1'],7888)
# print(np.average(mse))
# print(np.average(psnr))
# print(np.average(ssim))
# write_to_file(mse, '../quanti/mse_.txt')
# write_to_file(psnr, '../quanti/psnr_.txt')
# write_to_file(ssim, '../quanti/ssim_.txt')
# write_to_file(l1, '../quanti/l1_.txt')

# mse, psnr, ssim = evaluate_output(['../output_large_1', '../output_large_2'],7888)
# l1 = evaluate_l1(['../output_large_1', '../output_large_2'],7888)
# print(np.average(mse))
# print(np.average(psnr))
# print(np.average(ssim))
# write_to_file(mse, '../quanti/mse.txt')
# write_to_file(psnr, '../quanti/psnr.txt')
# write_to_file(ssim, '../quanti/ssim.txt')
# write_to_file(l1, '../quanti/l1.txt')
# #
# #
# pix_output = '/home/juliussurya/workspace/pix2pixHD'
# mse, psnr, ssim = evaluate_output_pix(['../output_large_1', pix_output],7888)
# l1 = evaluate_l1_pix(['../output_large_1', pix_output],7888)
# print(np.average(mse))
# print(np.average(psnr))
# print(np.average(ssim))
# write_to_file(mse, '../quanti/mse_pix.txt')
# write_to_file(psnr, '../quanti/psnr_pix.txt')
# write_to_file(ssim, '../quanti/ssim_pix.txt')
# write_to_file(l1, '../quanti/l1_pix.txt')


# ssim = read_list_from_file('../quanti/l1.txt')
# ssim_pix = read_list_from_file('../quanti/l1_.txt')
#
# print(np.average(ssim))
# print(np.average(ssim_pix))
#
# ssim = np.array(ssim)
# ssim_pix = np.array(ssim_pix)
#
# x_grid = np.linspace(0.0,1.0,1000)
#
# pdf = compute_kde(ssim, x_grid, bandwith=0.02)
# pdf_pix = compute_kde(ssim_pix, x_grid, bandwith=0.02)
# our = plt.plot(x_grid, pdf, color='blue', alpha=0.5, lw=3)
# baseline = plt.plot(x_grid, pdf_pix, color='green', alpha=0.5, lw=3)
# plt.yticks([])
# plt.tight_layout()
# plt.gca().legend(('Ours','pix2pixHD'), fontsize='xx-large')
# plt.rc('axes', labelsize=40)
# plt.gca().set_facecolor('xkcd:sky blue')
# plt.show()

# ('... pause ...')
#
# psnr = read_list_from_file('../quanti/psnr.txt')
# psnr_pix = read_list_from_file('../quanti/psnr_.txt')
#
# print(np.average(psnr))
# print(np.average(psnr_pix))
# x_grid = np.linspace(0.0,40,1000)
#
# score =  psnr
# score_pix = psnr_pix
# score = np.array(score)
# score_pix = np.array(score_pix)
#
# pdf = compute_kde(score, x_grid, bandwith=0.4)
# pdf_pix = compute_kde(score_pix, x_grid, bandwith=0.4)
# plt.yticks([])
# plt.tight_layout()
# plt.rc('axes', labelsize=40)
# our = plt.plot(x_grid, pdf, color='blue', alpha=0.5, lw=3)start
# baseline = plt.plot(x_grid, pdf_pix, color='green', alpha=0.5, lw=3)
# plt.gca().legend(('Ours','pix2pixHD'), fontsize='xx-large')
# plt.show()


# imdir = '/home/juliussurya/workspace/360pano2/output_temp/test_im9'
# write_output_single(imdir, 'trained_input.jpg',
#                     model_path='model_190712/model_n_medium_30000',
#                     net_type='medium')
                    # model_path = 'large_190723/model_fov2_large_10000_2',
                    # net_type='large')
# write_output_single(imdir)
# write_output_single(save_path, 'trained_input_gt.jpg',  model_path = 'large_190723/model_fov2_large_10000_2', net_type='large')

"""
Best model for medium - large - standard 
- model_190712/model_n_medium_30000
- large_190723/model_fov2_large_10000_2
Best model for large - overlap
- model_overlap_large_13000_latest
Best model for FOV estimation
- large_190723/model_fov2_large_10000_2
"""