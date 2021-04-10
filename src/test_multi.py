from util.base import *
from util.opt import Options
import util.utilities as utl
from util.utilities import min_visualize, calc_quanti, to_numpy, save_img, calc_l1
from dataset.dataset import Dataset
from model.Networks import Networks
from model.Networks2 import Networks2
from scipy.stats import gaussian_kde
import model.models as m
import model.ops as ops
import model.models3 as m3
import time
from convert.createEquiFromSquareFiles import C2E

opt = Options(sys.argv[0])
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = 'cuda:0'

def estimate_fov(im_dir, model_name, device):
    # Init network
    net = m.FOVnetwork().to(device)
    net.eval()

    # load pretrained model
    model_path = os.path.join(opt.model_path, model_name + '.pt')
    model = torch.load(model_path)
    net.load_state_dict(model, False)

    imglist = []
    for i in range(4):
        impath = os.path.join(im_dir, "img" + str(i) + ".jpg")
        imglist.append(read_img_to_tensor(impath))
        #img = cv2.imread(impath)
        #crop = img[20:180,20:180]
        #cv2.imwrite('.\\img\\_img'+str(i)+'.jpg',crop)

    #img_tensor = torch.cat([imglist[2], imglist[0], imglist[3], imglist[1]], dim=2)
    img_tensor = torch.cat([imglist[0], imglist[1], imglist[2], imglist[3]], dim=2)
    in_img = make_input_batch(img_tensor, device)
    in_img = ops.downsample(in_img,1)#2
    _, fov_out = net(in_img)
   
    fov = torch.argmax(fov_out)
    fov_pred = fov.item() * 2

    imglist = []
    for i in range(4):
        impath = os.path.join(im_dir, "img" + str(i) + ".jpg")
        img = cv2.imread(impath)
        fov = generate_pad_img(img,fov_pred)
        imglist.append(fov)
        cv2.imwrite('.\\img\\fov\\img'+str(i)+'.jpg',fov)
    
    h,w,_ = imglist[0].shape
    up = np.zeros((h, w, 3), np.uint8)
    down = np.zeros((h,w, 3), np.uint8)
    cv2.imwrite('.\\img\\fov\\posy.jpg', up)
    cv2.imwrite('.\\img\\fov\\negy.jpg', down)

    #img_horiz = np.hstack([imglist[2], imglist[0], imglist[3], imglist[1]])
    img_horiz = np.hstack([imglist[0], imglist[1], imglist[2], imglist[3]])
    img_cat = np.vstack([np.zeros_like(img_horiz), img_horiz, np.zeros_like(img_horiz)])
    #img_cat = utl.numpy_to_pano(img_cat)
    #outpath = os.path.join(im_dir, "img_out.jpg")
    #cv2.imwrite(outpath, img_cat)
    c2e = C2E()
    c2e.cube2equi()

def write_output_single(folder_path, im_name, model_name, net_type=None):
    # Init network
    generator = m3.GM().to(device)
    generator.eval()
    model_path = os.path.join(opt.model_path, model_name + '.pt')
    model = torch.load(model_path)
    generator.load_state_dict(model['Generator'], strict=True)

    # Init input
    im_name_splt = im_name.split('.')
    in_img = read_img_to_tensor('.\\img\\img_out.jpg')
    in_img = torch.unsqueeze(in_img, 0)
    in_img = in_img.to(device)
    in_img = ops.downsample(in_img)

    out_s, out_m = generator(in_img)

    # torchvision.utils.save_image(out_m, '/var/www/html/' + im_name_splt[0] + '_m.png', normalize=True)
    save_img_from_tensor(imdir + '\\trained_output.jpg', out_m)
    print("finished saving image", out_m.size())


def read_img_to_tensor(im_path):
    im = cv2.imread(im_path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = im / 127.5 - 1
    tsr = torch.from_numpy(im.transpose(2,0,1))
    tsr = tsr.type(torch.FloatTensor)
    return tsr

def make_input_batch(img_tensor, device, down=2):
    in_img = torch.unsqueeze(img_tensor, 0)
    in_img = in_img.to(device)
    return in_img

def generate_pad_img(img, fov):
    pad = int((256-fov)/2)
    img = cv2.resize(img, (fov,fov))
    img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT)
    return img

def save_img_from_tensor(im_path, img_tensor):
    img = to_numpy(img_tensor) * 255.0
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(im_path, img.astype(np.uint8))

imdir = '.\\img'
im_name = estimate_fov(imdir,'fov\\model_fov2_fov_2000', device)
write_output_single(imdir, "trained_input.jpg",
                    model_name='model_190712\\model_n_medium_30000',
                    net_type='medium')
