from model.Networks_compress import *
from util.base import *
from util.opt import Options
import util.utilities as utl
from util.utilities import cube_to_equirect
from dataset.dataset import Dataset
from model.Networks import Networks, Networks_Wnet, NetworksSegment
from model.Networks2 import Networks2
from model.ops import dstack

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

opt = Options(sys.argv[0])

# Define Dataset
# =================
data_loader = Dataset('train',resize=opt.resize).load_data()

# teacher_model = 'model_190712/model_n_medium_30000'
teacher_model = 'model_191105_medium_12'
# student_model = 'model_190712/model_n_medium_30000'
# student_model = 'compressed/teacher_student_1000'
student_model = 'model_191105_medium_12'

net = Networks_compress(device, teacher_model)
net.load_model(student_model, GAN=True, strict=False)
net.decompose()

for epoch in range(opt.total_epochs):
    for step, item in enumerate(data_loader):
        in_img, gt_img, gt_fov = item['input'], item['gt'], item['fov']

        if in_img.size()[0] != opt.train_batch:
            break

        net.load_input_batch(in_img)
        net.train()

        if step % 50 == 0:
            net.print_summary(epoch, step)
        if step % 100 == 0:
            net.write_img_summary(step)
        if step % 1000 == 0:
            net.save_model(step, model_name='ts_decompose_arch256')
