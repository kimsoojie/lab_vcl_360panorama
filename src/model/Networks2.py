from util.base import *
from model.models import *
from model.vgg import Vgg
import model.models as m
import model.models2 as m2
import model.models3 as m3
import model.ops as ops
from util.opt import  Options
import util.utilities as utl

opt = Options()

class Networks2():
    def __init__(self, device, phase):
        # self.net = VGG16().to(device)
        self.net = m.FOVnetwork().to(device)

        self.device = device
        self.optim = optim.Adam(self.net.parameters(), lr=opt.learn_rate,
                                betas=(opt.beta1,opt.beta2))

        self.phase = phase
        # self.writer = SummaryWriter(opt.train_log)
        self.loss_fn_BCE = nn.BCELoss()
        self.loss_fn_CE = nn.CrossEntropyLoss()
        self.loss_fn_L1 = nn.L1Loss()

    def load_input_batch(self, in_img, gt_img, gt_fov):
        self.in_img = in_img
        self.gt_img = gt_img
        self.gt_fov = gt_fov

        self.preprocess_input()

        self.in_img = ops.downsample(self.in_img, 2)
        self.gt_img = ops.downsample(self.gt_img, 2)

    def forward(self):
        print(self.in_img.size())
        self.pred_img, self.fov = self.net(self.in_img)
        self.fov = self.fov.view(-1,128)
        return self.fov, self.pred_img

    def compute_loss(self):
        self.loss_fov = self.loss_fn_CE(self.fov, self.gt_fov)
        self.loss_l1 = self.loss_fn_L1(self.pred_img, self.gt_img)
        self.loss = self.loss_fov + self.loss_l1

    def optimize(self):
        self.loss.backward()
        self.optim.step()

    def train_fov(self):
        self.net.zero_grad()
        self.pred_img, self.fov = self.net(self.in_img)
        self.fov = self.fov.view(-1,128)
        self.loss_fov = self.loss_fn_CE(self.fov, self.gt_fov)
        self.loss_l1 = self.loss_fn_L1(self.pred_img, self.gt_img)
        self.loss = self.loss_fov + self.loss_l1
        self.loss.backward()
        self.optim.step()

    def set_phase(self, phase):
        if phase == 'test':
            self.net.eval()
        else:
            self.net.train()

    def clear_gradient(self):
        self.net.zero_grad()

    def save_model(self, step, model_name='model'):
        print('Saving model at step ' + str(step))
        model_path = os.path.join(opt.model_path, model_name + '_' + str(step))
        torch.save(self.net.state_dict(), model_path + '.pt')
        print('Finished saving model')

    def load_model(self, model_name, strict=True):
        print('Loading trained model')
        model_path = os.path.join(opt.model_path, model_name + '.pt')
        model = torch.load(model_path)
        self.net.load_state_dict(model, strict=strict)
        print('Finished loading trained model')

    def preprocess_input(self):
        self.in_img = self.in_img.to(self.device)
        self.gt_img = self.gt_img.to(self.device)
        self.gt_fov = self.gt_fov.to(self.device)

    def print_structure(self):
        print(self.net)
        print('Phase : ' + self.phase)

    def print_summary(self, epoch, step):
        pred = torch.argmax(self.fov[0])
        gt = (self.gt_fov)
        gt = gt[0]
        print('Epoch %d [%d | %d] > Loss : %.5f Pred : %d GT : %d'
              %(epoch, step, opt.train_len, self.loss.item(), pred.item(), gt.item()))

    def write_img_summary(self, step):
        self.writer.add_image('Input', (self.in_img[0].squeeze(0)+1)/2, step)
        self.writer.add_image('Output', (self.pred_img[0].squeeze(0)+1)/2, step)
        self.writer.add_image('GT', (self.gt_img[0].squeeze(0)+1)/2, step)

