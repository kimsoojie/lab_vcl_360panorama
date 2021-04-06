import model.models3 as m3
from model.models import *
import model.ops as ops
from util.opt import Options
import util.utilities as utl
from tensorboardX import SummaryWriter

opt = Options()

def gram_matrix(tnsr):
    n, c, h, w = tnsr.size()
    feats = tnsr.view(n * c, h * w)
    G = torch.mm(feats, feats.t())
    return G.div(n * c * h * w)

def extract_gt_features(model_path):
    def hook(module, input, output):
        outputs = []
        outputs.append(output)

    layer_list = []
    m = model_path
    net = m3.GM()
    for mod in net.resblock:
        print(mod.block)
        input('...')
        for b in mod.block:
            if isinstance(b, nn.Conv2d):
                out = b.register_forward_hook(hook)
                print(out)
                input('....')
                out.remove()
                # gram_out = gram_matrix(b.weight)
                # print(gram_out.size())
    return

class Teacher(nn.Module):
    def __init__(self, trained_weights):
        super(Teacher, self).__init__()
        Gm = m3.GTestMobile()
        Gm.load_state_dict((trained_weights),strict=True)

        self.in_med0 = Gm.in_med0
        self.in_med1 = Gm.in_med1
        self.in_med2 = Gm.in_med2

        self.in_small0 = Gm.in_small0
        self.in_small1 = Gm.in_small1
        self.in_small2 = Gm.in_small2

        self.conv3 = Gm.conv3
        self.conv4 = Gm.conv4
        self.conv5 = Gm.conv5

        self.resblock0 = Gm.resblock[0]
        self.resblock1 = Gm.resblock[1]
        self.resblock2 = Gm.resblock[2]
        self.resblock3 = Gm.resblock[3]
        self.resblock4 = Gm.resblock[4]
        self.resblock5 = Gm.resblock[5]

        self.dconv1 = Gm.dconv1
        self.dconv2 = Gm.dconv2
        self.dconv3 = Gm.dconv3
        self.dconvs = Gm.dconvs
        self.dconvm = Gm.dconvm
        self.dconvl = Gm.dconvl

        self.outs = Gm.outs
        self.outm = Gm.outm
        self.activ = Gm.activ

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        # xs = ops.downsample(x)
        xs = F.avg_pool2d(x, 2, 2)
        med0 = self.in_med0(x)  # 256 x 512
        med1 = self.in_med1(med0)  # 256 x 512
        med2 = self.in_med2(med1)  # 128 x 256

        small0 = self.in_small0(xs)  # 128 x 256
        small1 = self.in_small1(small0)  # 128 x 256
        small2 = self.in_small2(small1)  # 64 x 128
        small2 = self.activ(small2 + med2)

        enc3 = self.conv3(small2)
        enc4 = self.conv4(enc3)
        enc5 = self.conv5(enc4)

        res0 = self.resblock0(enc5)
        res1 = self.resblock1(res0)
        res2 = self.resblock2(res1)
        res3 = self.resblock3(res2)
        res4 = self.resblock4(res3)
        res5 = self.resblock5(res4)

        dec1 = self.dconv1(res5)
        dec2 = self.dconv2(dec1)
        dec3 = self.dconv3(dec2)
        decs = self.dconvs(ops.dstack(dec3, small2))
        # decm = self.dconvm(ops.dstack(decs, med2))
        decs = ops.upsample(decs,2)
        decm = self.dconvm(decs)
        outm = self.outm(decm)
        outm = F.tanh(outm)

        int_feats = [res0, res1, res2, res3, res4, res5]
        return int_feats, outm


class Networks_compress():
    def __init__(self, device, teacher_model):
        self.device = device
        self.student = m3.GTestMobileStudent().to(device)
        teacher_weights = self.get_teacher_weight(teacher_model)
        self.teacher = Teacher(teacher_weights['Generator']).to(device)
        self.writer = SummaryWriter(opt.train_log)

        self.loss_fn_L1 = nn.L1Loss()
        self.loss_fn_L2 = nn.MSELoss()
        self.optim = optim.Adam(self.student.parameters(), lr=opt.learn_rate * 1,
                                betas=(opt.beta1, opt.beta2))

    def load_input_batch(self, x):
        self.in_img = ops.downsample(x, 2)
        self.in_img = self.in_img.to(self.device)

    def train(self):
        self.student.zero_grad()
        self.s_feats, self.s_out = self.student(self.in_img)
        self.t_feats, self.t_out = self.teacher(self.in_img)

        # Feature loss
        t_gram = 0.0
        for feats in self.t_feats:
            t_gram += gram_matrix(feats.detach())
        t_gram /= len(self.t_feats)
        s_gram = gram_matrix(self.s_feats)
        self.loss_feats = self.loss_fn_L2(t_gram, s_gram)
        self.s_gram = s_gram
        self.t_gram = t_gram

        # Feature loss2
        # self.loss_feats2 = 0.0
        # for t, s in zip(t_feats2,s_feats2):
        #     self.loss_feats2 += self.loss_fn_L2(t,s)

        # Out loss
        self.loss_out = self.loss_fn_L1(self.t_out.detach(), self.s_out)

        # Total loss
        self.total_loss = self.loss_out + 10 * self.loss_feats
        self.total_loss.backward()
        self.optim.step()

    def decompose(self):
        print('Decomposing Tensor....')
        self.student.decompose_layer()
        print(self.student)
        print(self.teacher)
        input('....')

    def save_model(self, step, model_name='model'):
        print('Saving model at step', str(step))
        model_path = os.path.join(opt.model_path, model_name + '_' + str(step) + '.pt')
        torch.save(self.student.state_dict(), model_path)
        print('Finished saving model')

    def load_model(self, model_name, GAN=False, strict=True):
        print('Loading trained model')
        model_path = os.path.join(opt.model_path, model_name + '.pt')
        if GAN == False:
            model = torch.load(model_path)
            self.student.load_state_dict(model, strict=strict)
        else:
            model = torch.load(model_path)
            self.student.load_state_dict(model['Generator'], strict=strict)
        print("Finished loading trained model")

    def get_teacher_weight(self, model_name):
        model_path = os.path.join(opt.model_path, model_name + '.pt')
        model = torch.load(model_path)
        return model

    def write_img_summary(self, step):
        # in_img = self.in_img[0,:,:,:]
        # out_student = self.s_out[0,:,:,:]
        # out_teacher = self.t_out[0,:,:,:]
        in_img = torchvision.utils.make_grid(self.in_img, normalize=True)
        out_student = torchvision.utils.make_grid(self.s_out, normalize=True)
        out_teacher = torchvision.utils.make_grid(self.t_out, normalize=True)
        s_gram = torchvision.utils.make_grid(self.s_gram.unsqueeze(0), normalize=True)
        t_gram = torchvision.utils.make_grid(self.t_gram.unsqueeze(0), normalize=True)
        self.writer.add_image('in/img', in_img, step)
        self.writer.add_image('out/student', out_student, step)
        self.writer.add_image('out/teacher', out_teacher, step)
        self.writer.add_image('Feats/student', s_gram, step)
        self.writer.add_image('Feats/teacher', t_gram, step)

        self.writer.add_scalar('L1', self.loss_out, step)

    def print_summary(self, epoch, step):
        print('Epoch %d [%d | %d] -> Loss : %.2f | Feat2 Loss : %.2f | Out Loss : %.2f'
              % (epoch, step, opt.train_len, self.total_loss, self.loss_feats, self.loss_out))
