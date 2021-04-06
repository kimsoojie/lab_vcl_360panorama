from util.base import *
from util.opt import Options
import util.utilities as utl
from util.utilities import min_visualize, calc_quanti, to_numpy, save_img, calc_l1
from dataset.dataset import Dataset
from model.Networks import Networks
from model.Networks2 import Networks2
from scipy.stats import gaussian_kde
import model.ops as ops
import model.models3 as m3
import time
import onnx
# import caffe2.python.onnx.backend as backend


opt = Options(sys.argv[0])
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = 'cuda:0'

def write_output_single(folder_path, im_name, model_name, net_type=None):
    # Init network
    generator = m3.GM().to(device)
    generator.eval()
    model_path = os.path.join(opt.model_path, model_name + '.pt')
    model = torch.load(model_path)
    generator.load_state_dict(model['Generator'], strict=True)
    # generator = make_onnx_backend(imdir, model_name, device)

    # Init input
    im_name_splt = im_name.split('.')
    in_img = read_img_to_tensor('/var/www/html/img_out.jpg')
    in_img = torch.unsqueeze(in_img, 0)
    in_img = in_img.to(device)
    in_img = ops.downsample(in_img)

    # out_s, out_m = generator.run(in_img.numpy())
    out_s, out_m = generator(in_img)

    # torchvision.utils.save_image(out_m, '/var/www/html/' + im_name_splt[0] + '_m.png', normalize=True)
    save_img_from_tensor(imdir + '/trained_output.jpg', out_m)
    # out_m = np.squeeze(out_m, dim=0)
    # out_m = np.transpose(out_m, (1,2,0))
    # out_m = (out_m + 1)/2 * 255.0
    # out_m = cv2.cvtColor(out_m, cv2.COLOR_RGB2BGR)
    # cv2.imwrite(folder_path + '/trained_output.jpg', out_m.astype(np.uint8))
    print("finished saving image", out_m.shape)

def postproc_img(img):
    out_m = np.squeeze(img, 0)
    out_m = np.transpose(out_m, (1,2,0))
    out_m = (out_m + 1)/2 * 255.0
    out_m = cv2.cvtColor(out_m, cv2.COLOR_RGB2BGR)
    return out_m

def read_img_to_tensor(im_path):
    im = cv2.imread(im_path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = im / 127.5 - 1
    tsr = torch.from_numpy(im.transpose(2,0,1))
    tsr = tsr.type(torch.FloatTensor)
    return tsr

def save_img_from_tensor(im_path, img_tensor):
    img = to_numpy(img_tensor) * 255.0
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(im_path, img.astype(np.uint8))

def convert_to_onnx(model_name):
    generator = m3.GTest()
    # generator = m3.GM()
    model_path = os.path.join(opt.model_path, model_name + '.pt')
    model = torch.load(model_path)
    generator.load_state_dict(model['Generator'])
    input_np = np.random.uniform(0, 1, (1, 3, 224, 224))
    input_var = torch.FloatTensor(input_np)
    output_path = os.path.join(opt.model_path, 'test_model' + ".onnx")
    torch.onnx.export(generator, input_var, output_path)



def convert_to_keras(model_name):
    from pytorch2keras import pytorch_to_keras
    import tensorflow as tf
    import onnx
    from onnx2keras import onnx_to_keras, check_torch_keras_error

    tf.keras.backend.set_image_data_format('channels_first')
    in_img = read_img_to_tensor('/var/www/html/img_out.jpg')
    in_img = torch.unsqueeze(in_img, 0)
    in_img = ops.downsample(in_img)

    model_path = os.path.join(opt.model_path, model_name + '.pt')
    onnx_path = os.path.join(opt.model_path, 'model_190712/test_model2' + '.onnx')
    keras_path = os.path.join(opt.model_path, model_name + '.h5')
    model = torch.load(model_path)

    # # Convert to ONNX
    generator = m3.GTest()
    generator.load_state_dict(model['Generator'])
    input_np = np.random.uniform(0, 1, (1, 3, 224, 224))
    # input_var = torch.FloatTensor(input_np)
    # torch.onnx.export(generator, (input_var), onnx_path, verbose=True,
    #                   input_names=['input'],
    #                   output_names=['output'])

    # convert to keras
    onnx_model = onnx.load(onnx_path)
    k_model = onnx_to_keras(onnx_model, ['input'], change_ordering=False)
    error = check_torch_keras_error(generator, k_model, input_np)
    print('Error: {0}'.format(error))  #  1e-6 :)
    input('... press next ...')

    k_model.summary()

    # inpvar = torch.autograd.Variable(torch.FloatTensor(inp))
    output_path = os.path.join(opt.model_path, 'model_190712/test_model' + ".h5")
    # k_model = pytorch_to_keras(generator, inpvar, [(3, 256, 512)], verbose=True)
    # print(output_path)
    # k_model.summary()
    # out = k_model.predict(in_img.numpy(),batch_size=1)
    # out_m = postproc_img(out)
    # cv2.imwrite('/var/www/html' + '/trained_output_keras.jpg', out_m.astype(np.uint8))
    # tf.keras.models.save_model(k_model, output_path)

def predict_keras(model_name):
    import tensorflow as tf
    in_img = read_img_to_tensor('/var/www/html/img_out.jpg')
    in_img = torch.unsqueeze(in_img, 0)
    in_img = ops.downsample(in_img)

    out_path = os.path.join(opt.model_path, model_name + '.tflite')
    model_path = os.path.join(opt.model_path, model_name + '.h5')
    # converter = tf.lite.TFLiteConverter.from_keras_model_file(model_path)
    # tflite_model = converter.convert()
    # open(out_path, "wb").write(tflite_model)
    # k_model = load_model(model_path, compile=False)
    # k_model.summary()
    # out = k_model.predict(in_img.numpy(), batch_size=1)
    # print(out.shape)

def make_onnx_backend(im_dir, model_name, device):
    model_path = os.path.join(opt.model_path, model_name + '.onnx')
    model = onnx.load(model_path)
    net = backend.prepare(model, device='CUDA')
    return net

def onnx2tf(model_name):
    import tensorflow as tf
    model_path = os.path.join(opt.model_path, model_name + '.pb')
    out_path = os.path.join(opt.model_path, model_name + '.tflite')
    converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
    tflite_model = converter.convert()
    open(out_path, "wb").write(tflite_model)
    # with tf.Session() as sess:
    #     with tf.gfile.GFile(model_path, 'rb') as f:
    #         graph_def = tf.GraphDef()
    #         graph_def.ParseFromString(f.read())
    #         g_in = tf.import_graph_def(graph_def, name='')
    #
    #     ns = [n.name for n in tf.get_default_graph().as_graph_def().node]
    #     for n in ns:
    #         print(n)
    #         input('....')
        # b = sess.graph.get_tensor_by_name('input:0')
        # print(b)

# convert_to_onnx('model_190712/model_n_medium_30000')
# convert_to_keras('model_190712/model_n_medium_30000')
# predict_keras('model_190712/test_model')
# onnx2tf('model_190712/test_model2')

imdir = '/var/www/html/'
write_output_single(imdir, 'trained_input.jpg',
                    model_name='model_190712/model_n_medium_30000',
                    net_type='medium')
