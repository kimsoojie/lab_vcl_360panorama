import os
import sys
from util.opt import Options
import tensorflow as tf
import onnx
from onnx_tf.backend import prepare
import numpy as np

opt = Options(sys.argv[0])

inp = np.random.rand(1,3,256,512)

# model_path = os.path.join(opt.model_path, 'model_190712/model_n_medium_30000.pb')
model_path = os.path.join(opt.model_path, 'model_190712/test_model.pb')
# model = onnx.load(model_path)
# model.ir_version = 2
# output = prepare(model).run(inp)

logdir =  '../log_tf'


with tf.Session() as sess:
    with tf.gfile.GFile(model_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')
        # graph_nodes = [n for n in graph_def.node]
    # writer = tf.summary.FileWriter(logdir)
    # writer.add_graph(sess.graph)

    for op in graph.get_operations():
        print(op.name)

    # for t in graph_nodes:
    #     print(t.name)
#
