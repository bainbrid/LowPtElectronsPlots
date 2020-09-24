# conda create -n tf tensorflow 
# conda create --clone tf --name tf-dev
# conda activate tf-dev
# conda install uproot
# conda install matplotlib
# conda install pillow (PIL not supported anymore, just use pillow as you would PIL, from PIL import ...)

import tensorflow as tf
import numpy as np
import uproot
import os
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
import collections 

################################################################################

def init() :
    global base_path
    global data_path
    global models_path
    base_path='/Users/bainbrid/Repositories/icenet/standalone'
    data_path = base_path+'/data'
    models_path = base_path+'/models'
    print('base_path:',f'{base_path}')
    print('data_path:',f'{data_path}')
    print('models_path:',f'{models_path}')
    print('TF version={}, CUDA={}'.format( tf.__version__, 
                                           tf.test.is_built_with_cuda(),
                                           #len(tf.config.list_physical_devices('GPU')) > 0, # GPU={}
                                           ))
    
def load_root_file(filename,VARS=['is_e','image_*']) :
    file = uproot.open(f'{data_path}/{filename}')
    tree = file['ntuplizer']['tree']
    events = tree.arrays(VARS, outputtype=collections.namedtuple)
    return events

def intensity(energy) : 
    return min(255, int(energy/0.1))

def save_img(array) :
    img = Image.fromarray(array)
    plt.imshow(np.asarray(img))
    plt.savefig(models_path+'/image.pdf')
    plt.clf()
    plt.close()

def event_to_protobuf(events,index) :
    example = tf.train.Example()
    assert index < events.is_e.shape[0], \
        "index: {:.0f}, events.is_e.shape[0]: {:.0f}".format(index,events.is_e.shape[0])
    example.features.feature['label'].int64_list.value.append(events.is_e[index])
    example.features.feature['clu_n'].int64_list.value.append(events.image_clu_n[index])
    example.features.feature['clu_eta'].float_list.value.extend(events.image_clu_eta[index])
    example.features.feature['clu_phi'].float_list.value.extend(events.image_clu_phi[index])
    example.features.feature['clu_e'].float_list.value.extend(events.image_clu_e[index])
    return example

def events_to_protobufs(events,nevent=-1) :
    nevent = min(nevent,len(events.is_e)) if nevent > 0 else len(events.is_e)
    protobufs = []
    for i in range(nevent) : 
        protobuf = event_to_protobuf(events,i)
        protobufs.append(protobuf)
    return protobufs

def protobuf_to_string(protobuf) :
    return protobuf.SerializeToString()

def string_to_protobuf(string) :
    return tf.train.Example.FromString(string)

def write_protobufs_to_tfrecord(protobufs,filename='output.tfrecords') :
    writer = tf.io.TFRecordWriter(filename)
    for protobuf in protobufs : writer.write(protobuf.SerializeToString())

def read_protobufs_from_tfrecord(filename='output.tfrecords') :
    description = {
        'label':   tf.io.FixedLenFeature([], tf.int64),
        'clu_n':   tf.io.FixedLenFeature([], tf.int64),
        'clu_eta': tf.io.VarLenFeature(tf.float32),
        'clu_phi': tf.io.VarLenFeature(tf.float32),
        'clu_e':   tf.io.VarLenFeature(tf.float32),
        }
    raw_protobufs = tf.data.TFRecordDataset(filename)
    protobufs = []
    for raw in raw_protobufs : 
        protobufs.append( tf.io.parse_single_example(raw,description) )
    return protobufs



################################################################################

if __name__ == "__main__" :

    init()
    events = load_root_file('temp_miniaod_test.root')
    protobufs = events_to_protobufs(events,nevent=10)
    write_protobufs_to_tfrecord(protobufs,filename='output.tfrecords')
    protobufs = read_protobufs_from_tfrecord(filename='output.tfrecords')

    for image_features in parsed_image_dataset:
        image_raw = image_features['image_raw'].numpy()
        display.display(display.Image(data=image_raw))

    print(len(protobufs))
    for protobuf in protobufs : print(protobuf) 
    
    #import time
    #start_time = time.time()
    #print("--- %s seconds ---" % (time.time() - start_time))


