from six.moves import cPickle
import lasagne
import theano
from theano import tensor as T
import numpy as np
import six

class CNNModel:

    def __init__(self, model_factory, model_weight_path):

        with open(model_weight_path, 'rb') as f:
            if six.PY2:
                model_params = cPickle.load(f)
            else:
                model_params = cPickle.load(f, encoding='latin1')

        self.input_size = model_params['input_size']
        self.img_size = model_params['img_size']

        net_input_size = (None, 1, self.input_size[0], self.input_size[1])
        self.model = model_factory.build_architecture(net_input_size,
                                                      model_params['params'])

        self.forward_util_layer = {}  

    def get_feature_vector(self, image, layer='fc2'):

        assert len(image.shape) == 2, "Input should have two dimensions: H x W"

        input = image[np.newaxis, np.newaxis]

        if layer not in self.forward_util_layer:
            inputs = T.tensor4('inputs')
            outputs = lasagne.layers.get_output(self.model[layer],
                                                inputs=inputs,
                                                deterministic=True)
            self.forward_util_layer[layer] = theano.function([inputs], outputs)

        out = self.forward_util_layer[layer](input)
        return out

    def get_feature_vector_multiple(self, images, layer='fc2'):

        images = np.asarray(images)
        assert len(images.shape) == 3, "Input should have three dimensions: N x H x W"

        input = np.expand_dims(images, axis=1)

        if layer not in self.forward_util_layer:
            inputs = T.tensor4('inputs')
            outputs = lasagne.layers.get_output(self.model[layer],
                                                inputs=inputs,
                                                deterministic=True)
            self.forward_util_layer[layer] = theano.function([inputs], outputs)

        out = self.forward_util_layer[layer](input)
        return out
