from cStringIO import StringIO
from IPython.display import clear_output, Image, display
from google.protobuf import text_format

import numpy as np
import scipy.ndimage as nd
import PIL.Image

import argparse

import sys

import caffe


def parseargs():
    # Kindly provided by Bystroushaak's `argparse builder`
    # http://kitakitsune.org/argparse_builder/argparse_builder.html
    parser = argparse.ArgumentParser(
        'Generate a "deepdream" image from a JPEG input.')
    parser.add_argument('--infile', '-i', required=True,
                        help='''The input image filename. Required. Passing in ''' +
                        '''the "hidden" parameter 'keys' instead of a filename ''' +
                        '''will dump the model's layer names to stdout.''')
    parser.add_argument('--outfile', '-o', default='out',
                        help='The output image filename. Defaults to <INFILE.png>.')
    parser.add_argument('--end-name', '-e', default='inception_4c/output',
                        help='The name of the layer to capture output from. For the ' +
                        'Google dataset, defaults to "inception_4c/output".')
    parser.add_argument('--iter-n', '-n', default=10, type=int,
                        help='Number of iterations per octave. Default 10.')
    parser.add_argument('--octaves', '-8', default=4, type=int,
                        help='Number of octaves. Default 4.')
    parser.add_argument('--oct-scale', '-s', default=1.4, type=float,
                        help='Octave scale. Default 1.4.')
    return parser.parse_args()


def savearray(a, filename, fmt='png'):
    a = np.uint8(np.clip(a, 0, 255))
    with open(filename, 'wb') as f:
        PIL.Image.fromarray(a).save(f, fmt)
        # display(Image(data=f.getvalue()))


# A couple of utility functions for converting to and from Caffe's input
# image layout.
def preprocess(net, img):
    return np.float32(np.rollaxis(img, 2)[::-1]) - net.transformer.mean['data']


def deprocess(net, img):
    return np.dstack((img + net.transformer.mean['data'])[::-1])


def make_step(net, step_size=1.5, end='inception_4c/output', jitter=32, clip=True):
    '''Basic gradient ascent step.'''
    # Input image is stored in Net's 'data' blob.
    src = net.blobs['data']
    dst = net.blobs[end]
    ox, oy = np.random.randint(-jitter, jitter + 1, 2)

    # Apply jitter shift.
    src.data[0] = np.roll(np.roll(src.data[0], ox, -1), oy, -2)
    net.forward(end=end)

    # Specify the optimization objective.
    dst.diff[:] = dst.data
    net.backward(start=end)
    g = src.diff[0]

    # Apply normalized ascent step to the input image.
    src.data[:] += step_size / np.abs(g).mean() * g
    src.data[0] = np.roll(
        np.roll(src.data[0], -ox, -1), -oy, -2)  # Unshift image.
    if clip:
        bias = net.transformer.mean['data']
        src.data[:] = np.clip(src.data, -bias, 255 - bias)


def deepdream(net, base_img, iter_n=10, octave_n=4, octave_scale=1.4, end='inception_4c/output', clip=True, **step_params):
    # Prepare base images for all octaves.
    octaves = [preprocess(net, base_img)]
    for i in xrange(octave_n - 1):
        octaves.append(
            nd.zoom(octaves[-1], (1, 1.0 / octave_scale, 1.0 / octave_scale), order=1))
    src = net.blobs['data']

    # Allocate image for network-produced details.
    detail = np.zeros_like(octaves[-1])
    for octave, octave_base in enumerate(octaves[::-1]):
        h, w = octave_base.shape[-2:]
        if octave > 0:
            # Upscale details from the previous octave.
            h1, w1 = detail.shape[-2:]
            detail = nd.zoom(detail, (1, 1.0 * h / h1, 1.0 * w / w1), order=1)
        src.reshape(1, 3, h, w)  # Resize the network's input image size.
        src.data[0] = octave_base + detail
        for i in xrange(iter_n):
            make_step(net, end=end, clip=clip, **step_params)
            # Visualization.
            vis = deprocess(net, src.data[0])
            # Adjust image contrast if clipping is disabled.
            if not clip:
                vis = vis * (255.0 / np.percentile(vis, 99.98))
            # showarray(vis)
            print octave, i, end, vis.shape
            clear_output(wait=True)

        # Extract details produced on the current octave.
        detail = src.data[0] - octave_base
    # Return the resulting image.
    return deprocess(net, src.data[0])


def main(args):
    try:
        # Substitute site-specific info here.
        model_path = '/home/vagrant/caffe/models/bvlc_googlenet/'
        net_fn = model_path + 'deploy.prototxt'
        param_fn = model_path + 'bvlc_googlenet.caffemodel'

        # Patching model to be able to compute gradients.
        # Note that you can also manually add "force_backward: true" line to
        # "deploy.prototxt".
        model = caffe.io.caffe_pb2.NetParameter()
        text_format.Merge(open(net_fn).read(), model)
        model.force_backward = True
        open('tmp.prototxt', 'w').write(str(model))
        net = caffe.Classifier('tmp.prototxt', param_fn,
                               # ImageNet mean, training set dependent.
                               mean=np.float32([104.0, 116.0, 122.0]),
                               # The reference model has channels in BGR order instead
                               # of RGB.
                               channel_swap=(2, 1, 0))
        if 'keys' == args.infile:
            print('\n'.join(net.blobs.keys()))
            return 0

        img = np.float32(PIL.Image.open(args.infile))
        output = deepdream(net, img, end=args.end_name, iter_n=args.iter_n,
                           octave_n=args.octaves, octave_scale=args.oct_scale)
        savearray(output, args.outfile)
        return 0

    except Exception as e:
        print(str(e))
        return 2


if __name__ == '__main__':
    args = parseargs()
    if args.outfile == 'out':
        # Split the filename from the extension and replace with 'png'.
        args.outfile = args.infile.split('.')[0] + '.png'
    sys.exit(main(args))
