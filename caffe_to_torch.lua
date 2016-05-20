-- Convert a pretrained Caffe model to a Torch model.

local argparse = require('argparse')
local loadcaffe = require('loadcaffe')

local parser = argparse() {
    description = 'Convert a pretrained Caffe model to a Torch model.'
}

parser:argument('caffe_prototxt', 'Caffe model prototxt')
parser:argument('caffe_model', 'Caffe model weights')
parser:argument('torch_output', 'Where to output the torch model.')
parser:option('--backend', 'Backend to use.', 'cudnn')

args = parser:parse()

model = loadcaffe.load(args.caffe_prototxt, args.caffe_model, args.backend)
torch.save(args.torch_output, model)
