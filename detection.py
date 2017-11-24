import argparse, time
import os
import json

import numpy as np

import cv2
import mxnet as mx

from .symbol.resnet import *
from .symbol.config import config
from .symbol.processing import bbox_pred, clip_boxes, nms

import logging


def ch_dev(arg_params, aux_params, ctx):
    """Copy parameters to new MXNet context (GPU or CPU)"""
    new_args = dict()
    new_auxs = dict()
    for k, v in list(arg_params.items()):
        new_args[k] = v.as_in_context(ctx)
    for k, v in list(aux_params.items()):
        new_auxs[k] = v.as_in_context(ctx)
    return new_args, new_auxs

def resize(im, target_size, max_size):
    """
    only resize input image to target size and return scale
    :param im: BGR image input by opencv
    :param target_size: one dimensional size (the short side)
    :param max_size: one dimensional max size (the long side)
    :return:
    """
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min)
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
    return im, im_scale

def write_image(detections, scale, original_image):
    """Write image with bounding boxes to file"""
    for i in range(detections.shape[0]):
        bbox = detections[i, :4]
        cv2.rectangle(original_image, (int(round(bbox[0]/scale)), int(round(bbox[1]/scale))),
                      (int(round(bbox[2]/scale)), int(round(bbox[3]/scale))),  (0, 255, 0), 2)
    cv2.imwrite("result.jpg", original_image)

def execute_detection(image_path, scale, max_scale, prefix, epoch, thresh, nms_thresh, gpu_id=None, cpu_id=None):
    original_image  = cv2.imread(image_path)
    transformed_image  = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    transformed_image, scale = resize(transformed_image.copy(), scale, max_scale)
    image_info = np.array([[transformed_image.shape[0], transformed_image.shape[1], scale]], dtype=np.float32)  # (h, w, scale)
    transformed_image = np.swapaxes(transformed_image, 0, 2)
    transformed_image = np.swapaxes(transformed_image, 1, 2)  # change to (c, h, w) order
    transformed_image = transformed_image[np.newaxis, :]  # extend to (n, c, h, w)

    #-------
    # Setup network
    #-------
    if gpu_id is not None:
        ctx = mx.gpu(gpu_id)
    elif cpu_id is not None:
        ctx = mx.cpu(cpu_id)
    else:
        raise TypeError('Either gpu_id or cpu_id need to be provided!')
    # Load parameters of trained model into RAM
    _, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
    # Move parameters to GPUs RAM
    arg_params, aux_params = ch_dev(arg_params, aux_params, ctx)
    # Create ResNet as MXNet Symbol
    sym = resnet_50(num_class=2)
    # Load image data into parameters on GPU
    arg_params["data"] = mx.nd.array(transformed_image, ctx)
    arg_params["im_info"] = mx.nd.array(image_info, ctx)
    # Bind ResNet to executor
    exe = sym.bind(ctx, arg_params, args_grad=None, grad_req="null", aux_states=aux_params)

    #-------
    # Run network
    #-------
    # Start time measurement
    tic = time.time()
    # Execute neural network
    exe.forward(is_train=False)
    # Group output
    output_dict = {name: nd for name, nd in zip(sym.list_outputs(), exe.outputs)}
    rois = output_dict['rpn_rois_output'].asnumpy()[:, 1:]  # first column is index
    scores = output_dict['cls_prob_reshape_output'].asnumpy()[0]
    bbox_deltas = output_dict['bbox_pred_reshape_output'].asnumpy()[0]
    # Predict bounding boxes
    pred_boxes = bbox_pred(rois, bbox_deltas)
    pred_boxes = clip_boxes(pred_boxes, (image_info[0][0], image_info[0][1]))
    cls_boxes = pred_boxes[:, 4:8]
    cls_scores = scores[:, 1]
    keep = np.where(cls_scores >= thresh)[0]
    cls_boxes = cls_boxes[keep, :]
    cls_scores = cls_scores[keep]
    detections = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
    keep = nms(detections.astype(np.float32), nms_thresh)
    detections = detections[keep, :]

    #End time measurement
    toc = time.time()

    print("time cost is:{}s".format(toc-tic))

    output_detections = detections.tolist()

    return output_detections, scale

def main():
    parser = argparse.ArgumentParser(description="use pre-trainned resnet model to classify one image")
    parser.add_argument('--img', type=str, default='test.jpg', help='input image/directory for classification')
    parser.add_argument('--gpu', type=int, default=0, help='the gpu id used for predict')
    parser.add_argument('--prefix', type=str, default='mxnet-face-fr50', help='the prefix of the pre-trained model')
    parser.add_argument('--epoch', type=int, default=0, help='the epoch of the pre-trained model')
    parser.add_argument('--thresh', type=float, default=0.1, help='the threshold of face score, set bigger will get more'
                                                                  'likely face result')
    parser.add_argument('--nms-thresh', type=float, default=0.3, help='the threshold of nms')
    parser.add_argument('--min-size', type=int, default=24, help='the min size of object')
    parser.add_argument('--scale', type=int, default=600, help='the scale of shorter edge will be resize to')
    parser.add_argument('--max-scale', type=int, default=1000, help='the maximize scale after resize')
    args = parser.parse_args()
    config.END2END = 1
    config.TEST.HAS_RPN = True
    config.TEST.RPN_MIN_SIZE = args.min_size
    config.SCALES = (args.scale, )
    config.MAX_SIZE = args.max_scale

    args_path = os.path.expanduser(args.img)

    if os.path.isfile(args_path):
        execute_detection(args.img,
                          args.scale,
                          args.max_scale,
                          args.prefix,
                          args.epoch,
                          args.thresh,
                          args.nms_thresh,
                          args.gpu_id
                         )

        print("Done.")

    elif os.path.isdir(args_path):
        dir_path = os.path.expanduser(args.img)
        image_paths = []
        for path in os.listdir(dir_path):
            absolute_path = os.path.abspath(os.path.join(dir_path, path))
            # Check if item is a valid image
            if cv2.imread(absolute_path) is not None:
                image_paths.append(absolute_path)
            else:
                print("Skipping path, not an image: {}".format(path))

        print("Images to process: ")
        for image_path in image_paths:
            print(image_path)

        all_detections = {}
        print("Starting processing.")
        for image_path in image_paths:
            detections, scale = \
                execute_detection(image_path,
                                  args.scale,
                                  args.max_scale,
                                  args.prefix,
                                  args.epoch,
                                  args.thresh,
                                  args.nms_thresh,
                                  args.gpu_id
                                 )
            all_detections[os.path.basename(image_path)] = {
                "scale" : scale,
                "detections" : detections
            }

        print("Writing results to file.")
        with open('results.json', 'w') as results_file:
            json.dump(all_detections, results_file, indent=2)

        print("Done.")

    else:
        print("{} is not a valid path to a file nor directory.".format(args.img))

if __name__ == "__main__":
    main()
