import torch
from predictor import COCODemo
import pandas as pd
import os
import cv2
import numpy as np
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.utils.visualize import vis_image
from PIL import Image
import time
import argparse
import pickle
import dill

os.environ["CUDA_VISIBLE_DEVICES"] = "15"


def format_prediction_string(boxes, scores):
    pred_strings = []
    for j in zip(scores, boxes):
        pred_strings.append("{0:.4f} {1} {2} {3} {4}".format(j[0], j[1][0], j[1][1], j[1][2], j[1][3]))

    return " ".join(pred_strings)


def main(cfg, args):

    demo_folder = os.path.split(os.path.split(cfg.MODEL.WEIGHT)[0])[-1]
    if not os.path.exists(demo_folder):
        os.mkdir('./' + demo_folder)

    result_dir = os.path.join('results', args.config_file.split(
        '/')[-1].split('.')[0], cfg.MODEL.WEIGHT.split('/')[-1].split('.')[0])

    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)

    coco_demo = COCODemo(
        cfg,
        confidence_threshold=0.9,
    )
    with open("/home/zhangziwei/wheat_det/coco_demo.pkl", "wb") as f:
        f = dill.dump(coco_demo, f, True, True, True)
    exit()

    image_dir = "/home/zhangziwei/wheat_det/test"
    # vocab_dir = testing_dataset[dataset_name]['test_vocal_dir']

    # load image and then run prediction
    # image_dir = '../datasets/ICDAR13/Challenge2_Test_Task12_Images/'
    imlist = os.listdir(image_dir)

    print('************* META INFO ***************')
    print('config_file:', args.config_file)
    print('result_dir:', result_dir)
    print('image_dir:', image_dir)
    print('weights:', cfg.MODEL.WEIGHT)
    print('***************************************')

    save = True
    num_images = len(imlist)
    cnt = 0
    results = []
    for image in imlist:
        # if image != 'rotate_747.jpg':
        #     continue
        impath = os.path.join(image_dir, image)
        print('image:', impath)
        img = cv2.imread(impath)
        cnt += 1
        tic = time.time()
        predictions, bounding_boxes, score = coco_demo.run_on_opencv_image(img)
        toc = time.time()

        print('time cost:', str(toc - tic)[:6], '|', str(cnt) + '/' + str(num_images))

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        bboxes_np = bounding_boxes.bbox.data.cpu().numpy()
        scores_np = score.cpu().numpy()

        # print(bboxes_np.shape)
        # print(bboxes_np)
        # print(scores_np.shape)
        # print(scores_np)

        for i, bboxes in enumerate(bboxes_np):

            boxes = bboxes_np[i]
            scores = scores_np[i]

            boxes = boxes[scores >= 0.7]
            scores = scores[scores >= 0.7]

            boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
            boxes[:, 3] = boxes[:, 3] - boxes[:, 1]

            result = {
                'image_id': image.split(".")[0],
                'PredictionString': format_prediction_string(boxes, scores)
            }

            results.append(result)

        width, height = bounding_boxes.size
        print(image)
        print(score)

        if save:
            pil_image = vis_image(Image.fromarray(img), bboxes_np)
            pil_image.save(f'./{demo_folder}/result_{image}.png')

    test_df = pd.DataFrame(results, columns=['image_id', 'PredictionString'])
    print(test_df.head())
    test_df.to_csv('submission.csv', index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PyTorch Object Detection Training")
    parser.add_argument(
        "--config-file",
        default="/home/zhangziwei/wheat_det/maskrcnn-benchmark/configs/pascal_voc/e2e_faster_rcnn_X_101_32x8d_FPN_1x.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--skip-test",
        default=False,
        help="Do not test the final model",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    main(cfg, args)
