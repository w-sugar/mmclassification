from argparse import ArgumentParser

from mmcls.apis import inference_model, init_model, show_result_pyplot
import json
import os
import cv2
import numpy as np

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

def image_bbox(gt_json, dt_json):
    gt_contents = json.load(open(gt_json, 'r'))
    dt_contents = json.load(open(dt_json, 'r'))
    image_id_name = {}
    for i in gt_contents['images']:
        image_id_name[i['id']] = i['file_name']
    
    image_id_results = {}
    for i in dt_contents:
        image_id_results.setdefault(i['image_id'], [])
        image_id_results[i['image_id']].append(i)
    return image_id_name, image_id_results

def main():
    # parser = ArgumentParser()
    # parser.add_argument('img', help='Image file')
    # parser.add_argument('config', help='Config file')
    # parser.add_argument('checkpoint', help='Checkpoint file')
    # parser.add_argument(
    #     '--device', default='cuda:0', help='Device used for inference')
    # args = parser.parse_args()

    gt_annotations = '/home/sugar/workspace/mmdetection/data/visdrone/annotations/coco-test-dev.json'
    det_result = '/home/sugar/workspace/mmdetection/work_dirs/cascade_rcnn_r50_fpn_1x_coco_del_ignore/del-ignore.bbox.json'
    config = '/home/sugar/workspace/mmclassification/configs/visdrone/resnet50_b32x8_imagenet.py'
    checkpoint = '/home/sugar/workspace/mmclassification/work_dirs/resnet50_b32x8_imagenet/epoch_5.pth'
    image_root = '/home/sugar/workspace/mmdetection/data/visdrone/images/VisDrone2019-DET-test-dev/images'
    det_resulty_new = '/home/sugar/workspace/mmdetection/work_dirs/cascade_rcnn_r50_fpn_1x_coco_del_ignore/del-ignore-new.bbox.json'
    # load-image
    image_id_name, image_id_results = image_bbox(gt_annotations, det_result)
    print('laod image done!')

    # build the model from a config file and a checkpoint file
    model = init_model(config, checkpoint, device='cuda:3')
    print('build model done!')

    det_new = []
    for image_id, results in image_id_results.items():
        image_name = image_id_name[image_id]
        img = cv2.imread(os.path.join(image_root, image_name))
        for result in results:
            bbox = result['bbox']
            if result['score'] < 0.3:
                det_new.append(result)
                continue
            x1 = bbox[0]
            y1 = bbox[1]
            x2 = x1 + bbox[2]
            y2 = y1 + bbox[3]
            cropped = img[int(y1):int(y2), int(x1):int(x2)]

            # test a single image
            result_cls = inference_model(model, cropped)
            if result_cls['pred_label'] == 0:
                continue
            score = result_cls['pred_score']
            category_id = result_cls['pred_label']
            dict_ = {
                'image_id': image_id,
                'bbox': bbox,
                'score': score,
                'category_id': category_id
            }
            det_new.append(dict_)
    with open(det_resulty_new, 'w') as w:
        json.dump(det_new, w, indent=2, cls=NpEncoder)

    # # show the results
    # show_result_pyplot(model, args.img, result)


if __name__ == '__main__':
    main()
