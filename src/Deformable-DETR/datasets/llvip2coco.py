import os
import json
import xml.etree.ElementTree as ET
from tqdm import tqdm
import argparse
import glob
import numpy as np
import random

# visible and ir in llvip are the same because they are aligned
# python voc2coco.py --annotation_path ./Annotations/ --json_save_path llvip_rgb_train.json --dataset ./visible/train/
# python voc2coco.py --annotation_path ./Annotations/ --json_save_path llvip_rgb_test.json --dataset ./visible/test/
def seed_everything(seed: int):

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


def parse_opt():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--annotation_path', type=str, default='/root/LLVIP/Annotations', help='folder containing xml files')
    parser.add_argument('--json_save_path', type=str, default='/root/LLVIP/LLVIP.json', help='json file')
    parser.add_argument('--dataset', type=str, default='./visible/train/', help='Img path')
    parser.add_argument('--seed', type=int, default=42, help='Seed for split the train/valid')
    parser.add_argument('--percentage-split', type=float, default=0.9, help='Split percentage for train/validate')
    opt = parser.parse_args()
    return opt


def addCatItem(name):
    global category_item_id
    category_item = dict()
    category_item['supercategory'] = 'none'
    category_item_id += 1
    category_item['id'] = category_item_id
    category_item['name'] = name
    coco['categories'].append(category_item)
    category_set[name] = category_item_id
    return category_item_id


def addImgItem(file_name, size):
    global image_id
    if file_name is None:
        raise Exception('Could not find filename tag in xml file.')
    if size['width'] is None:
        raise Exception('Could not find width tag in xml file.')
    if size['height'] is None:
        raise Exception('Could not find height tag in xml file.')
    image_id += 1
    image_item = dict()
    image_item['id'] = int(file_name.split(".")[0].split('_')[1]) if 'FLIR' else int(file_name.split(".")[0])
    image_item['file_name'] = file_name
    image_item['width'] = size['width']
    image_item['height'] = size['height']
    coco['images'].append(image_item)
    image_set.add(file_name)
    return image_id


def addAnnoItem(object_name, image_id, category_id, bbox):
    global annotation_id
    annotation_item = dict()
    annotation_item['segmentation'] = []
    seg = []
    # bbox[] is x,y,w,h
    # left_top
    seg.append(bbox[0])
    seg.append(bbox[1])
    # left_bottom
    seg.append(bbox[0])
    seg.append(bbox[1] + bbox[3])
    # right_bottom
    seg.append(bbox[0] + bbox[2])
    seg.append(bbox[1] + bbox[3])
    # right_top
    seg.append(bbox[0] + bbox[2])
    seg.append(bbox[1])
 
    annotation_item['segmentation'].append(seg)
 
    annotation_item['area'] = bbox[2] * bbox[3]
    annotation_item['iscrowd'] = 0
    annotation_item['ignore'] = 0
    annotation_item['image_id'] = int(image_id)
    annotation_item['bbox'] = bbox
    annotation_item['category_id'] = int(category_id)
    annotation_id += 1
    annotation_item['id'] = int(annotation_id)
    coco['annotations'].append(annotation_item)

def parseXmlFiles(xml_path,json_save_path,dataset_list):
    
    if('FLIR' in dataset_list[0]):
        dataset_list = [x.split('/')[-1].split('.')[0] for x in dataset_list]
    
    for f in tqdm(os.listdir(xml_path)):
        if not f.endswith('.xml'):
            continue
        
        if sum([f.split('.')[0] in x for x in dataset_list]) == 0:
           continue

        bndbox = dict()
        size = dict()
        current_image_id = None
        current_category_id = None
        file_name = None
        size['width'] = None
        size['height'] = None
        size['depth'] = None

        xml_file = os.path.join(xml_path, f)
        #print(xml_file)

        tree = ET.parse(xml_file)
        root = tree.getroot()
        if root.tag != ('Annotation' if 'FLIR' in dataset_list[0] else 'annotation'):
            raise Exception('pascal voc xml root element should be annotation, rather than {}'.format(root.tag))

        # elem is <folder>, <filename>, <size>, <object>
        for elem in root:
            current_parent = elem.tag
            current_sub = None
            object_name = None

            if elem.tag == 'folder':
                continue

            if elem.tag == 'filename':
                file_name = elem.text
                # file_name = file_name.split('.jpeg')[0] + '_PreviewData' + '.jpeg' if 'FLIR' in dataset_list[0] else file_name
                file_name = file_name.split('.jpeg')[0] + '_RGB' + '.jpg' if 'FLIR' in dataset_list[0] else file_name ## TO DO: SEPARATE CORRECTLY IR AND RGB FOR FLIR
                if file_name in category_set:
                    raise Exception('file_name duplicated')

            # add img item only after parse <size> tag
            elif current_image_id is None and file_name is not None and size['width'] is not None:
                if file_name not in image_set:
                    current_image_id = addImgItem(file_name, size)
                    #print('add image with {} and {}'.format(file_name, size))
                else:
                    raise Exception('duplicated image: {}'.format(file_name))
                    # subelem is <width>, <height>, <depth>, <name>, <bndbox>
            for subelem in elem:
                bndbox['xmin'] = None
                bndbox['xmax'] = None
                bndbox['ymin'] = None
                bndbox['ymax'] = None

                current_sub = subelem.tag
                if current_parent == 'object' and subelem.tag == 'name':
                    object_name = subelem.text
                    if object_name not in category_set:
                        current_category_id = addCatItem(object_name)
                    else:
                        current_category_id = category_set[object_name]

                elif current_parent == 'size':
                    if size[subelem.tag] is not None:
                        raise Exception('xml structure broken at size tag.')
                    size[subelem.tag] = int(subelem.text)

                # option is <xmin>, <ymin>, <xmax>, <ymax>, when subelem is <bndbox>
                for option in subelem:
                    if current_sub == 'bndbox':
                        if bndbox[option.tag] is not None:
                            raise Exception('xml structure corrupted at bndbox tag.')
                        bndbox[option.tag] = int(option.text)

                # only after parse the <object> tag
                if bndbox['xmin'] is not None:
                    if object_name is None:
                        raise Exception('xml structure broken at bndbox tag')
                    if current_image_id is None:
                        raise Exception('xml structure broken at bndbox tag')
                    if current_category_id is None:
                        raise Exception('xml structure broken at bndbox tag')
                    bbox = []
                    # x
                    bbox.append(bndbox['xmin'])
                    # y
                    bbox.append(bndbox['ymin'])
                    # w
                    bbox.append(bndbox['xmax'] - bndbox['xmin'])
                    # h
                    bbox.append(bndbox['ymax'] - bndbox['ymin'])
                    #print('add annotation with {},{},{},{}'.format(object_name, current_image_id, current_category_id, bbox))
                    addAnnoItem(object_name, int(f.split(".")[0].split('_')[1]) if 'FLIR' else int(f.split(".")[0])
                                , int(current_category_id), bbox)

    json.dump(coco, open(json_save_path, 'w'))


if __name__ == '__main__':
    opt = parse_opt()
    seed_everything(opt.seed)

    coco = dict()
    coco['images'] = []
    coco['type'] = 'instances'
    coco['annotations'] = []
    coco['categories'] = []

    category_set = dict()
    image_set = set()

    category_item_id = -1
    image_id = 00000
    annotation_id = 0

    dataset=opt.dataset
    dataset_list = glob.glob(dataset + '*.jpeg' if 'FLIR' in dataset else dataset + '*.jpg')
    ann_path=opt.annotation_path
    json_save_path=opt.json_save_path


    if not('FLIR' in dataset):
        if('train' in dataset):
            dataset_train, dataset_valid = np.split(dataset_list, [int(len(dataset_list)*opt.percentage_split)])
            valid_path = json_save_path.split('.')[0].split('train')[0]+'valid.json'

            parseXmlFiles(ann_path, json_save_path, dataset_train)

            coco = dict()
            coco['images'] = []
            coco['type'] = 'instances'
            coco['annotations'] = []
            coco['categories'] = []

            category_set = dict()
            image_set = set()

            category_item_id = -1
            image_id = 00000
            annotation_id = 0

            parseXmlFiles(ann_path, valid_path, dataset_valid)

        else:
            parseXmlFiles(ann_path,json_save_path, dataset_list)
    # FLIR    
    else:
        train_flag = False
        
        if train_flag:
            print(dataset.split('/JPEGImages')[0])
            import os
            with open(os.path.join(dataset.split('/JPEGImages')[0], 'align_train.txt')) as file:
                lines = [line.rstrip() for line in file]
            dataset_list = [x for x in dataset_list if x.split('/')[-1].split('.')[0] in lines]
            
            dataset_train, dataset_valid = np.split(dataset_list, [int(len(dataset_list)*opt.percentage_split)])
            valid_path = json_save_path.split('.')[0].split('train')[0]+'valid.json'

            parseXmlFiles(ann_path, json_save_path, dataset_train)

            coco = dict()
            coco['images'] = []
            coco['type'] = 'instances'
            coco['annotations'] = []
            coco['categories'] = []

            category_set = dict()
            image_set = set()

            category_item_id = -1
            image_id = 00000
            annotation_id = 0

            parseXmlFiles(ann_path, valid_path, dataset_valid)
            
            print(len(dataset_train), len(dataset_valid))

        else:
            import os
            with open(os.path.join(dataset.split('/JPEGImages')[0], 'align_validation.txt')) as file:
                lines = [line.rstrip() for line in file]
            dataset_list = [x for x in dataset_list if x.split('/')[-1].split('.')[0] in lines]
            parseXmlFiles(ann_path,json_save_path, dataset_list)