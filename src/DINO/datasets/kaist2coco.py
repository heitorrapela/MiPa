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
    parser.add_argument('--dataset', type=str, default='../../datasets/kaist_multispectral/', help='Img path')
    parser.add_argument('--train',action='store_true')
    parser.add_argument('--json_save_path', type=str, default=None, help='Annotation path')
    parser.add_argument('--modality', type=str, default='rgb', help='rgb or ir')
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
    
    image_item['id'] = image_id
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
    annotation_item['image_id'] = image_id
    annotation_item['bbox'] = bbox
    annotation_item['category_id'] = int(category_id)
    annotation_id += 1
    annotation_item['id'] = int(annotation_id)
    coco['annotations'].append(annotation_item)

def parseXmlFiles(xml_path, json_save_path, dataset_list, train_flag, modality_flag):
            
    for f in tqdm(xml_path):
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

        xml_file = f
        tree = ET.parse(xml_file)
        root = tree.getroot()
        

        if root.tag != 'annotation':
            raise Exception('pascal voc xml root element should be annotation, rather than {}'.format(root.tag))

        # elem is <folder>, <filename>, <size>, <object>
        for elem in root:
            current_parent = elem.tag
            current_sub = None
            object_name = None

            if elem.tag == 'folder':
                continue

            if elem.tag == 'filename':
                
                file_name = os.path.join('train' if train_flag else 'test' ,elem.text)
                temp_file_name = file_name.split('/')[:-1]
                temp_file_name = os.path.join('/'.join(temp_file_name), 'lwir' if modality_flag == 'ir' else 'visible', file_name.split('/')[-1])
                file_name = temp_file_name + '.png'
                
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
                bndbox['x'] = None
                bndbox['y'] = None
                bndbox['w'] = None
                bndbox['h'] = None

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
                
                #(subelem)
                
                for option in subelem:
                    if current_sub == 'bndbox':
                        #print(bndbox)
                        if bndbox[option.tag] is not None:
                            raise Exception('xml structure corrupted at bndbox tag.')
                        bndbox[option.tag] = int(option.text)

                # only after parse the <object> tag
                if bndbox['x'] is not None:
                    if object_name is None:
                        raise Exception('xml structure broken at bndbox tag')
                    if current_image_id is None:
                        raise Exception('xml structure broken at bndbox tag')
                    if current_category_id is None:
                        raise Exception('xml structure broken at bndbox tag')
                    bbox = []
                    # x
                    bbox.append(bndbox['x'])
                    # y
                    bbox.append(bndbox['y'])
                    # w
                    bbox.append(bndbox['w'])
                    # h
                    bbox.append(bndbox['h'])
                    #print('add annotation with {},{},{},{}'.format(object_name, current_image_id, current_category_id, bbox))
                    
                    if int(bndbox['w']) < 5 or int(bndbox['h']) < 5 or int(bndbox['h']*bndbox['w']) < 20:
                        continue
                    else:
                        addAnnoItem(object_name, current_image_id#'_'.join(f.split('/')[4:]).split('.')[0]#int(f.split(".")[0].split('_')[1]) if 'FLIR' else int(f.split(".")[0])
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
    
    import os
    data_set_path = os.path.join(dataset, 'train' if opt.train else 'test')
    
    if (opt.train):
        file_path = 'train-all-20-'
    else:
        file_path = 'test-all-20-'

    if(opt.modality == 'rgb'):
        file_path += 'rgb'
    else:
        file_path += 'ir'
    file_path += '.txt'
    
    final_path = os.path.join(data_set_path, file_path)
    
    with open(final_path) as file:
        lines = [line.rstrip() for line in file]
    
    
    lines = [os.path.join(data_set_path, x) for x in lines]

    dataset_list = [x + '.png' for x in lines]
    annotations_path = [x + '.xml' for x in lines]
    

    ## open file 
    opt.json_save_path = './kaist_'+opt.modality+'_train.json' if opt.train else './kaist_'+opt.modality+'_test.json'
    
    
    if(opt.train):
        dataset_train, dataset_valid = np.split(dataset_list, [int(len(dataset_list)*opt.percentage_split)])
        
        
        valid_path = opt.json_save_path.split('train')[0]+'valid.json'
        
        
        parseXmlFiles(annotations_path, opt.json_save_path, dataset_train, opt.train, opt.modality)

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
        
        annotations_path = [x.split('.png')[0]+'.xml' for x in dataset_valid]

        parseXmlFiles(annotations_path, valid_path, dataset_valid, opt.train, opt.modality)

    else:

        parseXmlFiles(annotations_path, opt.json_save_path, dataset_list, opt.train, opt.modality)
