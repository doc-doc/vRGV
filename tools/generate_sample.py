# ====================================================
# @Time    : 4/1/20 11:09 PM
# @Author  : Xiao Junbin
# @Email   : junbin@comp.nus.edu.sg
# @File    : generate_sample.py
# ====================================================

from util import *

def check_repeat_video(video_list_file):
    video_list = load_file(video_list_file)
    video_list = set(video_list)
    print(len(video_list))


def generate_sample(anno_dir, data_list_file, out_file):
    """

    :param anno_dir:
    :param out_file:
    :return:
    """
    video_list = load_file(data_list_file)
    output = []
    video_list = sorted(video_list)
    for vid, video in enumerate(video_list):
        vrelation = []
        file = osp.join(anno_dir, video+'.json')
        with open(file, 'r') as fin:
            anno = json.load(fin)
        # if vid == 291:
        #     print(anno.keys())
        nframe = anno['frame_count']
        width = anno['width']
        height = anno['height']
        subobjs = anno['subject/objects']
        tid2cls = {}
        for ins in subobjs:
            tid2cls[ins['tid']] = ins['category']
        relations = anno['relation_instances']
        rels = []
        for relation in relations:
            sub = tid2cls[relation['subject_tid']]
            obj = tid2cls[relation['object_tid']]
            pred = relation['predicate']
            rels.append('-'.join([sub, pred, obj]))
        rels = list(set(rels))
        vrelation = [[video, nframe, width, height, rel] for rel in rels]
        # vrelation = list(set(vrelation))
        # vrelation = sorted(vrelation)
        if vid == 0:
            output = vrelation
        else:
            output.extend(vrelation)
            # if vid==291:
            #     print(vrelation)
        # print(vid, video)

    print(len(output))
    with open(out_file, 'w') as fp:
        json.dump(output, fp)



def main():
    root_dir = '/storage/jbxiao/workspace/data/vidor-annotation/'
    out_dir = '../dataset/vidor/'
    data_split = 'val'

    anno_dir = osp.join(root_dir, data_split)

    data_list_file = osp.join(out_dir, '{}_list.json'.format(data_split))

    out_file = osp.join(out_dir, 'vrelation_{}.json'.format(data_split))

    # check_repeat_video(data_list_file)
    generate_sample(anno_dir, data_list_file, out_file)


if __name__ == "__main__":
    main()
