# ====================================================
# @Time    : 10/3/19 2:55 PM
# @Author  : Xiao Junbin
# @Email   : junbin@comp.nus.edu.sg
# @File    : visualize.py
# ====================================================

import os
import os.path as osp
import cv2
import json
import skvideo.io as vio


#colors = np.loadtxt('colors.txt', dtype=int, delimiter=' ')
#colors = colors.tolist()


def image_relation(pimage, captions, predicate, score):
    '''
    visualize the relation and track information
    '''
    colors = [[118, 55, 243], [29, 233, 182]]
    src_image = pimage.copy()
    image = pimage.copy()

    pad = 1
    font = 1.0
    text_color = [0, 0, 0]

    for b in range(len(captions)):

        caption = captions[b]
        frame_id = caption['fid']
        track_id = caption['track_id']
        object_name = caption['name']

        bbox = [int(b) for b in caption['bbox']]
        color = colors[track_id]

        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 4)

        bbox_caption = str(track_id) + '/' + object_name

        text_len = len(bbox_caption) * 13
        if (bbox[1] > 20):
            cv2.rectangle(image, (bbox[0] - pad, bbox[1] - 18), (bbox[0] + text_len, bbox[1]), color, -1)
            cv2.putText(image, bbox_caption, (bbox[0], bbox[1] - 6), cv2.FONT_HERSHEY_COMPLEX_SMALL, font, text_color)
        else:
            cv2.rectangle(image, (bbox[0] - pad, bbox[1]), (bbox[0] + text_len, bbox[1] + 20), color, -1)
            cv2.putText(image, bbox_caption, (bbox[0], bbox[1] + 15), cv2.FONT_HERSHEY_COMPLEX_SMALL, font, text_color)

    cv2.putText(image, str(captions[0]['fid']), (10, 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, font, [255, 0, 0])
    sp = 0
    relation = predicate +'>'+str(captions[1]['track_id'])
    str_score = str('{:.4f}'.format(score))
    length = len(relation) * 13
    len_score = len(str_score) * 13
    bbox = [int(b) for b in captions[0]['bbox']]
    if (bbox[1] > 20):

        cv2.rectangle(image, (bbox[2] - length, sp + bbox[1] + 2), (bbox[2], sp + bbox[1] + 35),
                      colors[captions[1]['track_id']], -1)
        cv2.putText(image, relation, (bbox[2] - length, sp + bbox[1] + 15),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, font, text_color)
        # cv2.rectangle(image, (bbox[2] - length, sp + bbox[1] + 2), (bbox[2], sp + bbox[1] + 20),
        #               colors[captions[1]['track_id']], -1)
        cv2.putText(image, str_score, (bbox[2] - length, sp + bbox[1] + 30),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, font, text_color)
    else:
        cv2.rectangle(image, (bbox[2] - length, sp + bbox[1] + 20), (bbox[2], sp + bbox[1] + 55),
                      colors[captions[1]['track_id']], -1)
        cv2.putText(image, relation, (bbox[2] - length, sp + bbox[1] + 35),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, font, text_color)
        # cv2.rectangle(image, (bbox[2] - length, sp + bbox[1] + 20), (bbox[2], sp + bbox[1] + 38),
        #               colors[captions[1]['track_id']], -1)
        cv2.putText(image, str_score, (bbox[2] - length, sp + bbox[1] + 50),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, font, text_color)
        # sp += 18

    cv2.addWeighted(image, 0.9, src_image, 0.1, 0, src_image)
    return src_image


def show_instance(instances, video_name, relation):
    """

    :param instance:
    :param relation:
    :return:
    """
    video_dir = '../vdata/JPEGImages/'
    video_name = osp.join(video_dir, video_name)
    spo = relation.split('-')
    for ins_id, so_pair in enumerate(instances):
        sub, obj, score = so_pair['sub'], so_pair['obj'], so_pair['score']
        duration = [sub[0][0], sub[-1][0]]

        for fid in range(duration[0], duration[1]):
            captions = []
            caption, caption_obj = {}, {}
            # subject
            caption['fid'] = fid
            caption['bbox'] = sub[fid - duration[0]][1:5]
            caption['name'] = spo[0]
            caption['track_id'] = 0
            captions.append(caption)
            # object
            caption_obj['fid'] = fid
            caption_obj['bbox'] = obj[fid - duration[0]][1:5]
            caption_obj['name'] = spo[2]
            caption_obj['track_id'] = 1
            captions.append(caption_obj)

            img = cv2.imread(osp.join(video_name, '{:06d}.JPEG'.format(fid)))

            image = image_relation(img, captions, spo[1], score)
            cv2.namedWindow(video_name)
            cv2.moveWindow(video_name, 50, 50)
            cv2.imshow(video_name, image)
            if cv2.waitKey(1) and 0xFF == ord('q'):
                break
        break
    cv2.destroyAllWindows()



def vis_video_relation(video_name, anno_file, query, save=False):
    """"""
    with open(anno_file, 'r') as fp:
        gt_relation = json.load(fp)

    print(video_name)
    vname = video_name.split('/')[-1]
    vanno = gt_relation[vname]
    query = '-'.join(query)

    fps = 20
    if save:
        v_name = '../vdata/vids/'+vname+'.mp4'
        # capture = vio.FFmpegReader(v_name)
        data = vio.ffprobe(v_name)['video']
        fps = data['@r_frame_rate']
    for relation, ins in vanno.items():
        # if relation != query: continue
        spo = relation.split('-')
        # instances = sorted(instances, key=lambda inst: inst['score'], reverse=True)
        print(relation, len(ins))
        if save:
            save_name  = 'demo/'+vname+'/'+relation+ '.mp4'
            save_dir = osp.dirname(save_name)
            if not osp.exists(save_dir):
                os.makedirs(save_dir)
            vid_out = vio.FFmpegWriter(save_name, inputdict={
                '-r': fps, }, outputdict={
                '-vcodec': 'libx264',
                '-pix_fmt': 'yuv420p',
                '-r': fps, })

        # for id, ins in enumerate(instances):
        duration = ins['duration']
        sub, obj = ins['sub'], ins['obj']
        if 'score' in ins:
            score = ins['score']
        else:
            score = 1.

        for fid in range(duration[0], duration[1]):
            captions = []
            caption, caption_obj = {},{}
            #subject
            caption['fid'] = fid
            caption['bbox'] = sub[fid - duration[0]]
            caption['name'] = spo[0]
            caption['track_id'] = 0
            captions.append(caption)
            #object
            caption_obj['fid'] = fid
            caption_obj['bbox'] = obj[fid-duration[0]]
            caption_obj['name'] = spo[2]
            caption_obj['track_id'] = 1
            captions.append(caption_obj)

            img = cv2.imread(osp.join(video_name, '{:06d}.JPEG'.format(fid)))
            image = image_relation(img, captions, spo[1], score)
            cv2.imshow(video_name, image)
            if cv2.waitKey(1) and 0xFF == ord('q'):
                break
            if save:
                image = image[:,:, ::-1]
                vid_out.writeFrame(image)

        if save:
            vid_out.close()
            break
    cv2.destroyAllWindows()


def main():

    root_dir = '../vdata/'
    video_dir = root_dir + 'JPEGImages/'
    # 00091008, 00119046, 00272006, 00125015, 00223001, 00091004, 00190000, 00142000, 00177011, 00416000
    video_name = video_dir + 'ILSVRC2015_train_00057005' #ILSVRC2015_train_00010001' #'ILSVRC2015_train_00225000'

    anno_file_pred = 'results/ground.json'

    #query = ('bicycle', 'move_beneath', 'person')
    query = ('person', 'ride', 'bicycle')
    vis_video_relation(video_name, anno_file_pred, query)


if __name__ == "__main__":
    main()


