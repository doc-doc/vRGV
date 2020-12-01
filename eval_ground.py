import os.path as osp
from evaluations.common import tiou
from evaluations.util import load_file
import generate_track_link

def eval_ground_scores(gt_relations, pred_relations, tiou_threshold):
    """

    :param gt_relations:
    :param pred_relations:
    :param tiou_threshold:
    :return:
    """
    # pred_relations = sorted(pred_relations, key=lambda x: x['score'], reverse=True)

    relation_num = len(gt_relations)
    predict, predict_sub, predict_obj = 0, 0, 0

    for relation, pred_trajs in pred_relations.items():
        pred_sub = pred_trajs['sub']
        pred_obj = pred_trajs['obj']
        flag, flag_s, flag_o = False, False, False

        gt_trajs = gt_relations[relation]

        # print(relation)

        for gt_traj in gt_trajs:
            gt_sub = gt_traj['sub']
            gt_obj = gt_traj['obj']
            s_tiou = tiou(pred_sub, gt_sub)
            o_tiou = tiou(pred_obj, gt_obj)
            r_iou = min(s_tiou, o_tiou)

            if r_iou >= tiou_threshold:
                flag = True
            if s_tiou >= tiou_threshold:
                flag_s = True
            if o_tiou >= tiou_threshold:
                flag_o = True
        if flag:
            predict += 1
        if flag_s:
            predict_sub += 1
        if flag_o:
            predict_obj += 1

    predict = predict / relation_num
    predict_sub = predict_sub /relation_num
    predict_obj = predict_obj /relation_num

    return predict, predict_sub, predict_obj, relation_num


def evaluate(groundtruth, prediction, tiou_threshold=0.5):
    """ evaluate visual relation detection and visual 
    relation tagging.
    """

    video_num = len(groundtruth)
    print('Computing grounding accuracy over {} videos...'.format(video_num))
    acc, acc_sub, acc_obj = 0.0, 0.0, 0.0

    gt_rnum = 0
    for qid, relation_gt in groundtruth.items():

        if qid not in prediction:
            continue
        relation_pred = prediction[qid]
        if len(relation_pred) == 0:
            continue

        video_acc, video_acc_sub, video_acc_obj, relation_num = eval_ground_scores(relation_gt, relation_pred, tiou_threshold)

        acc += video_acc
        acc_sub += video_acc_sub
        acc_obj += video_acc_obj
        gt_rnum += relation_num


    acc /= video_num
    acc_sub /= video_num
    acc_obj /= video_num

    print("Acc_S\t Acc_O\t Acc_R")

    print('{:.2f}\t {:.2f}\t {:.2f}'.format(acc_sub*100, acc_obj*100, acc*100))


def main():

    groundtruth_dir = 'dataset/vidvrd/'
    gt_file = osp.join(groundtruth_dir, 'gt_relation_frame.json')

    result_dir = 'results/'
    res_file = osp.join(result_dir, 'test_viterbi_1gap_04_batch.json')
    if not osp.exists(res_file):
        print('Generating ...')
        generate_track_link.main(res_file)

    grountruth = load_file(gt_file)
    prediction = load_file(res_file)

    evaluate(grountruth, prediction)


if __name__ == "__main__":
    main()

