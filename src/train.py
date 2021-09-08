import time
import os
import numpy as np

from keras.utils import generic_utils
import tensorflow as tf

import src.preprocessing as prep
import src.config as config
import src.utils as utils
import src.models as models
import src.losses as loss

np.random.seed(config.RANDOM_SEED)

# Training loop

def train_frcnn(rpn_model, detector_model, total_model, train_patch_list, rpn_model_eval, detector_model_eval, total_model_eval, val_patch_list, class_list, num_epochs, patches_folder_path, backbone, pixel_mean=None, resume_train=True):

    ######### build class_mapping

    class_mapping = {key:value for value, key in enumerate(class_list)}
    class_mapping['bg'] = len(class_mapping)


    ######### shuffle image list
    np.random.shuffle(train_patch_list)

    train_datagen = prep.get_anchor_gt(patches_folder_path, train_patch_list, backbone, pixel_mean=pixel_mean)

    iter_num = 0
    epoch_length = 2
    rpn_accuracy_rpn_monitor = []
    rpn_accuracy_for_epoch = []
    start_time = time.time()
    num_epochs = num_epochs
    losses = np.zeros((epoch_length, 5))
    scores = np.zeros((epoch_length, 2))


    ######### resume training

    if resume_train:
        previous_losses = np.load(f"{config.MODEL_WEIGHTS}/{backbone}/loss_history.npy")
        best_loss = min(previous_losses[:,:-1].sum(axis=1))
        del previous_losses

        previous_scores = np.load(f"{config.MODEL_WEIGHTS}/{backbone}/scores_history.npy")
        best_mAP = max(previous_scores[:,0])
        print(f'\nPrevious best loss: {best_loss} - best mAP: {best_mAP}')
        del previous_scores
    else:
        if (os.path.exists(f"{config.MODEL_WEIGHTS}/{backbone}/loss_history.npy") or os.path.exists(f"{config.MODEL_WEIGHTS}/{backbone}/scores_history.npy")):
            raise ValueError('There is already a loss history or a score history file. Please delete it first.')
        best_loss = np.Inf
        best_mAP = 0.

    ######### (re-)start training
    for epoch in range(num_epochs):

        progbar = generic_utils.Progbar(epoch_length)
        print('Epoch {}/{}'.format(epoch+1, num_epochs))

        #TODO: check if we need LR warmup

        while True:
            try:
                if len(rpn_accuracy_rpn_monitor) == epoch_length:
                    mean_overlapping_bboxes = float(sum(rpn_accuracy_rpn_monitor))/len(rpn_accuracy_rpn_monitor)
                    rpn_accuracy_rpn_monitor = []
                    print('Average number of overlapping bounding boxes from RPN = {} for {} previous iterations'.format(mean_overlapping_bboxes, epoch_length))
                    if mean_overlapping_bboxes == 0:
                        print('RPN is not producing bounding boxes that overlap the ground truth boxes. Check RPN settings or keep training.')
                
                image, [y_rpn_cls_true, y_rpn_reg_true], img_data_aug, _, _, patch_id = next(train_datagen)
                
                print(f'Starting rpn model training on patch {patch_id}')
                # print(img_data_aug)

                loss_rpn_tot, loss_rpn_cls, loss_rpn_regr = rpn_model.train_on_batch(image, [y_rpn_cls_true, y_rpn_reg_true])
                # Get predicted rpn from rpn model [rpn_cls, rpn_regr]
                P_rpn = rpn_model.predict_on_batch(image)
                
                # R: bboxes (shape=(300,4))
                # Convert rpn layer to roi bboxes
                R = utils.rpn_to_roi(P_rpn[0], P_rpn[1], use_regr=True, max_boxes=config.nms_max_boxes, overlap_thresh=0.7) #TODO: try with a lower threshold
                # print(R)
                # # note: calc_iou converts from (x1,y1,x2,y2) to (x,y,w,h) format
                # # X2: bboxes with iou > config.classifier_min_overlap for all gt bboxes in 2000 non_max_suppression bboxes
                # # Y1: one hot encode for bboxes from above => x_roi (X)
                # # Y2: corresponding labels and corresponding gt bboxes
                X2, Y1, Y2, IoUs = utils.calc_iou(R, img_data_aug, class_mapping)
                # print('IoUs:',IoUs)
                # tf.print('Y2 shape = ', Y2.shape, output_stream=sys.stderr, sep=',', summarize=-1)

                
                # print(IoUs)
                if IoUs is not None:
                    print(f'Best IoU found in this run: {max(IoUs)}')

                # print(X2, Y1, Y2)
                # print(Y2)

                # If X2 is None means there are no matching bboxes
                if X2 is None:
                    rpn_accuracy_rpn_monitor.append(0)
                    rpn_accuracy_for_epoch.append(0)
                    continue
                
                # Find out the positive anchors and negative anchors
                neg_samples = np.where(Y1[0, :, -1] == 1) # --> these are bg anchors
                pos_samples = np.where(Y1[0, :, -1] == 0)

                # tf.print('Y2 = ', Y2[0,pos_samples,...], output_stream=sys.stderr, sep=',', summarize=-1)


                if len(neg_samples) > 0:
                    neg_samples = neg_samples[0]
                else:
                    neg_samples = []

                if len(pos_samples) > 0:
                    pos_samples = pos_samples[0]
                else:
                    pos_samples = []

                rpn_accuracy_rpn_monitor.append(len(pos_samples))
                rpn_accuracy_for_epoch.append((len(pos_samples)))

                # Here we select 64 proposals in the 1:3 ratio:
                # 64/4*3 positive and 64/4*1 negative
                num_pos_samples=(config.num_rois//4)*2
                # num_pos_samples=(config.num_rois//4)*3

                # print('config.num_rois:',config.num_rois)
                # print('pos threshold:',num_pos_samples)
                # print('neg threshold:',(config.num_rois//4))

                if config.num_rois > 1:
                    # If number of positive anchors is smaller than num_rois*3/4, take all positive samples and then resample until num_rois*3/4; 
                    # else randomly choose num_pos_samples pos samples
                    if len(pos_samples) < num_pos_samples:
                        selected_pos_samples = pos_samples.tolist()
                        # selected_pos_samples += np.random.choice(pos_samples, num_pos_samples-len(pos_samples), replace=True).tolist()
                    else:
                        selected_pos_samples = np.random.choice(pos_samples, num_pos_samples, replace=False).tolist()
                    
                    # Randomly choose (num_rois - num_pos) neg samples
                    try:
                        selected_neg_samples = np.random.choice(neg_samples, config.num_rois - len(selected_pos_samples), replace=False).tolist()
                    except:
                        selected_neg_samples = np.random.choice(neg_samples, config.num_rois - len(selected_pos_samples), replace=True).tolist()
                    
                    # Save all the pos and neg samples in sel_samples
                    sel_samples = selected_pos_samples + selected_neg_samples
                else:
                    # in the extreme case where num_rois = 1, we pick a random pos or neg sample
                    selected_pos_samples = pos_samples.tolist()
                    selected_neg_samples = neg_samples.tolist()
                    if np.random.randint(0, 2):
                        sel_samples = np.random.choice(neg_samples)
                    else:
                        sel_samples = np.random.choice(pos_samples)

                print('Selected pos samples:', len(selected_pos_samples))
                print('Selected neg samples:', len(selected_neg_samples))

                # training_data          => [X, X2[:, sel_samples, :]]
                # labels                 => [Y1[:, sel_samples, :], Y2[:, sel_samples, :]]
                #  X                     => img_data resized image
                #  X2[:, sel_samples, :] => num_rois (4 in here) bboxes which contains selected neg and pos
                #  Y1[:, sel_samples, :] => one hot encode for num_rois bboxes which contains selected neg and pos
                #  Y2[:, sel_samples, :] => labels and gt bboxes for num_rois bboxes which contains selected neg and pos
                print('Starting detector model training on batch')
                # print('sel_samples:', sel_samples)

                # print(f'X2 shape: {X2.shape}')
                # print('Y2:',Y2[:, sel_samples, :])
                loss_detector_tot, loss_detector_cls, loss_detector_regr, detector_class_acc, _ = detector_model.train_on_batch([image, X2[:, sel_samples, :]], [Y1[:, sel_samples, :], Y2[:, sel_samples, :]]) 
                # [P_cls, P_regr] = detector_model.predict([image, X2[:, sel_samples, :]])
                # print(P_regr)

                losses[iter_num, 0] = loss_rpn_cls
                losses[iter_num, 1] = loss_rpn_regr

                losses[iter_num, 2] = loss_detector_cls
                losses[iter_num, 3] = loss_detector_regr
                losses[iter_num, 4] = detector_class_acc

                iter_num += 1

                progbar.update(iter_num, [
                ('rpn_cls', np.mean(losses[iter_num-1, 0])), 
                ('rpn_regr', np.mean(losses[iter_num-1, 1])), 
                ('detector_cls', np.mean(losses[iter_num-1, 2])), 
                ('detector_regr', np.mean(losses[iter_num-1, 3])), 
                ("average number of objects", len(selected_pos_samples))])

                if iter_num == epoch_length:

                    loss_rpn_cls = np.mean(losses[:, 0])
                    loss_rpn_regr = np.mean(losses[:, 1])
                    loss_detector_cls = np.mean(losses[:, 2])
                    loss_detector_regr = np.mean(losses[:, 3])
                    detector_class_acc = np.mean(losses[:, 4])

                    losses_to_save = [[loss_rpn_cls, loss_rpn_regr, loss_detector_cls, loss_detector_regr, detector_class_acc]]
                    mean_overlapping_bboxes = float(sum(rpn_accuracy_for_epoch)) / len(rpn_accuracy_for_epoch)
                    rpn_accuracy_for_epoch = []
                    curr_loss = loss_rpn_cls + loss_rpn_regr + loss_detector_cls + loss_detector_regr

                    print('Mean number of bounding boxes from RPN overlapping ground truth boxes: {}'.format(mean_overlapping_bboxes))
                    print('Classifier accuracy for bounding boxes from RPN: {}'.format(detector_class_acc))
                    print('Loss RPN classifier: {}'.format(loss_rpn_cls))
                    print('Loss RPN regression: {}'.format(loss_rpn_regr))
                    print('Loss Detector classifier: {}'.format(loss_detector_cls))
                    print('Loss Detector regression: {}'.format(loss_detector_regr))
                    print('Total loss: {}, wrt best loss: {} '.format(curr_loss, best_loss))
                    print('Elapsed time: {}'.format(time.time() - start_time))

                    if resume_train:
                        previous_losses = np.load(f"{config.MODEL_WEIGHTS}/{backbone}/loss_history.npy")
                        loss_hist = np.concatenate((previous_losses, losses_to_save), axis=0)
                        counter = previous_losses.shape[0] + 1
                    else:
                        loss_hist = losses_to_save
                        counter = 1
                    print('saving loss')
                    np.save(f"{config.MODEL_WEIGHTS}/{backbone}/loss_history.npy", loss_hist)
                    total_model.save_weights(f'{config.MODEL_WEIGHTS}/{backbone}/0_frcnn_{backbone}.h5')
                    
                    print('Validating model on validation set...')

                    models.load_weigths(rpn_model_eval, detector_model_eval, backbone, checkpoint=f'0_frcnn_{backbone}.h5')
                    models.compile_models(rpn_model_eval, detector_model_eval, total_model_eval, rpn_losses=[loss.rpn_loss_cls, loss.rpn_loss_regr], detector_losses=[loss.detector_loss_cls, loss.detector_loss_regr], class_list=class_list)


                    _, mAP, mPrecision = utils.evaluate_model(rpn_model_eval, detector_model_eval, backbone, val_patch_list, class_list, map_threshold=.5, acceptance_treshold=.5)

                    if (mPrecision == 0. and mAP == 1.):
                        mAP=0.
                    
                    scores_to_save = [[mAP, mPrecision]]
                    print(scores_to_save)

                    if resume_train:
                        previous_scores = np.load(f"{config.MODEL_WEIGHTS}/{backbone}/scores_history.npy")
                        scores_hist = np.concatenate((previous_scores, scores_to_save), axis=0)
                    else:
                        scores_hist = scores_to_save
                        resume_train = True
                    
                    np.save(f"{config.MODEL_WEIGHTS}/{backbone}/scores_history.npy", scores_hist)

                    iter_num = 0

                    # save weights
                    if curr_loss < best_loss:
                        print('Total loss decreased from {} to {}, saving weights'.format(best_loss,curr_loss))
                        best_loss = curr_loss
                        total_model.save_weights(f'{config.MODEL_WEIGHTS}/{backbone}/loss_{counter}_frcnn_{backbone}.h5')
                    if (mAP > best_mAP and mPrecision > 0):
                        print('mAP decreased from {} to {}, saving weights'.format(best_mAP,mAP))
                        best_mAP = mAP
                        total_model.save_weights(f'{config.MODEL_WEIGHTS}/{backbone}/map_{counter}_frcnn_{backbone}.h5')
                    
                    start_time = time.time()
                    break

            except Exception as e:
                print('Exception: {}'.format(e))
                # return
                continue

    print('Training complete.')
