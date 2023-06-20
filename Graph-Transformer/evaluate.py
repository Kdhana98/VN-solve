import time

import torch
import numpy as np
from fairseq import checkpoint_utils, utils, options, tasks
from fairseq.logging import progress_bar
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
import ogb
import sys
import os
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score, recall_score, accuracy_score
import pandas as pd
import sys
from os import path

sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
from pretrain import load_pretrained_model

import logging

# replace this file with the evaluate.py in Graphormer/graphormer/evaluate/evaluate.py

def eval(args, use_pretrained, checkpoint_path=None, logger=None):
    cfg = convert_namespace_to_omegaconf(args)
    np.random.seed(cfg.common.seed)
    utils.set_torch_seed(cfg.common.seed)

    # initialize task
    task = tasks.setup_task(cfg.task)
    model = task.build_model(cfg.model)

    # load checkpoint
    if use_pretrained:
        model_state = load_pretrained_model(cfg.task.pretrained_model_name)
    else:
        model_state = torch.load(checkpoint_path)["model"]
    model.load_state_dict(
        model_state, strict=True, model_cfg=cfg.model
    )
    del model_state

    model.to(torch.cuda.current_device())
    # load dataset
    split = args.split
    task.load_dataset(split)
    batch_iterator = task.get_batch_iterator(
        dataset=task.dataset(split),
        max_tokens=cfg.dataset.max_tokens_valid,
        max_sentences=cfg.dataset.batch_size_valid,
        max_positions=utils.resolve_max_positions(
            task.max_positions(),
            model.max_positions(),
        ),
        ignore_invalid_inputs=cfg.dataset.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=cfg.dataset.required_batch_size_multiple,
        seed=cfg.common.seed,
        num_workers=cfg.dataset.num_workers,
        epoch=0,
        data_buffer_size=cfg.dataset.data_buffer_size,
        disable_iterator_cache=False,
    )
    itr = batch_iterator.next_epoch_itr(
        shuffle=False, set_dataset_epoch=False
    )
    progress = progress_bar.progress_bar(
        itr,
        log_format=cfg.common.log_format,
        log_interval=cfg.common.log_interval,
        default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple")
    )
    df_time = pd.DataFrame(columns=['run', 'time'])
    # infer
    y_pred = []
    y_true = []
    tt = 0
    with torch.no_grad():
        model.eval()
        for i, sample in enumerate(progress):
            sample = utils.move_to_cuda(sample)
            st = time.time()
            y = model(**sample["net_input"])[:, 0, :].reshape(-1)
            y_pred.extend(y.detach().cpu())
            et = time.time()
            new_row_time = pd.DataFrame({'run': [i],
                                         'time': [float(et)-float(st)]})
            df_time = df_time._append(new_row_time, ignore_index=True)
            y_true.extend(sample["target"].detach().cpu().reshape(-1)[:y.shape[0]])
            torch.cuda.empty_cache()
            tt = tt + et - st

    print ("time: " + str(tt))
    df_time.to_csv('graphormer_time.csv')
    # save predictions
    y_pred = torch.Tensor(y_pred)
    y_true = torch.Tensor(y_true)

    # evaluate pretrained models
    if use_pretrained:
        if cfg.task.pretrained_model_name == "pcqm4mv1_graphormer_base":
            evaluator = ogb.lsc.PCQM4MEvaluator()
            input_dict = {'y_pred': y_pred, 'y_true': y_true}
            result_dict = evaluator.eval(input_dict)
            logger.info(f'PCQM4Mv1Evaluator: {result_dict}')
        elif cfg.task.pretrained_model_name == "pcqm4mv2_graphormer_base":
            evaluator = ogb.lsc.PCQM4Mv2Evaluator()
            input_dict = {'y_pred': y_pred, 'y_true': y_true}
            result_dict = evaluator.eval(input_dict)
            logger.info(f'PCQM4Mv2Evaluator: {result_dict}')
    else:
        if args.metric == "auc":
            auc = roc_auc_score(y_true, y_pred)
            tensor_to_numpy = y_true.numpy()
            ones = np.count_nonzero(tensor_to_numpy == 1)

            #print ('auc: '+ str(auc))
            logger.info(f"auc: {auc}")
            return auc
        elif args.metric == "mae":
            mae = np.mean(np.abs(y_true - y_pred))
            logger.info(f"mae: {mae}")
        elif args.metric == 'F1':
            f1 = f1_score(y_true, y_pred)
            print (f1)
            return f1
        elif args.metric == 'recall':
            recall = recall_score(y_true, y_pred)
            return recall
        elif args.metric == 'accuracy':
            accuracy = accuracy_score(y_true, y_pred)
            return accuracy
        elif args.metric == 'all':
            auc_1 = roc_auc_score(y_true, y_pred)
            thresholds = np.linspace(0, 1, 100)  # Range of threshold values
            best_threshold = None
            best_auc = -1
            for threshold in thresholds:
                y_pred_2 = (y_pred >= threshold)
                auc = roc_auc_score(y_true, y_pred_2)
                if auc > best_auc:
                    best_auc = auc
                    best_threshold = threshold
            y_pred_final = (y_pred >= best_threshold)
            f1 = f1_score(y_true, y_pred_final)
            recall = recall_score(y_true, y_pred_final)
            accuracy = accuracy_score(y_true, y_pred_final)
            res = {'auc': auc_1, 'accuracy': accuracy, 'F1': f1, 'recall': recall}
            return res
        else:
            raise ValueError(f"Unsupported metric {args.metric}")

def main():
    parser = options.get_training_parser()
    parser.add_argument(
        "--split",
        type=str,
    )
    parser.add_argument(
        "--metric",
        type=str,
    )
    args = options.parse_args_and_arch(parser, modify_parser=None)
    logger = logging.getLogger(__name__)
    if args.pretrained_model_name != "none":
        eval(args, True, logger=logger)
    elif hasattr(args, "save_dir"):
        #print (os.getcwd())
        if args.metric == 'all':
            print ('hi')
            df = pd.DataFrame(columns=['EpochNumber', 'auc', 'accuracy', 'F1', 'Recall'])
            bb = os.listdir(args.save_dir)
            for checkpoint_fname in os.listdir(args.save_dir):
                if checkpoint_fname.endswith('_best.pt'):
                #if checkpoint_fname.endswith('t137.pt'):
                    checkpoint_path = Path(args.save_dir) / checkpoint_fname
                    logger.info(f"evaluating checkpoint file {checkpoint_path}")
                    res = eval(args, False, checkpoint_path, logger)
                    df.loc[len(df.index)] = [checkpoint_fname, res['auc'], res['accuracy'], res['F1'], res['recall']]
                        #if checkpoint_fname.endswith('_best.pt'):
                            #df.loc[len(df.index)] = [checkpoint_fname, res['auc'], res['accuracy'], res['F1'], res['recall']]
                            #print (res)

        else:
            df = pd.DataFrame(columns=['EpochNumber', 'AUC'])
            for checkpoint_fname in os.listdir(args.save_dir):
                checkpoint_path = Path(args.save_dir) / checkpoint_fname
                logger.info(f"evaluating checkpoint file {checkpoint_path}")
                auc = eval(args, False, checkpoint_path, logger)
                df.loc[len(df.index)] = [checkpoint_fname, auc]
                if checkpoint_fname.endswith('_best.pt'):
                    print (auc)

        #df.to_csv('best_checkpoint_test_small_800_200_7.csv')
        #df.to_csv('best_F1_test_medium_800_200_29.csv')
        #df.to_csv('validation_medium_800_200_11.csv')



if __name__ == '__main__':
    main()
