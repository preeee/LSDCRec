import os
import torch
import numpy as np

import multiprocessing as mp    # 2024-5-11 本地python37

from model import MODEL_DICT
from trainers import Trainer
from utils import EarlyStopping, check_path, set_seed, parse_args, set_logger
from dataset import get_seq_dic, get_dataloader, get_rating_matrix

def main():

    args = parse_args()
    log_path = os.path.join(args.output_dir, args.train_name + '.log')
    logger = set_logger(log_path)

    set_seed(args.seed)
    check_path(args.output_dir)     # 检查目录是否存在，如果不存在就创建它

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda

    seq_dic, max_item, num_users = get_seq_dic(args)
    args.item_size = max_item + 1               # item_size：物品总数
    args.num_users = num_users + 1
    # ↑ 为什么item_size=max_item+1
    # 无论原始数据集中，item ID 是从0 还是1 开始，设置 item_size 为 max_item+1 都是正确的。
    # 这样做是为了确保Embedding层有足够的容量来处理所有可能的索引，因为Embedding层的索引是从0开始的。 +1是为了这个0索引。

    args.checkpoint_path = os.path.join(args.output_dir, args.train_name + '.pt')
    args.same_target_path = os.path.join(args.data_dir, args.data_name+'_same_target.npy')
    ################# 添加意图标签路径
    # args.intent_labels_path = os.path.join(args.data_dir, args.data_name + '_intent_labels.npy')  # 添加意图标签路径
    ###################
    train_dataloader, eval_dataloader, test_dataloader = get_dataloader(args,seq_dic)        # 获取训练、验证、测试dataloader

    logger.info(str(args))       # 打印参数信息
    model = MODEL_DICT[args.model_type.lower()](args=args)          # 根据模型名称获取对应模型的构造函数，并传入参数args
    logger.info(model)          # 打印模型信息
    trainer = Trainer(model, train_dataloader, eval_dataloader, test_dataloader, args, logger)

    args.valid_rating_matrix, args.test_rating_matrix = get_rating_matrix(args.data_name, seq_dic, max_item)

    # eval
    if args.do_eval:
        if args.load_model is None:
            logger.info(f"No model input!")
            exit(0)
        else:
            args.checkpoint_path = os.path.join(args.output_dir, args.load_model + '.pt')       # 构建要加载的模型的路径
            trainer.load(args.checkpoint_path)              # 加载模型
            logger.info(f"Load model from {args.checkpoint_path} for test!")
            scores, result_info = trainer.test(0)           # 对加载的模型进行测试，得到测试结果

    # train
    else:
        early_stopping = EarlyStopping(args.checkpoint_path, logger=logger, patience=args.patience, verbose=True)
        for epoch in range(args.epochs):

            trainer.train(epoch)
            scores, _ = trainer.valid(epoch)    # 在每个 epoch 结束后，调用valid()方法对模型在验证集上进行评估，返回评估结果 scores, 其中 _ 是 get_full_sort_score() 返回的str(post_fix)
            # evaluate on NDCG@20
            early_stopping(np.array(scores[-1:]), trainer.model)    # scores[-1:] 表示取 scores 数组的最后一个元素（通常是最新的评估结果），这里具体是 NDCG@20
            # early_stopping(np.array(scores[4:5]), trainer.model)    # scores[4:5] 表示 Recall@20

            if early_stopping.early_stop:
                logger.info("Early stopping")
                break                            # 发生早停，则结束训练

        # 2025.11.29 可视化权重历史记录
        # 保存权重历史记录（如果模型有可学习权重参数）
        if hasattr(trainer, 'weight_history') and len(trainer.weight_history['epochs']) > 0:
            weight_file = trainer.save_weight_history()
            logger.info(f"Weight history saved. Use 'python visualize_weights.py --weight_file {weight_file}' to visualize.")

        logger.info("---------------Test Score---------------")
        trainer.model.load_state_dict(torch.load(args.checkpoint_path))
        scores, result_info = trainer.test(0)

    logger.info(args.train_name)    # 打印训练名称
    logger.info(result_info)        # 打印测试结果信息


# 用服务器运行时，运行这一行代码
main()


# 在本地运行时，需要添加这段代码，否则会报错
# if __name__ == '__main__':
#     mp.freeze_support()  # Add this line
#     main()