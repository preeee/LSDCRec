import tqdm
import torch
import numpy as np
import json
import os

from torch.optim import Adam
from metrics import recall_at_k, ndcg_k

class Trainer:
    def __init__(self, model, train_dataloader, eval_dataloader, test_dataloader, args, logger):
        super(Trainer, self).__init__()         # 父类是object

        self.args = args
        self.logger = logger
        self.cuda_condition = torch.cuda.is_available() and not self.args.no_cuda
        self.device = torch.device("cuda" if self.cuda_condition else "cpu")

        self.model = model
        if self.cuda_condition:
            self.model.cuda()       # 将模型放到GPU上

        # Setting the train and test data loader
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader

        # self.data_name = self.args.data_name
        betas = (self.args.adam_beta1, self.args.adam_beta2)
        self.optim = Adam(self.model.parameters(), lr=self.args.lr, betas=betas, weight_decay=self.args.weight_decay)

        self.logger.info(f"Total Parameters: {sum([p.nelement() for p in self.model.parameters()])}")
        
        # 2025.11.29 可视化权重历史记录
        # 初始化参数记录（用于可视化）
        self.weight_history = {
            'weight_long_term': [],
            'weight_short_term': [],
            'epochs': []
        }

    def train(self, epoch):
        self.iteration(epoch, self.train_dataloader, train=True)

    def valid(self, epoch):
        self.args.train_matrix = self.args.valid_rating_matrix          # 验证集的评分矩阵
        return self.iteration(epoch, self.eval_dataloader, train=False)

    def test(self, epoch):
        self.args.train_matrix = self.args.test_rating_matrix           # 测试集的评分矩阵
        return self.iteration(epoch, self.test_dataloader, train=False)

    def save(self, file_name):
        torch.save(self.model.cpu().state_dict(), file_name)
        self.model.to(self.device)

    def load(self, file_name):
        original_state_dict = self.model.state_dict()
        self.logger.info(original_state_dict.keys())
        new_dict = torch.load(file_name)
        self.logger.info(new_dict.keys())
        for key in new_dict:
            original_state_dict[key]=new_dict[key]
        self.model.load_state_dict(original_state_dict)
    
    # 2025.11.29 可视化权重历史记录
    def save_weight_history(self, file_path=None):
        """保存权重参数历史记录到JSON文件"""
        if file_path is None:
            # 默认保存路径：output目录下
            output_dir = getattr(self.args, 'output_dir', 'output')
            os.makedirs(output_dir, exist_ok=True)
            train_name = getattr(self.args, 'train_name', 'model')
            file_path = os.path.join(output_dir, f'{train_name}_weight_history.json')
        
        with open(file_path, 'w') as f:
            json.dump(self.weight_history, f, indent=2)
        
        self.logger.info(f"Weight history saved to {file_path}")
        return file_path

    # seq_out是模型BSARec的输出，相当于seq_output = self.forward(input_ids)中的seq_output
    def predict_full(self, seq_out):
        # 1、test_item_emb的形状是[item_num, hidden_size]
        test_item_emb = self.model.item_embeddings.weight   # 获取模型中的item_embeddings的权重
        # 2、seq_out的形状是[batch, seq_len, hidden_size ]
        # import pdb; pdb.set_trace()
        # 3、rating_pred的形状是[batch, seq_len, item_num]，item_num是候选item的数量（也可能是所有item的数量）
        rating_pred = torch.matmul(seq_out, test_item_emb.transpose(0, 1))      # 矩阵相乘，得到预测评分
        return rating_pred

    # 计算Recall和NDCG
    def get_full_sort_score(self, epoch, answers, pred_list):
        recall, ndcg = [], []
        for k in [5, 10, 15, 20]:
            recall.append(recall_at_k(answers, pred_list, k))
            ndcg.append(ndcg_k(answers, pred_list, k))
        post_fix = {
            "Epoch": epoch,
            "Recall@5": '{:.4f}'.format(recall[0]), "NDCG@5": '{:.4f}'.format(ndcg[0]),
            "Recall@10": '{:.4f}'.format(recall[1]), "NDCG@10": '{:.4f}'.format(ndcg[1]),
            "Recall@20": '{:.4f}'.format(recall[3]), "NDCG@20": '{:.4f}'.format(ndcg[3])
        }
        self.logger.info(post_fix)

        return [recall[0], ndcg[0], recall[1], ndcg[1], recall[3], ndcg[3]], str(post_fix)



    def iteration(self, epoch, dataloader, train=True):

        # 根据train参数的布尔值，确定当前是处于train模式还是test模式
        str_code = "train" if train else "test"
        # Setting the tqdm progress bar
        rec_data_iter = tqdm.tqdm(enumerate(dataloader),
                                  desc="Mode_%s:%d" % (str_code, epoch),
                                  total=len(dataloader),
                                  bar_format="{l_bar}{r_bar}")

        # train
        if train:
            self.model.train()
            rec_loss = 0.0      # 累加每个batch的loss

            # 遍历每个batch，以一个batch为单位
            for i, batch in rec_data_iter:
                # 0. batch_data will be sent into the device(GPU or CPU)
                batch = tuple(t.to(self.device) for t in batch)     # 将batch中的数据放到GPU上

                user_ids, input_ids, answers, neg_answer, same_target = batch   # 提取batch中的各个张量
                loss = self.model.calculate_loss(input_ids, answers, neg_answer, same_target, user_ids)

                self.optim.zero_grad()      # 1.梯度清零
                loss.backward()             # 2.反向传播
                self.optim.step()           # 3.更新参数
                rec_loss += loss.item()     # 4.累加loss

            # 2025.11.29 可视化权重历史记录
            # 记录可学习权重参数（如果模型有这些参数）
            weight_long_term_val = None
            weight_short_term_val = None
            if hasattr(self.model, 'weight_long_term') and hasattr(self.model, 'weight_short_term'):
                weight_long_term_val = self.model.weight_long_term.item()
                weight_short_term_val = self.model.weight_short_term.item()
                self.weight_history['weight_long_term'].append(weight_long_term_val)
                self.weight_history['weight_short_term'].append(weight_short_term_val)
                self.weight_history['epochs'].append(epoch)
            # 2025.11.29 可视化权重历史记录 结束
            
            post_fix = {
                "epoch": epoch,             # 当前epoch
                "rec_loss": '{:.4f}'.format(rec_loss / len(rec_data_iter)),     # 平均每个batch的loss
            }
            
            # 2025.11.29 可视化权重历史记录
            # 如果有权重参数，添加到日志中
            if weight_long_term_val is not None:
                post_fix["weight_long_term"] = '{:.4f}'.format(weight_long_term_val)
                post_fix["weight_short_term"] = '{:.4f}'.format(weight_short_term_val)

            if (epoch + 1) % self.args.log_freq == 0:    # 每隔log_freq个epoch，打印一次日志
                self.logger.info(str(post_fix))         # 将损失值记录到日志中

        # test，不会计算梯度或更新参数，只进行前向传播，生成预测结果
        else:
            self.model.eval()
            pred_list = None
            answer_list = None

            for i, batch in rec_data_iter:
                batch = tuple(t.to(self.device) for t in batch)             # batch是一个元组，包含多个张量
                user_ids, input_ids, answers, _, _ = batch                  # 我们只对前三个张量感兴趣
                recommend_output = self.model.predict(input_ids, user_ids)  # predict()是SequentialRecModel的方法
                recommend_output = recommend_output[:, -1, :]               # 序列推荐的结果：取二维张量的最后一个时间步的所有元素

                rating_pred = self.predict_full(recommend_output)           # 计算预测评分
                rating_pred = rating_pred.cpu().data.numpy().copy()         # 将预测评分移动到CPU上，并转为numpy数组
                batch_user_index = user_ids.cpu().numpy()

                try:
                    rating_pred[self.args.train_matrix[batch_user_index].toarray() > 0] = 0     # 将已经交互过的item的评分设置为0
                except: # bert4rec
                    rating_pred = rating_pred[:, :-1]
                    rating_pred[self.args.train_matrix[batch_user_index].toarray() > 0] = 0

                # reference: https://stackoverflow.com/a/23734295, https://stackoverflow.com/a/20104162
                # argpartition time complexity O(n)  argsort O(nlogn)
                # The minus sign "-" indicates a larger value.
                ind = np.argpartition(rating_pred, -20)[:, -20:]                                # 返回每一行中最高的20个评分的索引
                # Take the corresponding values from the corresponding dimension
                # according to the returned subscript to get the sub-table of each row of topk
                arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]                # 返回索引对应的评分
                # Sort the sub-tables in order of magnitude.
                arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]        # 对评分进行排序，返回排序后的索引
                # retrieve the original subscript from index again
                batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]    # 根据索引重新获取排序后的评分在原始数组中的位置

                if i == 0:      # 第一个batch
                    pred_list = batch_pred_list
                    answer_list = answers.cpu().data.numpy()
                else:           # 后续的batch
                    pred_list = np.append(pred_list, batch_pred_list, axis=0)
                    answer_list = np.append(answer_list, answers.cpu().data.numpy(), axis=0)

            return self.get_full_sort_score(epoch, answer_list, pred_list)
