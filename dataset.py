import tqdm
import numpy as np
import torch
import os
from scipy.sparse import csr_matrix
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import random

# user_seq通常是二级列表，每一个元素是一个用户的交互序列 Su
class RecDataset(Dataset):
    def __init__(self, args, user_seq, test_neg_items=None, data_type='train'):
        self.args = args
        self.user_seq = []                      # 二级列表，每一个元素是一个用户的交互序列
        self.max_len = args.max_seq_length      #序列的最大长度
        self.user_ids = []
        self.contrastive_learning = args.model_type.lower() in ['fearec', 'duorec']
        self.data_type = data_type

        ################## 加载或计算意图标签

        # if os.path.exists(args.intent_labels_path):
        #     self.intent_labels = np.load(args.intent_labels_path, allow_pickle=True)
        # else:
        #     print("Start computing intent labels")
        #     self.intent_labels = self.compute_intent_labels(user_seq)
        #     np.save(args.intent_labels_path, self.intent_labels)

        ###################

        # 在处理训练集（train）时 ↓
        # 如果 seq 长度是 100，max_len 为50，seq[-52:-2] 会提取 seq 序列中的第 48 个元素到第 98 个元素（总共 50 个元素），input_ids 将会是 seq[48:98]
        # 例如，如果 input_ids = [1, 2, 3, 4]，那么 self.user_seq 将逐步存储 [1], [1, 2], [1, 2, 3], [1, 2, 3, 4]
        # 这4个子序列，对于每个子序列，self.user_ids 中都会添加相应的用户ID，因此会有4次相同的 user 值被添加到 self.user_ids 中
        # 这样做的目的是生成不同长度的子序列，使模型能够学习处理不同长度的序列
        if self.data_type=='train':
            for user, seq in enumerate(user_seq):
                input_ids = seq[-(self.max_len + 2):-2]     # input_ids的长度为 max_len，是seq的子序列
                for i in range(len(input_ids)):
                    self.user_seq.append(input_ids[:i + 1])
                    self.user_ids.append(user)              # user_ids是一个列表，存储了每个用户的ID ，这些ID会重复，因为input_ids有多个子序列
        elif self.data_type=='valid':
            for sequence in user_seq:
                self.user_seq.append(sequence[:-1])         # 验证集的user_seq是原始数据的子序列，去掉最后一个元素
        else:
            self.user_seq = user_seq                        # 测试集的user_seq是原始数据

        #### ↓ 对比学习的部分
        self.test_neg_items = test_neg_items
        ####
        if self.contrastive_learning and self.data_type=='train':
            if os.path.exists(args.same_target_path):
                self.same_target_index = np.load(args.same_target_path, allow_pickle=True)
            else:
                print("Start making same_target_index for contrastive learning")
                self.same_target_index = self.get_same_target_index()
                self.same_target_index = np.array(self.same_target_index)
                np.save(args.same_target_path, self.same_target_index)

    ################ 计算意图标签
    # def compute_intent_labels(self, user_seq):
    #     from sklearn.cluster import KMeans
    #     user_embeddings = []
    #     for seq in user_seq:
    #         # 假设每个序列的最后一个元素是用户的行为嵌入
    #         user_embeddings.append(seq[-1])
    #
    #     user_embeddings = np.array(user_embeddings)  # 转换为 NumPy 数组
    #
    #     # 使用 KMeans 聚类算法对用户的行为嵌入进行聚类
    #     kmeans = KMeans(n_clusters=5, random_state=42)
    #     intent_labels = kmeans.fit_predict(user_embeddings)
    #
    #     return intent_labels


    ####
    def get_same_target_index(self):
        num_items = max([max(v) for v in self.user_seq]) + 2    # 获取最大的item ID，然后 +2
        same_target_index = [[] for _ in range(num_items)]   # 生成了 num_items 个空列表，并将这些空列表组成一个列表，即二级列表
        
        user_seq = self.user_seq[:]     # 浅拷贝，不改变原始数据
        tmp_user_seq = []
        for i in tqdm.tqdm(range(1, num_items)):
            for j in range(len(user_seq)):
                if user_seq[j][-1] == i:
                    same_target_index[i].append(user_seq[j])
                else:
                    tmp_user_seq.append(user_seq[j])
            user_seq = tmp_user_seq
            tmp_user_seq = []

        return same_target_index        # 返回一个二级列表，

    def __len__(self):
        return len(self.user_seq)


    def __getitem__(self, index):
        items = self.user_seq[index]    # 获取第index个用户的交互序列 :items
        input_ids = items[:-1]          # 去掉该交互序列的最后一个item元素，作为模型的输入序列
        answer = items[-1]              # 获取最后一个item元素，作为模型的目标项（针对序列推荐任务），即gound truth

        seq_set = set(items)            # 将items转换为集合，去重
        #### ↓args中没有直接设置item_size这个参数，因为item_size是在数据处理的过程中求得的。item_size=max_item+1
        neg_answer = neg_sample(seq_set, self.args.item_size)   # 生成一个负样本


        # padding
        pad_len = self.max_len - len(input_ids)     # 计算需要填充的长度
        input_ids = [0] * pad_len + input_ids       # 在input_ids前面填充0，使其长度为max_len
        input_ids = input_ids[-self.max_len:]       # 截取input_ids的后max_len个元素
        assert len(input_ids) == self.max_len       # 检查input_ids的长度是否为max_len

        # 将数据转换为张量
        # cur_tensors是一个元组，包含了5个张量
        if self.data_type in ['valid', 'test']:
            cur_tensors = (
                torch.tensor(index, dtype=torch.long),          # 此时的index是用户ID
                torch.tensor(input_ids, dtype=torch.long),      # input_ids是用户的交互序列，是模型的输入
                torch.tensor(answer, dtype=torch.long),         # ground truth
                torch.zeros(0, dtype=torch.long), # not used，占位符，不会对模型的训练产生影响
                torch.zeros(0, dtype=torch.long), # not used，占位符
            )

        ####
        elif self.contrastive_learning:
            sem_augs = self.same_target_index[answer]
            sem_aug = random.choice(sem_augs)
            keep_random = False
            for i in range(len(sem_augs)):
                if sem_augs[0] != sem_augs[i]:
                    keep_random = True

            while keep_random and sem_aug == items:
                sem_aug = random.choice(sem_augs)

            sem_aug = sem_aug[:-1]
            pad_len = self.max_len - len(sem_aug)
            sem_aug = [0] * pad_len + sem_aug
            sem_aug = sem_aug[-self.max_len:]
            assert len(sem_aug) == self.max_len

            cur_tensors = (
                torch.tensor(self.user_ids[index], dtype=torch.long),
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(answer, dtype=torch.long),
                torch.tensor(neg_answer, dtype=torch.long),
                torch.tensor(sem_aug, dtype=torch.long)
            )

        # train
        else:
            cur_tensors = (
                torch.tensor(self.user_ids[index], dtype=torch.long),  # 不同于验证集和测试集直接使用 index 作为用户 ID，这里通过 self.user_ids 来获取用户 ID
                torch.tensor(input_ids, dtype=torch.long),             # input_ids是用户的交互序列，是模型的输入序列
                torch.tensor(answer, dtype=torch.long),                # 正样本
                torch.tensor(neg_answer, dtype=torch.long),            # 负样本
                torch.zeros(0, dtype=torch.long), # not used
            )

        ################# 添加意图标签
        # intent_label = self.intent_labels[self.user_ids[index]]
        # cur_tensors += (torch.tensor(intent_label, dtype=torch.long),)  # 将intent_label转换为张量，并添加到cur_tensors中
        ##########################

        return cur_tensors

# 生成负样本
# item_set: 一个用户的交互序列，大概率是正样本
# item_size: item的总数，
# 返回一个负样本，这个负样本不在item_set中
def neg_sample(item_set, item_size):
    item = random.randint(1, item_size - 1)
    while item in item_set:                         # 循环结束的条件：item不在item_set中，即 item是一个负样本
        item = random.randint(1, item_size - 1)
    return item

# 生成一个用于验证集的稀疏评分矩阵rating_matrix
def generate_rating_matrix_valid(user_seq, num_users, num_items):
    # three lists are used to construct sparse matrix
    row = []
    col = []
    data = []
    for user_id, item_list in enumerate(user_seq):
        for item in item_list[:-2]:     # 列表的从头开始到倒数第三个元素（不包括倒数第二个和倒数第一个元素）的子列表
            row.append(user_id)
            col.append(item)
            data.append(1)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    rating_matrix = csr_matrix((data, (row, col)), shape=(num_users, num_items))

    return rating_matrix

# 生成一个用于测试集的稀疏评分矩阵rating_matrix
def generate_rating_matrix_test(user_seq, num_users, num_items):
    # three lists are used to construct sparse matrix
    row = []
    col = []
    data = []
    for user_id, item_list in enumerate(user_seq):
        for item in item_list[:-1]: # 除了最后一个元素之外的每个元素
            row.append(user_id)
            col.append(item)
            data.append(1)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    rating_matrix = csr_matrix((data, (row, col)), shape=(num_users, num_items))

    return rating_matrix

# 获取两个评分矩阵
def get_rating_matrix(data_name, seq_dic, max_item):
    
    num_items = max_item + 1
    valid_rating_matrix = generate_rating_matrix_valid(seq_dic['user_seq'], seq_dic['num_users'], num_items)
    test_rating_matrix = generate_rating_matrix_test(seq_dic['user_seq'], seq_dic['num_users'], num_items)

    return valid_rating_matrix, test_rating_matrix

# 这个方法 not used
def get_user_seqs_and_max_item(data_file):
    lines = open(data_file).readlines()         # 读取数据文件的所有行
    lines = lines[1:]                           # 去掉第一行（可能是列名）
    user_seq = []
    item_set = set()
    for line in lines:
        user, items = line.strip().split('	', 1)
        items = items.split()                   # 将items字符串按空格分割，得到一个item列表
        items = [int(item) for item in items]
        user_seq.append(items)                  # 将items子列表添加到user_seq中
        item_set = item_set | set(items)
    max_item = max(item_set)
    return user_seq, max_item

# 返回用户交互序列（二级列表）、物品最大ID、用户总数
def get_user_seqs(data_file):
    lines = open(data_file).readlines()             # 读取数据文件的所有行
    user_seq = []                                   # 用于存储所有的用户交互序列
    item_set = set()                                # 用于存储所有的item
    for line in lines:
        user, items = line.strip().split(' ', 1)
        items = items.split(' ')                 # 将items字符串按空格分割，得到一个item列表
        items = [int(item) for item in items]    # 将item列表中的每个元素转换为int类型
        user_seq.append(items)                   # 将items子列表添加到user_seq中
        item_set = item_set | set(items)
    max_item = max(item_set)                    # max_item是item ID的最大值
    num_users = len(lines)                      # 用户数等于数据文件的行数

    return user_seq, max_item, num_users        # 返回用户交互序列（二级列表）、item ID的最大值、用户总数量

# 返回一个字典，包含用户交互序列（二级列表）和用户总数
def get_seq_dic(args):

    args.data_file = args.data_dir + args.data_name + '.txt'
    user_seq, max_item, num_users = get_user_seqs(args.data_file)
    seq_dic = {'user_seq':user_seq, 'num_users':num_users }     # 将用户交互序列（二级列表）和用户总数存储在字典中

    return seq_dic, max_item, num_users

def get_dataloader(args,seq_dic):

    # 调用 DataLoader 来获取一个批次时，RecDataset中的 __getitem__ 方法会被调用 batch_size 次，以此获取该批次中的所有样本。
    train_dataset = RecDataset(args, seq_dic['user_seq'], data_type='train')
    train_sampler = RandomSampler(train_dataset)        # 随机采样
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size, num_workers=args.num_workers)

    eval_dataset = RecDataset(args, seq_dic['user_seq'], data_type='valid')
    eval_sampler = SequentialSampler(eval_dataset)      # 顺序采样
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size, num_workers=args.num_workers)

    test_dataset = RecDataset(args, seq_dic['user_seq'], data_type='test')
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size, num_workers=args.num_workers)

    return train_dataloader, eval_dataloader, test_dataloader
