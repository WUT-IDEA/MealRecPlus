import torch
import numpy as np


# How to eval

# 1. State Metrics before testing
# nutritionMetrics = [NutritionMetric('./data/' + CONFIG['dataset_name'] + '/nutrition/', topK=20),
#                     NutritionMetric('./data/' + CONFIG['dataset_name'] + '/nutrition/', topK=40),
#                     NutritionMetric('./data/' + CONFIG['dataset_name'] + '/nutrition/', topK=80)]

# 2. Calculate them for each test batch
# (pred_b: prediction result, users: test user indexs)
# for metric in nutritionMetrics:
#     metric(pred_b, users)

# 3. Print them after each test epoch
# for metric in nutritionMetrics:
#     metric.print_metric(epoch)

class _Nutrition_Metric:
    def __init__(self):
        self.start()

    def start(self):
        self._cnt = 0

    def stop(self):
        self._metric = self._sum / self._cnt

    @property
    def metric(self):
        return self._metric

    def __call__(self, scores, bundle_idx):
        raise NotImplementedError

    def get_title(self):
        raise NotImplementedError


class NutritionMetric:

    def __init__(self, data_path, category_num=3, topK=100):
        self.topK = topK
        self.category_num = category_num
        self.bundle_cnt = 0
        self.user_cnt = 0

        self.who_sum = 0
        self.fsa_sum = 0

        self.diff_who_sum = 0
        self.diff_fsa_sum = 0

        self.item_who, self.item_fsa, self.bundle_who, self.bundle_fsa, self.user_who, self.user_fsa = NutritionData(data_path)

        self.metric_who_mean = 0
        self.metric_fsa_mean = 0

        self.metric_who_diff = 0
        self.metric_fsa_diff = 0

        # fairness as ranking exposure
        self.fairness_metric_group_a_sum = 0
        self.fairness_metric_group_b_sum = 0
        self.fairness_metric_sum = 0
        self.fairness_metric_count = 0
        self.fairness_metric = 0

    def get_title(self):
        return "Nutrition_Metric@{}".format(self.topk)

    # 定义了__call__之后，我们就可以用类名的方式调用以下的方法
    def __call__(self, pred, user_index):
        pred = pred.cpu()
        user_index = user_index.cpu()
        _, bundle_index = torch.topk(pred, self.topK)
        self.bundle_cnt += pred.shape[0] * self.topK

        self.user_cnt += user_index.shape[0]
        self.who_sum += self.bundle_who[bundle_index].sum()
        self.fsa_sum += self.bundle_fsa[:, 0][bundle_index].sum()

        self.diff_who_sum += (self.bundle_who[:, 0][bundle_index].mean(-1) - self.user_who[:, 0][user_index]).sum()
        self.diff_fsa_sum += (self.bundle_fsa[:, 0][bundle_index].mean(-1) - self.user_fsa[:, 0][user_index]).sum()

        sorted_index = torch.argsort(torch.from_numpy(self.bundle_fsa[:, 0]))
        group_a_mask = torch.BoolTensor(np.isin(range(self.bundle_who[:, 0].shape[0]), sorted_index[:int(len(sorted_index) / 2)].numpy()))
        group_b_mask = ~group_a_mask

        utility = 1.0 / torch.arange(1, 1 + self.topK, dtype=torch.float32)
        # utility = torch.ones(100, dtype=torch.float32)
        # utility = 1.0 / torch.log(torch.arange(2, 2 + self.topK, dtype=torch.float32))


        for i in range(bundle_index.shape[0]):
            a = utility[group_a_mask[bundle_index][i]].sum()
            b = utility[group_b_mask[bundle_index][i]].sum()
            self.fairness_metric_group_a_sum += a
            self.fairness_metric_group_b_sum += b
            # self.fairness_metric_sum += abs(a - b) / (a + b)
            self.fairness_metric_count += 1
            # self.fairness_metric_group_a_sum += utility[group_a_mask[bundle_index][i]].sum()
            # self.fairness_metric_group_b_sum += utility[group_b_mask[bundle_index][i]].sum()

    def stop(self):
        self.metric_who_mean = self.who_sum / self.bundle_cnt
        self.metric_fsa_mean = self.fsa_sum / self.bundle_cnt

        self.metric_who_diff = self.diff_who_sum / self.user_cnt
        self.metric_fsa_diff = self.diff_fsa_sum / self.user_cnt

    def print_metric(self, epoch):
        self.stop()
        print("==================top-{}===epoch-{}==start==============".format(self.topK, epoch))
        print('who/fsa mean: {:.4f}  {:.4f}'.format(self.metric_who_mean, self.metric_fsa_mean), end='\n')

        # print('fairness exposure metric: {:.4f}'.format(self.fairness_metric), end='\n')
        print('ranking exposure gap: {:.4f}'.format((self.fairness_metric_group_a_sum - self.fairness_metric_group_b_sum) / (self.fairness_metric_group_a_sum + self.fairness_metric_group_b_sum)), end='\n')
        print('ranking exposure mean, group a: {:.4f}, group b: {:.4f}'.format(self.fairness_metric_group_a_sum / self.fairness_metric_count, self.fairness_metric_group_b_sum / self.fairness_metric_count),
              end='\n')

        print("==================top-{}===epoch-{}===end=============".format(self.topK, epoch))
        print("\n")


def NutritionData(data_path):
    with open(data_path + '/item_who.txt', 'r') as f:
        item_who = np.array(list(map(lambda s: tuple(float(i) for i in s[:-1].split('\t')), f.readlines())))
    with open(data_path + '/item_fsa.txt', 'r') as f:
        item_fsa = np.array(list(map(lambda s: tuple(float(i) for i in s[:-1].split('\t')), f.readlines())))
    with open(data_path + '/bundle_who.txt', 'r') as f:
        bundle_who = np.array(list(map(lambda s: tuple(float(i) for i in s[:-1].split('\t')), f.readlines())))
    with open(data_path + '/bundle_fsa.txt', 'r') as f:
        bundle_fsa = np.array(list(map(lambda s: tuple(float(i) for i in s[:-1].split('\t')), f.readlines())))
    with open(data_path + '/user_who.txt', 'r') as f:
        user_who = np.array(list(map(lambda s: tuple(float(i) for i in s[:-1].split('\t')), f.readlines())))
    with open(data_path + '/user_fsa.txt', 'r') as f:
        user_fsa = np.array(list(map(lambda s: tuple(float(i) for i in s[:-1].split('\t')), f.readlines())))
    return item_who, item_fsa, bundle_who, bundle_fsa, user_who, user_fsa
