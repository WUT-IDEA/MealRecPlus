#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
import numpy as np
import scipy.sparse as sp
from torch.utils.data import Dataset

# User -> U
# Meal -> M
# Course -> S
# Category -> C

def NutritionData(data_path):
    with open(data_path + '/course_who.txt', 'r') as f:
        course_who = np.array(list(map(lambda s: tuple(float(i) for i in s[:-1].split('\t')), f.readlines())))
    with open(data_path + '/course_fsa.txt', 'r') as f:
        course_fsa = np.array(list(map(lambda s: tuple(float(i) for i in s[:-1].split('\t')), f.readlines())))
    with open(data_path + '/meal_who.txt', 'r') as f:
        meal_who = np.array(list(map(lambda s: tuple(float(i) for i in s[:-1].split('\t')), f.readlines())))
    with open(data_path + '/meal_fsa.txt', 'r') as f:
        meal_fsa = np.array(list(map(lambda s: tuple(float(i) for i in s[:-1].split('\t')), f.readlines())))
    with open(data_path + '/user_who.txt', 'r') as f:
        user_who = np.array(list(map(lambda s: tuple(float(i) for i in s[:-1].split('\t')), f.readlines())))
    with open(data_path + '/user_fsa.txt', 'r') as f:
        user_fsa = np.array(list(map(lambda s: tuple(float(i) for i in s[:-1].split('\t')), f.readlines())))
    return course_who, course_fsa, meal_who, meal_fsa, user_who, user_fsa

def print_statistics(X, string):
    print('>'*10 + string + '>'*10 )
    print('Average interactions', X.sum(1).mean(0).item())
    nonzero_row_indice, nonzero_col_indice = X.nonzero()
    unique_nonzero_row_indice = np.unique(nonzero_row_indice)
    unique_nonzero_col_indice = np.unique(nonzero_col_indice)
    print('Non-zero rows', len(unique_nonzero_row_indice)/X.shape[0])
    print('Non-zero columns', len(unique_nonzero_col_indice)/X.shape[1])
    print('Matrix density', len(nonzero_row_indice)/(X.shape[0]*X.shape[1]))


class BasicDataset(Dataset):

    def __init__(self, path, name, task, neg_sample):
        self.path = path
        self.name = name
        self.task = task
        self.neg_sample = neg_sample
        self.num_users, self.num_meals, self.num_courses, self.num_categories = self.__load_data_size()

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __load_data_size(self):
        if self.name == 'MealRec+H':
            return [1575, 3817, 7280, 3]
        elif self.name == 'MealRec+L':
            return [1928, 3578, 10589, 3]
        # with open(os.path.join(self.path, self.name, '{}_data_size.txt'.format(self.name)), 'r') as f:
        #     return [int(s) for s in f.readline().split('\t')][:4]
    def load_U_M_interaction(self):
        with open(os.path.join(self.path, self.name, 'user_meal_{}.txt'.format(self.task)), 'r') as f:
            return list(map(lambda s: tuple(int(i) for i in s[:-1].split('\t')), f.readlines()))
    def load_U_S_interaction(self):
        with open(os.path.join(self.path, self.name, 'user_course.txt'), 'r') as f:
            return list(map(lambda s: tuple(int(i) for i in s[:-1].split('\t')), f.readlines()))
    def load_M_S_affiliation(self):
        with open(os.path.join(self.path, self.name, 'meal_course.txt'), 'r') as f:
            return list(map(lambda s: tuple(int(i) for i in s[:-1].split('\t')), f.readlines()))
    def load_S_C_affiliation(self):
        with open(os.path.join(self.path, self.name, 'course_category.txt'), 'r') as f:
            return list(map(lambda s: tuple(int(i) for i in s[:-1].split('\t')), f.readlines()))


class MealTrainDataset(BasicDataset):
    def __init__(self, path, name, course_data, assist_data, seed=None):
        super().__init__(path, name, 'train', 1)
        # U-M
        self.U_M_pairs = self.load_U_M_interaction()
        indice = np.array(self.U_M_pairs, dtype=np.int32)
        values = np.ones(len(self.U_M_pairs), dtype=np.float32)
        self.ground_truth_u_m = sp.coo_matrix(
            (values, (indice[:, 0], indice[:, 1])), shape=(self.num_users, self.num_meals)).tocsr()

        print_statistics(self.ground_truth_u_m, 'U-M statistics in train')

    def __getitem__(self, index):
        user_m, pos_meal = self.U_M_pairs[index]
        all_meals = [pos_meal]
        while True:
            i = np.random.randint(self.num_meals)
            if self.ground_truth_u_m[user_m, i] == 0 and not i in all_meals:
                all_meals.append(i)
                if len(all_meals) == self.neg_sample+1:
                    break
        return torch.LongTensor([user_m]), torch.LongTensor(all_meals)

    def __len__(self):
        return len(self.U_M_pairs)


class MealTestDataset(BasicDataset):
    def __init__(self, path, name, train_dataset, task='test'):
        super().__init__(path, name, task, None)
        # U-M
        self.U_M_pairs = self.load_U_M_interaction()
        indice = np.array(self.U_M_pairs, dtype=np.int32)
        values = np.ones(len(self.U_M_pairs), dtype=np.float32)
        self.ground_truth_u_m = sp.coo_matrix(
            (values, (indice[:, 0], indice[:, 1])), shape=(self.num_users, self.num_meals)).tocsr()

        print_statistics(self.ground_truth_u_m, 'U-M statistics in test')

        self.train_mask_u_m = train_dataset.ground_truth_u_m
        self.users = torch.arange(self.num_users, dtype=torch.long).unsqueeze(dim=1)
        self.meals = torch.arange(self.num_meals, dtype=torch.long)
        assert self.train_mask_u_m.shape == self.ground_truth_u_m.shape

    def __getitem__(self, index):
        return index, torch.from_numpy(self.ground_truth_u_m[index].toarray()).squeeze(),  \
            torch.from_numpy(self.train_mask_u_m[index].toarray()).squeeze(),  \

    def __len__(self):
        return self.ground_truth_u_m.shape[0]

class ItemDataset(BasicDataset):
    def __init__(self, path, name, seed=None):
        super().__init__(path, name, 'train', 1)
        # U-S
        self.U_S_pairs = self.load_U_S_interaction()
        indice = np.array(self.U_S_pairs, dtype=np.int32)
        values = np.ones(len(self.U_S_pairs), dtype=np.float32)
        self.ground_truth_u_s = sp.coo_matrix(
            (values, (indice[:, 0], indice[:, 1])), shape=(self.num_users, self.num_courses)).tocsr()

        print_statistics(self.ground_truth_u_s, 'U-S statistics')

    def __getitem__(self, index):
        user_s, pos_course = self.U_S_pairs[index]
        all_courses = [pos_course]
        while True:
            j = np.random.randint(self.num_courses)
            if self.ground_truth_u_s[user_s, j] == 0 and not j in all_courses:
                all_courses.append(j)
                if len(all_courses) == self.neg_sample+1:
                    break

        return torch.LongTensor([user_s]), torch.LongTensor(all_courses)

    def __len__(self):
        return len(self.U_S_pairs)


class AffiliationDataset(BasicDataset):
    def __init__(self, path, name):
        super().__init__(path, name, None, None)
        # M-S
        self.M_S_pairs = self.load_M_S_affiliation()
        indice = np.array(self.M_S_pairs, dtype=np.int32)
        values = np.ones(len(self.M_S_pairs), dtype=np.float32)
        self.ground_truth_m_s = sp.coo_matrix(
            (values, (indice[:, 0], indice[:, 1])), shape=(self.num_meals, self.num_courses)).tocsr()

        print_statistics(self.ground_truth_m_s, 'M-S statistics')


class CategoryDataset(BasicDataset):
    def __init__(self, path, name):
        super().__init__(path, name, None, None)
        self.S_C_pairs = self.load_S_C_affiliation()
        indice = np.array(self.S_C_pairs, dtype=np.int32)
        values = np.ones(len(self.S_C_pairs), dtype=np.float32)
        self.ground_truth_s_c = sp.coo_matrix(
            (values, (indice[:, 0], indice[:, 1])), shape=(self.num_courses, self.num_categories)).tocsr()

        print_statistics(self.ground_truth_s_c, 'S-C statistics')


def get_dataset(path, name, task='tune', seed=123):
    affiliation_data = AffiliationDataset(path, name)
    print('finish loading affiliation data')
    category_data = CategoryDataset(path, name)
    print('finish loading course data')
    course_data = ItemDataset(path, name, seed=seed)
    print('finish loading course data')
    meal_train_data = MealTrainDataset(path, name, course_data, affiliation_data, seed=seed)
    print('finish loading meal train data')
    meal_test_data = MealTestDataset(path, name, meal_train_data, task=task)
    print('finish loading meal test data')

    return meal_train_data, meal_test_data, course_data, affiliation_data, category_data

