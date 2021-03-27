import os
import copy
import logging

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
logger = logging.getLogger(__name__)


def name2module(model, name):
    name = name.split('.')
    for part in name:
        if hasattr(model, part):
            model = getattr(model, part)
        elif isinstance(model, nn.ModuleDict):
            model = model[part]
        elif isinstance(model, nn.ModuleList):
            model = model[int(part)]
        else:
            raise NotImplementedError

        if isinstance(model, nn.Linear) or isinstance(model, nn.LayerNorm):
            break
    return model


def get_model_parameters(model: nn.Module):
    # 获取初始模型中所有的模块
    for name, p in model.named_parameters():
        if 'layers' in name:
            yield name.rstrip('.bias').rstrip('.weight')


def get_parent_module(model, name):
    name = name.split('.')
    parent = model
    for part in name:
        parent = model
        if hasattr(model, part):
            model = getattr(model, part)
        elif isinstance(model, nn.ModuleDict):
            model = model[part]
        elif isinstance(model, nn.ModuleList):
            model = model[int(part)]
        else:
            raise NotImplementedError

        if isinstance(model, nn.Linear) or isinstance(model, nn.LayerNorm):
            break
    return parent


class ModelView:
    def __init__(self, model):
        self.model = model
        # 初始化的时候，为全共享模型
        names = list(set(list(get_model_parameters(model))))
        self.container = {
            n: model.keys for n in names
        }
        self.gradients = {lang_pair: {} for lang_pair in model.keys}
        self.split_all = os.environ.get('SPLIT_ALL', 'FALSE') == 'TRUE'

    def accum_gradient(self, lang_pair):
        cur_model = self.model.models[lang_pair]
        names = list(set(list(get_model_parameters(cur_model))))
        for n in names:
            module = name2module(cur_model, n)
            grad = torch.cat([module.weight.grad.view(-1), module.bias.grad.view(-1)]).data.cpu()
            self.gradients[lang_pair][n] = grad + self.gradients[lang_pair].get(n, 0)

    def auto_split(self, optimizer):
        logger.info('Detect split parameters by grad')
        # 根据梯度，计算每个模块的散度
        divergences = {}
        for name, lang_pairs in self.container.items():
            # name是模块的全名，lang_pairs是这个模块被多少语言对共享。
            # 如果把name中的lang_pair变为lang_pairs中的lang_pair，那实际指向的是同一个模块
            short_name = ".".join(name.split('.')[2:])
            module_gradients = {lang_pair: self.gradients[lang_pair][short_name] for lang_pair in lang_pairs}
            divergences[name] = calculate_div(module_gradients)

        count = len([a for a in divergences.values() if a[1] > 0])
        logger.info('Cos similarity < 0: {} / {}'.format(count, len(divergences.items())))

        # 按距离排序，从大到小，[-1, 1]，-1表示距离最小。
        sorted_divergences = sorted(divergences.items(), key=lambda item: -item[1][1])
        # 所有距离>0的module
        sorted_divergences = [d for d in sorted_divergences if d[1][1] > 0]

        if len(sorted_divergences) == 0:
            logger.info('Skip split due to similarity > 0.')
            return

        for best_name, (best_lang_pairs, best_score) in sorted_divergences:
            logger.info('Split shared parameters: {}'.format(best_name))
            logger.info('This parameter is shared by {}'.format(','.join(best_lang_pairs[0] + best_lang_pairs[1])))
            logger.info('After split: {}   {}'.format(','.join(best_lang_pairs[0]), ','.join(best_lang_pairs[1])))
            logger.info('Cos similarity is {}'.format(-best_score))
            self.split_module(best_name, best_lang_pairs, optimizer)

            if not self.split_all:
                break

    def split_module(self, module_to_split, split_lang_pairs, optimizer):
        # 1. 修改container的内容
        # best_name: models.$lang_pair.*
        # best_lang_pairs：两组语言
        module_base_lang_pair = module_to_split.split(".")[1]
        if module_base_lang_pair in split_lang_pairs[0]:
            pass
        elif module_base_lang_pair in split_lang_pairs[1]:
            split_lang_pairs[0], split_lang_pairs[1] = split_lang_pairs[1], split_lang_pairs[0]
        else:
            raise NotImplementedError

        self.container[module_to_split] = split_lang_pairs[0]
        # 新的参数以best_lang_pairs[1][0]为base
        new_name = ".".join([module_to_split.split(".")[0], split_lang_pairs[1][0]] + module_to_split.split(".")[2:])
        self.container[new_name] = split_lang_pairs[1]

        # 2. 新建参数
        lang_pairs = self.container[module_to_split]
        parent_module = get_parent_module(self.model, module_to_split)
        shared_module = getattr(parent_module, module_to_split.split(".")[-1])
        device = list(shared_module.parameters())[0].device
        new_module = copy.deepcopy(shared_module).to(device)
        setattr(parent_module, module_to_split.split(".")[-1], new_module)

        # 3. 给其他语言也共享了
        for lang_pair in lang_pairs[1:]:
            module_name = ".".join([module_to_split.split(".")[0], lang_pair] + module_to_split.split(".")[2:])
            parent_module = get_parent_module(self.model, module_name)
            setattr(parent_module, module_name.split(".")[-1], new_module)

        # 4. 添加到优化器中
        optimizer.optimizer.add_param_group({"params": new_module.parameters()})


def calculate_div(module_gradients):
    # 分成两个类别，返回类别对应的语言，以及类间距离，使用-cos_sim
    # 取值范围：[-1, 1]，-1表示最相似，1表示最不相似
    lang_pairs = list(module_gradients.keys())
    if len(lang_pairs) < 2:
        return [], -1

    # convert to numpy
    for k in module_gradients.keys():
        module_gradients[k] = module_gradients[k].numpy()

    # 使用层次聚类

    # 1. 每个语言都是一个类
    clusters = [[lang] for lang in list(module_gradients.keys())]

    while True:
        if len(clusters) == 2:
            break
        # 计算每两个cluster之间的距离，记录最小的
        min_distance = 1
        min_cluster_index = (-1, -1)
        for i, cluster_i in enumerate(clusters):
            for j, cluster_j in enumerate(clusters):
                if i <= j:
                    break
                distance = calculate_distance(module_gradients, cluster_i, cluster_j)
                if distance < min_distance:
                    min_distance = distance
                    min_cluster_index = (i, j)

        # 合并最小的两个类
        clusters[min_cluster_index[0]] = clusters[min_cluster_index[0]] + clusters[min_cluster_index[1]]
        del clusters[min_cluster_index[1]]

    return clusters, calculate_distance(module_gradients, clusters[0], clusters[1])


def calculate_distance(module_gradients, langs_1, langs_2):
    l1 = [module_gradients[k] for k in langs_1]
    l2 = [module_gradients[k] for k in langs_2]
    l1_point = np.mean(l1, axis=0)
    l2_point = np.mean(l2, axis=0)
    return cos_sim(l1_point, l2_point)


def cos_sim(vector_a, vector_b):
    # [-1, 1]，-1表示相似，1表示不相似
    return -cosine_similarity([vector_a, vector_b])[0, 1]
