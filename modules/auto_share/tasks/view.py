import logging
import copy
import torch
import torch.nn as nn
from fairseq import utils
import numpy as np
from collections import OrderedDict
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


def name2module(module, name):
    def _generator(_module):
        for part in name.split('.'):
            _module = getattr(_module, part)
            yield _module
    return [module] + list(_generator(module))


class ModelView:
    def __init__(self, model, args):
        self.model = model
        self.split_all = utils.eval_bool(args.split_all)
        self.threshold = args.split_threshold
        self.granularity = args.split_granularity

        # 初始化的时候，为全共享模型
        self.container = {name: model.keys for name in self.get_module_names(model)}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gradients = {lang_pair: {} for lang_pair in model.keys}

    def get_module_names(self, model: nn.Module):
        # 获取初始模型中所有的参数path
        all_param_names = [name for name in dict(model.named_parameters()).keys() if 'layers' in name]
        if self.granularity == 'parameter':
            # 所有参数的path
            pass
        elif self.granularity == 'module':
            # 精确到一个小模块，如Linear和LayerNorm，即包含weight和bias(可选)的名字
            all_param_names = [param.rstrip('.weight') for param in all_param_names if 'weight' in param]
        elif self.granularity == 'layer':
            def _get_layer_name(_param_name):
                _param_name = _param_name.split('.')
                _layer_index = _param_name.index('layers')  # encoder.layers.0, _layer_index=1, 取[:3]=[0,1,2]
                return '.'.join(_param_name[:_layer_index+2])
            all_param_names = [_get_layer_name(param) for param in all_param_names]
            all_param_names = list(OrderedDict.fromkeys(all_param_names))
        else:
            raise NotImplementedError('granularity error')
        return all_param_names

    def extract_gradient_from_module(self, module):
        if self.granularity == 'parameter':  # 拆分的是参数层级的
            assert torch.is_tensor(module)
            return module.grad.view(-1).data.cpu()
        else:
            assert isinstance(module, nn.Module), module
            return torch.cat([p.grad.view(-1) for p in module.parameters()]).data.cpu()

    def accum_gradient(self, lang_pair):
        cur_model = self.model.models[lang_pair]
        for name in self.get_module_names(cur_model):
            module_tree = name2module(cur_model, name)
            grad = self.extract_gradient_from_module(module_tree[-1])
            self.gradients[lang_pair][name] = grad + self.gradients[lang_pair].get(name, 0)

    def auto_split(self):
        logger.info('Detect split parameters by grad')
        # 根据梯度，计算每个模块的散度
        divergences = {}
        for name, lang_pairs in self.container.items():
            # name是模块的全名，lang_pairs是这个模块被多少语言对共享。
            # 如果把name中的lang_pair变为lang_pairs中的lang_pair，那实际指向的是同一个模块
            short_name = ".".join(name.split('.')[2:])
            module_gradients = {lang_pair: self.gradients[lang_pair][short_name] for lang_pair in lang_pairs}
            divergences[name] = calculate_div(module_gradients)

        # 按距离排序，从大到小，[-1, 1]，-1表示距离最小，所有距离>T的module
        sorted_divergences = [d for d in sorted(divergences.items(), key=lambda item: -item[1][1]) if d[1][1] > self.threshold]

        if len(sorted_divergences) == 0:
            logger.info('Skip split due to similarity > {}.'.format(self.threshold))
        else:
            logger.info('Cos similarity < {}: {} / {}'.format(-self.threshold, len(sorted_divergences), len(divergences.items())))
            for best_name, (best_lang_pairs, best_score) in sorted_divergences:
                logger.info('Split shared parameters: {}'.format(best_name))
                logger.info('This parameter is shared by {}'.format(','.join(best_lang_pairs[0] + best_lang_pairs[1])))
                logger.info('After split: {}   {}'.format(','.join(best_lang_pairs[0]), ','.join(best_lang_pairs[1])))
                logger.info('Cos similarity is {}'.format(-best_score))
                self.split_module(best_name, best_lang_pairs)

                if not self.split_all:
                    break

        # 计算完了，清空梯度
        self.gradients = {lang_pair: {} for lang_pair in self.model.keys}

    def split_module(self, module_to_split, split_lang_pairs):
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
        module_tree = name2module(self.model, module_to_split)
        new_module = copy.deepcopy(module_tree[-1]).to(self.device)
        setattr(module_tree[-2], module_to_split.split(".")[-1], new_module)

        # 3. 给其他语言也共享了
        for lang_pair in lang_pairs[1:]:
            module_name = ".".join([module_to_split.split(".")[0], lang_pair] + module_to_split.split(".")[2:])
            module_tree = name2module(self.model, module_to_split)
            setattr(module_tree[-2], module_name.split(".")[-1], new_module)


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
