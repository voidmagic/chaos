import logging
import copy
import torch
import torch.nn as nn
from collections import OrderedDict
from sklearn.cluster import AgglomerativeClustering


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
        self.split_count = args.split_count
        self.granularity = args.split_granularity

        # 初始化的时候，为全共享模型
        self.container = {name: model.keys for name in self.get_module_names(model)}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gradients = {lang_pair: {} for lang_pair in model.keys}

    def clear_gradient(self):
        self.gradients = {lang_pair: {} for lang_pair in self.model.keys}

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
            # 按顺序拼接
            grads = [p.grad.view(-1) for _, p in sorted(module.named_parameters(), key=lambda pair: pair[0])]
            return torch.cat(grads).data.cpu()

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
        for name, lang_pairs in self.container.items():   # name是模块的全名，lang_pairs是这个模块被多少语言对共享。
            # 如果把name中的lang_pair变为lang_pairs中的lang_pair，那实际指向的是同一个模块
            short_name = ".".join(name.split('.')[2:])    # name: 'models.en-de.encoder.layers.0'  short_name: 'encoder.layers.0'
            module_gradients = {lang_pair: self.gradients[lang_pair][short_name] for lang_pair in lang_pairs}
            divergences[name] = calculate_div(module_gradients)

        # 按距离排序，从大到小，-1表示距离最小，所有距离>T的module
        sorted_divergences = [d for d in sorted(divergences.items(), key=lambda item: -item[1][1])]
        logger.info('Top 10 cosine distance: ')
        for best_name, (_, best_score) in sorted_divergences[:10]:
            logger.info('{}: {}'.format(best_name, best_score))
        for best_name, (best_lang_pairs, best_score) in sorted_divergences[:self.split_count]:
            logger.info('Split shared parameters: {}'.format(best_name))
            logger.info('This parameter is shared by {}'.format(','.join(best_lang_pairs[0] + best_lang_pairs[1])))
            logger.info('After split: {}   {}'.format(','.join(best_lang_pairs[0]), ','.join(best_lang_pairs[1])))
            logger.info('Cosine distance is {}'.format(best_score))
            yield self.split_module(best_name, best_lang_pairs)

    def split_module(self, module_to_split, split_lang_pairs):
        # 1. 修改container的内容
        # 旧的参数以lang_pairs[0][i]为base
        if module_to_split.split(".")[1] in split_lang_pairs[1]:
            split_lang_pairs[0], split_lang_pairs[1] = split_lang_pairs[1], split_lang_pairs[0]

        self.container[module_to_split] = split_lang_pairs[0]
        # 新的参数以lang_pairs[1][0]为base
        new_name = ".".join([module_to_split.split(".")[0], split_lang_pairs[1][0]] + module_to_split.split(".")[2:])
        self.container[new_name] = split_lang_pairs[1]

        # 2. 新建参数
        module_tree = name2module(self.model, module_to_split)
        new_module = copy.deepcopy(module_tree[-1]).to(self.device)

        # 3. 给第二个聚类中的语言，赋予该模块
        for lang_pair in split_lang_pairs[1]:
            module_name = ".".join([module_to_split.split(".")[0], lang_pair] + module_to_split.split(".")[2:])
            module_tree = name2module(self.model, module_name)
            setattr(module_tree[-2], module_name.split(".")[-1], new_module)
        return new_name, module_to_split


def calculate_div(module_gradients):
    """
    对于一个特定模块，由L种语言共享，module_gradients就是在这个模块上，每个语言对应的梯度。
    本函数对其进行聚类，最后分为两个类别，并返回类间距离。
    :param module_gradients: dict of {lang_pair: gradient}
    :return: [[cluster_1], [cluster_2]], distance
    """
    if len(module_gradients) < 2:
        return [], -1

    cluster = AgglomerativeClustering(linkage='average', affinity='cosine', n_clusters=2, compute_distances=True)
    lang_pairs, gradients = zip(*module_gradients.items())
    labels = cluster.fit_predict(torch.stack(gradients).numpy())
    cluster_0 = [lang_pair for lang_pair, label in zip(lang_pairs, labels) if label == 0]
    cluster_1 = [lang_pair for lang_pair, label in zip(lang_pairs, labels) if label == 1]
    return [cluster_0, cluster_1], cluster.distances_[-1]
