import torch.nn.functional as F
import math
import random
import os
import numpy as np
import sklearn.model_selection as sklearn
import torch
from torch.distributions import Bernoulli
from model.PropensityNet import MiddleModel, TopModel


def binary_classification_loss(concat_true, concat_pred):
    t_true = concat_true[:, 1]

    t_pred = concat_pred[:, 2]
    t_pred = (t_pred + 0.001) / 1.002

    losst = torch.sum(F.binary_cross_entropy(t_pred, t_true, reduction='none'))

    return losst


def regression_loss(concat_true, concat_pred):
    y_true = concat_true[:, 0]
    t_true = concat_true[:, 1]

    y0_pred = concat_pred[:, 0]
    y1_pred = concat_pred[:, 1]

    loss0 = torch.sum((1. - t_true) * torch.square((y_true - y0_pred)))
    loss1 = torch.sum(t_true * torch.square((y_true - y1_pred)))

    return loss0 + loss1


def ned_loss(concat_true, concat_pred):
    t_true = concat_true[:, 1]

    t_pred = concat_pred[:, 1]
    return torch.sum(F.binary_cross_entropy(t_pred, t_true, reduction='none'))


def dead_loss(concat_true, concat_pred):
    return regression_loss(concat_true, concat_pred)


def dragonnet_loss_binarycross(concat_true, concat_pred):
    return regression_loss(concat_true, concat_pred) + binary_classification_loss(concat_true, concat_pred)


def treatment_accuracy(concat_true, concat_pred):
    t_true = concat_true[:, 1]
    t_pred = concat_pred[:, 2]
    return t_pred.argmax(dim=1).eq(t_true).sum().item()


def track_epsilon(concat_true, concat_pred):
    epsilons = concat_pred[:, 3]
    return torch.abs(torch.mean(epsilons))


def make_tarreg_loss(ratio=1., dragonnet_loss=dragonnet_loss_binarycross):
    def tarreg_ATE_unbounded_domain_loss(concat_true, concat_pred):
        vanilla_loss = dragonnet_loss(concat_true, concat_pred)

        y_true = concat_true[:, 0]
        t_true = concat_true[:, 1]

        y0_pred = concat_pred[:, 0]
        y1_pred = concat_pred[:, 1]
        t_pred = concat_pred[:, 2]

        epsilons = concat_pred[:, 3]
        t_pred = (t_pred + 0.01) / 1.02
        # t_pred = tf.clip_by_value(t_pred,0.01, 0.99,name='t_pred')

        y_pred = t_true * y1_pred + (1 - t_true) * y0_pred

        h = t_true / t_pred - (1 - t_true) / (1 - t_pred)

        y_pert = y_pred + epsilons * h
        targeted_regularization = torch.sum(torch.square(y_true - y_pert))

        # final
        loss = vanilla_loss + ratio * targeted_regularization
        return loss

    return tarreg_ATE_unbounded_domain_loss


def convert_df_to_np_arr(data):
    return data.to_numpy()


def test_train_split(covariates_X, treatment_Y, split_size=0.8):
    return sklearn.train_test_split(covariates_X, treatment_Y, train_size=split_size, random_state=1234)


def convert_to_tensor(X, Y):
    tensor_x = torch.stack([torch.Tensor(i) for i in X])
    tensor_y = torch.from_numpy(Y)
    processed_dataset = torch.utils.data.TensorDataset(tensor_x, tensor_y)
    return processed_dataset


def convert_to_tensor_DCN(X, ps_score, Y_f, Y_cf):
    tensor_x = torch.stack([torch.Tensor(i) for i in X])
    tensor_ps_score = torch.from_numpy(ps_score)
    tensor_y_f = torch.from_numpy(Y_f)
    tensor_y_cf = torch.from_numpy(Y_cf)
    processed_dataset = torch.utils.data.TensorDataset(tensor_x, tensor_ps_score,
                                                       tensor_y_f, tensor_y_cf)
    return processed_dataset


def concat_np_arr(X, Y, axis=1):
    return np.concatenate((X, Y), axis)


def get_device():
    return torch.device("cpu")#("cuda:1" if torch.cuda.is_available() else "cpu")


def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()


def get_shanon_entropy(prob):
    if prob == 1:
        return -(prob * math.log(prob))
    elif prob == 0:
        return -((1 - prob) * math.log(1 - prob))
    else:
        return -(prob * math.log(prob)) - ((1 - prob) * math.log(1 - prob))


def get_dropout_probability(entropy, gama=1):
    return 1 - (gama * 0.5) - (entropy * 0.5)


def get_dropout_mask(prob, x):
    return Bernoulli(torch.full_like(x, 1 - prob)).sample() / (1 - prob)


"""
input：按rank sorted的model params的list
output:纵向拼接的params
"""


def paramStack(vector_list):
    start = vector_list[0]
    if len(vector_list) == 1:
        return start
    for i in range(1, len(vector_list)):
        start = torch.cat((start, vector_list[i]), dim=-1)
    return start


def seed_torch(seed=1034):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


def get_propensity_score(path, concat_bottom):
    device = get_device()
    middle_model = MiddleModel(25).to(device)
    m_path = path + "middle_model_epoch50.pth"
    t_path = path + "top_model_epoch50.pth"
    # init middle model
    middle_model.load_state_dict(torch.load(m_path))
    middle_model.eval()
    middle_out = middle_model(concat_bottom)
    # init top model
    top_model = TopModel("eval").to(device)
    top_model.load_state_dict(torch.load(t_path))
    top_model.eval()
    pred = top_model(middle_out)

    prob = torch.chunk(pred, 2, dim=1)[1]

    return prob
