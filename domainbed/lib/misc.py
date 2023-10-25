# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
Things that don't belong anywhere else
"""

import hashlib
import json
import os
import sys
from shutil import copyfile
from collections import OrderedDict, defaultdict
from numbers import Number
import torchvision
import operator
import seaborn as sns
import torch.nn.functional as F
import matplotlib.pyplot as pl
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
import numpy as np
import torch
import torch.nn as nn
import tqdm
from collections import Counter
# from domainbed import algorithms
# from domainbed.visiontransformer import VisionTransformer
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import matplotlib


plt.rcParams["font.family"] = "DejaVu Serif"

# from calibration_metrics import ECELoss,SCELoss
from scipy.special import softmax

def compute_calibration(true_labels, pred_labels, confidences, num_bins=10):
    """Collects predictions into bins used to draw a reliability diagram.
    Arguments:
        true_labels: the true labels for the test examples
        pred_labels: the predicted labels for the test examples
        confidences: the predicted confidences for the test examples
        num_bins: number of bins
    The true_labels, pred_labels, confidences arguments must be NumPy arrays;
    pred_labels and true_labels may contain numeric or string labels.
    For a multi-class model, the predicted label and confidence should be those
    of the highest scoring class.
    Returns a dictionary containing the following NumPy arrays:
        accuracies: the average accuracy for each bin
        confidences: the average confidence for each bin
        counts: the number of examples in each bin
        bins: the confidence thresholds for each bin
        avg_accuracy: the accuracy over the entire test set
        avg_confidence: the average confidence over the entire test set
        expected_calibration_error: a weighted average of all calibration gaps
        max_calibration_error: the largest calibration gap across all bins
    """
    assert(len(confidences) == len(pred_labels))
    assert(len(confidences) == len(true_labels))
    assert(num_bins > 0)

    bin_size = 1.0 / num_bins
    bins = np.linspace(0.0, 1.0, num_bins + 1)
    indices = np.digitize(confidences, bins, right=True)

    bin_accuracies = np.zeros(num_bins, dtype=float)
    bin_confidences = np.zeros(num_bins, dtype=float)
    bin_counts = np.zeros(num_bins, dtype=int)

    for b in range(num_bins):
        selected = np.where(indices == b + 1)[0]
        if len(selected) > 0:
            bin_accuracies[b] = np.mean(true_labels[selected] == pred_labels[selected])
            bin_confidences[b] = np.mean(confidences[selected])
            bin_counts[b] = len(selected)

    avg_acc = np.sum(bin_accuracies * bin_counts) / np.sum(bin_counts)
    avg_conf = np.sum(bin_confidences * bin_counts) / np.sum(bin_counts)

    gaps = np.abs(bin_accuracies - bin_confidences)
    ece = np.sum(gaps * bin_counts) / np.sum(bin_counts)
    mce = np.max(gaps)

    return { "accuracies": bin_accuracies, 
             "confidences": bin_confidences, 
             "counts": bin_counts, 
             "bins": bins,
             "avg_accuracy": avg_acc,
             "avg_confidence": avg_conf,
             "expected_calibration_error": ece,
             "max_calibration_error": mce }


def _reliability_diagram_1(bin_data, 
                                  draw_ece, draw_bin_importance, draw_averages, 
                                  title, figsize, dpi, return_fig):
    """Draws a reliability diagram and confidence histogram using the output
    from compute_calibration()."""
    figsize = (figsize[0], figsize[1])

    fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True, figsize=figsize, dpi=dpi, 
                           gridspec_kw={"height_ratios": [1]})

    plt.tight_layout()
    plt.subplots_adjust(hspace=-0.1)

    _reliability_diagram_subplot(ax, bin_data, draw_ece, draw_bin_importance, 
                                 title=title)

    # Draw the confidence histogram upside down.
    orig_counts = bin_data["counts"]
    bin_data["counts"] = -bin_data["counts"]
    # _confidence_histogram_subplot(ax[1], bin_data, draw_averages, title="")
    # bin_data["counts"] = orig_counts

    # # Also negate the ticks for the upside-down histogram.
    # new_ticks = np.abs(ax[1].get_yticks()).astype(int)
    # ticks_loc = ax[1].get_yticks().tolist()
    # ax[1].set_yticks(ax[1].get_yticks().tolist())
    # ax[1].set_yticklabels(new_ticks)    

    plt.show()

    if return_fig: return fig

def _reliability_diagram_subplot(ax, bin_data, 
                                 draw_ece=True, 
                                 draw_bin_importance=False,
                                 title="Reliability Diagram", 
                                 xlabel="Confidence", 
                                 ylabel="Expected Accuracy"):
    """Draws a reliability diagram into a subplot."""
    accuracies = bin_data["accuracies"]
    confidences = bin_data["confidences"]
    counts = bin_data["counts"]
    bins = bin_data["bins"]

    bin_size = 1.0 / len(counts)
    positions = bins[:-1] + bin_size/2.0

    widths = bin_size
    alphas = 0.3
    min_count = np.min(counts)
    max_count = np.max(counts)
    normalized_counts = (counts - min_count) / (max_count - min_count)

    if draw_bin_importance == "alpha":
        alphas = 0.2 + 0.8*normalized_counts
    elif draw_bin_importance == "width":
        widths = 0.1*bin_size + 0.9*bin_size*normalized_counts

    colors = np.zeros((len(counts), 4))
    colors[:, 0] = 60 / 255.
    colors[:, 1] = 60 / 255.
    colors[:, 2] = 60 / 255.
    colors[:, 3] = alphas

    # gap_plt = ax.bar(positions, np.abs(accuracies - confidences), 
    #                  bottom=np.minimum(accuracies, confidences), width=widths,
    #                  edgecolor=colors, color=colors, linewidth=1, label="Gap")

    # gap_plt = ax.bar(positions, np.abs(accuracies - confidences), 
    #                  bottom=accuracies, width=widths,
    #                  edgecolor=colors, color=colors, linewidth=1, label="Gap")

    acc_plt = ax.bar(positions, accuracies, bottom=0, width=widths,
                     edgecolor="navy", color="skyblue", alpha=1.0, linewidth=3,
                     label="Accuracy")

    gap_plt = ax.bar(positions, positions - accuracies, 
                     bottom=accuracies, width=widths-(bin_size/3), hatch = "/\/\/\/", 
                     edgecolor="red", color="lightcoral", linewidth=2, label="Gap")

    # conf_plt = ax.bar(positions, confidences, bottom=0, width=widths,
    #                  edgecolor="black", color="skyblue", alpha=1.0, linewidth=1,
    #                  label="Confidence")

    ax.set_aspect("equal")
    ax.plot([0,1], [0,1], linestyle = "--", color="gray", linewidth=5)
    
    if draw_ece:
        ece = (bin_data["expected_calibration_error"] * 100)
        ax.text(0.5, 0.55, "ECE=%.2f" % ece, color="black", 
                ha="right", va="bottom", transform=ax.transAxes, fontsize=25, fontname='DejaVu Serif')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    #ax.set_xticks(bins)
    plt.rcParams["font.family"] = "DejaVu Serif"
    ax.set_title(title, fontsize=25, fontname='DejaVu Serif')
    ax.set_xlabel(xlabel, fontsize=25, fontname='DejaVu Serif')
    ax.set_ylabel(ylabel, fontsize=25, fontname='DejaVu Serif')
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_xticklabels([0, 0.2, 0.4, 0.6, 0.8, 1])

    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)

    ax.tick_params(axis='both', which='major', labelsize=25)

    ticks_font = matplotlib.font_manager.FontProperties(family='DejaVu Serif', style='normal', size=20, weight='normal', stretch='normal')


    for label in ax.get_xticklabels():
        label.set_fontproperties(ticks_font)

    for label in ax.get_yticklabels():
        label.set_fontproperties(ticks_font)

    ax.legend(handles=[acc_plt, gap_plt], fontsize = 25)

def _confidence_histogram_subplot(ax, bin_data,
                                  draw_averages=True,
                                  title="Examples per bin", 
                                  xlabel="Confidence",
                                  ylabel=r"# of Examples ($10^3$)",
                                  xlow=0,
                                  xhigh=1):
    """Draws a confidence histogram into a subplot."""
    import numpy as np
    from matplotlib import rc,rcParams
    counts = bin_data["counts"]
    bins = bin_data["bins"]
    # write # of Examples (10^3) in math notation and assign it to a string
    # ylabel = r"# of Examples ($10^3$)"

    bin_size = 1.0 / len(counts)
    positions = bins[:-1] + bin_size/2.0

    ax.bar(positions, counts, width=bin_size * 0.95, edgecolor="black", linewidth=3)
   
    ax.set_xlim(xlow, xhigh)
    ax.set_title(title, fontsize=25)
    ax.set_xlabel(xlabel, fontsize=25)
    ax.set_ylabel(ylabel, fontsize=25)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_xticklabels([0, 0.2, 0.4, 0.6, 0.8, 1])

    # ax.set_yticks([0, 2000, 4000, 6000, 8000, 10000])
    # ax.set_yticklabels([0, 2, 4, 6, 8, 10])

    # divide yticks by 1000 to get the correct number of examples
    # ax.set_yticklabels([0, 2, 4, 6, 8, 10])

    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    ax.set_xlim(0, 1.0)

    ax.tick_params(axis='both', which='major', labelsize=25)

    if "CIFAR10)" in title:
        ax.set_ylim(0, 10)
    elif "CIFAR100)" in title:
        # y_ticks = np.arange(0, 4.5, 0.5)
        # y_tick_labels = [str(tick) for tick in y_ticks]
        # plt.set_yticks(y_ticks)
        # plt.set_yticklabels(y_tick_labels)
        ax.set_ylim(0, 4)
    elif "SVHN" in title:
        ax.set_ylim(0, 25)
    else:
        ax.set_ylim(0, 10)

    if draw_averages:
        label = "Accuracy = " + str("{:.2f}".format(bin_data["avg_accuracy"]))
        label = "Accuracy"
        acc_plt = ax.axvline(x=bin_data["avg_accuracy"], ls="solid", lw=5, 
                             c="black", label=label)
        label = "Avg. confidence = " + str("{:.4f}".format(bin_data["avg_confidence"]))
        label = "Confidence"
        conf_plt = ax.axvline(x=bin_data["avg_confidence"], ls="dotted", lw=5, 
                              c="red", label=label)
        ax.legend(handles=[acc_plt, conf_plt], fontsize = 20)
        print("%.2f , %.2f"%(bin_data["avg_confidence"], bin_data["avg_accuracy"]))


def _reliability_diagram_2(bin_data, draw_ece, draw_bin_importance, draw_averages, title, figsize, dpi, return_fig):
    """Draws a reliability diagram and confidence histogram using the output
    from compute_calibration()."""
    figsize = (figsize[0], figsize[1])

    fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True, figsize=figsize, dpi=dpi, 
                           gridspec_kw={"height_ratios": [1]})

    plt.tight_layout()
    plt.subplots_adjust(hspace=-0.1)

    # _reliability_diagram_subplot(ax, bin_data, draw_ece, draw_bin_importance, 
    #                              title=title)

    # Draw the confidence histogram upside down.
    orig_counts = bin_data["counts"]
    bin_data["counts"] = -bin_data["counts"]
    _confidence_histogram_subplot(ax, bin_data, draw_averages, title=title)
    bin_data["counts"] = orig_counts

    # Also negate the ticks for the upside-down histogram.
    new_ticks = np.abs(ax.get_yticks()).astype(int)
    ticks_loc = ax.get_yticks().tolist()
    ax.set_yticks(ax.get_yticks().tolist())
    ax.set_yticklabels(new_ticks)    

    # plt.show()

    if return_fig: return fig

class BrierScore():
    def __init__(self) -> None:
        pass

    def loss(self, outputs, targets):
        K = outputs.shape[1]
        one_hot = np.eye(K)[targets]
        probs = softmax(outputs, axis=1)
        return np.mean( np.sum( (probs - one_hot)**2 , axis=1) )


class CELoss(object):

    def compute_bin_boundaries(self, probabilities = np.array([])):

        #uniform bin spacing
        if probabilities.size == 0:
            bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
            self.bin_lowers = bin_boundaries[:-1]
            self.bin_uppers = bin_boundaries[1:]
        else:
            #size of bins 
            bin_n = int(self.n_data/self.n_bins)

            bin_boundaries = np.array([])

            probabilities_sort = np.sort(probabilities)  

            for i in range(0,self.n_bins):
                bin_boundaries = np.append(bin_boundaries,probabilities_sort[i*bin_n])
            bin_boundaries = np.append(bin_boundaries,1.0)

            self.bin_lowers = bin_boundaries[:-1]
            self.bin_uppers = bin_boundaries[1:]


    def get_probabilities(self, output, labels, logits):
        #If not probabilities apply softmax!
        if logits:
            self.probabilities = softmax(output, axis=1)
        else:
            self.probabilities = output

        self.labels = labels
        self.confidences = np.max(self.probabilities, axis=1)
        self.predictions = np.argmax(self.probabilities, axis=1)
        self.accuracies = np.equal(self.predictions,labels)

    def binary_matrices(self):
        idx = np.arange(self.n_data)
        #make matrices of zeros
        pred_matrix = np.zeros([self.n_data,self.n_class])
        label_matrix = np.zeros([self.n_data,self.n_class])
        #self.acc_matrix = np.zeros([self.n_data,self.n_class])
        pred_matrix[idx,self.predictions] = 1
        label_matrix[idx,self.labels] = 1

        self.acc_matrix = np.equal(pred_matrix, label_matrix)


    def compute_bins(self, index = None):
        self.bin_prop = np.zeros(self.n_bins)
        self.bin_acc = np.zeros(self.n_bins)
        self.bin_conf = np.zeros(self.n_bins)
        self.bin_score = np.zeros(self.n_bins)

        if index == None:
            confidences = self.confidences
            accuracies = self.accuracies
        else:
            confidences = self.probabilities[:,index]
            accuracies = (self.labels == index).astype("float")


        for i, (bin_lower, bin_upper) in enumerate(zip(self.bin_lowers, self.bin_uppers)):
            # Calculated |confidence - accuracy| in each bin
            in_bin = np.greater(confidences,bin_lower.item()) * np.less_equal(confidences,bin_upper.item())
            self.bin_prop[i] = np.mean(in_bin)

            if self.bin_prop[i].item() > 0:
                self.bin_acc[i] = np.mean(accuracies[in_bin])
                self.bin_conf[i] = np.mean(confidences[in_bin])
                self.bin_score[i] = np.abs(self.bin_conf[i] - self.bin_acc[i])

class MaxProbCELoss(CELoss):
    def loss(self, output, labels, n_bins = 15, logits = True):
        self.n_bins = n_bins
        super().compute_bin_boundaries()
        super().get_probabilities(output, labels, logits)
        super().compute_bins()

#http://people.cs.pitt.edu/~milos/research/AAAI_Calibration.pdf
class ECELoss(MaxProbCELoss):

    def loss(self, output, labels, n_bins = 15, logits = True):
        super().loss(output, labels, n_bins, logits)
        return np.dot(self.bin_prop,self.bin_score)

class MCELoss(MaxProbCELoss):
    
    def loss(self, output, labels, n_bins = 15, logits = True):
        super().loss(output, labels, n_bins, logits)
        return np.max(self.bin_score)

#https://arxiv.org/abs/1905.11001
#Overconfidence Loss (Good in high risk applications where confident but wrong predictions can be especially harmful)
class OELoss(MaxProbCELoss):

    def loss(self, output, labels, n_bins = 15, logits = True):
        super().loss(output, labels, n_bins, logits)
        return np.dot(self.bin_prop,self.bin_conf * np.maximum(self.bin_conf-self.bin_acc,np.zeros(self.n_bins)))


#https://arxiv.org/abs/1904.01685
class SCELoss(CELoss):

    def loss(self, output, labels, n_bins = 15, logits = True):
        sce = 0.0
        self.n_bins = n_bins
        self.n_data = len(output)
        self.n_class = len(output[0])

        super().compute_bin_boundaries()
        super().get_probabilities(output, labels, logits)
        super().binary_matrices()

        for i in range(self.n_class):
            super().compute_bins(i)
            sce += np.dot(self.bin_prop, self.bin_score)

        return sce/self.n_class

class TACELoss(CELoss):

    def loss(self, output, labels, threshold = 0.01, n_bins = 15, logits = True):
        tace = 0.0
        self.n_bins = n_bins
        self.n_data = len(output)
        self.n_class = len(output[0])

        super().get_probabilities(output, labels, logits)
        self.probabilities[self.probabilities < threshold] = 0
        super().binary_matrices()

        for i in range(self.n_class):
            super().compute_bin_boundaries(self.probabilities[:,i]) 
            super().compute_bins(i)
            tace += np.dot(self.bin_prop,self.bin_score)

        return tace/self.n_class

#create TACELoss with threshold fixed at 0
class ACELoss(TACELoss):

    def loss(self, output, labels, n_bins = 15, logits = True):
        return super().loss(output, labels, 0.0 , n_bins, logits)

def l2_between_dicts(dict_1, dict_2):
    assert len(dict_1) == len(dict_2)
    dict_1_values = [dict_1[key] for key in sorted(dict_1.keys())]
    dict_2_values = [dict_2[key] for key in sorted(dict_1.keys())]
    return (
            torch.cat(tuple([t.view(-1) for t in dict_1_values])) -
            torch.cat(tuple([t.view(-1) for t in dict_2_values]))
    ).pow(2).mean()


class MovingAverage:

    def __init__(self, ema, oneminusema_correction=True):
        self.ema = ema
        self.ema_data = {}
        self._updates = 0
        self._oneminusema_correction = oneminusema_correction

    def update(self, dict_data):
        ema_dict_data = {}
        for name, data in dict_data.items():
            data = data.view(1, -1)
            if self._updates == 0:
                previous_data = torch.zeros_like(data)
            else:
                previous_data = self.ema_data[name]

            ema_data = self.ema * previous_data + (1 - self.ema) * data
            if self._oneminusema_correction:
                # correction by 1/(1 - self.ema)
                # so that the gradients amplitude backpropagated in data is independent of self.ema
                ema_dict_data[name] = ema_data / (1 - self.ema)
            else:
                ema_dict_data[name] = ema_data
            self.ema_data[name] = ema_data.clone().detach()

        self._updates += 1
        return ema_dict_data


def make_weights_for_balanced_classes(dataset):
    counts = Counter()
    classes = []
    for _, y in dataset:
        y = int(y)
        counts[y] += 1
        classes.append(y)

    n_classes = len(counts)

    weight_per_class = {}
    for y in counts:
        weight_per_class[y] = 1 / (counts[y] * n_classes)

    weights = torch.zeros(len(dataset))
    for i, y in enumerate(classes):
        weights[i] = weight_per_class[int(y)]

    return weights


def pdb():
    sys.stdout = sys.__stdout__
    import pdb
    print("Launching PDB, enter 'n' to step to parent function.")
    pdb.set_trace()


def seed_hash(*args):
    """
    Derive an integer hash from all args, for use as a random seed.
    """
    args_str = str(args)
    return int(hashlib.md5(args_str.encode("utf-8")).hexdigest(), 16) % (2 ** 31)


def print_separator():
    print("=" * 80)


def print_row(row, colwidth=10, latex=False):
    if latex:
        sep = " & "
        end_ = "\\\\"
    else:
        sep = "  "
        end_ = ""

    def format_val(x):
        if np.issubdtype(type(x), np.floating):
            x = "{:.10f}".format(x)
        return str(x).ljust(colwidth)[:colwidth]

    print(sep.join([format_val(x) for x in row]), end_)


class _SplitDataset(torch.utils.data.Dataset):
    """Used by split_dataset"""

    def __init__(self, underlying_dataset, keys):
        super(_SplitDataset, self).__init__()
        self.underlying_dataset = underlying_dataset
        self.keys = keys

    def __getitem__(self, key):
        return self.underlying_dataset[self.keys[key]]

    def __len__(self):
        return len(self.keys)


def split_dataset(dataset, n, seed=0):
    """
    Return a pair of datasets corresponding to a random split of the given
    dataset, with n datapoints in the first dataset and the rest in the last,
    using the given random seed
    """
    assert (n <= len(dataset))
    keys = list(range(len(dataset)))
    np.random.RandomState(seed).shuffle(keys)
    keys_1 = keys[:n]
    keys_2 = keys[n:]
    return _SplitDataset(dataset, keys_1), _SplitDataset(dataset, keys_2)


def random_pairs_of_minibatches(minibatches):
    perm = torch.randperm(len(minibatches)).tolist()
    pairs = []

    for i in range(len(minibatches)):
        j = i + 1 if i < (len(minibatches) - 1) else 0

        xi, yi = minibatches[perm[i]][0], minibatches[perm[i]][1]
        xj, yj = minibatches[perm[j]][0], minibatches[perm[j]][1]

        min_n = min(len(xi), len(xj))

        pairs.append(((xi[:min_n], yi[:min_n]), (xj[:min_n], yj[:min_n])))

    return pairs


def accuracy(network, loader, weights, device,noise_sd=0.5,addnoise=False):
    correct = 0
    total = 0
    weights_offset = 0

    network.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            if(addnoise):
                x=x + torch.randn_like(x, device='cuda') * noise_sd
            p = network.predict(x)
            if weights is None:
                batch_weights = torch.ones(len(x))
            else:
                batch_weights = weights[weights_offset: weights_offset + len(x)]
                weights_offset += len(x)
            batch_weights = batch_weights.to(device)
            # print(p.shape)
            if len(p.shape)==1:
                p = p.reshape(1,-1)
            if p.size(1) == 1:

                # if p.size(1) == 1:
                correct += (p.gt(0).eq(y).float() * batch_weights.view(-1, 1)).sum().item()
            else:
                # print('p hai ye', p.size(1))
                correct += (p.argmax(1).eq(y).float() * batch_weights).sum().item()
            total += batch_weights.sum().item()
    network.train()

    return correct / total


def two_model_analysis(network,network_comp, loader, weights, device,noise_sd=0.5,addnoise=False,env_name="env0"):
    correct = 0
    total = 0
    weights_offset = 0
    pred_cls_all=[]
    pred_comp_cls_all=[]
    y_all=[]
    all_x=[]
    network.eval()
    network_comp.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            if(addnoise):
                x=x + torch.randn_like(x, device='cuda') * noise_sd
            p = network.predict(x)
            p_comp=network_comp.predict(x)
            pred_cls=torch.argmax(p, dim=1)
            pred_comp_cls=torch.argmax(p_comp, dim=1)
            pred_cls_all.append(pred_cls)
            pred_comp_cls_all.append(pred_comp_cls)
            y_all.append(y)
            all_x.append(x)
            if weights is None:
                batch_weights = torch.ones(len(x))
            else:
                batch_weights = weights[weights_offset: weights_offset + len(x)]
                weights_offset += len(x)
            batch_weights = batch_weights.to(device)
            # print(p.shape)
            if len(p.shape)==1:
                p = p.reshape(1,-1)
            if p.size(1) == 1:

                # if p.size(1) == 1:
                correct += (p.gt(0).eq(y).float() * batch_weights.view(-1, 1)).sum().item()
            else:
                # print('p hai ye', p.size(1))
                correct += (p.argmax(1).eq(y).float() * batch_weights).sum().item()
            total += batch_weights.sum().item()
    network.train()
    print("predicted_cls")
    pred_cls_all=torch.cat(pred_cls_all,dim=0).cpu().numpy()
    pred_comp_cls_all=torch.cat(pred_comp_cls_all,dim=0).cpu().numpy()
    y_all=torch.cat(y_all,dim=0).cpu().numpy()
    all_x=torch.cat(all_x,dim=0)
    correct_classes=[]
    selected_images=[]
    for i in range(pred_cls_all.size):
        if(pred_cls_all[i]!=y_all[i] and pred_comp_cls_all[i]==y_all[i]):
            correct_classes.append(pred_comp_cls_all[i])
            torchvision.utils.save_image(all_x[i],'two_model_analysis_neg/test_env:'+env_name+'_cls:'+str(pred_comp_cls_all[i])+"_num_"+str(i)+'.png')
        else:
            correct_classes.append(-1)
    pred_corr=np.array([pred_cls_all==y_all]).astype('int')
    pred_comp_corr=np.array([pred_comp_cls_all==y_all]).astype('int')
    
    pred_only_correct=pred_corr-pred_comp_corr
    pred_only_correct[pred_only_correct<0]=0
    only_correct_in_algo=np.count_nonzero(pred_only_correct)
    print(np.count_nonzero(pred_only_correct))
    print(total)
    print(Counter(correct_classes))
    print("classes:",Counter(y_all))
    
    return only_correct_in_algo / total


def loss_ret(network, loader, weights, device,noise_sd=0.5,addnoise=False):
    correct = 0
    total = 0
    weights_offset = 0
    total_loss=0
    network.eval()
    counter=0
    with torch.no_grad():
        for x, y in loader:
            counter+=1
            x = x.to(device)
            y = y.to(device)
            # p,output_rb = network.network(x,flatness=True)
            if(addnoise):
                x=x + torch.randn_like(x, device='cuda') * noise_sd
            p = network.predict(x)
            
            loss=F.cross_entropy(p,y)
            # rb_loss = F.kl_div(
            #     F.log_softmax(output_rb / 6.0, dim=1),
            #     ## RB output cls token, original network output cls token
            #     F.log_softmax(p / 6.0, dim=1),
            #     reduction='sum',
            #     log_target=True
            # ) * (6.0 * 6.0) / output_rb.numel()

            total_loss+=loss
            # +0.5*rb_loss
            if weights is None:
                batch_weights = torch.ones(len(x))
            else:
                batch_weights = weights[weights_offset: weights_offset + len(x)]
                weights_offset += len(x)
            batch_weights = batch_weights.to(device)
            # print(p.shape)
            if len(p.shape)==1:
                p = p.reshape(1,-1)
            if p.size(1) == 1:

                # if p.size(1) == 1:
                correct += (p.gt(0).eq(y).float() * batch_weights.view(-1, 1)).sum().item()
            else:
                # print('p hai ye', p.size(1))
                correct += (p.argmax(1).eq(y).float() * batch_weights).sum().item()
            total += batch_weights.sum().item()
    network.train()

    return total_loss/(counter*1.0),correct / total

def confusionMatrix(network, loader, weights, device, output_dir, env_name, algo_name,args,algorithm_class,dataset,hparams):
    trials=3
    
    
    if algo_name is None:
        algo_name = type(network).__name__
    conf_mat_all=[]
    
    for i in range(trials):
        pretrained_path=args.pretrained
        pretrained_path=pretrained_path[:-14]+str(i)+pretrained_path[-13:]
        network = algorithm_class(dataset.input_shape, dataset.num_classes,
            len(dataset) - len(args.test_envs), hparams,pretrained_path) #args.pretrained
        network.to(device)
        correct = 0
        total = 0
        weights_offset = 0
        y_pred = []
        y_true = []
        
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device)
                y = y.to(device)
                p = network.predict(x)[-1]
                pred = p.argmax(1)
                y_true = y_true + y.to("cpu").numpy().tolist()
                y_pred = y_pred + pred.to("cpu").numpy().tolist()
                # print(y_true)
                # print("hashf")
                # print(y_pred)
                if weights is None:
                    batch_weights = torch.ones(len(x))
                else:
                    batch_weights = weights[weights_offset: weights_offset + len(x)]
                    weights_offset += len(x)
                batch_weights = batch_weights.to(device)
                if p.size(1) == 1:
                    # if p.size(1) == 1:
                    correct += (p.gt(0).eq(y).float() * batch_weights.view(-1, 1)).sum().item()
                else:
                    # print('p hai ye', p.size(1))
                    correct += (p.argmax(1).eq(y).float() * batch_weights).sum().item()
                total += batch_weights.sum().item()
        
        conf_mat = confusion_matrix(y_true, y_pred)
        # print(confusion_matrix(y_true, y_pred))
        conf_mat_all.append(conf_mat)
        print(conf_mat, 'cf_matrix')
    conf_mat=(conf_mat_all[0]+conf_mat_all[1]+conf_mat_all[2])/(trials*1.0)
    conf_mat=conf_mat.astype('int')
    print(conf_mat, 'cf_matrix_average')
    conf_mat=conf_mat/np.sum(conf_mat,axis=1,keepdims=True) #percentage calculator

    sn.set(font_scale=20)  # for label size
    plt.figure(figsize=(90, 90))
    # sn.heatmap(conf_mat, cbar=False,square=True, annot=True,annot_kws={"size": 90},fmt='d',xticklabels=['DG','EP','GF','GT','HR','HS','PR'],yticklabels=['DG','EP','GF','GT','HR','HS','PR'])  # font size
    ax=sn.heatmap(conf_mat, cmap="Blues", cbar=False,linewidths=4, square=True, annot=True,fmt='.1%',annot_kws={"size": 155},xticklabels=['0','1','2','3','4'],yticklabels=['0','1','2','3','4'])  # font size
    # ax=sn.heatmap(conf_mat, cbar=True, cmap="Blues",annot=True,fmt='.1%',annot_kws={"size": 90},linewidths=4, square = True, xticklabels=['0','1','2','3','4','5','6'],yticklabels=['0','1','2','3','4','5','6'])  # font size
    # ax=sn.heatmap(conf_mat, cbar=True, cmap="Blues",annot=True,fmt='.1%',annot_kws={"size": 90},linewidths=4, square = True, xticklabels=['0','1','2','3','4','5','6'],yticklabels=['0','1','2','3','4','5','6'])  # font size
    plt.yticks(rotation=0)
    ax.xaxis.tick_top() # x axis on top
    ax.xaxis.set_label_position('top')
    ax.axhline(y=0, color='k',linewidth=10)
    ax.axhline(y=conf_mat.shape[1], color='k',linewidth=10)

    ax.axvline(x=0, color='k',linewidth=10)
    ax.axvline(x=conf_mat.shape[1], color='k',linewidth=10)
    # plt.show()
    plt.savefig('Confusion_matrices/'+algo_name+env_name+'.png',bbox_inches='tight')


    

    return correct / total

# cmap='summer'


def TsneFeatures(network, loader, weights, device, output_dir, env_name, algo_name):
 

    correct = 0
    total = 0
    weights_offset = 0
    network.eval()
    Features=[[] for _ in range(12)]
    labels=[]
    if algo_name is None:
        algo_name = type(network).__name__
    try:
        Transnetwork = network.network
    except:
        Transnetwork = network.network_original
    with torch.no_grad():
        count = 0 
        for x, y in loader:
            print(x.shape)
            x = x.to(device)
            y = y.to(device)
            count += 1

            if count == 5:
                break

            p,F = Transnetwork(x,return_feat=True)
            p=p[-1]
            for i in range(len(F)):

                Features[i].append(F[i])
            labels.append(y)
            if weights is None:
                batch_weights = torch.ones(len(x))
            else:
                batch_weights = weights[weights_offset: weights_offset + len(x)]
                weights_offset += len(x)
            batch_weights = batch_weights.to(device)
            if p.size(1) == 1:
                # if p.size(1) == 1:
                correct += (p.gt(0).eq(y).float() * batch_weights.view(-1, 1)).sum().item()
            else:
                # print('p hai ye', p.size(1))
                correct += (p.argmax(1).eq(y).float() * batch_weights).sum().item()
            total += batch_weights.sum().item()
    network.eval()
    labels=torch.cat(labels).cpu().detach().numpy()
    Features_all=[[] for _ in range(12)]
    for i in range(len(Features)):
        Features_all[i]=torch.cat(Features[i],dim=0).cpu().detach().numpy()
    # print(labels)
    print(labels.shape)
    
    name_conv=env_name

    # print(y)
    # print(len(y))
    # print(len(Features))
    # print(Features[0].shape)
    return Features_all,labels

def plot_block_accuracy2(network, loader, weights, device, output_dir, env_name, algo_name):
    # print(network)

    if algo_name is None:
        algo_name = type(network).__name__
    try:
        network = network.network
    except:
        network = network.network_original
    correct = [0] * len(network.blocks)
    total = [0] * len(network.blocks)
    weights_offset = [0] * len(network.blocks)

    network.eval()

    all_targets = []
    all_outputs = []
    all_confidences = []

    with torch.no_grad():
        features = []
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            p1 = network.acc_for_blocks(x)
            out = p1[-1]
            # add softmax
            out = F.softmax(out, dim=1)
            all_confidences.append(out)
            # take the index of the max log-probability from out
            out_index = out.argmax(dim=1)
            all_outputs.append(out_index)
            all_targets.append(y)
            # print('out_index: ', out_index)
            

            # print('label: ', y)
            # print('out: ', out)
            # print('label: ', y.shape)
            # print('out: ', out.shape)
            # print(np.max(out.cpu().numpy(), axis=1), 'max')

            features.append(network.forward_features(x)[-1])
            # print(len(network.forward_features(x)))

            # if all_targets is None:
            #     all_outputs = out_index.cpu().numpy()
            #     all_targets = y.cpu().numpy()
            #     all_confidences = out.cpu().numpy()
            # else:
            #     all_targets = np.concatenate([all_targets, y.cpu().numpy()], axis=0)
            #     all_outputs = np.concatenate([all_outputs, out.cpu().numpy(),], axis=0)
            #     all_confidences = np.concatenate([all_confidences, out.cpu().numpy(),], axis=0)

            
            # print(out)
            for count, p in enumerate(p1):
                if weights is None:
                    batch_weights = torch.ones(len(x))
                else:
                    batch_weights = weights[weights_offset[count]: weights_offset[count] + len(x)]
                    weights_offset[count] += len(x)
                batch_weights = batch_weights.to(device)
                # print(p.size, 'p size')
                # if p.size(1) == 1:
                if p.size(1) == 1:
                    correct[count] += (p.gt(0).eq(y).float() * batch_weights.view(-1, 1)).sum().item()
                else:
                    # print('p hai ye', p.size(1))
                    correct[count] += (p.argmax(1).eq(y).float() * batch_weights).sum().item()
                total[count] += batch_weights.sum().item()

    all_predictions = torch.cat(all_outputs, dim=0).cpu().numpy()
    all_targets = torch.cat(all_targets, dim=0).cpu().numpy()
    all_confidences = torch.cat(all_confidences, dim=0).cpu().numpy()
    
    title = algo_name + " " + env_name
    device = 0

    accuracies = []
    confidences = []
    counts = []
    bins = []
    avg_accuracy = []
    avg_confidence = []
    expected_calibration_error = []
    max_calibration_error = []

    bin_data = compute_calibration(all_predictions, all_targets, all_confidences, num_bins=15)

    accuracies.append(bin_data["accuracies"])
    confidences.append(bin_data["confidences"])
    counts.append(bin_data["counts"])
    bins.append(bin_data["bins"])
    avg_accuracy.append(bin_data["avg_accuracy"])
    avg_confidence.append(bin_data["avg_confidence"])
    expected_calibration_error.append(bin_data["expected_calibration_error"])
    max_calibration_error.append(bin_data["max_calibration_error"])

    bin_data["accuracies"] = np.mean(accuracies, axis=0)
    bin_data["confidences"] = np.mean(confidences, axis=0)
    bin_data["counts"] = np.mean(counts, axis=0)/1000.0
    bin_data["bins"] = np.mean(bins, axis=0)
    bin_data["avg_accuracy"] = np.mean(avg_accuracy, axis=0)
    bin_data["avg_confidence"] = np.mean(avg_confidence, axis=0)
    bin_data["expected_calibration_error"] = np.mean(expected_calibration_error, axis=0)
    bin_data["max_calibration_error"] = np.mean(max_calibration_error, axis=0)


    fig1 = _reliability_diagram_1(bin_data, draw_ece=True, draw_bin_importance="alpha",
                            draw_averages=True, title=title, figsize=(6, 6), 
                            dpi=100, return_fig=True)

    fig2 = _reliability_diagram_2(bin_data, draw_ece=True, draw_bin_importance="alpha",
                            draw_averages=True, title=title, figsize=(6, 6), 
                            # draw_averages=True, title=title, figsize=(6, 4), 
                            dpi=100, return_fig=True)

    fig1.tight_layout()
    fig2.tight_layout()

    if not os.path.exists("./histograms/{}".format(env_name)):
        os.makedirs("./histograms/{}".format(env_name))

    fig1.savefig("./histograms/{}/{}_1.png".format(env_name, title))
    fig2.savefig("./histograms/{}/{}_2.png".format(env_name, title))

    print("Done")


    # print("Accuracies: ", accuracies)
    
    # print('all_outputs: ', torch.cat(all_outputs, dim=0).shape)
    # print('all_targets: ', torch.cat(all_targets, dim=0).shape)
    # print('all_confidences: ', torch.cat(all_confidences, dim=0).shape)
    
    # eces = ECELoss().loss(all_outputs, all_targets, n_bins=15)
    # cces = SCELoss().loss(all_outputs, all_targets, n_bins=15)

    # print('ECE_loss: ',eces)
    # print('SCE_loss: ',cces)

    features = torch.cat(features, dim=0)


    res = [i / j for i, j in zip(correct, total)]
    print(algo_name, ":", env_name, ":blockwise accuracies:", res)
    plt.plot(res)
    plt.title(algo_name)
    plt.xlabel('Block#')
    plt.ylabel('Acc')
    plt.ylim(0.0,1.0)
    plt.savefig(output_dir + "/" + algo_name + "_" + env_name + "_" + 'acc.png')
    return res
    return features


class Tee:
    def __init__(self, fname, mode="a"):
        self.stdout = sys.stdout
        self.file = open(fname, mode)

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)
        self.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()


class ParamDict(OrderedDict):
    """Code adapted from https://github.com/Alok/rl_implementations/tree/master/reptile.
    A dictionary where the values are Tensors, meant to represent weights of
    a model. This subclass lets you perform arithmetic on weights directly."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, *kwargs)

    def _prototype(self, other, op):
        if isinstance(other, Number):
            return ParamDict({k: op(v, other) for k, v in self.items()})
        elif isinstance(other, dict):
            return ParamDict({k: op(self[k], other[k]) for k in self})
        else:
            raise NotImplementedError

    def __add__(self, other):
        return self._prototype(other, operator.add)

    def __rmul__(self, other):
        return self._prototype(other, operator.mul)

    __mul__ = __rmul__

    def __neg__(self):
        return ParamDict({k: -v for k, v in self.items()})

    def __rsub__(self, other):
        # a- b := a + (-b)
        return self.__add__(other.__neg__())

    __sub__ = __rsub__

    def __truediv__(self, other):
        return self._prototype(other, operator.truediv)
    

class MMD(nn.Module):
    def __init__(self, kernel_mul = 2.0, kernel_num = 5):
        super(MMD, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0])+int(target.size()[0])
        total = torch.cat([source, target], dim=0)

        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)
    
    def forward(self, source, target):
        batch_size = int(source.size()[0])
        kernels = guassian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY -YX)
        return loss

# class MMD_loss(nn.Module):
# 	def __init__(self, kernel_mul = 2.0, kernel_num = 5):
# 		super(MMD_loss, self).__init__()
# 		self.kernel_num = kernel_num
# 		self.kernel_mul = kernel_mul
# 		self.fix_sigma = None
# 		return

#     def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
# 		# n_samples = int(source.size()[0])+int(target.size()[0])
#     	total = torch.cat([source, target], dim=0)

#     	total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
#     	total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
#     	L2_distance = ((total0-total1)**2).sum(2) 
#     	if fix_sigma:
#     		bandwidth = fix_sigma
#     	else:
#     		bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
#     	bandwidth /= kernel_mul ** (kernel_num // 2)
#     	bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
#     	kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
#     	return sum(kernel_val)

#     def forward(self, source, target):
#     	batch_size = int(source.size()[0])
#     	kernels = guassian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
#     	XX = kernels[:batch_size, :batch_size]
#     	YY = kernels[batch_size:, batch_size:]
#     	XY = kernels[:batch_size, batch_size:]
#     	YX = kernels[batch_size:, :batch_size]
#     	loss = torch.mean(XX + YY - XY -YX)
#     	return loss
