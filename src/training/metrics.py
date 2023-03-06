#references
# https://github.com/bilalsal/blocks/blob/master/Places365_example.ipynb
# https://github.com/cbernecker/medium/blob/main/confusion_matrix.ipynb

import os
import torch
import logging
from datetime import datetime
import time

try:
    from sklearn.metrics import confusion_matrix
    import pandas as pd
    import numpy as np
    import seaborn as sn
    from matplotlib import pylab as plt
except Exception as e:
    logging.warning("Error loading extended metrics libraries: extended metrics will fail")
    logging.warning(e)

def reorder_matrix(mat, showdendrogram = False):
    import scipy.cluster.hierarchy as sch
    Y = sch.linkage(mat, method='centroid')
    Z = sch.dendrogram(Y, orientation='left', no_plot= not showdendrogram)

    index = Z['leaves']
    mat = mat[index,:]
    mat = mat[:,index]
    return mat, index
    # Plot distance matrix.

def show_matrix(args, mat, title):
    fig = plt.figure(figsize=(28, 28), dpi=200)
    fig.suptitle(title, ha = "center", fontsize = 20)

    axmatrix = fig.add_axes([0, 0, 0.9,0.9], label='axes1')
    im = axmatrix.matshow(mat, aspect='auto',  origin='lower')
    # axmatrix.set_xticks([0, 50, 100, 150, 200, 250, 300, 350])
    # axmatrix.set_yticks([0, 50, 100, 150, 200, 250, 300, 350])
    # axmatrix.set_xticklabels([0, 50, 100, 150, 200, 250, 300, 350])
    # axmatrix.set_yticklabels([0, 50, 100, 150, 200, 250, 300, 350])
    res = str(datetime.now())[:19]
    res = res.translate({ord(":"): "-", ord(" "):"_"})
    axcolor = fig.add_axes([0.95,0,0.02,0.9])
    plt.colorbar(im, cax=axcolor)
    save_path = os.path.join(args.conf_path, 'clustered_confusion_matrix_{}.svg'.format(res))
    fig.savefig(save_path, format='svg', dpi=200)

# def show_5x5_img_grid(indices):
#     curr_row = 0
#     fig = plt.figure(figsize=(28, 28), dpi=1200)
#     axarr = fig.subplots(5, 5)
#     for  i in range(25):
#          col = i % 5
#          index = indices[i]
#          implot = axarr[col,curr_row].imshow(sample_img[index])
#          axarr[col,curr_row].set_title(classes[index], fontsize = 8)
#          axarr[col,curr_row].axis('off')
#          if col == 4:
#              curr_row += 1

def log_confusion_matrix(args, output, labels):
    # print("output before")
    # print(output)
    # print(output.size())
    output = output.topk(max((1, 5)), 1, True, True)
    # print("output after topk")
    output = output[1].t()[0].data.cpu()
    output = np.array(output, ndmin=1)  
    #output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
    labels = labels.data.cpu()
    labels = np.array(labels, ndmin=1)
    # if isinstance(labels, int):
    #     labels = [labels]
    # print("output after, labels")
    #print(output, output.shape)
    #print(labels, labels.shape)
    # assert len(labels) == len(output)
    args.y_pred.extend(output) # Save Prediction
    args.y_true.extend(labels) # Save Truth

def write_confusion_matrix(args, output, labels, classes):
    #confusion matrix
    cf_matrix = confusion_matrix(args.y_true, args.y_pred)
    if len(cf_matrix) != len(classes):
        classes = np.array(classes)
        classes = classes[np.unique(args.y_pred + args.y_true).tolist()].tolist()
    df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) * 10, index = [i for i in classes],
                        columns = [i for i in classes])
    df_cm = df_cm.div(df_cm.sum(axis=1), axis=0)
    args.conf_path = os.path.join(args.log_base_path, "confusion_matrix")
    if not os.path.exists(args.conf_path):
        os.mkdir(args.conf_path)
    res = str(datetime.now())[:19]
    res = res.translate({ord(":"): "-", ord(" "):"_"})
    logging.info('Writing confusion matrix')
    df_cm.to_csv(os.path.join(args.conf_path, "confusion_matrix_{}.csv".format(res)), index=False)
    per_class_acc = pd.Series(np.diag(df_cm), index=[df_cm.index, df_cm.columns])
    per_class_acc = pd.DataFrame(per_class_acc).transpose()
    per_class_acc.columns = [''.join(col[1:]) for idx, col in enumerate(per_class_acc.columns.values)]
    per_class_acc.to_csv(os.path.join(args.conf_path, "per_class_acc_{}.csv".format(res)), index=False)
    font_size = round(1 * 100//len(classes), 2)
    if font_size < 0.1:
        font_size = 0.1
    sn.set(font_scale=font_size)
    #if len(classes) < 201:
    start = time.time()
    print("Saving confusion matrix: this might take a while...")
    plt.figure(figsize = (168,80), dpi=200)
    sn.heatmap(df_cm, annot=True)
    plt.savefig(os.path.join(args.conf_path, "confusion_matrix_{}.svg".format(res)), format='svg', dpi=200)        
    plt.close('all')
    print("Saving confusion matrix: done in {} seconds".format(time.time() - start))
    #class-class clustering matrix
    logging.info('Saving class-class clustering matrix')
    logit_concat = np.concatenate(args.logits, axis=0)
    corr_mat_logits = np.corrcoef(logit_concat, rowvar=False)
    corr_mat_logits[corr_mat_logits < 0] = 0 # not quite necessary, but helps sharpen the blocks
    try:
        corr_mat_logits, indices_logits = reorder_matrix(corr_mat_logits)
        show_matrix(args, corr_mat_logits, 'Logit-based Similarity Matrix')
    except Exception as e:
        logging.warning("Clustering matrix did not save")
        logging.warning(e)


