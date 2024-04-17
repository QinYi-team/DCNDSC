import numpy as np
import pandas as pd
import torch

# check accuracy of model,if error_analysis return confuse matrix
def accuracy(model, loader, device, error_analysis=False):
    # save error samples predicted
    ys = np.array([])
    y_pred = np.array([])
    y_label = np.array([])
    cf_matrix = None
    correct_num = 0
    # start validate model
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x.resize_(x.size()[0], 1, x.size()[1])
            x, y = x.float(), y.long()
            x, y = x.to(device), y.to(device)
            # prediect section
            score = model(x)
            pred = score.max(1, keepdim=True)[1]
            correct_num += pred.eq(y.view_as(pred)).sum().item()
            pred_values = pred.cpu().numpy()
            y_label = np.append(y_label, pred_values)
            # confuse matrix: label and pred
            if error_analysis:
                b = y.cpu()
                a = np.array(b)
                ys = np.append(ys, a)
                y_pred = np.append(y_pred, np.array(pred.cpu()))

    acc = float(correct_num) / len(loader.dataset)
    # confuse matrix
    if error_analysis:
        cf_matrix = pd.crosstab(y_pred, ys, margins=True)
    print('num: %d / %d correct (%.2f)'%(correct_num, len(loader.dataset), 100*acc))
    # print(y_label)
    return acc, cf_matrix
