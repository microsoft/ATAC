import torch

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise TypeError('Boolean value expected.')


def torch_method(torch_fun):
    def wrapped_fun(x):
        with torch.no_grad():
            return torch_fun(torch.Tensor(x)).numpy()
    return wrapped_fun


import numpy as np
import csv
def read_attr_from_csv(csv_path, attr, delimiter=','):
    with open(csv_path) as csv_file:
        reader = csv.reader(csv_file, delimiter=delimiter)
        try:
            row = next(reader)
        except Exception:
            return None
        if attr not in row:
            return None
        idx = row.index(attr)  # the column number for this attribute
        vals = []
        for row in reader:
            vals.append(row[idx])

    vals = [np.nan if v=='' else v for v in vals]
    return np.array(vals, dtype=np.float64)
