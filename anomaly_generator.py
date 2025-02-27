import numpy as np
import torch

def gen_add_anomaly(timeseries, bias, num_anomalies=1, min_val=None, max_val=None):
    # Select iid random index to add bias (Q1)
    if isinstance(timeseries, torch.Tensor):
        indicies = torch.randint(len(timeseries), (num_anomalies,))
        anomaly = timeseries[indicies] + bias
        if min_val is not None:
            anomaly = torch.maximum(anomaly, torch.tensor(min_val))
        if max_val is not None:
            anomaly = torch.minimum(anomaly, torch.tensor(max_val))
    else:
        indicies = np.random.randint(len(timeseries), size=(num_anomalies,))
        anomaly = timeseries[indicies] + bias
        if min_val is not None:
            anomaly = np.maximum(anomaly, min_val)
        if max_val is not None:
            anomaly = np.minimum(anomaly, max_val)
    timeseries[indicies] = anomaly

def gen_mult_anomaly(timeseries, bias, num_anomalies=1, min_val=None, max_val=None):
    # Select iid random index to multiply bias (Q1)
    if isinstance(timeseries, torch.Tensor):
        indicies = torch.randint(len(timeseries), (num_anomalies,))
        anomaly = timeseries[indicies] * bias
        if min_val is not None:
            anomaly = torch.maximum(anomaly, torch.tensor(min_val))
        if max_val is not None:
            anomaly = torch.minimum(anomaly, torch.tensor(max_val))
    else:
        indicies = np.random.randint(len(timeseries), size=(num_anomalies,))
        anomaly = timeseries[indicies] * bias
        if min_val is not None:
            anomaly = np.maximum(anomaly, min_val)
        if max_val is not None:
            anomaly = np.minimum(anomaly, max_val)
    timeseries[indicies] = anomaly

def gen_sustained_add(timeseries, bias, len_anomaly=4, min_val=None, max_val=None):
    # Select len_anomaly consecutive indicies to add bias (Q2)
    start_index = np.random.randint(len(timeseries)-len_anomaly+1)
    anomaly = timeseries[start_index:start_index+len_anomaly] + bias
    if isinstance(timeseries, torch.Tensor):
        if min_val is not None:
            anomaly = torch.maximum(anomaly, torch.tensor(min_val))
        if max_val is not None:
            anomaly = torch.minimum(anomaly, torch.tensor(max_val))
    else:
        if min_val is not None:
            anomaly = np.maximum(anomaly, min_val)
        if max_val is not None:
            anomaly = np.minimum(anomaly, max_val)
    timeseries[start_index:start_index+len_anomaly] = anomaly

def gen_sustained_mult(timeseries, bias, len_anomaly=4, min_val=None, max_val=None):
    # Select len_anomaly consecutive indicies to multiply bias (Q2)
    start_index = np.random.randint(len(timeseries)-len_anomaly+1)
    anomaly = timeseries[start_index:start_index+len_anomaly] * bias
    if isinstance(timeseries, torch.Tensor):
        if min_val is not None:
            anomaly = torch.maximum(anomaly, torch.tensor(min_val))
        if max_val is not None:
            anomaly = torch.minimum(anomaly, torch.tensor(max_val))
    else:
        if min_val is not None:
            anomaly = np.maximum(anomaly, min_val)
        if max_val is not None:
            anomaly = np.minimum(anomaly, max_val)
    timeseries[start_index:start_index+len_anomaly] = anomaly

def gen_bleed_add(timeseries, bias, bias_step, len_anomaly=4, min_val=None, max_val=None):
    # Select len_anomaly consecutive indicies to add increasing bias (Q3)
    start_index = np.random.randint(len(timeseries)-len_anomaly+1)
    if isinstance(timeseries, torch.Tensor):
        anomaly = timeseries[start_index:start_index+len_anomaly] + bias + torch.arange(len_anomaly) * bias_step
        if min_val is not None:
            anomaly = torch.maximum(anomaly, torch.tensor(min_val))
        if max_val is not None:
            anomaly = torch.minimum(anomaly, torch.tensor(max_val))
    else:
        anomaly = timeseries[start_index:start_index+len_anomaly] + bias + np.arange(len_anomaly) * bias_step
        if min_val is not None:
            anomaly = np.maximum(anomaly, min_val)
        if max_val is not None:
            anomaly = np.minimum(anomaly, max_val)
    timeseries[start_index:start_index+len_anomaly] = anomaly

def gen_bleed_mult(timeseries, bias, bias_step, len_anomaly=4, min_val=None, max_val=None):
    # Select len_anomaly consecutive indicies to multiply increasing bias (Q3)
    start_index = np.random.randint(len(timeseries)-len_anomaly+1)
    if isinstance(timeseries, torch.Tensor):
        anomaly = timeseries[start_index:start_index+len_anomaly] * (bias + torch.arange(len_anomaly) * bias_step)
        if min_val is not None:
            anomaly = torch.maximum(anomaly, torch.tensor(min_val))
        if max_val is not None:
            anomaly = torch.minimum(anomaly, torch.tensor(max_val))
    else:
        anomaly = timeseries[start_index:start_index+len_anomaly] * (bias + np.arange(len_anomaly) * bias_step)
        if min_val is not None:
            anomaly = np.maximum(anomaly, min_val)
        if max_val is not None:
            anomaly = np.minimum(anomaly, max_val)
    timeseries[start_index:start_index+len_anomaly] = anomaly