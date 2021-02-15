import pandas as pd
import numpy as np
import os
import pickle as pkl
from tqdm import tqdm

feature_list = [0, 1, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
def append_snippet(df, index, start, end, out_list):
    copy = df.iloc[start:end].values[:, feature_list]
    if copy.shape[0] != (end-start):
        Exception('bad play length')
        return
    
    out_list.append({'index': index, 'play': copy})

def get_sequences(data_folder, time_length, stride):
    # time_length: number of frames to save
    # stride: how many time steps to skip
    
    file_list = os.listdir(data_folder)
    
    data_list = []    
    for file in tqdm(file_list):
        if file.endswith(".pkl"):
            load_file = data_folder + "/" + file
            game_df = pkl.load(open(load_file, "rb"))
            
            for play_index, entry in game_df.iterrows():
                possession_df = entry['possession']
                T = len(possession_df.index)
                N = T//stride
                for n in range(N):
                    start = n*stride
                    end = start + time_length
                    append_snippet(possession_df, play_index, start, end, data_list)
    
    trajectory_df = pd.DataFrame(data_list)
    return trajectory_df

def save_sequences(save_path, data_folder, time_length, stride):
    # save path: where to save plays
    # time_length: number of frames to save
    # stride: how many time steps to skip
    
    save_df = get_sequences(data_folder, time_length, stride)
    
    pkl.dump((save_df), open(save_path, 'wb'))


def extract_from_df(df, label, n_total, n_train, n_val, 
                    X_train, y_train, X_val, y_val, X_test, y_test):
    i = 0
    for _, p in df.iterrows():
        inp_vals = p['play'].values[:, feature_list]
        if i < n_train:
            X_train.append(inp_vals)
            y_train.append(label)
        elif i < n_val:
            X_val.append(inp_vals)
            y_val.append(label)
        else:
            X_test.append(inp_vals)
            y_test.append(label)
        i = i+1
        if i > n_total:
            break

def get_classifier_data(mode, load_path):    
    X_train = []; y_train = [];
    X_val = []; y_val = [];
    X_test = []; y_test = [];
    arglist = [X_train, y_train, X_val, y_val, X_test, y_test]
    
    (screen_df_orig, handoff_df_orig, negative_df_orig) = pkl.load(open(load_path, 'rb'))
    
    screen_df = screen_df_orig
    handoff_df = handoff_df_orig
    negative_df = negative_df_orig
    
    n_screens = len(screen_df)
    n_s_train = int(n_screens * 0.8)
    n_s_val = int(n_screens * 0.9)
    n_handoffs = len(handoff_df)
    n_h_train = int(n_handoffs * 0.8)
    n_h_val = int(n_handoffs * 0.9)
    n_others = n_s_train + n_s_val + len(negative_df)*0.1
    
    if mode == 'both':
        extract_from_df(screen_df, [0], n_screens, n_s_train, n_s_val, *arglist)
        extract_from_df(handoff_df, [1], n_handoffs, n_h_train, n_h_val, *arglist)
        extract_from_df(negative_df, [2], n_others, n_s_train, n_s_val, *arglist)
        n_classes = 3
    elif mode == 'p&r':
        extract_from_df(screen_df, [0], n_screens, n_s_train, n_s_val, *arglist)
        extract_from_df(negative_df, [1], n_screens, n_s_train, n_s_val, *arglist)
        n_classes = 2
    elif mode == 'handoff':
        extract_from_df(handoff_df, [0], n_handoffs, n_h_train, n_h_val, *arglist)
        extract_from_df(negative_df, [1], n_handoffs, n_h_train, n_h_val, *arglist)
        n_classes = 2
    
    X_train = np.array(X_train, dtype='float')
    y_train = np.array(y_train, dtype='float')
    X_val = np.array(X_val, dtype='float')
    y_val = np.array(y_val, dtype='float')
    X_test = np.array(X_test, dtype='float')
    y_test = np.array(y_test, dtype='float')
    
    return X_train, y_train, X_val, y_val, X_test, y_test, n_classes