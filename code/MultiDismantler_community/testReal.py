import argparse
import os
import re
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn

from MultiDismantler_torch import MultiDismantler
def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

g_type = "GMM"
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", default="../../results/community_cost/MultiDismantler_real",
                help="path to output file")
args = vars(ap.parse_args())

def find_latest_ckpt(models_dir, prefix="nrange"):
    latest_iter = -1
    latest_path = None
    if not os.path.isdir(models_dir):
        return None
    for fname in os.listdir(models_dir):
        if not fname.startswith(prefix) or not fname.endswith(".ckpt"):
            continue
        m = re.search(r"nrange_30_50_iter_(\d+)\.ckpt", fname)
        if m:
            val = int(m.group(1))
            if val > latest_iter:
                latest_iter = val
                latest_path = os.path.join(models_dir, fname)
    return latest_path

def GetSolution(STEPRATIO, MODEL_FILE):
    ######################################################################################################################
    ##................................................Get Solution (model).....................................................
    dqn = MultiDismantler()
    data_test_path = './data/real/'
    data_test_name = [
        # 'fao_trade_multiplex',
        # 'celegans_connectome_multiplex',
        # 'fb-tw',
        # 'homo_genetic_multiplex',
        # 'sacchpomb_genetic_multiplex',
        # 'Sanremo2016_final_multiplex',
        # "arxiv_netscience_multiplex",
        # "EUAirTransportation_multiplex",
        # "humanHIV1_genetic_multiplex",
        "Padgett-Florentine-Families_multiplex",
        "netsci_co-authorship_multiplex",
        "Lazega-Law-Firm_multiplex",
    ]
    date_test_n = [
        # 214,
        # 279,
        # 1043,
        # 18222,
        # 4092,
        # 56562,
        # 14488,
        # 450,
        # 1005,
        16,
        1400,
        71,
    ]
    data_test_layer = [
        # (3,24),
        # (2,3),
        # (1,2),
        # (1,2),
        # (4,6),
        # (1,2),
        # (4, 8),
        # (1, 11),
        # (1, 5),
        (1, 2),
        (1, 2),
        (1, 3),
    ]
    # resolve model file
    if MODEL_FILE:
        model_file = "./models/g0-1_10w_TORCH-Model_GMM_30_50/{}".format(MODEL_FILE)
    else:
        model_file = find_latest_ckpt('./models/g0-1_10w_TORCH-Model_GMM_30_50')
    if model_file is None:
        raise FileNotFoundError("No checkpoint found under ./models for community model")
    ## save_dir
    save_dir = args['output']
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    ## begin computing...
    print ('The best model is :%s'%(model_file))
    dqn.LoadModel(model_file)
    for j in range(len(data_test_name)):
        df = pd.DataFrame(np.arange(2*len(data_test_name)).reshape((2,len(data_test_name))),index=['time','score'], columns=data_test_name)
        #################################### modify to choose which stepRatio to get the solution
        stepRatio = STEPRATIO
        print ('\nTesting dataset %s'%data_test_name[j])
        data_test = data_test_path + data_test_name[j] + '.edges'
        solution, time, score= dqn.EvaluateRealData(model_file, data_test, save_dir, stepRatio,date_test_n[j],data_test_layer[j])
        df.iloc[0,j] = time
        df.iloc[1,j] = score
        print('Data:%s, time:%.2f, audc:%.6f'%(data_test_name[j], time, score))
        save_dir_local = save_dir + '/StepRatio_%.4f' % stepRatio
        if not os.path.exists(save_dir_local):
            os.mkdir(save_dir_local)
        df.to_csv(save_dir_local + '/time&audc_%s.csv'% data_test_name[j], encoding='utf-8', index=False)


def main():
    outputpath = f"{args['output']}"    
    model_file_ckpt = None  # auto-pick latest if not provided
    GetSolution(0, model_file_ckpt)


if __name__=="__main__":
    cudnn.benchmark = True
    cudnn.deterministic = False
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    main()
