import glob
import os
import numpy as np
import pandas as pd

from logging import getLogger, StreamHandler, DEBUG
# logger = getLogger(__name__)
# handler = StreamHandler()
# handler.setLevel(DEBUG)
# logger.setLevel(DEBUG)
# logger.addHandler(handler)
# logger.propagate = False

# # logger = getLogger(__name__)
# handler = StreamHandler()
# handler.setLevel(DEBUG)
# logger.setLevel(DEBUG)
# logger.addHandler(handler)
# logger.propagate = False
# # logger.debug('hello')

def make_dataset(loglevel='INFO'):
    """
    train data 
        - Exp:06,11, Loc:01-06, Direc:left&right
    test data
        - Exp:02,03,04,05, Loc:01~06, Dirc:left&right => A
        - Exp:07,08,09,10, Loc:01~06, Dirc:left&right => B
    としてデータを作ります。
    
    Parameters
    ----------
    Returns
    -------
    train_feat_arr_list (list) 
    train_flow_arr_list (list)
    test_A_feat_arr_list (list) 
    test_A_flow_arr_list (list)
    test_B_feat_arr_list (list)
    test_B_flow_arr_list (list)
    """

    data_path = '/work-hmcomm/project/nedo2020_yokogawa/data/PilotPlant_Experiments_2020-0309/feature/extracted_on_2020-04-21_melspectrogram/'
    data_path = data_path + '*npy'
    data_path_list = sorted(glob.glob(data_path))

    train_set = set(['Exp06', 'Exp11'])
    test_A_set = set(['Exp02','Exp03','Exp04','Exp05', ])
    test_B_set = set(['Exp07','Exp08','Exp09','Exp10', ])

    train_path_list = []
    test_A_path_list = []
    test_B_path_list = []
    for i in data_path_list:
        if 'Exp01' not in i and '_7_' not in i:
            exp = os.path.basename(i).split('_')[3]
            if exp in train_set: 
                train_path_list.append(i)
            elif exp in test_A_set:
                test_A_path_list.append(i)
            elif exp in test_B_set:
                test_B_path_list.append(i)
        else:
            pass
#             logger.debug(f'{i}\n')


    # data 確認
#     logger.debug(f'n_train_data:\n{len(train_path_list)}\ntrain data:\n{train_path_list}\n')
#     logger.debug(f'n test A data:\n{len(test_A_path_list)}\ntest A data:\n{test_A_path_list}\n')
#     logger.debug(f'n test B data:\n{len(test_B_path_list)}\ntest B data:\n{test_B_path_list}\n')


    train_arr_list = [np.load(i) for i in train_path_list]
    test_A_arr_list = [np.load(i) for i in test_A_path_list]
    test_B_arr_list = [np.load(i) for i in test_B_path_list]

    train_name_list = [os.path.splitext(os.path.basename(i))[0] for i in train_path_list]
    test_A_name_list = [os.path.splitext(os.path.basename(i))[0] for i in test_A_path_list]
    test_B_name_list = [os.path.splitext(os.path.basename(i))[0] for i in test_B_path_list]
    
    print(f'data set size\n')

    print(f'train_arr_list len: {len(train_arr_list)}')
    print(f'train_arr_list shape: {train_arr_list[0].shape}')
    print(f'test_A_arr_list len: {len(test_A_arr_list)}')
    print(f'test_A_arr_list shape: {test_A_arr_list[0].shape}')
    print(f'test_B_arr_list len: {len(test_B_arr_list)}')
    print(f'test_B_arr_list shaoe: {test_B_arr_list[0].shape}')

#     logger.info(f'train_arr_list len: {len(train_arr_list)}')
#     logger.info(f'train_arr_list shape: {train_arr_list[0].shape}')
#     logger.info(f'test_A_arr_list len: {len(test_A_arr_list)}')
#     logger.info(f'test_A_arr_list shape: {test_A_arr_list[0].shape}')
#     logger.info(f'test_B_arr_list len: {len(test_B_arr_list)}')
#     logger.info(f'test_B_arr_list shaoe: {test_B_arr_list[0].shape}')

    def make_df(name_list, arr_list):
        set_arr = []
        for i, file_name in enumerate(name_list):
#             print(f'processing for {file_name} ...')
            loc_id = file_name.split('_')[1]
            exp_id = file_name.split('_')[3]
            feat_arr = arr_list[i]
            set_arr.append([exp_id, loc_id, feat_arr])
        return pd.DataFrame(set_arr, columns=['ExperimentID', 'LocationID', 'feat_arr'])

    train_feature_df = make_df(train_name_list, train_arr_list)
    test_A_feature_df = make_df(test_A_name_list, test_A_arr_list)
    test_B_feature_df = make_df(test_B_name_list, test_B_arr_list)

    train_feature_df.head()

    # flow data load

    pkl_path = '/work-hmcomm/project/nedo2020_yokogawa/data/PilotPlant_Experiments_2020-0309/feature/'
    pkl_path = pkl_path + 'fastd-standard-features-complete_left-right-mean_S200msW400msHan_flow_meta-data_labeled_df.pkl'
    df = pd.read_pickle(pkl_path)
    df = df.loc[:,['ExperimentID', 'LocationID', 'flow_array']]

    df.head()

    train_df = pd.merge(train_feature_df, df, how='left')
    test_A_df = pd.merge(test_A_feature_df, df, how='left')
    test_B_df = pd.merge(test_B_feature_df, df, how='left')
    print('\n')
    print('*'*30)
    print(f'train_df : {len(train_df)}')
    display(train_df.head())
    print(f'test_A_df : {len(test_A_df)}')
    display(test_A_df.head())
    print(f'test_B_df : {len(test_B_df)}')
    display(test_B_df.head())

    # 長さ合わせ　
    def organize_dataflame_size(dataflame):
#         print('\nmodification data length...')
        for i in range(len(dataflame)):
            feat_len = len(dataflame.iloc[i,2][0,:])
            flow_len = len(dataflame.iloc[i,3])
#             print('Size before modification')
#             print(len(dataflame.iloc[i,2][0,:]))
#             print(len(dataflame.iloc[i,3])) 
            if not feat_len == flow_len:
                dataflame.iloc[i,2] = dataflame.iloc[i,2][:,:flow_len]

#                 print('-'*20)
#                 print('modification')
#                 print('Size after modification')
#                 print(dataflame.iloc[i,2].shape)
#                 print(dataflame.iloc[i,3].shape) 
            else:
                pass 
#                 print('-'*20)
#                 print('No modification\n\n')
#     print('\n')
#     print(f'modification train data')
    display(organize_dataflame_size(train_df))
#     print(f'modification test A data')
    display(organize_dataflame_size(test_A_df))
#     print(f'modification test B  data')
    display(organize_dataflame_size(test_B_df))

    # Exp06 location:6 のデータを落とす
    train_df = train_df.drop([20,21])
    train_df

    train_feat_arr_list = [i for i in train_df['feat_arr']]
    train_flow_arr_list = [i for i in train_df['flow_array']]

    test_A_feat_arr_list = [i for i in test_A_df['feat_arr']]
    test_A_flow_arr_list = [i for i in test_A_df['flow_array']]

    test_B_feat_arr_list = [i for i in test_B_df['feat_arr']]
    test_B_flow_arr_list = [i for i in test_B_df['flow_array']]
    
    
    return train_df, test_A_df, test_B_df 