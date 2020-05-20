import os
import sys
import yaml

import numpy as np
import pandas as pd
from random import randint
import random
import matplotlib.pyplot as plt
import matplotlib as mpl

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split 

# model import 
lib_path = '/work-hmcomm/project/nedo2020_yokogawa/Git/yokogawa_ryamaguchi/nedo2020-yokogawass/script/RY_module/model/'
sys.path.insert(0,lib_path)

import ResCNN

class model_CNN_Res:
    def __init__(self, 
                 config_path = './config.yml',
#                  net = None
                ):
        
        self.load_config(config_path)
        self.net = ResCNN.ResNetCNN(channel_in = self.channel_in,
                                    channel = self.channel,
                                    channel_res = self.channel_res,
                                    n_res_block = self.n_res_block,
                                    n_cnn_block = self.n_cnn_block,
                                    n_dense_block = self.n_dense_block,
                                    n_dense_hidden = self.n_dense_hidden,
                                    num_classes = self.num_classes,
                                    is_bn_dense = self.is_bn_dense,
                                   )
        
#         self.net.apply(self.init_weights)
        self.update_device()
        self.net.to(self.device)
#         self.save_model_path = './'
        
        self.train_loss_list = []
        self.train_acc_list = []
        self.valid_loss_list = []
        self.valid_acc_list = []
        self.y_pred_list = []
#         self.learning_rate = learning_rate
#         self.net = net
        
        
        self.net_loss = nn.MSELoss()
#         optimizer = torch.optim.Adam(net.parameters(), lr=self.learning_rate)
#         criterion = nn.MSELoss()
#          cpu = torch.device('cpu')
    
    # networkの重みの初期化
    def init_weights(self,m):
        if type(m) == nn.Linear:
            torch.nn.init.kaiming_uniform_(m.weight)
            torch.nn.init.uniform_(m.bias, a=-1.0, b=0.0)

    # Parameter の load
    def load_config(self, config_path):
        f = open(config_path, 'r+')
        data = yaml.load(f)
        
        self.channel_in = data.get('channel_in')
        self.channel = data.get('channel')
        self.channel_res = data.get('channel_res') 
        self.n_res_block = data.get('n_res_block')
        self.n_cnn_block = data.get('n_cnn_block')
        self.n_dense_block = data.get('n_dense_block')
        self.n_dense_hidden = data.get('n_dense_hidden')
        self.num_classes = data.get('num_classes')
        self.is_bn_dense = data.get('is_bn_dense')
 
        self.save_path = data.get('save_path')
        self.save_model_path = data.get('save_path') + data.get('save_model_name')
        self.is_init_weights = data.get('is_init_weights')
        self.is_early_stopping = data.get('is_early_stopping')
        self.is_normalize = data.get('is_normalize')
        self.batch_size = data.get('batch_size')
        self.epochs = data.get('epochs')
        self.iteration_per_epoch = data.get('iteration_per_epoch')
        self.learning_rate = data.get('learning_rate')
        self.device_id = data.get('device_id')
        
        self.hwin = data.get('hwin')
        self.vwin = data.get('vwin')
        
        
    def update_device(self):
        if torch.cuda.is_available():
            self.device = torch.device('cuda:'+str(self.device_id))
            print('GPU available\n')
            print(f'GPU id : {self.device_id}')
        else:
            self.device = torch.device('cpu')
            print('GPU NOT available\n')

    def progressBar(self,current_iter, total_iter, bar_length=30):
        percent = 100.0 * (current_iter+1) / total_iter
        sys.stdout.write("\rCompleted: [{:{}}] {:>3}%  {}/{} "
                     .format('='*int(percent/(100.0/bar_length)),
                             bar_length, int(percent),
                             current_iter+1,total_iter))
        sys.stdout.flush()      
   

    def origGen(self, num, vmin, hmin, vmax, hmax):
        """
        
        原点配列の算出

        Parameters 
        ----------
        num  : いくつのpointを使うか
        vmin : 取得したいメルスペクトログラムの原点の最小のメル数
        hmin : 取得したいメルスペクトログラムの原点の最小の幅
        vmax : 取得したいメルスペクトログラムの原点の最大のメル数
        hmax : 取得したいメルスペクトログラムの原点の最大の幅

        Return
        ------
        orig_list(np.array) : 原点の配列
        """
        orig_list = []
        for i in range(num):
            vorig = randint(vmin,vmax)
            horig = randint(hmin,hmax)
            orig_list.append(np.array([vorig,horig]).reshape(1,-1))
        return np.concatenate(orig_list,axis=0)

    def extractImg(self, D,orig_array, vwin, hwin):
        """
        Parameters
        ----------
        D : メルスペクトログラムのarray
        orig_array : 取得するarraynの原点
        vwin :
        hwin : 

        Return
        ------
        array_list(np.array) : メルスペクトログラムのimageの配列のリスト
        """
        array_list = []
        for array in orig_array:
            img = D[array[0]:array[0]+vwin,array[1]-int(hwin/2):array[1]+int(hwin/2)]
            array_list.append(img.reshape(1,*img.shape))
        return np.concatenate(array_list,axis=0)
     
    def extractFlow(self, D, orig_array):
        """
        Parameters
        ----------
        D(np.array) : 流速データの配列
        orig_array)(np.array) : origGenで算出した原点配列 
        
        Returns
        -------
        flow list(np.array) : 原点配列に沿った流速のデータの配列
        
        """
        flow_list = []
        for array in orig_array:
            idx = array[1]
            flow = D[idx]
            flow_list.append(flow)
        return np.array(flow_list)

    def train_valid_split(self, x_arr, y_arr, n_batch, vwin, hwin, random_state=11):
        """
        train data と valid data に分割
        batch 数に応じた train data と valid data を生成
        extractImg と extractFlow を使用
        
        Parameters
        ----------
        X(np.array) : 学習データ 
        y(np.array) : 教師データ
        n_batch : バッチ数
        vwin : image 抽出するときの縦幅
        hwin : image 抽出するときの横幅
        random_state : train valid に分割するときの random state
        
        Returns
        X_train(np.array)
        X_valid(np.array) :
        y_train(np.array) : 
        y_valid(np.array) :
        -------
        
        
        """
        # 全点の原点配列を生成
        orig = np.array([[0,i] for i in np.arange(int(hwin/2), 
                                                  x_arr.shape[1]-int(hwin/2)-1,
                                                  1)])

        # 全点の原点配列をtrainとvalidに分ける
        # random_stateは固定
        orig_train, orig_valid = train_test_split(orig, random_state=random_state)

        # origtrain と origvalid から n_batch 分の配列を生成する
        orig_train_batch = np.array(random.sample(list(orig_train), n_batch))
        orig_valid_batch = np.array(random.sample(list(orig_valid), n_batch))


        # n_batch 分の原点配列から image を切り取る
        X_train = self.extractImg(x_arr, orig_train_batch, vwin=vwin, hwin=hwin)
        X_valid = self.extractImg(x_arr, orig_valid_batch, vwin=vwin, hwin=hwin)

        # n_batch 分の原点配列に沿った 流速のデータを抽出する
        y_train = self.extractFlow(y_arr, orig_train_batch)
        y_valid = self.extractFlow(y_arr, orig_valid_batch)

        return X_train, y_train, X_valid, y_valid

    def _normalize(self, x):
        """
        Parameters
        ----------
        x(bach_size, 1, vwin, hwin) :
        """
        x_mean = np.mean(x, axis=(2,3))
        x_std  = np.std(x, axis=(2,3))
        x_norm = (x.squeeze() - x_mean[:,np.newaxis])/x_std[:,np.newaxis]
        
        return x_norm.reshape(1, *x_norm.shape).transpose(1,0,2,3)
        
    
    def fit(self, X_train, y_train):
        """
        Parameters
        ----------
        X_train : np.arrayのリスト
        y_train : np.arrayのリスト
        """
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
#         criterion = nn.MSELoss()
        criterion = self.net_loss
        early_stop_count = 0 
        iter_per_epoch = int(np.ceil(self.iteration_per_epoch/self.batch_size))
        # 重みの初期化
        if self.is_init_weights:
            self.net.apply(self.init_weights)
            print('init weights')
        
        for epoch in range(self.epochs):
            # Init every epoch
            train_loss = 0 
            train_acc = 0
            val_loss = 0
            val_acc = 0 
            
            ############
            # Training
            ###########
            # Switch to training mode
            self.net.train()
            valid_data = []
            for i in range(iter_per_epoch):
#                 np.random.seed(seed=11)
                
                # ランダムに idx を指定
                idx = randint(1, len(X_train)) -1 
                # train data の抽出
                train_X, train_y, valid_X, valid_y = self.train_valid_split(X_train[idx], y_train[idx], n_batch=self.batch_size, vwin=self.vwin, hwin=self.hwin, random_state=11)
                  
                
                train_X = train_X.reshape(1, *train_X.shape).transpose(1,0,2,3)
                valid_X = valid_X.reshape(1, *valid_X.shape).transpose(1,0,2,3)
                
                # 標準化
                if self.is_normalize:
                    train_X = self._normalize(train_X)
                    valid_X = self._normalize(valid_X)
                
                
                X = torch.Tensor(train_X)
                y = torch.Tensor(train_y)
                
                X = X.to(self.device)
                y = y.to(self.device)

                # Reset gradient
                optimizer.zero_grad()
                
                # Forward propagation
                y_hat = torch.squeeze(self.net(X))
                
                # Calculation loss
                loss = criterion(y_hat, y)
                train_loss += loss.item()
                
                # Back propagation
                loss.backward()
                
                # Update weights
                optimizer.step()
                
                valid_data.append((valid_X, valid_y))
                
                self.progressBar(i, total_iter = iter_per_epoch, bar_length=30)
                
            avg_train_loss = train_loss / iter_per_epoch
#             self.avg_train_acc = train_acc/ iter_per_epoch
            ##############
            # Validation
            ##############
            # switch to validation mode
            self.net.eval()
            with torch.no_grad():
                for i, data in enumerate(valid_data):
                    valid_x = data[0]
                    valid_y = data[1]
                   
                    X = torch.Tensor(valid_x)
                    y = torch.Tensor(valid_y)

                    X = X.to(self.device)
                    y = y.to(self.device)

                    # Forward propagation
                    y_hat = torch.squeeze(self.net(X))

                    # Calculation loss
                    loss = criterion(y_hat, y)
                    val_loss += loss.item()
                    
                    
#                 for i in range(int(iter_per_epoch*0.2)):
#                     # ランダムに idx を指定
#                     idx = randint(1, len(X_train)) -1 
#                     # valid data の抽出
#                     _, _, X_valid, y_valid = self.train_valid_split(X_train[idx], y_train[idx], n_batch=self.batch_size, vwin=self.vwin, hwin=self.hwin)
#                     X_valid = X_valid.reshape(1, *X_valid.shape).transpose(1,0,2,3)
 
#                     X = torch.Tensor(X_valid)
#                     y = torch.Tensor(y_valid)

#                     X = X.to(self.device)
#                     y = y.to(self.device)

#                     # Forward propagation
#                     y_hat = torch.squeeze(self.net(X))

#                     # Calculation loss
#                     loss = criterion(y_hat, y)
#                     val_loss += loss.item()

            
                avg_valid_loss = val_loss / len(valid_data)
                
            
            print(f'Epoch [{epoch+1}/{self.epochs}], train_loss: {avg_train_loss:.4f}, valid_loss: {avg_valid_loss:.4f}')
            
            self.train_loss_list.append(avg_train_loss)
            self.valid_loss_list.append(avg_valid_loss)
            
            early_stop_count += 1             
            # save model
            if avg_valid_loss == np.min(np.array(self.valid_loss_list)):
                self.save_model()
                early_stop_count = 0
                
                
            # early stopping
            if self.is_early_stopping:
                print(f'early_stop_count: {early_stop_count}/{int(self.epochs*0.1)}')
                print(f'epoch: {epoch+1}')

                if early_stop_count == int(self.epochs*0.1):
                    print('Early Stopping')
                    break
            
            
            
    def predict(self, X_test, y_test):
        self.y_pred_list = []
        if os.path.isfile(self.save_model_path):
            print('Loading Weights')
            self.net.load_state_dict(torch.load(self.save_model_path))
        
        self.test_orig_list = np.array([[[0,i]] for i in np.arange(int(self.hwin/2), 
                                                            X_test.shape[1]-int(self.hwin/2)-1,
                                                            1)])
        X_test_list = [self.extractImg(X_test, orig, vwin=self.vwin, hwin=self.hwin) for orig in self.test_orig_list]
        y_test_list = [self.extractFlow(y_test, orig) for orig in self.test_orig_list]
        for X in X_test_list:
            # Switch to predict mode
            self.net.eval()
            with torch.no_grad():
                X = X.reshape(1, *X.shape)
                
                 # 標準化
                if self.is_normalize:
                    X = self._normalize(X)
               
                X = torch.Tensor(X)
#                 y = torch.Tensor(y_test)

                X = X.to(self.device)

                y_pred = torch.squeeze(self.net(X))
                y_pred = y_pred.to('cpu').numpy()
                self.y_pred_list.append(y_pred)
            

            
        
        return np.array(self.y_pred_list), np.array(y_test_list)
#         return test_orig_list, X_test_list

    def save_model(self):
        print(f'Saving Model to {self.save_model_path}')
        torch.save(self.net.state_dict(), self.save_model_path)
        
    
    def plot_learnig_curve(self):
        epoch = np.linspace(1, len(self.train_loss_list), num=len(self.train_loss_list))
        fig, axs = plt.subplots(1,1, figsize=(10,4))
        axs.plot(epoch, self.train_loss_list, label='train')
        axs.plot(epoch, self.valid_loss_list, label='valid')
        axs.set_xlabel('epoch')
        axs.set_ylabel('loss')
        axs.legend()
        
    def calc_score(self, y_pred, y_test):
        score = []
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        mape = round(mape, 2)
        print(f'MAPE score : {mape}')
        score.append(mape)
    
        maape = np.mean(np.arctan(np.abs((y_test - y_pred) / y_test))) * 100
        maape = round(maape, 2)        
        print(f"MAAPE score : {maape}")
        score.append(maape)
    
        return np.array(score)
 
    
    def visualize_score(self, np_scores, save_path, np_var=[]):

        #  visualize
        x_flow = [1.0, 2.0, 3.0, 3.9]
        x = np.arange(1, 5)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].set_title('MAPE')
        axes[0].set_xlabel('flow(m/s)')
        axes[0].set_ylim(0,50)
        axes[0].bar(x, np_scores[:,0], width=0.3, tick_label = x_flow)
        axes[0].yaxis.set_major_formatter(mpl.ticker.PercentFormatter(100))
        for x_, y_ in zip(x, np_scores[:,0]):
            axes[0].text(x_, y_, y_, ha='center', va='bottom')
        axes[1].set_title('MAAPE')
        axes[1].set_xlabel('flow(m/s)')
        axes[1].set_ylim(0,50)
        axes[1].bar(x, np_scores[:,1], width=0.3, tick_label = x_flow)
        axes[1].yaxis.set_major_formatter(mpl.ticker.PercentFormatter(100))
        for x_, y_ in zip(x, np_scores[:,1]):
            axes[1].text(x_, y_, y_, ha='center', va='bottom')
        # fig.show()

        figname = 'score.png'
        fig.savefig(save_path + figname)

        #  calc score
        mean_ = []
        mape_mean_ = np.mean(np_scores[:,0])
        print(f'MAPE mean : {round(mape_mean_, 2)}')
        mean_.append(mape_mean_)
        maape_mean_ = np.mean(np_scores[:,1])
        print(f'MAAPE mean : {round(maape_mean_, 2)}')
        mean_.append(maape_mean_)

        score_index = [1.0, 2.0, 3.0, 3.9]
        score_column = ['MAPE', 'MAAPE']
        # np_scores = np.append(np_scores, mean_)
        df_score = pd.DataFrame(np_scores, index=score_index, columns=score_column)
        df_score.loc['mean'] = mean_
        if(len(np_var) == len(np_scores)):
            np_var = np.append(np_var, np.mean(np_var))
            df_score['model_var'] = np_var

        csvname = 'score.csv'
        df_score.to_csv(save_path + csvname)

        return df_score
