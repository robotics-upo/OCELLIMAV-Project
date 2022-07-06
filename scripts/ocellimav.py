# -*- coding: utf-8 -*-
#!/usr/bin/python3

################################################################################################################################################################
# OCELLIMAV 2021                                                                                                                                               #
#                                                                                                                                                              #
# Script to train CNNBiGRU on Airsim synthetic dataset.                                                                                                        #
# Versions:                                                                                                                                                    #
#       Numpy : 1.13.1                                                                                                                                         #
#       Matplotlib: 0.3.0                                                                                                                                      #
#       Python: 3.6.5                                                                                                                                          #
#       Tensorflow: 1.13.1                                                                                                                                        #
#                                                                                                                                                              #
# How to execute it (in a terminal):                                                                                                                           #
#          python3 train_synthetic_2021.py  --set=training --sequences 00 02 --model_dir='path/to/model/experiment/'                                           #
#                                                                                                                                                              #
#--------------------------------------------------------------------------------------------------------------------------------------------------------------#
__author__ = "Macarena Merida-Floriano"                                                                                                                        #
__email__ = "mmerflo@upo.es"                                                                                                                                   #
################################################################################################################################################################


# =============== Imports =============== #
from keras.models import Model, load_model
from keras.layers import Input, Conv2D, Dropout, MaxPooling2D, Flatten, concatenate, Dense, GRU, TimeDistributed, Bidirectional
from keras import optimizers
from keras.callbacks import ModelCheckpoint
import numpy as np
import os
import argparse
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import time

# =============== Define class =============== #

class ocellimav():
    def __init__(self, args):
        assert os.path.exists(args.general_data_dir + args.data_type), "ERROR: data path does not exists!"
        self.data_path = args.general_data_dir + args.data_type + '/'
        if args.modelcheckpoint == True: 
            if os.path.exists(args.modelcheckpoint_path) != True:
                os.mkdir('../models/modelcheckpoint/')
        if args.mode == 'test':
            assert os.path.exists(args.saved_model_path), "ERROR: saved model path does not exists!"
        self.seqs = args.sequences #[os.path.splitext(f)[0] for f in list(args.sequences)] 

    def test_mode(self, args, net_inputs, net_labels):
        '''
        Function to test saved model. Metrics and plots are generated.
        '''
        print('\n')
        print('Loading model ....')
        print('\n')
        print('\n')
        model_path = args.saved_model_path + args.saved_model_test
        model = load_model(model_path)
        print(model.summary())
        print('Model loaded')

        mse_test = model.evaluate(net_inputs, net_labels)
        print("Testing %s data MSE = %.4f" % (args.data_type, mse_test))

        output = model.predict(net_inputs)
        error = (output - net_labels)**2
        mse = np.mean(error, axis=0)
        sem = np.std(error, axis=0)/np.sqrt(len(error))
        loss = np.mean(mse)
        sem_loss = np.mean(sem)
        time.sleep(2)
        print("Error in x-axis = %.4f +/- %.4f " % (mse[0], sem[0]))
        print("Error in y-axis = %.4f +/- %.4f " % (mse[1], sem[1]))
        print("Error in z-axis = %.4f +/- %.4f " % (mse[2], sem[2]))
        print("Total loss = %.4f +/- %.4f\n\n " % (loss, sem_loss))

        if args.save_metrics == True:
            orig_stdout = sys.stdout
            f = open(args.save_metrics_path + "metrics_%s.txt" % args.scenario, 'w')
            sys.stdout = f
            print(' Model %s evaluated on %s datasets' % (args.saved_model_test, args.data_type))
            print(' Evaluated on Sequences %s' % self.seqs)
            print(' ----------------------------------------------------------')
            print('|                      General info                        |')
            print(' ----------------------------------------------------------')
            print('| General Mean Squared Error = %.4f +/- %.4f               ' % (loss, sem_loss))
            print(' ----------------------------------------------------------')
            print('|                      X coordinate                        |')
            print(' ----------------------------------------------------------')
            print('| Predicted x MSE = %.4f +/- %.4f                          ' % (mse[0], sem[0]))
            print(' ----------------------------------------------------------')
            print('|                      Y coordinate                        |')
            print(' ----------------------------------------------------------')
            print('| Predicted y MSE = %.4f +/- %.4f                          ' % (mse[1], sem[1]))
            print(' ----------------------------------------------------------')
            print('|                      Z coordinate                        |')
            print(' ----------------------------------------------------------')
            print('| Predicted z MSE = %.4f +/- %.4f                          ' % (mse[2], sem[2]))
            print(' ----------------------------------------------------------')            
            sys.stdout = orig_stdout
            f.close()
            print("Metrics saved")
        time.sleep(3)
        if args.save_plots == True:
            # Plot configuration
            label_size=18
            mpl.rcParams['xtick.labelsize'] = label_size
            mpl.rcParams['ytick.labelsize'] = label_size
            plt.rcParams["figure.figsize"] = (15,9)
            xtime = np.array(range(0, net_labels.shape[0])).astype(np.float)
            xtime = xtime/30.0
            axis = ['x', 'y', 'z']
            if os.path.exists(args.save_metrics_path + 'plots/') != True:
                    os.mkdir(args.save_metrics_path + 'plots/')
            # Plots
            for i in range(3):
                plt.plot(xtime, net_labels[:,i], 'r')
                plt.plot(xtime, output[:,i], 'b')
                plt.legend(['Ground-truth', 'Predictions'], fontsize=15)
                plt.xlabel('Time (s)', fontsize=20)
                ylab = '$\omega_' + axis[i] +'$'
                plt.ylabel(ylab + r'($\frac{rad}{s}$)', fontsize=20)
                #plt.ylabel(r'$\omega_{0}$ ($\frac{rad}{s}$)'.format(axis[i]), fontsize=20)
                plt.savefig(args.save_metrics_path + 'plots/' + "w_%s_%s_seqs_%s" % (axis[i], args.data_type, self.seqs), dpi=100)
                plt.show()
            time.sleep(1)
            for i in range(3):
                plt.subplot(3, 1, i+1)
                plt.plot(xtime, net_labels[:,i], 'r')
                plt.plot(xtime, output[:,i], 'b')
                plt.legend(['Ground-truth', 'Predictions'], fontsize=8,loc=3)
                plt.xlabel('Time (s)', fontsize=20)
                ylab = '$\omega_' + axis[i] +'$'
                plt.ylabel(ylab + r'($\frac{rad}{s}$)', fontsize=20)
            plt.savefig(args.save_metrics_path + 'plots/' + "W_%s_seqs_%s" % (args.data_type, self.seqs))
            plt.show()
            print("Plots saved")
            time.sleep(3)
        return print("Model tested.")

    def define_network(self, input_shape, n_cnnfilters, n_kernels, n_padding, n_strides, n_maxpool, n_dense, n_gru):
        '''
        Function to create network, just in case args.mode == 'train'
        '''
        input_network = Input(shape=input_shape)
        x = TimeDistributed(Conv2D(n_cnnfilters[0], n_kernels[0], padding=str(n_padding[0]), strides=n_strides[0], activation='relu', data_format='channels_first', name='{0}cnnbigru'.format(n_cnnfilters[0])))(input_network)
        x = TimeDistributed(Dropout(0.2))(x)
        if len(n_cnnfilters) > 1:
            for L in range(1, len(n_cnnfilters)):
                if n_maxpool[L] == True:
                    x = TimeDistributed(MaxPooling2D((2,2), name='%d_mp' % n_cnnfilters[L]))(x)
                    x = TimeDistributed(Dropout(0.2))(x)
                x = TimeDistributed(Conv2D(n_cnnfilters[L], n_kernels[L], padding=n_padding[L], strides=n_strides[L], activation='relu', data_format='channels_first', name='%dcnnbigru' % n_cnnfilters[L]))(x)
                x = TimeDistributed(Dropout(0.2))(x)
        x = TimeDistributed(Flatten())(x)
        for d in range(len(n_dense)):
            x = TimeDistributed(Dense(n_dense[d], activation='relu', kernel_initializer='uniform', name='%d_dense' % n_dense[d]))(x)
            x = TimeDistributed(Dropout(0.2))(x)
        x = Bidirectional(GRU(n_gru, use_bias=True, kernel_initializer='uniform', name='bigru'), merge_mode='ave')(x)
        x = Dropout(0.2)(x)
        out = Dense(3, kernel_initializer='uniform', name='outputs')(x)
        model = Model(input_network, out)
        print('\n')
        time.sleep(1)
        print('Network created with architecture:\n')
        print(model.summary())
        time.sleep(4)
        return model

    def train_mode(self, args, net_inputs, net_labels):
        '''
        Function to train the designed network
        '''
        print('\n')
        print('##########################################')
        print('#            Training mode               #')
        print('##########################################')
        print('\n')
        print('Preparing trainnig mode ........')
        print('\n')
        input_data = net_inputs
        labels_data = net_labels
        model = self.define_network(input_data.shape[1:], [40, 20], [(3,3),(2,2)], ['same', 'valid'], [(1,2), 2], [0, 1], [100, 50, 20], 40)
        print('\n')
        loss_fun = 'mse'
        time.sleep(2)
        print('Compiling model with lr=%d' % args.lr)
        print('Loss function: %s' % loss_fun)
        print('\n')
        print('\n')
        adam = optimizers.Adam(lr = args.lr, clipnorm=1., clipvalue=0.5)
        model.compile(optimizer=adam, loss=loss_fun)
        if args.modelcheckpoint == True:
            checkpoint = ModelCheckpoint(args.modelcheckpoint_path + 'model.{epoch:02d}-{val_loss:.4f}.hdf5', monitor='val_loss', save_best_only = True, save_weights_only=False)
            history = model.fit(input_data, labels_data, validation_split=0.2,epochs=args.epochs, batch_size=args.batch_size, verbose=1, callbacks=[checkpoint])
        if args.modelcheckpoint == False:
            history = model.fit(input_data, labels_data, validation_split=0.2,epochs=args.epochs, batch_size=args.batch_size, verbose=1)
        #Plot history
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss. Model #%s' % (args.model_num))
        plt.ylabel('Loss' + r'($\frac{rad^2}{s^2}$)')
        plt.xlabel('Epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(args.modelcheckpoint_path + 'plots/' + "training_plot_#%d" % (args.model_num)) #check this name for savingfig in this case
        plt.show()
        print("Plots saved")
        time.sleep(3)
        return print("Network trained. Best models saved at %s" % args.modelcheckpoint_path)

    def load_data(self, args):
        '''
        Function to load data specified in sequences input argument
        '''
        #print(self.seqs)
        #print(type(self.seqs[0]))
        print('##########################################')
        print('#            Data loading                #')
        print('##########################################')
        print('\n')
        print('Sequences %s are going to be load:' % self.seqs)
        time.sleep(1)
        single_inputs = []
        sequenced_inputs = []
        labels = []
        for s in self.seqs:
            print('Loading data %d ...' % s)
            data = np.load(self.data_path + 'data%d.npz' % s)
            single_inputs.append(data["simple_inputs"])
            sequenced_inputs.append(data["inputs_seq"])
            labels.append(data["labels"])
        single_inputs = np.concatenate(single_inputs, axis=0)
        sequenced_inputs = np.concatenate(sequenced_inputs, axis=0)
        labels = np.concatenate(labels, axis=0)
        if args.data_type == 'synthetic':
            wx = -labels[:,1]
            wy = labels[:,2]
            wz = -labels[:,0]
            labels = np.empty((wx.shape[0], 3))
            labels[:,0] = wx
            labels[:,1] = wy
            labels[:,2] = wz
        print('\n')
        print('All sequences loaded.')
        time.sleep(1)
        print('Samples loaded:\n ................. %d' % single_inputs.shape[0])
        print('Sequenced inputs shape:\n ................. {0}'.format(sequenced_inputs.shape))
        print('Single inputs shape:\n ................. {0}'.format(single_inputs.shape))
        print('Labels shape:\n ................. {0}'.format(labels.shape))
        time.sleep(3)
        return single_inputs, sequenced_inputs, labels

    def main_function(self, args):
        '''
        Main function
        '''
        print('\n')
        print('\n')
        print("----------------------- OCELLIMAV 2021 -----------------------")
        print("Author = Macarena Merida-Floriano")
        print("Email: mmerflo@upo.es")
        print("--------------------------------------------------------------")
        print('\n')
        print('\n')
        time.sleep(2)
        single_inputs, sequenced_inputs, labels = self.load_data(args)

        if args.mode == 'train':
            self.train_mode(args,sequenced_inputs, labels)
        if args.mode == 'test':
            self.test_mode(args, sequenced_inputs, labels)
        return("Job done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OCELLIMAV main script to train and test a network')
    parser.add_argument('--general_data_dir', type=str, default='../data/processed/', help = 'Directory path to general data')
    parser.add_argument('--data_type', type=str, help='Wether synthetic or real data to train or test')
    parser.add_argument('--mode', type=str, default='train', help='Wether to train or test the network')
    parser.add_argument('--modelcheckpoint', type=bool, default=False, help='Wether to use or not modelcheckpoint function')
    parser.add_argument('--modelcheckpoint_path', type=str, default='../models/modelcheckpoint/', help='Modelcheckpoint directory path')
    parser.add_argument('--saved_model_path', type=str, default='../models/', help='Path to saved model main directory')
    parser.add_argument('--saved_model_test', type=str, help='Name of the model to be tested')
    parser.add_argument('--scenario', type=str, default='outdoor', help='Wether the testing set corresponds to an outdoor, indoor or porch scenario')
    parser.add_argument('--sequences', type=int, nargs='+', help='Data sequences to use. Format example: --sequences 0 1 2')
    parser.add_argument('--save_plots', type=bool, default=False, help='Wether to save or not generated plots')
    parser.add_argument('--save_metrics', type=bool, default=False, help='Wether to save or not metrics')
    parser.add_argument('--save_metrics_path', default='..models/', type=str, help='Path directory to save metrics and plots from testing model')
    parser.add_argument('--epochs', type=int, default=400, help='Number of epochs to train the network')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size to train the network')
    parser.add_argument('--model_num', type=int, help='Model attempt number to keep track')
    parser.add_argument('--lr', type=int, help='Learning rate for learning. When fine-tuning the network with real data use lr = 0.000001, otherwise lr=0.001')
    args = parser.parse_args()
    ocelli = ocellimav(args)
    ocelli.main_function(args)
