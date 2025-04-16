import os
import numpy as np
import matplotlib
matplotlib.use('Agg') # No pictures displayed
import matplotlib.pyplot as plt
plt.ioff()
import tensorflow as tf
import gc
import os

os.environ["KERAS_BACKEND"] = "tensorflow"
import keras
from keras import layers
from keras.optimizers import Adam, SGD, Adagrad, Adamax
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.models import Model, load_model, save_model
from keras.callbacks import EarlyStopping, Callback, LearningRateScheduler
from keras import backend as K
from keras.optimizers.schedules import ExponentialDecay, PiecewiseConstantDecay, CosineDecay
from keras.utils import plot_model

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class LearningRateLogger(Callback):
    def __init__(self, verbose=1):
        super().__init__()
        self.verbose = verbose

    def on_train_begin(self, logs={}):
        self.lrates = []

    def on_epoch_end(self, epoch, logs={}):
        optimizer = self.model.optimizer
        lr = optimizer.learning_rate

        if isinstance(lr, keras.optimizers.schedules.LearningRateSchedule):
            lrate = float(lr(optimizer.iterations).numpy())
        elif hasattr(lr, 'numpy'):  #  Variable
            lrate = float(lr.numpy())
        else:  # float 
            lrate = float(lr)        

        self.lrates.append(lrate)
        if self.verbose == 1:
            print('\nEpoch {} Lr {}'.format(epoch + 1, lrate))

class Normalize:
    def everyMinMax(self, data, datanumber):
        data = data.astype(np.float32)
        for i in range(datanumber):
            data[i] = (data[i] - data[i].min()) / (data[i].max()-data[i].min())
        return data
    
    def MinMax(self, data, datanumber, L):
        data = data.astype(np.float32)
        data = data.reshape(-1,1)
        data = (data - data.min()) / (data.max()-data.min())
        data = data.reshape(datanumber, L)
        return data
        
    def everyStandard(self, data, datanumber):
        data = data.astype(np.float32)
        for i in range(datanumber):
            data[i] = (data[i]-data[i].mean()) / data[i].std()
        return data
    
    def Standard(self, data, datanumber, L):
        data = data.astype(np.float32)
        data = data.reshape(-1,1)
        data = (data - data.mean()) / data.std()
        data = data.reshape(datanumber, L)
        return data


class TeO6:
    def __init__(self,train_step,batch_size=1024,epochs=300,learning_rate=1e-4,cluster=3,targetFolder='/rhome/jimmy0111/jimmy/TeO6/cluster_3/'):
        self.train_step = train_step   
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate 
        self.cluster = cluster
        self.targetFolder = targetFolder 



    def opendata(self,):
        train = np.load('/rhome/jimmy0111/jimmy/TeO6/DATA/three_data.npz')
        self.train_data = train['data']
        train_label = train['label']
        self.train_label = keras.utils.to_categorical(train_label, self.cluster)

        test = np.load('/rhome/jimmy0111/jimmy/TeO6/DATA/data.npz')
        self.test_data, self.X= test['data'], test['x']

        self.L = len(self.train_data[0])
        self.datanumber = len(self.test_data)
        self.trainnumber = len(self.train_data)

    def Normal(self,): 
        self.train_data = Normalize().everyStandard(self.train_data, self.trainnumber)
        self.test_data = Normalize().everyStandard(self.test_data, self.datanumber)

       
    def supervised_1d_model(self,L): # 1D_integral_model
        input_shape = (L,)
        inputs = layers.Input(shape=input_shape)
   
        x = layers.Dense(1024, activation='relu')(inputs)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dense(16, activation='relu')(x)
        x = layers.Dense(8, activation='relu')(x)
        outputs = layers.Dense(self.cluster, activation='softmax')(x)
        model = Model(inputs=inputs, outputs=outputs)
        model.summary()
        return model

    
    def train(self, ):

        # learning_rate = ExponentialDecay(learning_rate, 1000, 0.1, staircase=True)
        # learning_rate = CosineDecay(learning_rate, 500, alpha=0.1)
        learning_rate = self.learning_rate
        model = self.supervised_1d_model(L=self.L)

        callback = EarlyStopping(monitor="val_loss", min_delta=1e-8, patience=20, verbose=1, mode="auto") #baseline=0.0012
        self.lr_logger = LearningRateLogger(verbose=1)
        
        model.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=learning_rate), metrics=["accuracy"])
        History=model.fit(self.train_data, self.train_label, batch_size=self.batch_size, epochs=self.epochs, validation_split=0.2,callbacks=[callback,self.lr_logger],verbose=2)

        Loss = History.history['loss']
        Val_loss = History.history['val_loss']
        Predict = model.predict(self.test_data)
        Y1,Y2,Y3 = Predict[:,0],Predict[:,1],Predict[:,2]

        keras.backend.clear_session()
        del model, History, Predict
        gc.collect()  
        
        return Loss, Val_loss, Y1, Y2, Y3

    def plotloss(self, Loss, Val_loss, i):
        plt.figure(figsize=(8,4.5),dpi = 200)
        plt.plot(Loss,'o-',markersize=2.5)
        plt.plot(Val_loss,'o-',markersize=2.5)
        plt.title('loss')
        plt.xlabel('Epoch')
        plt.ylabel('loss')
        plt.legend(['Train_loss', 'Test_val_loss'], loc='upper left', fontsize=6)
        plt.show()
        loss_path = self.targetFolder+'/loss/'
        if not os.path.exists(loss_path):
            os.makedirs(loss_path)
        plt.savefig(os.path.join(loss_path,  '{i}_loss.png'.format(i=i)))
        plt.close()

        plt.figure(figsize=(8,4.5),dpi = 200)
        plt.plot(np.log10(Loss),'o-',markersize=2.5)
        plt.plot(np.log10(Val_loss),'o-',markersize=2.5)
        plt.title('log(loss)')
        plt.xlabel('Epoch')
        plt.ylabel('log(loss)')
        plt.legend(['Train_loss', 'Test_val_loss'], loc='upper left', fontsize=6)
        plt.show()
        log_loss_path = self.targetFolder+'/log(loss)/'
        if not os.path.exists(log_loss_path):
            os.makedirs(log_loss_path)
        plt.savefig(os.path.join(log_loss_path,  '{i}_log(loss).png'.format(i=i)))
        plt.close()

    def plotphase(self, Y1, Y2, Y3):
        n_2 = -1 # when temperature = 2, n_2 = 5000
        plt.figure(figsize=(8,4.5),dpi = 200)
        plt.scatter(self.X[:n_2],Y1[:n_2],s=0.5,color='r',label='a')
        plt.scatter(self.X[:n_2],Y2[:n_2],s=0.5,color='g',label='b')
        plt.scatter(self.X[:n_2],Y3[:n_2],s=0.5,color='b',label='c')
        plt.title('Phase')
        plt.xticks(ticks=[0,0.25,0.5,0.75,1],label=["0","0.25","0.5","0.75","1.25"])
        plt.xlabel('X')
        plt.ylabel('Probability')
        plt.show()
        plt.savefig(self.targetFolder+'phase.png')
        plt.close()

    def start(self,):
        path = self.targetFolder
        if not os.path.exists(path):
            os.makedirs(path)
        self.opendata()
        self.Normal()

        Y1_sum, Y2_sum, Y3_sum = np.zeros(self.datanumber), np.zeros(self.datanumber), np.zeros(self.datanumber)
        All_Tc = np.zeros(self.train_step)
        for i in range(self.train_step):
            Loss, Val_loss, Y1, Y2, Y3 = self.train()
            Y1_sum += Y1
            Y2_sum += Y2
            Y3_sum += Y3
            if i % 10 == 0:
                self.plotloss(Loss, Val_loss, i)

        Y1, Y2, Y3 = Y1_sum/self.train_step, Y2_sum/self.train_step, Y3_sum/self.train_step
        self.plotphase(Y1, Y2, Y3)    


if __name__ == '__main__':
    
    GPU_number = 0
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.set_visible_devices(gpus[GPU_number], 'GPU')  # USE GPU 0 or 1 
            print("Correct use", gpus[GPU_number])

        except RuntimeError as e:
            print(e)

    TeO6(train_step=1).start()
