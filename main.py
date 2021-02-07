from model import *
from lib import method as M
import os
from keras.callbacks import ModelCheckpoint
import mrcfile
import numpy as np
import argparse
from glob import glob
from keras.models import load_model
import time
import math
import cv2
import tensorflow as tf
from keras import backend as K

parser=argparse.ArgumentParser(description='T')
parser.add_argument('--suffix',type=str,default='png')
parser.add_argument('--input_dir',type=str)
parser.add_argument("--input_X",type=str)
parser.add_argument("--input_Y",type=str)
parser.add_argument('--mode',type=str,default='train')
parser.add_argument('--gpu',type=str,default='7')
parser.add_argument('--output',type=str,default='outputtest.mrc')
parser.add_argument('--output_dir',type=str)
parser.add_argument('--rate',type=float,default=1)
parser.add_argument('--mag',type=float,default=10)
parser.add_argument('--size',type=int,default=100)
parser.add_argument('--model_name',type=str,default='modeltest.h5')
parser.add_argument('--x',type=int,default=0)
parser.add_argument('--y',type=int,default=0)
parser.add_argument('--z',type=int,default=0)
parser.add_argument('--b',type=int,default=16)
parser.add_argument('--epochs',type=int,default=1)
parser.add_argument('--batch_size',type=int,default=1)
parser.add_argument('--s',type=int,default=60)
parser.add_argument('--loss',type=str,default='mse')
parser.add_argument('--pre_load',type=str,default='')
parser.add_argument('--optimizer',type=str,default='adam')

parser.add_argument('--load_weights',type=str,default='')
parser.add_argument('--weights',type=str,default='weights.h5')
args=parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
#=============DEFINE==================

batch_size=args.batch_size
epochs=args.epochs


#===========prepare datasets==================#
def read_pngs(dir,suffix):
  files=glob(dir+'/*.'+suffix) 
  L=len(files)
  if L==0:
    print('no file exists')
    return -1,-1,-1,-1,-1
  else:
    print(str(len(files))+' files in ',dir)
    data0=M.single_2d(cv2.imread(files[0]))
    x,y=data0.shape
    raw_input=np.zeros(dtype=data0.dtype,shape=[L,x,y,1])
    for filei in range(L):
      img=M.single_2d(cv2.imread(files[filei]))
      raw_input[filei,:]=img[:].reshape([x,y,1])
    return raw_input,L,files
    
#---------------
input_X=args.input_dir+'/X/'
input_Y=args.input_dir+'/Y/'

X_train,L,files=read_pngs(input_X,args.suffix)


#=======================================#
shape=(512,512,1)
Single_shape=1,512,512,1
output_shape=512,512
#==

#=====================================

if args.mode=='train':
  Y_train=read_pngs(input_Y,args.suffix)[0]
  if args.pre_load!='':
    model=load_model(args.pre_load)
  else:
    model=unet(shape)

  if args.load_weights!='':
    model.load_weights(args.load_weights)
  model.fit(X_train,Y_train,batch_size=batch_size,epochs=epochs,shuffle=1)
  if args.weights=='':
    model.save(args.model_name)
  else:
    model.save_weights(args.weights)

if args.mode=='test':
  M.create_newdir(args.output_dir)
  if args.load_weights=='':  
    model=load_model(args.model_name)
  else:
    model=unet(shape)
    model.load_weights(args.load_weights)
  
  for i in range(L):
    Y_out=model.predict(X_train[i].reshape(Single_shape))
    Y_out=Y_out.reshape(output_shape)
   
    outname=args.output_dir+'/'+M.get_name(files[i])+'_out.'+args.suffix
    print('writing :',outname)
    cv2.imwrite(outname,Y_out)   
