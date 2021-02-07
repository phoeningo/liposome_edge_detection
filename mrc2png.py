from lib import method as M

import numpy as np
import mrcfile

import cv2
import argparse
from numba import cuda,jit


parser=argparse.ArgumentParser(description='T')
parser.add_argument("--input_dir",type=str)
parser.add_argument('--draw_dir',type=str)
parser.add_argument("--output_dir",type=str,default='')
parser.add_argument('--label',type=str,default='G')
parser.add_argument('--suffix',type=str,default='mrc')
parser.add_argument('--step',type=int,default=5)
parser.add_argument('--start',type=int,default=0)
parser.add_argument('--end',type=int,default=9999)
args=parser.parse_args()
#resize-2-1024

files,L=M.file_read(args.input_dir,args.suffix)
L=min(args.end,L)
L-=args.start
#if L>0:
 # M.create_dir(args.output_dir)
inx=1
for eachfile in files:
  true_name=M.get_name(eachfile)
  index=true_name.split('_')[1]
  if args.end>int(index) and int(index)>args.start:

    img=M.raw(eachfile)
    img=cv2.GaussianBlur(img,(7,7),7)
  #print(img.shape)
    print(inx,'/',L)
    inx+=1
    cv2.imwrite(args.output_dir+'/'+true_name+'.png',cv2.resize(M.map2uint8(img),(512,512)))
    img=0
