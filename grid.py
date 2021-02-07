import cv2
from lib import method as M
import argparse
from numba import cuda,jit
import numpy as np

parser=argparse.ArgumentParser(description='T')
parser.add_argument("--input_dir",type=str)
parser.add_argument('--draw_dir',type=str)
parser.add_argument("--output_dir",type=str,default='')
parser.add_argument('--label',type=str,default='G')
parser.add_argument('--suffix',type=str,default='png')
parser.add_argument('--step',type=int,default=5)
parser.add_argument('--thre',type=int,default=50)
parser.add_argument('--er',type=int,default=7)
parser.add_argument('--dr',type=int,default=3)
parser.add_argument('--w',type=int,default=30)
parser.add_argument('--mode',type=str,default='dual')
parser.add_argument('--rename',type=str,default='')
parser.add_argument('--minz',type=int,default=64)
parser.add_argument('--graphname',type=str)
args=parser.parse_args()


str1='data_\n\nloop_\n_rlnCoordinateX #1\n_rlnCoordinateY #2\n'
str2='_rlnCoordinateZ #3\n_rlnMicrographName #4\n'
@jit
def gen_grid(shape,step):
  g=np.zeros(shape=shape)
  x,y=g.shape
  gx=int(x/(step-1))
  gy=int(y/step)
  for i in range(gx):
    for j in range(gy):
      g[i*(step-1)+1,j*step+1]=1
  return g

@jit
def grid(arr,step):
  temp_grid=gen_grid(arr.shape,step)
  x,y=arr.shape
  for i in range(x):
    for j in range(y):
      arr[i,j]*=temp_grid[i,j]
  return arr

@jit
def bw(arr):
  x,y=arr.shape
  for i in range(x):
    for j in range(y):
      if arr[i,j]>thre:
        arr[i,j]=255
      else:
        arr[i,j]=0
  return arr


#@jit
def write_star(filename,arr):
  if args.mode=='tomo':
    starfile=open(args.output_dir+'/'+rename,'a')
    #starfile.write(str1)
    #starfile.write(str2)
  else:
    starfile=open(filename,'w')
    print('writing',filename)
    starfile.write(str1)
  #if args.mode=='tomo':
  # starfile.write(str2)
  x,y=arr.shape
  if args.mode=='star' or args.mode=='dual':
    for i in range(x):
      for j in range(y):
        if arr[i,j]>args.w:
       #5760,4092
          true_x=float(j+1)/512*5760
          true_y=float(i+1)/512*4092
          starfile.write(str(true_x)+' '+str(true_y)+'\n')


  if args.mode=='tomo':
      #  print('---checkpoint 1 ----')
    true_name=M.get_name(filename)
    slice_index=int(true_name.split('_')[2])

#    print('scanning slice :',slice_index, '           ---remain :')
    print('scanning slice :',slice_index)
    if slice_index>args.minz:
      for i in range(x):
        for j in range(y):
          if arr[i,j]>args.w:
            true_x=float(j)/512*768
            true_y=float(i)/512*742
            true_z=slice_index
            starfile.write(str(true_x)+' '+str(true_y)+' '+str(true_z)+' '+g_name+'\n')
  #print('--checkpoint 2 ---')
  starfile.flush()
  starfile.close()
        
rename=args.rename
g_name=args.graphname
thre=args.thre
ori_list,L=M.file_read(args.input_dir,args.suffix)
ori_list,L=M.file_read(args.input_dir,args.suffix)
file_count=0

M.create_dir(args.output_dir)
e_kernel=np.ones((args.er,args.er))
d_kernel=np.ones((args.dr,args.dr))

file_count=0
for eachfile in ori_list:
  
  label=M.single_2d(cv2.imread(eachfile))
  #label=cv2.resize(label,(5760,4092))
  label=bw(label)
  label=cv2.erode(label,e_kernel)
  label=bw(label)

  grid_image=grid(label,args.step)
  grid_image=bw(grid_image)
  true_name=M.get_name(eachfile)
  if args.mode=='dual':
    #cv2.imwrite(args.output_dir+'/'+true_name+'_grid.png',grid_image)
    write_star(args.output_dir+'/'+true_name+'.star',grid_image)
  if args.mode=='star':
    write_star(args.output_dir+'/'+true_name+'.star',grid_image)
  if args.mode=='tomo':
    write_star(args.output_dir+'/'+true_name+'.star',grid_image)

