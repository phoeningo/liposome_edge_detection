import cv2
from lib import method as M
import argparse
#from numba import jit,cuda

parser=argparse.ArgumentParser(description='T')
parser.add_argument("--input_dir",type=str)
parser.add_argument('--draw_dir',type=str)
parser.add_argument("--output_dir",type=str,default='')
parser.add_argument('--label',type=str,default='')
parser.add_argument('--suffix',type=str,default='png')
parser.add_argument('--step',type=int,default=5)
args=parser.parse_args()




def get_one_label(filename):
  image_ori=cv2.imread(filename)
  #OR use M.get_name()
  true_name=M.get_name(eachfile)
  print(true_name)
  print(args.draw_dir)
  draw_path=args.draw_dir+'/'+args.label+true_name+'.'+args.suffix
  print(draw_path)
  image_draw=cv2.imread(draw_path)
  single_ori=M.single_2d(image_ori)
  single_draw=M.single_2d(image_draw)
  temp_img=single_draw-single_ori
  temp_res=single_ori-temp_img
  temp_out=single_draw-temp_res
  return temp_out

'''
@jit
def gen_grid(shape,step):
  g=np.zeros(shape=shape)
  x,y=g.shape
  gx=int(x/step)
  gy=int(y/step)
  for i in range(gx):
    for j in range(gy):
      g[i*step,j*step]=1
  return g

@jit
def grid(arr,step):
  temp_grid=gen_grid(arr.shape,step)
  x,y=arr.shape
  for i in range(x):
    for j in range(y):
      arr[i,j]*=temp_grid[i,j]
  return arr
  
'''


ori_list,L=M.file_read(args.input_dir,args.suffix)
M.create_newdir(args.output_dir)
for eachfile in ori_list:
  label=get_one_label(eachfile)
  #grid_image=grid(label,args.step)
  true_name=M.get_name(eachfile)
  #cv2.imwrite(true_name+'_grid.png',grid_image)
  cv2.imwrite(args.output_dir+'/'+true_name+'.png',label)


