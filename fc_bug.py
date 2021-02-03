import cv2
import sys
import os
import random
import glob
import numpy as np
from scipy.fftpack import dct

def create_bug(filename):
    src_img = cv2.imread(filename)
    height, width = src_img.shape[:2]

    #--------------------------------
    # crop image
    #--------------------------------
    DIVISION_NUM = 32
    block_size = int(width / DIVISION_NUM)

    crop_size = [height % block_size, width % block_size]
    start_pos = [int(crop_size[0] / 2), int(crop_size[1] / 2)]
    end_pos = [height - (crop_size[0] - start_pos[0]), width - (crop_size[1] - start_pos[1])]
    src_img = src_img[start_pos[0]:end_pos[0], start_pos[1]:end_pos[1], :]

    height, width = src_img.shape[:2]
    
    # parameters
    dst_img = np.zeros((height, width, 3), dtype = np.uint8)

    num_of_blocks = int((height / block_size) * (width / block_size))
    block_num_y = int(height / block_size)
    block_num_x = int(width / block_size)

    block_order = []

    # initial position
    for i in range(num_of_blocks):
        block_order.append(i)

    #--------------------------------
    # random shuffle
    #--------------------------------
    for i in range(int(num_of_blocks/1.5)):
        block_order[random.randrange(num_of_blocks)] = random.randrange(num_of_blocks)

    #--------------------------------
    # similar block paste
    #--------------------------------
    gray_img = cv2.cvtColor(src_img, cv2.COLOR_RGB2GRAY)

    # find low and high frequency blocks(DCT)
    dct_dc = []
    dct_ac = []
    for dst_block_y in range(block_num_y):
        for dst_block_x in range(block_num_x):
            xx = dst_block_x * block_size
            yy = dst_block_y * block_size
            dct_img = dct(dct(gray_img[yy:yy+block_size, xx:xx+block_size], axis=0, norm='ortho'), axis=1, norm='ortho')

            dct_dc.append(int(dct_img[0, 0]))   # DC coeff
            dct_ac.append(int(dct_img[int(block_size/2), int(block_size/2)]))   # AC(middle) coeff

    dc_max_idx = dct_dc.index(max(dct_dc))
    ac_max_idx = dct_ac.index(sorted(dct_ac)[-random.randrange(5)])

    # choose replaced block
    dst_block = dc_max_idx
    dst_block_y, dst_block_x = divmod(dst_block, block_num_x) # q, mod

    # templete matching
    dst_y = dst_block_y * block_size
    dst_x = dst_block_x * block_size

    res = cv2.matchTemplate(gray_img, gray_img[dst_y:(dst_y+block_size), dst_x:(dst_x+block_size)], cv2.TM_SQDIFF_NORMED)
    loc = np.where(res < 0.1)

    # paste
    src_block = ac_max_idx
    match_num = 0
    for pt in zip(*loc[::-1]):
        if (pt[0] % block_size) == 0 and (pt[1] % block_size) == 0:
            block_no = int((pt[1] * block_num_x + pt[0]) / block_size)
            block_order[block_no] = src_block

            match_num += 1

    #--------------------------------
    # vertical paste
    #--------------------------------
    if match_num < 20:
        for i in range(3):
            col = random.randrange(int(width / block_size))
            
            for j in range(random.randrange(1, 3)):
                src_block = random.randrange(num_of_blocks)

                for k in range(col + j, num_of_blocks, int(width / block_size)):
                    block_order[k] = src_block

    #--------------------------------
    # random change color
    #--------------------------------
    for i in range(50):
        target_block = random.randrange(num_of_blocks)
        target_block_y, target_block_x = divmod(target_block, block_num_x) # q, mod
        target_img_y = target_block_y * block_size
        target_img_x = target_block_x * block_size
        rgb = random.randrange(3)
        src_img[target_img_y:(target_img_y+block_size), target_img_x:(target_img_x+block_size), rgb] -= 128

    #--------------------------------
    # create output image
    #--------------------------------
    for dst_block_y in range(block_num_y):
        for dst_block_x in range(block_num_x):
            curr_pos = dst_block_y * block_num_x + dst_block_x
            src_block_y, src_block_x = divmod(block_order[curr_pos], block_num_x) # q, mod

            src_img_y = src_block_y * block_size
            src_img_x = src_block_x * block_size
            dst_img_y = dst_block_y * block_size
            dst_img_x = dst_block_x * block_size

            dst_img[dst_img_y:(dst_img_y+block_size), dst_img_x:(dst_img_x+block_size), :] = src_img[src_img_y:(src_img_y+block_size), src_img_x:(src_img_x+block_size), :]

    #--------------------------------
    # save image
    #--------------------------------
    dirname = os.path.dirname(filename)
    if dirname == '': dirname = '.'
    basename = os.path.basename(filename)
    outpath = '{}/out_{}.jpg'.format(dirname, basename[:-4])
    cv2.imwrite(outpath, dst_img)
    print('output:', outpath)


if __name__ == '__main__':
    args = sys.argv
    if len(args) < 2:
        print("Please input file_name or directory_name.")
        exit()
    else:
        filename = args[1]

    # directory
    if os.path.isdir(filename):
        types = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        paths = []
        for t in types:
            paths.extend(glob.glob(os.path.join(filename+'/', t)))

        for p in paths:
            print('input:', p)
            if os.path.basename(p)[:4] == 'out_':
                print('skip')
                continue
            create_bug(p)

    # single file
    else:
        create_bug(filename)
