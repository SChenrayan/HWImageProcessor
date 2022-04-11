
# coding: utf-8

# In[7]:


import cv2 as cv
import numpy as np
import sys
import timeit
import math
from pynq import allocate
from pynq import Overlay

# scaleFactor offset = 0x20
SCALE_FACTOR_OFF = 0x20
# kernel offset = 0x10
KERNEL_OFF = 0x10

OVERLAY_PATH = "conv2d_hw.bit"
OVERLAY = Overlay(OVERLAY_PATH)


# In[40]:


def to_int32_arr(nums):
    arr = np.ndarray(shape=(3), dtype=np.int32)
    arr[0] = 0
    arr[1] = 0
    arr[2] = 0
    
    for index, item in enumerate(nums):
        num = item & 0xff
        arr[index // 4] |= num << (8 * (index % 4))
    
    return arr

# segments an image or a channel of an image into 100 pixel
# wide chunks
def segment_image(img):
    # image length
    cols = len(img[0])            # image width

    num_chunks = math.ceil(cols/100)

    # split array into sub-arrays 
    split_indices =[]
    for i in range(1, num_chunks):
        split_indices.append(100 * i)
    chunks = np.split(img, split_indices, 1)

    return chunks

def assemble_chunks(chunks):
    all_chunks = np.concatenate(chunks, axis=1) 
    print("Overall chunk shape: ", all_chunks.shape)

    return all_chunks

def conv_dma(in_buf):
    if OVERLAY is None: 
        return in_buf.copy()
        
    # break up in_buff to fit in dma
    num_chunks = math.ceil(len(in_buf)/10000)

    # split array into sub-arrays 
    split_indices =[]
    for i in range(1, num_chunks):
        split_indices.append(10000 * i)
    buf_chunks = np.split(in_buf, split_indices, 0)
    
    # process each part of total buffer
    processed = np.array([], dtype=np.uint32)
    dma = OVERLAY.convDMA.axi_dma_0
    for buf in buf_chunks:
        out_buf = allocate(shape=buf.shape, dtype=np.uint32)
        dma.sendchannel.transfer(buf)
        dma.recvchannel.transfer(out_buf)
        dma.sendchannel.wait()
        dma.recvchannel.wait()
        processed = np.append(processed, out_buf)
    
    return processed

def stream_chunks(chunks): 
    new_chunks = []
    i = 0
    for chunk in chunks:
        print("Working on chunk ", i)
        print("chunk shape: ", chunk.shape)
        print("chunk element 0: ", chunk[0])
        i = i+1
        flattened = chunk.flatten()
        print("flattened chunk shape: ", flattened.shape)
        print("flattened first 3 elements: ", flattened[0], flattened[1], flattened[2])
        packed_32_bit = pack_32(flattened)
        print(packed_32_bit)
        inbuf = allocate(shape=packed_32_bit.shape, dtype=np.uint32)
        inbuf[:] = packed_32_bit
        out_chunk = conv_dma(inbuf)
        new_chunks.append(np.reshape(out_chunk, chunk.shape))

    return new_chunks

def pack_32(array):
    packed = np.packbits(array)
    print(packed)
    return packed.astype(np.int32)

def write_axilite(kernel, scaleFactor):
    np_kern = np.array(kernel, dtype=np.int8)
    conv_ip = OVERLAY.convDMA.conv2d_0
    arr = to_int32_arr(np_kern)
    conv_ip.write(KERNEL_OFF, int(arr[0]))
    conv_ip.write(KERNEL_OFF + 4, int(arr[1]))
    conv_ip.write(KERNEL_OFF + 8, int(arr[2]))
    
    conv_ip.write(SCALE_FACTOR_OFF, scaleFactor)

def hw_conv(img, kernel, scaleFactor): 
    b = img[:,:,0]
    g = img[:,:,1]
    r = img[:,:,2]

    write_axilite(kernel, scaleFactor)
    
    chunks = segment_image(img)
    new_chunks = stream_chunks(chunks)
    output_chunks = assemble_chunks(new_chunks)

    return output_chunks

def main(): 
    image_filename = "food.png"
    out_image_filename = "media/food_sobel.png"
    image = cv.imread(image_filename)
    sharpen_kernel = [0,-1,0,-1,5,-1,0,-1,0]
    sobel_filter = [-1, -2, -1, 0, 0, 0, 1, 2, 1]
    gaussian_blur_kernel = [1,2,1,2,4,2,1,2,1]   #scale factor is 16
    new_img = hw_conv(image, sobel_filter, 1)
    cv.imwrite(filename=out_image_filename, img=new_img)


# In[41]:


identity_kernel = [0,0,0,0,1,0,0,0,0]
sharpen_kernel = [0,-1,0,-1,5,-1,0,-1,0]
mean_blur_kernel = [1,1,1,1,1,1,1,1,1]       #scale factor is 9
laplacian_kernel = [0,1,0,1,-4, 1, 0, 1, 0]
gaussian_blur_kernel = [1,2,1,2,4,2,1,2,1]   #scale factor is 16
sobel_filter = [-1, -2, -1, 0, 0, 0, 1, 2, 1]


# In[42]:


main()

