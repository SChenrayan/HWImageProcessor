
# coding: utf-8

# In[65]:


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

OVERLAY_PATH = "actually_fixed.bit"
OVERLAY = Overlay(OVERLAY_PATH)


# In[85]:


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

    num_chunks = math.ceil(cols/98)

    # split array into sub-arrays 
    split_indices =[]
    for i in range(1, num_chunks):
        split_indices.append(98 * i)
    chunks = np.split(img, split_indices, 1)
    
    #set edge pixels to zero to prevent artifacts
    chunks = pad_zeros(chunks)

    return chunks

def pad_zeros(chunks):
    padded_chunks=[]
    for chunk in chunks:
        row, col = np.shape(chunk)
        #add columns of zeros to begin and end using hstack
        zeros_col = np.zeros((row,1), dtype=np.uint8)
        #zeros_again = np.zeros((row,1), dtype=np.uint8)
        
        zeros_row = np.zeros((1, col+2), dtype=np.uint8)
        padded = np.hstack((zeros_col, chunk, zeros_col))
        padded = np.vstack((zeros_row, padded, zeros_row))
        padded_chunks.append(padded)
    return padded_chunks

def assemble_chunks(chunks):
    resized_chunks = []
    for chunk in chunks:
        row, col = np.shape(chunk)
        chunk = np.delete(chunk, 0, axis=1)
        chunk = np.delete(chunk, col-2, axis=1)
        chunk = np.delete(chunk, 0, axis=0)
        chunk = np.delete(chunk, row-2, axis=0)
        print("chopped chunk shape: ", np.shape(chunk))
        resized_chunks.append(chunk)
    all_chunks = np.concatenate(resized_chunks, axis=1) 
    #print("Overall chunk shape: ", all_chunks.shape)

    return all_chunks

def assemble_channels(b, g, r): 
    all_channels = np.stack((b, g, r), axis=2)
    print("Image shape: ", all_channels.shape)

    return all_channels

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
    processed = np.array([], dtype=np.uint8)
    dma = OVERLAY.convDMA.axi_dma_0
    for buf in buf_chunks:
        out_buf = allocate(shape=buf.shape, dtype=np.uint8)
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
        #print("Working on chunk ", i)
        i = i+1
        flattened = chunk.flatten()
        inbuf = allocate(shape=flattened.shape, dtype=np.uint8)
        inbuf[:] = flattened
        out_chunk = conv_dma(inbuf)
        new_chunks.append(np.reshape(out_chunk, chunk.shape))

    return new_chunks

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
    
    channels = [b, g, r]
    new_channels = []
    for channel in channels:
        chunks = segment_image(channel)
        new_chunks = stream_chunks(chunks)
        new_channels.append(assemble_chunks(new_chunks))

    return assemble_channels(new_channels[0], new_channels[1], new_channels[2])

def img_filter(infile, outfile, kernel, scaleFactor):
    print("starting hw...")
    image = cv.imread(infile)
    if image is None:
        print("Could not find image")
        return
    new_img = hw_conv(image, kernel, scaleFactor)
    cv.imwrite(filename=outfile, img=new_img)
    print("ended hw.")
    
def sw_img_filter(infile, outfile, kernel, scaleFactor):
    print("starting cv...")
    image = cv.imread(infile)
    if image is None:
        print("Could not find image")
        return
    np_kernel = np.reshape(np.array(kernel), (3,3))
    new_img = cv.filter2D(image, -1, np_kernel/scaleFactor)
    cv.imwrite(filename=outfile, img=new_img)
    print("ended cv.")


# In[86]:


identity_kernel = [0,0,0,0,1,0,0,0,0]
sharpen_kernel = [0,-1,0,-1,5,-1,0,-1,0]
mean_blur_kernel = [1,1,1,1,1,1,1,1,1]       #scale factor is 9
laplacian_kernel = [0,1,0,1,-4, 1, 0, 1, 0]
gaussian_blur_kernel = [1,2,1,2,4,2,1,2,1]   #scale factor is 16
sobel_filter = [-1, -2, -1, 0, 0, 0, 1, 2, 1]


# In[87]:


infile = "media/food.png"
outfile = "media/food_padded_zeros_hw.png"
sw_outfile = "media/food_sw.png"
img_filter(infile, outfile, sharpen_kernel, 1)
#sw_img_filter(infile, sw_outfile, sobel_filter, 1)


# In[5]:


kernels = [(identity_kernel, 1, "identity"), (sharpen_kernel, 1, "sharpen"), (mean_blur_kernel, 9, "mean_blur"), (laplacian_kernel, 1, "laplacian"), (gaussian_blur_kernel, 16, "gaussian_blur"), (sobel_filter, 1, "sobel")]

    

