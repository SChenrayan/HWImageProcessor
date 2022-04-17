
# coding: utf-8

# In[47]:


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
MAX_CHUNK_SIZE = 2500


# In[60]:


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
    num_chunks = math.ceil(len(in_buf)/MAX_CHUNK_SIZE)

    # split array into sub-arrays 
    split_indices =[]
    for i in range(1, num_chunks):
        split_indices.append(MAX_CHUNK_SIZE * i)
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
        buf_shape = len(chunk) * len(chunk[0])
        inbuf = allocate(shape=buf_shape, dtype=np.uint32)
        pixel_count = 0
        for row in chunk:
            for pixel in row:
                packed_pixel = pack_pixel(pixel)
                inbuf[pixel_count] = packed_pixel
                pixel_count = pixel_count + 1
        print("inbuf", inbuf)
        out_chunk = conv_dma(inbuf)
        # TODO: For each packed pixel, unpack it
        unpacked_pixel = np.ndarray(shape=(len(out_chunk), 3), dtype=np.uint8)
        pixel_count = 0
        for pixel in out_chunk:
            unpacked = unpack_pixel(pixel)
            unpacked_pixel[pixel_count][0] = unpacked[0]
            unpacked_pixel[pixel_count][1] = unpacked[1]
            unpacked_pixel[pixel_count][2] = unpacked[2]
            pixel_count = pixel_count + 1
        i = i+1
            
        new_chunks.append(np.reshape(unpacked_pixel, chunk.shape))

    return new_chunks

# pack 3 8-bit values into 1  32-bit value.
# The upper 8 bits are zeroed since the
# hardware block ignores them
def pack_pixel(bgr):
    #print("pre-pack", bgr)
    b = bgr[0] & 0xff
    g = bgr[1] & 0xff
    r = bgr[2] & 0xff
    res = int(0)
    res |= b 
    res = res << 8
    res |= g
    res = res << 8
    res |= r
    #print("packed", res)
    return int(res)

# Extracts 3 8-bit values from 32-bit packed
# bgr val
def unpack_pixel(packed):
    b = (packed >> 16) & 0xff
    g = (packed >> 8) & 0xff
    r = packed & 0xff
    return [b, g, r]

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
    
    print(img.shape)
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


# In[61]:


identity_kernel = [0,0,0,0,1,0,0,0,0]
sharpen_kernel = [0,-1,0,-1,5,-1,0,-1,0]
mean_blur_kernel = [1,1,1,1,1,1,1,1,1]       #scale factor is 9
laplacian_kernel = [0,1,0,1,-4, 1, 0, 1, 0]
gaussian_blur_kernel = [1,2,1,2,4,2,1,2,1]   #scale factor is 16
sobel_filter = [-1, -2, -1, 0, 0, 0, 1, 2, 1]


# In[62]:


main()

