
# coding: utf-8

# In[1]:


import cv2 as cv
import numpy as np
import sys
from pynq import allocate


# In[2]:


image_filename = "sunflower.png"
image = cv.imread(image_filename)
print(image.shape)
print("Bytes: ",sys.getsizeof(image))


# In[47]:


print(len(image))
print(len(image[0]))
print(len(image[0][0]))
print(image.shape[0])
print(image.shape[1])
print(image.shape[2])


# In[54]:


def apply_kernel_to_img(kern, img): 
    # allocate enough space in same shape as img
    filtered_img = allocate(shape=(img.shape), dtype=np.uint8)
    
    # for each pixel, apply kernel. Intentionally very very slow
    for row_idx in np.arange(0, 200):
        for pixel_idx in np.arange(0, 200): 
            new_blue = apply_kernel_to_pixel(kern, img, row_idx, pixel_idx, 0)
            new_green = apply_kernel_to_pixel(kern, img, row_idx, pixel_idx, 1)
            new_red = apply_kernel_to_pixel(kern, img, row_idx, pixel_idx, 2)
            
            filtered_img[row_idx][pixel_idx][0] = np.uint8(new_blue)
            filtered_img[row_idx][pixel_idx][1] = np.uint8(new_green)
            filtered_img[row_idx][pixel_idx][2] = np.uint8(new_red)
    
    print("Done loop")
    return filtered_img


# In[55]:


def apply_kernel_to_pixel(kern, img, row_idx, pixel_idx, channel):
    return img.item((row_idx, pixel_idx, channel))


# In[56]:


def run_naive_blur(img):
    kernel = np.ones((3,3),np.float32)/9
    return apply_kernel_to_img(kernel, img)


# In[57]:


def run_opencv_blur(img):
    kernel = np.ones((3,3),np.float32)/9
    return cv.filter2D(img,-1,kernel)


# In[61]:


get_ipython().magic('timeit run_opencv_convolution(image)')
opencv_blurred = run_opencv_convolution(image)
cv.imwrite("opencv_blurred.png", opencv_blurred)
print("Saved opencv_blurred...")

get_ipython().magic('timeit run_naive_convolution(image) -n 1')
naive_blurred = run_naive_convolution(image) # terribly terribly slow!
cv.imwrite("naive_blurred.png", naive_blurred)
print("Saved naive_blurred...")


# In[59]:


get_ipython().system('cat /proc/meminfo | grep MemFree')


# In[63]:


get_ipython().magic('timeit -n 1 run_naive_convolution(image)')


# In[64]:


get_ipython().magic('timeit run_opencv_convolution(image)')

