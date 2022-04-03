import cv2 as cv
import numpy as np
import sys
import timeit
import math
# from pynq import allocate

# saturates/clamps value to 0 - 255
def saturate(val):
    if (val > 255):
        return 255
    elif (val < 0): 
        return 0
    return val

# 
def in_bounds(k_mid, img, row_idx, pixel_idx):
    img_max_rows = img.shape[0] - 1
    img_max_cols = img.shape[1] - 1
    row_in_bounds = (row_idx - k_mid) >= 0 and (row_idx + k_mid) < img_max_rows
    col_in_bounds = (pixel_idx - k_mid) >= 0 and (pixel_idx + k_mid) < img_max_cols

    return row_in_bounds and col_in_bounds

# Applies the kernel to the given pixel across red/blue/green channels.
# PNG images are stored as BGR images (blue in channel 0, green in 1, red in 2)
def apply_kernel_to_pixel(kern, img, row_idx, pixel_idx, scale_factor):
    k_mid = int(len(kern) / 2) # assuming square kernel
    blue = 0
    green = 0
    red = 0
    if not in_bounds(k_mid, img, row_idx, pixel_idx): # bounds checking
        return img[row_idx][pixel_idx] # return pixel untransformed if not in bounds

    for row in range(-k_mid, k_mid + 1): # if k_mid = 2, then range(-2, 3) = [-2, -1, 0, 1, 2]
        for col in range(-k_mid, k_mid + 1):
            pixel = img[row_idx + row][pixel_idx + col]
            blue = blue + int(pixel[0] * kern[k_mid + row][k_mid + col] / scale_factor)
            green = green + int(pixel[1] * kern[k_mid + row][k_mid + col] / scale_factor)
            red = red + int(pixel[2] * kern[k_mid + row][k_mid + col] / scale_factor)

    return (saturate(blue), saturate(green), saturate(red))

# Iterates over each pixel of img and applies the given kernel. 
def apply_kernel_to_img(kern, img, scale_factor): 
    # allocate enough space in same shape as img
    filtered_img = np.zeros(shape=(img.shape), dtype=np.uint8)
    # on Pynq, this should be replaced by allocate call.
    
    # for each pixel, apply kernel. Intentionally very very slow
    for row_idx in np.arange(0, img.shape[0]):
        for pixel_idx in np.arange(0, img.shape[1]): 
            new_blue, new_green, new_red = apply_kernel_to_pixel(kern, img, row_idx, pixel_idx, scale_factor)
            
            filtered_img[row_idx][pixel_idx][0] = np.uint8(new_blue)
            filtered_img[row_idx][pixel_idx][1] = np.uint8(new_green)
            filtered_img[row_idx][pixel_idx][2] = np.uint8(new_red)

    return filtered_img

# Our convolution implementation
def run_naive_blur(img):
    kernel = np.ones((3,3),np.uint8)
    return apply_kernel_to_img(kernel, img, 9)

# Open CV's convolution implementation - many times faster!
def run_opencv_blur(img):
    kernel = np.ones((3,3),np.float32)/9
    return cv.filter2D(img,-1,kernel)

# segments an image or a channel of an image into 100 pixel
# wide chunks
def segment_image(img):
    rows = len(img)               # image length
    cols = len(img[0])            # image width

    num_chunks = math.ceil(cols/100)

    # split array into sub-arrays 
    split_indices =[]
    for i in range(1, num_chunks):
        split_indices.append(100 * i)
    print("split indices: ", split_indices)
    chunks = np.split(img, split_indices, 1)
    print("Chunk shape: ", chunks[0].shape)

    return chunks

# TODO: Assemble channel/img from chunks
# TODO: Assemble image from channels

def main(): 
        
    image_filename = "sunflower.png"
    image = cv.imread(image_filename)
    print(image.shape)
    print("Total bytes of img: ",sys.getsizeof(image))
    print("Image width: ", len(image[0]))
    print("Image length: ", len(image))

    # get channels of image
    b = image[:,:,0]
    g = image[:,:,1]
    r = image[:,:,2]

    blue_chunks = segment_image(b)
    for chunk in blue_chunks:
        print("chunk shape: ", chunk.shape)

    '''
    opencv_blurred = run_opencv_blur(image)
    cv.imwrite("opencv_blurred.png", opencv_blurred)
    print("Saved opencv_blurred...")

    naive_blurred = run_naive_blur(image) # terribly terribly slow on the Pynq!
    cv.imwrite("naive_blurred.png", naive_blurred)
    print("Saved naive_blurred...")

    opencv_time = timeit.timeit(lambda: run_opencv_blur(image), number=1)
    naive_time = timeit.timeit(lambda: run_naive_blur(image), number=1)

    print("OpenCV Time: ", opencv_time)
    print("Naive Time: ", naive_time)
    '''


if __name__ == "__main__": 
    main()

