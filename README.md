# HWImageProcessor
Project for EECE4632

## Description, so far
As of 4/6/22, we've gotten the core of our hardware and software working. We can process any image, regardless of size, and convolve an arbitrary 3x3 kernel over it (allowing us to perform basic image processing operations such as blurring, sharpening, edge detection, etc). 

### Data Pre-processing in Software
To satisfy the DMA's memory constraints, the image is pre-processed on the software side. First, the three R, G, and B channels of the image are extracted. Each channel is then split into chunks that are 100 pixels wide using the `segment_image` function. Depending on the length of the image, these chunks may still be too big for the DMA block to handle. To make sure the data can fit in the DMA, each chunk is flattened and segmented further to have a maximum length of 10000 pixels.

### Convolution in Hardware

### Data Post-processing in Software

### Known Issues
- There's an issue with some of our image filtering. Our hunch is that the kernels with negative values are not working properly, or otherwise leading to some overflow when our convolution IP does signed/unsigned operations. 

### Plan
- Lots of room for optimization!
  - Our current HLS code contains basically no optimizations, and we definitely have the opportunity to do some design space exploration here similar to HW4.
  - Pixel packing
    - Currently, we stream each channel of each pixel (an unsigned 8 bit value). Instead of operating on each channel independently, we can pack each channel together and stream 24-bit pixels instead. 
    - The hardware IP block will have to do a little more work to de-package and re-package the pixel, but we then would only iterate over the image once, instead of thrice (once for each channel). 
    - This will hopefully significantly improve performance.
  - Optimizing pre- and post-processing on the PS side
    - These are expensive operations that require manipulation of thousands of pixels.
    - For instance, finding a way to increase the DMA's buffer size will allow us to stream more pixels at once - don't have to break up into as many chunks on the PS.
  - Parallel convolution blocks
    - Instead of having one convolution block process all pixels, can we use multiple blocks to process chunks of the image in parallel?   
- New features
  - Implementing IP for different types of operations that also rely on kernel application
    - Image de-noising (erosion and dilation)
  - More user friendly PS interface

### Usage - Project Update 2

Copy the entire `project_update_two` directory to the pynq board. With the bitstream and hwh files, you can immediately run the `conv2d_hw.ipnyb` notebook. Modify the appropriate input and output file variable names as needed. For best results, stick to PNG images. The `media` directory has examples you can use as input files. 
