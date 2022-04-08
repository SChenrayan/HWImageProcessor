# HWImageProcessor
Project for EECE4632

## Description, so far
As of 4/6/22, we've gotten the core of our hardware and software working. We can process any image, regardless of size, and convolve an arbitrary 3x3 kernel over it (allowing us to perform basic image processing operations such as blurring, sharpening, edge detection, etc). 

### Hardware side

Our convolution IP block takes in three inputs - the input pixel, the `kernel` (currently limited to 3x3), and the `scaleFactor`. There is one output - the output pixel. The input and output pixels are transmitted through the AXI streaming interface, while the `kernel` and `scaleFactor` are transmitted via axilite. For maximum portability, we assume that the input pixel stream describes an image chunk that is 100 pixels wide (100 pixels is arbitrary - this number could just as easily have been 150 or 200, and could be changed going forward, perhaps as a part of our design-space exploration). Constraining the image chunk width allows us to avoid variable-length arrays and loops, which could degrade the performance of our hardware. Any image can be handled by breaking it up into chunks 100 pixels wide and sequentially feeding the chunks through our hardware block; however, segmenting/assembling the image must occur as a part of the software pre- and post-processing stage. 

### Software side

The software side of our design takes care of file operations (i.e. reading and writing image files); we make use of OpenCV, which provides enough basic functionality for us to convert image to pixel arrays.

#### Pre-processing and post-processing

To satisfy the DMA's memory constraints, the pixel-array is pre-processed on the software side. First, the three R, G, and B channels of the image are extracted. Each channel is then split into chunks that are 100 pixels wide using the `segment_image` function. Depending on the length of the image, these chunks may still be too big for the DMA block to handle. To make sure the data can fit in the DMA, each chunk is flattened and segmented further to have a maximum length of 10000 bytes.

Once the convolution IP block returns its output, we have to re-assemble the pixel channels from each chunk, essentially performing the reverse operations of the pre-processing step. We then put together each channel, and write the filtered image out to the desired file.  

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
