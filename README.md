# HWImageProcessor
Project for EECE4632

## Final Project Update - 5/4/22

### Usage

See the usage notes for `Project Update 3` since not much has changed to be able to run the project

### Changes since last update

- Testing: validation and timing
- Attempt at implementing zero-padding (see `conv2D_hw_with_padding.py`)

## Project Update 3 - 4/21/22

### Usage 

Please see the pngs, notebook, bitstream, hwh files in the `project_update_three` directory for an optimized version of our convolution IP. You might have to fiddle around with the filenames - for instance, make sure the overlay is set correctly.
```
OVERLAY = "actually_fixed.bit"
```

Similarly, you might need to adjust the image filepaths (we recommend using `sunflower.png` and `food.png` as inputs but if you're feeling adventurous, any PNG should work fine). As always, feel free to contact us if you're having setup troubles.

For a more detailed description of the project, please see the report. 

## Project Update 2 (4/7/22)
As of 4/6/22, we've gotten the core of our hardware and software working. We can process any image, regardless of size, and convolve an arbitrary 3x3 kernel over it (allowing us to perform basic image processing operations such as blurring, sharpening, edge detection, etc). 

### Hardware side

Our convolution IP block takes in three inputs - the input pixel, the `kernel` (currently limited to 3x3), and the `scaleFactor`. There is one output - the output pixel. The input and output pixels are transmitted through the AXI streaming interface, while the `kernel` and `scaleFactor` are transmitted via axilite. For maximum portability, we assume that the input pixel stream describes an image chunk that is 100 pixels wide (100 pixels is arbitrary - this number could just as easily have been 150 or 200, and could be changed going forward, perhaps as a part of our design-space exploration). Constraining the image chunk width allows us to avoid variable-length arrays and loops, which could degrade the performance of our hardware. Any image can be handled by breaking it up into chunks 100 pixels wide and sequentially feeding the chunks through our hardware block; however, segmenting/assembling the image must occur as a part of the software pre- and post-processing stage. 

### Software side

The software side of our design takes care of file operations (i.e. reading and writing image files); we make use of OpenCV, which provides enough basic functionality for us to convert image to pixel arrays.

#### Pre-processing and post-processing

To satisfy the DMA's memory constraints, the pixel-array is pre-processed on the software side. First, the three R, G, and B channels of the image are extracted. Each channel is then split into chunks that are 100 pixels wide using the `segment_image` function. Depending on the length of the image, these chunks may still be too big for the DMA block to handle. To make sure the data can fit in the DMA, each chunk is flattened and segmented further to have a maximum length of 10000 bytes.

Once the convolution IP block returns its output, we have to re-assemble the pixel channels from each chunk, essentially performing the reverse operations of the pre-processing step. We then put together each channel, and write the filtered image out to the desired file.  

### Plan
- Testing and performance analysis 
- New features
  - More user friendly PS interface

### Usage - Project Update 2 4/7/22

Copy the entire `project_update_two` directory to the pynq board. With the bitstream and hwh files, you can immediately run the `conv2d_hw.ipnyb` notebook. Modify the appropriate input and output file variable names as needed. For best results, stick to PNG images. The `media` directory has examples you can use as input files. 



