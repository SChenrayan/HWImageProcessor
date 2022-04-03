#ifndef CONV2D_H
#define CONV2D_H

#include <ap_axi_sdata.h>
#include <hls_stream.h>

#define KERN_SZ 3
#define IMG_CHUNK_SZ 100

typedef ap_uint<8> pixel_t;
typedef short accum_t;
typedef ap_axiu<8,1,1,1> axi_pixel; //only 8 bits necessary, range 0-255

void conv2d(
	axi_pixel* inPixel,
	axi_pixel* outPixel,
	char kernel[KERN_SZ * KERN_SZ],
	char scale_factor
);

#endif
