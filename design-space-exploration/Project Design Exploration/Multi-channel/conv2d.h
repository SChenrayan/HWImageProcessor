#ifndef CONV2D_H
#define CONV2D_H

#include <ap_axi_sdata.h>
#include <hls_stream.h>

#define KERN_SZ 3
#define IMG_CHUNK_SZ 100
#define PIXEL_BUF_SZ (2 * IMG_CHUNK_SZ + KERN_SZ)

typedef ap_uint<24> pixel_t;
typedef ap_uint<8> pixel_channel_t;
typedef short accum_t;
typedef ap_axiu<32,1,1,1> axi_pixel_stream; //only 24 bits necessary, even though 32 bits wide

#define BLUE(pixel) ((pixel >> 16) & 0xFF)
#define GREEN(pixel) ((pixel >> 8) & 0xFF)
#define RED(pixel) ((pixel) & 0xFF)

void conv2d(
	axi_pixel_stream* inPixel,
	axi_pixel_stream* outPixel,
	char kernel[KERN_SZ * KERN_SZ],
	char scale_factor
);

#endif
