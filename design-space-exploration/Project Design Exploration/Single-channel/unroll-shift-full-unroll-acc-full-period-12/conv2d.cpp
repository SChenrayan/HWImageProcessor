#include "conv2d.h"
#include <ap_axi_sdata.h>
#include <hls_stream.h>
#include "ap_int.h"

pixel_t clamp_8bit(accum_t val) {
#pragma HLS INLINE
	pixel_t ret;
	if (val > 255) {
		ret = 255;
	}
	else if (val < 0) {
		ret = 0;
	}
	else {
		ret = (pixel_t) (val & 0xFF);
	}

	return ret;
}

void conv2d(
	axi_pixel* inPixel,
	axi_pixel* outPixel,
	char kernel[KERN_SZ * KERN_SZ],
	char scaleFactor
) {
#pragma HLS INTERFACE axis register both port=inPixel
#pragma HLS INTERFACE axis register both port=outPixel
#pragma HLS INTERFACE s_axilite port=kernel
#pragma HLS INTERFACE s_axilite port=scaleFactor
#pragma HLS INTERFACE ap_ctrl_none port=return

#pragma HLS ARRAY_PARTITION variable=kernel complete dim=1

	static pixel_t pixelBuf[2 * IMG_CHUNK_SZ + KERN_SZ];
#pragma HLS ARRAY_PARTITION variable=pixelBuf complete dim=1
	unsigned char i, j;
	accum_t acc = 0;

	// Two things:
	// 1: Add inPixel to pix_buf
Shift_Loop:
	for (i = 0; i < (2 * IMG_CHUNK_SZ + KERN_SZ - 1); i++) {
#pragma HLS UNROLL
		pixelBuf[i] = pixelBuf[i + 1];
	}
	pixelBuf[i] = inPixel->data;

	// 2: Perform convolution if applicable
Convolution_Loop:
	for (i = 0; i < KERN_SZ; i++) { // i = row
#pragma HLS UNROLL
		for (j = 0; j < KERN_SZ; j++) { //j = col
#pragma HLS UNROLL
			acc += pixelBuf[i * IMG_CHUNK_SZ + j] * kernel[i * KERN_SZ + j];
		}
	}
	// div by scale factor
	acc = acc / scaleFactor;

	// output accumulator
	outPixel->data = clamp_8bit(acc);
	outPixel->keep = inPixel->keep;
	outPixel->strb = inPixel->strb;
	outPixel->last = inPixel->last;
	outPixel->dest = inPixel->dest;
	outPixel->id = inPixel->id;
	outPixel->user = inPixel->user;
}
