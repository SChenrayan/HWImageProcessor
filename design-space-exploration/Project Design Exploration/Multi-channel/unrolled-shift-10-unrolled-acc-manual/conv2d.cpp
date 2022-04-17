#include "conv2d.h"
#include <ap_axi_sdata.h>
#include <hls_stream.h>
#include "ap_int.h"

pixel_t build_pixel(pixel_channel_t blue, pixel_channel_t green, pixel_channel_t red) {
	pixel_t output = 0;
	output = blue;
	output = output << 8;
	output |= green;
	output = output << 8;
	output |= red;

	return output;
}

pixel_channel_t clamp_8bit(accum_t p) {
	if (p > 255) {
		return 255;
	}
	else if (p < 0) {
		return 0;
	}

	return (pixel_channel_t) (p & 0xFF);
}

void conv2d(
	axi_pixel_stream* inPixel,
	axi_pixel_stream* outPixel,
	char kernel[KERN_SZ * KERN_SZ],
	char scaleFactor
) {
#pragma HLS INTERFACE axis register both port=inPixel
#pragma HLS INTERFACE axis register both port=outPixel
#pragma HLS INTERFACE s_axilite port=kernel
#pragma HLS INTERFACE s_axilite port=scaleFactor
#pragma HLS INTERFACE ap_ctrl_none port=return

#pragma HLS ARRAY_PARTITION variable=kernel complete dim=1

	static pixel_t pixelBuf[PIXEL_BUF_SZ];
#pragma HLS ARRAY_PARTITION variable=pixelBuf complete dim=1

	unsigned char i;
	accum_t red_acc = 0, blue_acc = 0, green_acc = 0;
	pixel_t currentPixel;
	char currentKernelVal;
	pixel_channel_t red, blue, green;
	pixel_t output;

	// Two things:
	// 1: Add inPixel to pix_buf
Shift_Loop:
	for (i = 0; i < (PIXEL_BUF_SZ - 1); i++) {
#pragma HLS PIPELINE
#pragma HLS LOOP unroll factor=10
		pixelBuf[i] = pixelBuf[i + 1];
	}
	pixelBuf[i] = inPixel->data & 0xFFFFFF; //lower 24 bits

	// 2: Perform convolution if applicable
//Convolution_Loop:
//	for (i = 0; i < KERN_SZ; i++) { // i = row
//		for (j = 0; j < KERN_SZ; j++) { //j = col
//			currentPixel = pixelBuf[i * IMG_CHUNK_SZ + j];
//			currentKernelVal = kernel[i * KERN_SZ + j];
//
//			blue = (currentPixel >> 16) & 0xFF;
//			green = (currentPixel >> 8) & 0xFF;
//			red = currentPixel & 0xFF;
//
//			blue_acc += blue * currentKernelVal;
//			red_acc += red * currentKernelVal;
//			green_acc += green * currentKernelVal;
//		}
//	}
	blue_acc += BLUE(pixelBuf[0]) * kernel[0];
	blue_acc += BLUE(pixelBuf[1]) * kernel[1];
	blue_acc += BLUE(pixelBuf[2]) * kernel[2];
	blue_acc += BLUE(pixelBuf[100] * kernel[3]);
	blue_acc += BLUE(pixelBuf[101] * kernel[4]);
	blue_acc += BLUE(pixelBuf[102] * kernel[5]);
	blue_acc += BLUE(pixelBuf[200] * kernel[6]);
	blue_acc += BLUE(pixelBuf[201] * kernel[7]);
	blue_acc += BLUE(pixelBuf[202] * kernel[8]);

	red_acc += RED(pixelBuf[0]) * kernel[0];
	red_acc += RED(pixelBuf[1]) * kernel[1];
	red_acc += RED(pixelBuf[2]) * kernel[2];
	red_acc += RED(pixelBuf[100] * kernel[3]);
	red_acc += RED(pixelBuf[101] * kernel[4]);
	red_acc += RED(pixelBuf[102] * kernel[5]);
	red_acc += RED(pixelBuf[200] * kernel[6]);
	red_acc += RED(pixelBuf[201] * kernel[7]);
	red_acc += RED(pixelBuf[202] * kernel[8]);

	green_acc += GREEN(pixelBuf[0]) * kernel[0];
	green_acc += GREEN(pixelBuf[1]) * kernel[1];
	green_acc += GREEN(pixelBuf[2]) * kernel[2];
	green_acc += GREEN(pixelBuf[100] * kernel[3]);
	green_acc += GREEN(pixelBuf[101] * kernel[4]);
	green_acc += GREEN(pixelBuf[102] * kernel[5]);
	green_acc += GREEN(pixelBuf[200] * kernel[6]);
	green_acc += GREEN(pixelBuf[201] * kernel[7]);
	green_acc += GREEN(pixelBuf[202] * kernel[8]);

	blue_acc = blue_acc / scaleFactor;
	red_acc = red_acc / scaleFactor;
	green_acc = green_acc / scaleFactor;

	red = clamp_8bit(red_acc);
	blue = clamp_8bit(blue_acc);
	green = clamp_8bit(green_acc);

	output = build_pixel(blue, green, red);

	// output accumulator
	outPixel->data = output;
	outPixel->keep = inPixel->keep;
	outPixel->strb = inPixel->strb;
	outPixel->last = inPixel->last;
	outPixel->dest = inPixel->dest;
	outPixel->id = inPixel->id;
	outPixel->user = inPixel->user;
}
