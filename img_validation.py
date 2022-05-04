import cv2 as cv
import numpy as np
import argparse as ap

TOLERANCE = 10
THRESHOLD = 10

SIGNIFICANT_ERR_PERCENT = 10

def percent_err(base, test):
    # print("base: ", base, " test: ", test)
    diff = abs(test - base)
    if base < THRESHOLD or test < THRESHOLD:
        return 100 if diff > TOLERANCE else 0

    err = 100 * diff / base
    return err

def pixel_diff(basepx, testpx):
    zipped = list(zip(basepx, testpx))
    total_err = 0
    for pair in zipped:
        total_err = total_err + percent_err(int(pair[0]), int(pair[1]))

    ret = total_err / len(zipped)
    return ret

def percent_diff(base, test):
    if not base.shape == test.shape:
        print("[ diff ] - Shape mismatch between base and test images (must be same shape)")
        return -1

    rows = base.shape[0]
    cols = base.shape[1]
    total_err = 0
    significant_err_count = 0
    err_count = 0
    for row in range(rows - 1):
        for col in range(cols - 1):
            basepix = base[row][col]
            testpix = test[row + 1][col + 1]
            err = pixel_diff(basepix, testpix)
            if err > SIGNIFICANT_ERR_PERCENT:
                significant_err_count = significant_err_count + 1

            if not err == 0: 
                err_count = err_count + 1

            total_err = total_err + err
    
    total_pix = (rows * cols)
    avg_err = total_err / total_pix # average error per pixel
    percent_significant = (significant_err_count / total_pix) * 100 # percentage of pixels with significant error
    percent_err = (err_count / total_pix) * 100 # percentage of pixels with any error
    return avg_err, percent_significant, percent_err

parser = ap.ArgumentParser(description="Filenames to process")
parser.add_argument("baseimg", type=str, help="The base image file to compare to")
parser.add_argument("testimg", type=str, help="The image file to test")

args = parser.parse_args()

if __name__ == "__main__":
    base_img = cv.imread(args.baseimg)
    test_img = cv.imread(args.testimg)

    if (base_img is None) or (test_img is None): 
        print("[ main ] - cannot read base img or test img")
        exit(1)

    err, sig, total = percent_diff(base_img, test_img)
    
    print("Average percent error: ", err, "%")
    print("Percentage of pixels with some error: ", total, "%")
    print("Percentage of pixels with significant error: ", sig, "%")

