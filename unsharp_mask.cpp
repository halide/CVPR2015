// Download a Halide distribution from halide-lang.org and untar it in
// the current directory. Then you should be able to compile this
// file with:
//
// c++ -g unsharp_mask.cpp -std=c++11 -L halide/bin/ -lHalide `libpng-config --cflags --ldflags` -lopencv_core -lopencv_imgproc -O3
//
// You'll also need a multi-megapixel png image to run this on. Name
// it input.png and put it in this directory.

// Include the Halide language
#include "halide/include/Halide.h"
using namespace Halide;

#include <iostream>

// Some support code for timing and loading/saving images
#include "halide/tutorial/image_io.h"
#include "halide/tutorial/clock.h"

// Include OpenCV for timing comparison
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>

int main(int argc, char **argv) {
    Image<float> in = load<float>("input.png");

    // Define a 7x7 Gaussian Blur with a repeat-edge boundary condition.
    float sigma = 1.5f;
    
    Var x, y, c;
    Func kernel;
    kernel(x) = exp(-x*x/(2*sigma*sigma)) / (sqrtf(2*M_PI)*sigma);

    Func in_bounded = BoundaryConditions::repeat_edge(in);

    Func gray;
    gray(x, y) = max(in_bounded(x, y, 0),
                     max(in_bounded(x, y, 1),
                         in_bounded(x, y, 2)));
    
    Func blur_y;
    blur_y(x, y) = (kernel(0) * gray(x, y) +
                    kernel(1) * (gray(x, y-1) +
                                 gray(x, y+1)) +
                    kernel(2) * (gray(x, y-2) +
                                 gray(x, y+2)) +
                    kernel(3) * (gray(x, y-3) +
                                 gray(x, y+3)));
    
    Func blur_x;
    blur_x(x, y) = (kernel(0) * blur_y(x, y) +
                    kernel(1) * (blur_y(x-1, y) +
                                 blur_y(x+1, y)) +
                    kernel(2) * (blur_y(x-2, y) +
                                 blur_y(x+2, y)) +
                    kernel(3) * (blur_y(x-3, y) +
                                 blur_y(x+3, y)));

    Func sharpen;
    sharpen(x, y) = 2 * gray(x, y) - blur_x(x, y);   

    Func ratio;
    ratio(x, y) = sharpen(x, y) / gray(x, y);
    
    Func result;
    result(x, y, c) = ratio(x, y) * in(x, y, c);
    
    // Schedule it.
    blur_y.compute_at(result, y).vectorize(x, 8);
    ratio.compute_at(result, y).vectorize(x, 8);
    result.vectorize(x, 8).parallel(y).reorder(x, c, y);
    
    // Benchmark the pipeline.
    Image<float> output(in.width(),
                        in.height(),
                        in.channels());
    for (int i = 0; i < 10; i++) {
        double t1 = current_time();
        result.realize(output);
        double t2 = current_time();
        std::cout << "Time: " << (t2 - t1) << '\n';
    }
    
    save(output, "output.png");

    // Time OpenCV doing the same thing.
    {
        cv::Mat input_image = cv::imread("input.png");
        input_image.convertTo(input_image, CV_32FC3);
        cv::Mat output_image;
        
        double best = 1e10;
        for (int i = 0; i < 10; i++) {
            double t1 = current_time();

            cv::Mat channels[3];
            cv::split(input_image, channels);
            cv::Mat gray = cv::max(channels[0], cv::max(channels[1], channels[2]));            

            cv::Mat blurry(gray.size(), CV_32FC1);
            GaussianBlur(gray, blurry, cv::Size(7, 7),
                         1.5f, 1.5f, cv::BORDER_REPLICATE);

            cv::Mat sharp = 2*gray - blurry;

            cv::Mat out_channels[3];
            cv::Mat ratio = sharp/gray;
            for (int c = 0; c < 3; c++) {
                out_channels[c] = channels[c].mul(ratio);
            }
            cv::merge(out_channels, 3, output_image);

            double t2 = current_time();
            best = std::min(best, t2 - t1);
        }
        std::cout << "OpenCV time: " << best << "\n";

        output_image.convertTo(output_image, CV_8UC3);
        cv::imwrite("opencv_output.png", output_image);
    }
    
    return 0;
}
