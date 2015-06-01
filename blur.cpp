// Download a Halide distribution from halide-lang.org and untar it in
// the current directory. Then you should be able to compile this
// file with:
//
// c++ -g blur.cpp -std=c++11 -L halide/bin/ -lHalide `libpng-config --cflags --ldflags` -lopencv_core -lopencv_imgproc -O3
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

    Func blur_y;
    blur_y(x, y, c) = (kernel(0) * in_bounded(x, y, c) +
                       kernel(1) * (in_bounded(x, y-1, c) +
                                    in_bounded(x, y+1, c)) +
                       kernel(2) * (in_bounded(x, y-2, c) +
                                    in_bounded(x, y+2, c)) +
                       kernel(3) * (in_bounded(x, y-3, c) +
                                    in_bounded(x, y+3, c)));
    
    Func blur_x;
    blur_x(x, y, c) = (kernel(0) * blur_y(x, y, c) +
                       kernel(1) * (blur_y(x-1, y, c) +
                                    blur_y(x+1, y, c)) +
                       kernel(2) * (blur_y(x-2, y, c) +
                                    blur_y(x+2, y, c)) +
                       kernel(3) * (blur_y(x-3, y, c) +
                                    blur_y(x+3, y, c)));

    // Schedule it.
    kernel.compute_root();
    blur_x.compute_root().vectorize(x, 8).parallel(y);
    blur_y.compute_at(blur_x, y).vectorize(x, 8);

    // Print out pseudocode for the pipeline.
    blur_x.compile_to_lowered_stmt("blur.html", {in}, HTML);
    
    // Benchmark the pipeline.
    Image<float> output(in.width(),
                        in.height(),
                        in.channels());
    for (int i = 0; i < 10; i++) {
        double t1 = current_time();
        blur_x.realize(output);
        double t2 = current_time();
        std::cout << "Time: " << (t2 - t1) << '\n';
    }
    
    save(output, "output.png");

    // Time OpenCV doing the same thing.
    {
        cv::Mat input_image = cv::Mat::zeros(in.width(), in.height(), CV_32FC3);
        cv::Mat output_image = cv::Mat::zeros(in.width(), in.height(), CV_32FC3);
        
        double best = 1e10;
        for (int i = 0; i < 10; i++) {
            double t1 = current_time();
            GaussianBlur(input_image, output_image, cv::Size(7, 7),
                         1.5f, 1.5f, cv::BORDER_REPLICATE);
            double t2 = current_time();
            best = std::min(best, t2 - t1);
        }
        std::cout << "OpenCV time: " << best << "\n";
    }
    
    return 0;
}
