// Download a Halide distribution from halide-lang.org and untar it in
// the current directory. Then you should be able to compile this
// file with:
//
// c++ -g brighten.cpp -std=c++11 -L halide/bin/ -lHalide `libpng-config --cflags --ldflags` -O3
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

int main(int argc, char **argv) {
    Image<float> in = load<float>("input.png");

    Func brighter;
    Var x, y, c;
    brighter(x, y, c) = pow(in(x, y, c), 0.8f);

    brighter.vectorize(x, 8).parallel(y);
    
    Image<float> output(in.width(), in.height(), in.channels());
    for (int i = 0; i < 10; i++) {
        double t1 = current_time();
        brighter.realize(output);
        double t2 = current_time();
        std::cout << "Time: " << (t2 - t1) << "\n";
    }
    
    save(output, "output.png");
    
    return 0;
}
