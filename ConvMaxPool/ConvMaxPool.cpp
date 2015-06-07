#include <Halide.h>
using namespace Halide;

class ConvMaxPool : public Generator<ConvMaxPool> {
public:
    
    Func build() {
        return Func();
    }
};

auto gen = RegisterGenerator<ConvMaxPool>("ConvMaxPool.cpp");