#include <Halide.h>

using namespace Halide;

class IirBlur : public Generator<IirBlur> {
public:
    // The input is a 3D single precision float buffer, the first two
    // dimensions are the pixels, and the third is the color channel.
    ImageParam input{Float(32), 3, "input"};
    // This parameter is the strength of the blur.
    Param<float> A{"A"};

    // Declare our free variables.
    Var x, y, c;

    Func blur_transpose(Func f, Expr height) {
        // Define the blur: first, just copy f to blur.
        Func blur;
        blur(x, y, c) = f(x, y, c);

        // Blur down the columns.
        RDom ry(1, height - 1);
        blur(x, ry, c) = A*blur(x, ry, c) + (1 - A)*blur(x, ry - 1, c);

        // And back up the columns.
        Expr flip_ry = height - ry - 1;
        blur(x, flip_ry, c) =
            A*blur(x, flip_ry, c) + (1 - A)*blur(x, flip_ry + 1, c);

        // Transpose the resulting blurred image.
        Func transpose;
        transpose(x, y, c) = blur(y, x, c);
        
        // Schedule.
        // First, split transpose into groups of 8 rows of pixels.
        Var yo, yi;
        transpose.compute_root().split(y, yo, yi, 8);

        // Parallelize the groups of rows.
        transpose.parallel(yo);

        // Compute the blur at each strip of rows (columns before the
        // transpose).
        blur.compute_at(transpose, yo);

        // Vectorize across x for all steps of blur.
        blur.vectorize(x, 8);
        blur.update(0).vectorize(x, 8);
        blur.update(1).vectorize(x, 8);

        return transpose;
    }

    Func build() {
        // Wrap the input image in a func.
        Func input_func;
        input_func(x, y, c) = input(x, y, c);

        // Blur down the columns and transpose.
        Func blur_y = blur_transpose(input_func, input.height());

        // Blur down the columns again (rows after the transpose
        // above), and then transpose back to the original
        // orientation.
        Func blur = blur_transpose(blur_y, input.width());

        return blur;
    }
};

auto gen = RegisterGenerator<IirBlur>("IirBlur");
