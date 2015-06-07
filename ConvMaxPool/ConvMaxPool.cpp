#include <Halide.h>
using namespace Halide;

class ConvMaxPool : public Generator<ConvMaxPool> {
public:
    // This is the input image to be convolved with the filter bank.
    ImageParam img{Float(32), 2, "img"};
    
    // This is the filter bank. The first two dimensions are the kernel
    // indices, the last dimension is the filter index.
    ImageParam filters{Float(32), 3, "filters"};
    
    // How large to pool the results.
    Param<int> pool_size{"pool_size"};

    Func build() {
        // Declare our free variables: pixel indices x, y, and filter index i.
        Var x, y, i;

        // Add a zero padding boundary condition to the input.
        Func img_bounded = BoundaryConditions::constant_exterior(img, 0);
        
        // We want to vectorize across the filters dimension. To do
        // this, we need to transpose the input filter array such that
        // the filter index is the innermost (contiguous) dimension.
        Func filters_reordered("filters_reordered");
        filters_reordered(x, y, i) = filters(x, y, i);

        Expr filter_width = filters.width();
        Expr filter_height = filters.height();

        // Define the convolution of the input 'img' with each filter in 'filters'.
        Func conv2;
        RDom rf(0, filter_width, 0, filter_height);
        conv2(x, y, i) =
            sum(img_bounded(x + rf.x - filter_width / 2, y + rf.y - filter_height / 2) * filters_reordered(rf.x, rf.y, i));

        // Define the maximum of each pool of conv2 results.
        Func max_pooled;
        RDom rp(0, pool_size, 0, pool_size);
        max_pooled(x, y, i) =
            maximum(conv2(pool_size * x + rp.x, pool_size * y + rp.y, i));

        // Schedule the pipeline. We want to reorder the computations
        // such that each filter is computed as the innermost loop.
        max_pooled.reorder(i, x, y);
        filters_reordered.compute_root().reorder_storage(i, x, y);

        // Vectorize across the filter dimension.
        max_pooled.vectorize(i, 8);

        // Parallelize by running 8 rows of the output at a time.
        max_pooled.parallel(y, 8);

        // Apply the boundary condition to the input as required for each
        // pooled result.
        img_bounded.compute_at(max_pooled, x);

        return max_pooled;
    }
};

auto gen = RegisterGenerator<ConvMaxPool>("ConvMaxPool");
