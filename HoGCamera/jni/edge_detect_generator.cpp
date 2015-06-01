#include "Halide.h"

namespace {

class EdgeDetect : public Halide::Generator<EdgeDetect> {
public:
    ImageParam input{ UInt(8), 2, "input" };

    Func build() {
        Var x, y;

        Func clamped = Halide::BoundaryConditions::repeat_edge(input);

        // Upcast to 16-bit
        Func in16;
        in16(x, y) = cast<int16_t>(clamped(x, y));
        
        // Gradients in x and y.
        Func gx;
        Func gy;
        gx(x, y) = (in16(x + 1, y) - in16(x - 1, y)) / 2;
        gy(x, y) = (in16(x, y + 1) - in16(x, y - 1)) / 2;

        // Gradient magnitude.
        Func grad_mag;
        grad_mag(x, y) = (gx(x, y) * gx(x, y) + gy(x, y) * gy(x, y));

        const int H = 32;
        const int B = 9;       
        
        // Mapping from gradient to histogram bucket
        Func grad_bucket;
        grad_bucket(x, y) = cast<int>(round((atan2(y, x) * B / M_PI))) % B;
        
        // Make a histogram of gradients per tile
        Func hog;
        Var tx, ty, i;
        {
            RDom r(0, H, 0, H);       
            hog(tx, ty, i) = 0;
            Expr px = tx*H + r.x, py = ty*H + r.y;
            hog(tx, ty, grad_bucket(gx(px, py), gy(px, py))) += grad_mag(px, py);
        }

        // Compute the inverse of a local sum to help normalize
        Func hog_normalize_factor;
        {
            RDom r(-1, 3, -1, 3, 0, B);
            hog_normalize_factor(tx, ty) = (255.0f * 9) / sum(hog(tx + r.x, ty + r.y, r.z));
        }

        Func hog_normalized;
        {
            hog_normalized(tx, ty, i) = cast<uint8_t>(clamp(hog(tx, ty, i) *
                                                            hog_normalize_factor(tx, ty), 0, 255));
        }
        
        // Draw the result
        Func result;
        result(x, y) = hog_normalized(x/H, y/H, grad_bucket(H/2 - y%H, x%H - H/2));
        
        grad_bucket.compute_root().memoize();

        gx.compute_at(hog, tx).vectorize(x, 16);
        gy.compute_at(hog, tx).vectorize(x, 16);
        grad_mag.compute_at(hog, tx).vectorize(x, 16);
        hog.compute_root().update().parallel(ty);

        hog_normalize_factor.compute_root().vectorize(tx, 4).parallel(ty);

        hog_normalized.compute_root().vectorize(tx, 8).parallel(i);

        result.compute_root().parallel(y, 8);
        
        return result;
    }
};

Halide::RegisterGenerator<EdgeDetect> register_edge_detect{ "edge_detect" };

}  // namespace
