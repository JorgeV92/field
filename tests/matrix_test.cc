#include "test_helpers.h"

#include <sstream>
#include <stdexcept>

#include "field/linalg/matrix.h"

int main() {
    using field::linalg::Matrix;
    using field::linalg::Vector;

    Matrix<double> a{{1.0, 2.0}, {3.0, 4.0}};
    Matrix<double> b{{2.0, 0.0}, {1.0, 2.0}};

    auto c = a* b;
    FIELD_EXPECT_NEAR(c(0,0), 4.0, 1e-9);
    
    return EXIT_SUCCESS;
}