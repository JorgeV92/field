#include "test_helpers.h"

#include <cmath>
#include <stdexcept>

#include "field/linalg/vector.h"

int main() {
    using field::linalg::Vector;

    Vector<double> a{1.0, 2.0, 3.0};
    Vector<double> b{4.0, 5.0, 6.0};

    FIELD_EXPECT_TRUE(a.Size() == 3);
    FIELD_EXPECT_TRUE(!a.Empty());

    return EXIT_SUCCESS;
}