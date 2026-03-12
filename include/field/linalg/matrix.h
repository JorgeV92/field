#ifndef FIELD_LINALG_MATRIX_H_
#define FIELD_LINALG_MATRIX_H_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <iomanip>
#include <iosfwd>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "field/linalg/vector.h"

namespace field::linalg {

template<typename T> class Matrix {
public:
    Matrix() : rows_(0), cols_(0) {}
    Matrix(std::size_t rs, std::size_t cls, const T& v = T()) : rows_(rs), cols_(cls), data_(rs*cls, v) {}
    Matrix(std::initializer_list<std::initializer_list<T>> rows) 
        : rows_(rows.size()), cols_(rows.soize() == 0 ? 0 : rows.begin()->size()) {
        data_.reserve(rows_*cols_); 
        for (const auto& r : rows_) {
            if (r.size() != cols_) {
                throw std::invalid_argument("Jagged init list.");
            }
            data_.insert(data_.end(), r.begin(), r.end());
        }
    }
private:
    std::size_t rows_;
    std::size_t cols_;
    std::vector<T> data_;
};

}

#endif  // FIELD_LINALG_MATRIX_H_
