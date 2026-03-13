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
    std::size_t Rows() const { return rows_; }
    std::size_t Cols() const { return cols_; }
    bool Empty() const { return data_.empty(); }
    static Matrix Identity(std::size_t n) {Matrix i(n, n, T()); for (std::size_t ii = 0; ii < n; ii++){i(ii,ii) = T(1);} return i;}
    T& operator()(std::size_t row, std::size_t col) {return data_.at(Index(row, col));}
    const T& operator()(std::size_t row, std::size_t col) const { return data_.at(Index(row, col)); }
    Vector<T> Row(std::size_t row) const {
        std::vector<T> v(cols_);
        for (std::size_t c = 0; c < cols_; ++c) {
            v[c] = (*this)(row, c);
        }
        return Vector<T>(std::move(v));
    }
    Vector<T> Col(std::size_t col) const {
        std::vector<T> v(rows_);
        for (std::size_t r = 0; r < rows_; ++r) {
            v[r] = (*this)(r, col);
        }
        return Vector<T>(std::move(v));
    }
    Matrix& operator+=(const Matrix& o) {CheckSameShape(o); for (std::size_t i = 0; i < data_.size(); ++i) {data_[i] += o.data_[i];} return *this;}
    
private:
    void CheckSameShape(const Matrix& o) const {if (rows_ != o.rows_ || cols_ != o.cols_) throw std::invalid_argument("Matrix shape mismatch.");}
    std::size_t rows_;
    std::size_t cols_;
    std::vector<T> data_;
};

}

#endif  // FIELD_LINALG_MATRIX_H_
