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
    Matrix& operator-=(const Matrix& o) {CheckSameShape(o); for (std::size_t i = 0; i < data_.size(); ++i) {data_[i] -= o.data_[i];} return *this;}
    Matrix& operator*=(const T& x) {for (T& v : data_) v *= x; return *this;}
    Matrix& operator/=(const T& x) {for (T& v : data_) v /= x; return *this;}
    Matrix operator+(const Matrix& o) const {Matrix cpy(*this); cpy += o; return cpy;}
    Matrix operator-(const Matrix& o) const {Matrix cpy(*this); cpy -= o; return cpy;}
    Matrix operator*(const T& x) const {Matrix cpy(*this); cpy *= x; return cpy;}
    Matrix operator/(const T& y) const {Matrix cpy(*this); cpy /= y; return cpy;}
    Matrix operator*(const Matrix& o) const {
        if (cols_ != o.rows_) throw std::invalid_argument("Matrix multi shape mismatch.");
        Matrix res(rows_, o.cols_, T());
        for (std::size_t i = 0; i < rows_; ++i) {
            for (std::size_t k = 0; k < cols_; ++k) {
                const T& lhs = (*this)(i, k);
                for (std::size_t j = 0; j < o.cols_; ++j) {
                    res(i, j) += lhs * o(k, j);
                }
            }
        }
        return res;
    }
    Vecotr<T> operator*(const Vector<T>& vec) const {
        if (cols_ != vec.Size()) throw std::invalid_argument("Matrix-vector multi shape mismatch.");
        std::vector<T> res(rows_, T());
        for (std::size_t i = 0; i < rows_; ++i) {
            for (std::size_t j = 0; j < cols_; ++j) {
                res[i] += (*this)(i, j) * vec[j];
            }
        }
        return Vector<T>(std::move(res));
    }
    Matrix Hadamard(const Matrix& o) const {
        CheckSameShape(o);
        Matrix res(rows_, cols_, T());
        for (std::size_t i = 0; i < data_.size(); ++i) {
            res.data_[i] = data_[i] * o.data_[i];
        }
        return res;
    }
    Matrix Transpose() const {
        Matrix T(cols_, rows_, T());
        for (std::size_t i = 0; i < rows_; ++i) {
            for (std::size_t j = 0; j < cols_; ++j) {
                T(j, i) = (*this)(i, j);
            }
        }
        return T;
    }
    T Trace() const {_CheckArgs(); T trace = T(); for (std::size_t i = 0; i < rows_; ++i){trace += (*this)(i,i);} return trace;}
    T Sum() const {T sum = T(); for (const T& v : data_){sum += v;} return sum;}
    template<typename UF> Matrix Apply(UF fn) const {
        Matrix res(rows_, cols_, T());
        for (std::size_t i = 0; i < data_.size(); ++i) {
            res.data_[i] = fn(data_[i]);
        }
        return res;
    }
    std::size_t Rank() const {
        Matrix<T> reduced(*this);
        const T zero = T();
        std::size_t rank_ = 0;
        std::size_t pivot_row = 0;
        for (std::size_t pivot_col = 0; pivot_col < cols_ && pivot_col < rows_; ++pivot_col) {
            std::size_t selected = pivot_row;
            while (selected < rows_ && reduced(selected, pivot_col) == zero) {++selected;}
            if (selected == rows_) continue;
            if (selected != pivot_row) reduced.SwapRows(selected, pivot_row);
            T pivot = reduced(pivot_row, pivot_col);
            for (std::size_t row = pivot_row + 1; row < rows_; ++row) {
                if (reduced(row, pivot_col) == zero) continue;
                T factor = reduced(row, pivot_col) / pivot;
                for (std::size_t col = pivot_col; col < cols_; ++col) {
                    reduced(row, col) -= factor * reduced(pivot_row, col);
                }
            }
            ++rank_;
            ++pivot_row;
        }
        return rank_;
    }
    T Determinant() const {
        CheckSquare();
        Matrix<T> reduced(*this);
        const T zero = T();
        T determinant = T(1);
        int swap_count = 0;
        for (std::size_t pivot_col = 0 ; pivot_col < cols_; ++pivot_col) {
            std::size_t selected = pivot_col;
            while (selected < rows_ && reduced(selected, pivot_col) == zero) {++selected;}
            if (selected == rows_) {return T();}
            if (selected != pivot_col) {reduced.SwapRows(selected, pivot_col); ++swap_count;}
            T pivot = reduced(pivot_col, pivot_col);
            determinant *= pivot;
            for (std::size_t row = pivot_col + 1; row < rows_; ++row) {
                if (reduced(row, pivot_col) == zero) continue;
                T factor = reduced(row, pivot_col) / pivot;
                for (std::size_t col = pivot_col; col < cols_; ++col) 
                    reduced(row, col) -= factor * reduced(pivot_col, col);
            }
        }
        if (swap_count % 2 != 0)
            determinant = -determinant;
        return determinant;
    }
    Matrix Inverse() const {
        CheckSquare();
        Matrix<T> left(*this);
        Matrix<T> right = Matrix<T>::Identity(rows_);
        const T zero = T();
        for (std::size_t pivot_col = 0; pivot_col < cols_; ++pivot_col) {
            std::size_t selected = pivot_col;
            while (selected < rows_ && left(selected, pivot_col) == zero) {++selected;}
            if (selected == rows_) throw std::domain_error("Matrix is singular and cannot be inverted.");
            if (selected != pivot_col) {left.SwapRows(selected, pivot_col); right.SwapRows(selected, pivot_col);}
            T pivot = left(pivot_col, pivot_col);
            for (std::size_t col = 0; col < cols_; ++col) {
                left(pivot_col, col) /= pivot;
                right(pivot_col, col) /= pivot;
            }
            for (std::size_t row = 0; row < rows_; ++row) {
                if (row == pivot_col || left(row, pivot_col) == zero) continue;
                T factor = left(row, pivot_col);
                for (std::size_t col = 0; col < cols_; ++col) {
                    left(row, col) -= factor * left(pivot_col, col);
                    right(row, col) -= factor * right(pivot_col, col);
                }
            }
        }
        return right;
    }
private:
    void CheckSameShape(const Matrix& o) const {if (rows_ != o.rows_ || cols_ != o.cols_) throw std::invalid_argument("Matrix shape mismatch.");}
    void CheckSquare() const {if (rows_ != cols_) throw std::invalid_argument("Matrix must be square.");}
    void SwapRows(std::size_t l, std::size_t r) {if(l == r) return; for (std::size_t col = 0; col < cols_; ++col){std::swap((*this)(l, col), (*this)(r, col));}}
    std::size_t Index(std::size_t row, std::size_t col) const {if (row >= rows_ || col >= cols_) throw std::out_of_range("Matrix index out of range.");}
    std::size_t rows_;
    std::size_t cols_;
    std::vector<T> data_;
};

}

#endif  // FIELD_LINALG_MATRIX_H_
