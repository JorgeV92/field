#ifndef FIELD_LINALG_TENSOR3_H_
#define FIELD_LINALG_TENSOR3_H_

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace field::linalg {

template<typename T> class Tensor3 {
public:
    Tensor3() : dim0_(0), dim1_(0), dim2_(0) {}
    Tensor3(std::size_t d0, std::size_t d1, std::size_t d2, const T& v = T()) 
        : dim0_(d0), dim1_(d1), dim2_(d2), data_(d0*d1*d2, v) {}
    std::size_t Dim0() const {return dim0_;}
    std::size_t Dim1() const {return dim1_;}
    std::size_t Dim2() const {return dim2_;}
    T& operator()(std::size_t i, std::size_t j, std::size_t k) {return data_.at(Index(i,j,k));}
    const T& operator()(std::size_t i, std::size_t j, std::size_t k) const {return data_.at(Index(i,j,k));}
private:
    std::size_t Index(std::size_t i, std::size_t j, std::size_t k) const {
        if (i >= dim0_ ||j >= dim1_ || k >= dim2_) throw std::out_of_range("Tensor3 index out of range.");
        return (i * dim1_ + j) * dim2_ + k;
    }
    std::size_t dim0_;
    std::size_t dim1_;
    std::size_t dim2_;
    std::vector<T> data_;
};

}

#endif  // FIELD_LINALG_TENSOR3_H_