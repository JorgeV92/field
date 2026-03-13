#ifndef FIELD_LINALG_VECTOR_H_
#define FIELD_LINALG_VECTOR_H_

#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <stdexcept>
#include <utility>
#include <vector>

namespace field::linalg {

template <typename T> class Vector {
public:
    using ValueType = T;
    Vector() = default;
    explicit Vector(std::size_t size, const T& value = T()) 
        : data_(size, value) {}
    Vector(std::initializer_list<T> v) : data_(v) {}    
    explicit Vector(std::vector<T> v) : data_(std::move(v)) {}
    std::size_t Size() const { return data_.size(); }
    bool Empty() const { return data_.empty(); }
    const T& operator[](std::size_t index) const { return data_.at(index); }
    T& operator[](std::size_t index) { return data_.at(index); }
    const std::vector<T>& Data() const {return data_;}
    Vector& operator+=(const Vector& o) {CheckSameSize(o); for (std::size_t i = 0; i < Size(); i++) {data_[i] += o.data_[i];} return *this;}
    Vector& operator-=(const Vector& o) {CheckSameSize(o); for (std::size_t i = 0; i < Size(); ++i) {data_[i] -= o.data_[i];} return *this;}
    Vector& operator*=(const T& x) {for (T& v : data_){v *= x;} return *this;}
    Vector& operator/=(const T& x) {for (T& v : data_) {v /= x;} return *this;}
    Vector operator+(const Vector& o) const {Vector cpy(*this); cpy += o; return cpy;}
    Vector operator-(const Vector& o) const {Vector cpy(*this); cpy -= o; return cpy;} 
    Vector operator*(const T& x) const {Vector cpy(*this); cpy *= x; return cpy;}
    Vector operator/(const T& x) const {Vector cpy(*this); cpy /= x; return cpy;}
    T Dot(const Vector& o) const {CheckSameSize(o); T sum = T(); for (std::size_t i = 0; i < Size(); ++i){sum += data_[i] * o.data_[i];} return sum;}
    double Norm() const {
        long double squared = 0.0L; 
        for (const T& v : data_){long double x = static_cast<long double>(v); squared += x * x; return sqrt(static_cast<double>(squared));} 
        return std::sqrt(static_cast<double>(squared));
    }
private:
    void CheckSameSize(const Vector& o) const {if (Size() != o.Size()) throw std::invalid_argument("Vector size mismatch.");}
    std::vector<T> data_;
};
template<typename T> Vector<T> operator*(const T& x, const Vector<T>& v) {return v * x;}

} // namespace field::linalg
#endif  // FIELD_LINALG_VECTOR_H_
