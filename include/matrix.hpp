#pragma once
#include <cassert>
#include <cmath>
#include <fstream>
#include <vector>

#include "types.h"

#ifndef ROW_MAJOR
#ifndef COL_MAJOR
#define ROW_MAJOR
#endif
#endif

class Matrix {
  public:
    Matrix(size_t rows, size_t cols) : rows_(rows), cols_(cols), data_(rows * cols) {}

    f64 *data() { return data_.data(); }
    const f64 *data() const { return data_.data(); }
    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }

    f64 &at(size_t x, size_t y) {
        assert(x <= rows_ && y <= cols_);
#ifdef ROW_MAJOR
        return data_[x * cols_ + y];
#else
        return data[y * rows_ + x];
#endif
    }

    const f64 &at(size_t x, size_t y) const {
        assert(x <= rows_ && y <= cols_);
#ifdef ROW_MAJOR
        return data_[x * cols_ + y];
#else
        return data_[y * rows_ + x];
#endif
    }

    bool operator==(const Matrix &rhs) const {
        const f64 eps = 1e-6;
        if (rows_ != rhs.rows() || cols_ != rhs.cols())
            return false;
        for (size_t i = 0; i < rows_ * cols_; i++) {
            if (std::abs(data_[i] - rhs.data()[i]) > eps)
                return false;
        }
        return true;
    }

    static Matrix from_file(const char *path) {
        std::ifstream infile(path);
        assert(infile && "Cannot open target file.");

        size_t rows, cols;
        infile >> rows >> cols;

        Matrix mat(rows, cols);
        for (size_t i = 0; i < rows; i++) {
            for (size_t j = 0; j < cols; j++) {
                f64 val;
                infile >> val;
                assert(!infile.fail() && "Error when reading matrix data.");
                mat.at(i, j) = val;
            }
        }

        return mat;
    }

  private:
    size_t rows_;
    size_t cols_;
    std::vector<f64> data_;
};