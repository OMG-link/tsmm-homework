#pragma once
#include <cassert>
#include <cmath>
#include <fstream>
#include <vector>

#include "types.h"

enum StoreMode { ROW_MAJOR, COL_MAJOR };

class Matrix {
  public:
    Matrix() = delete;
    Matrix(size_t rows, size_t cols, StoreMode store_mode = ROW_MAJOR)
        : rows_(rows), cols_(cols), data_(rows * cols), store_mode_(store_mode) {}

    f64 *data() { return data_.data(); }
    const f64 *data() const { return data_.data(); }
    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }
    StoreMode store_mode() const { return store_mode_; }

    f64 &at(size_t x, size_t y) {
        assert(x <= rows_ && y <= cols_);
        if (store_mode_ == ROW_MAJOR) {
            return data_[x * cols_ + y];
        } else {
            return data_[y * rows_ + x];
        }
    }

    const f64 &at(size_t x, size_t y) const {
        assert(x <= rows_ && y <= cols_);
        if (store_mode_ == ROW_MAJOR) {
            return data_[x * cols_ + y];
        } else {
            return data_[y * rows_ + x];
        }
    }

    void set_store_mode(StoreMode store_mode) {
        if (store_mode == store_mode_)
            return;
        Matrix result(rows_, cols_, store_mode);
        for (int i = 0; i < rows_; i++) {
            for (int j = 0; j < cols_; j++) {
                result.at(i, j) = this->at(i, j);
            }
        }
        *this = std::move(result);
        return;
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
    StoreMode store_mode_;
    size_t rows_;
    size_t cols_;
    std::vector<f64> data_;
};