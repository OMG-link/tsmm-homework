#pragma once
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>

#include "utils.h"

enum StoreMode { ROW_MAJOR, COL_MAJOR };

class Matrix {
  public:
    Matrix() = delete;
    ~Matrix() { free(data_); }
    Matrix(const Matrix &rhs) : store_mode_(rhs.store_mode_), rows_(rhs.rows_), cols_(rhs.cols_), data_(nullptr) {
        data_ = (f64 *)malloc_aligned((rows_ * cols_) * sizeof(f64), 64);
        memcpy(data_, rhs.data_, (rows_ * cols_) * sizeof(f64));
    }
    Matrix(Matrix &&rhs) : store_mode_(rhs.store_mode_), rows_(rhs.rows_), cols_(rhs.cols_), data_(rhs.data_) {
        rhs.rows_ = 0;
        rhs.cols_ = 0;
        rhs.data_ = nullptr;
    }
    Matrix &operator=(const Matrix &rhs) {
        free(data_);
        data_ = nullptr;
        store_mode_ = rhs.store_mode_;
        rows_ = rhs.rows_;
        cols_ = rhs.cols_;
        data_ = (f64 *)malloc_aligned((rows_ * cols_) * sizeof(f64), 64);
        memcpy(data_, rhs.data_, (rows_ * cols_) * sizeof(f64));
        return *this;
    }
    Matrix &operator=(Matrix &&rhs) {
        free(data_);
        data_ = nullptr;
        store_mode_ = rhs.store_mode_;
        rows_ = rhs.rows_;
        cols_ = rhs.cols_;
        data_ = rhs.data_;
        rhs.data_ = nullptr;
        return *this;
    }
    Matrix(size_t rows, size_t cols, StoreMode store_mode = ROW_MAJOR)
        : store_mode_(store_mode), rows_(rows), cols_(cols), data_(nullptr) {
        data_ = (f64 *)malloc_aligned((rows_ * cols_) * sizeof(f64), 64);
    }

    f64 *data() { return data_; }
    const f64 *data() const { return data_; }
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
        for (size_t i = 0; i < rows_; i++) {
            for (size_t j = 0; j < cols_; j++) {
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
    f64 *data_;
};