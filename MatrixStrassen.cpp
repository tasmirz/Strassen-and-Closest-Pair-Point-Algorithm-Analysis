#ifndef MATRIX_STRASSEN_CPP
#define MATRIX_STRASSEN_CPP
#include <future>
#include <stdexcept>
#include <vector>

using namespace std;

struct MatrixStrassen {
  vector<vector<int>> mat;
  int dim = 0;
  static int optimizer;

  MatrixStrassen() {}

  MatrixStrassen(int dim) : dim(dim) {
    mat = std::vector<std::vector<int>>(this->dim,
                                        std::vector<int>(this->dim, 0));
  }

  MatrixStrassen(vector<vector<int>> mat, int dim) : mat(mat), dim(dim) {
    int _dim = dim == 1 ? 1 : 1 << (32 - __builtin_clz(dim - 1));

    if (_dim != dim) {
      vector<vector<int>> _mat(_dim, vector<int>(_dim, 0));
      for (int i = 0; i < dim; i++)
        for (int j = 0; j < dim; j++) _mat[i][j] = mat[i][j];

      mat = _mat;
      this->dim = _dim;
    }
  }

  MatrixStrassen mul(MatrixStrassen other) {
    std::vector<std::vector<int>> res(dim, std::vector<int>(dim));
    for (int i = 0; i < dim; i++)
      for (int j = 0; j < dim; j++)
        for (int k = 0; k < dim; k++) res[i][j] += mat[i][k] * other.mat[k][j];
    return MatrixStrassen(res, dim);
  }

  MatrixStrassen operator+(MatrixStrassen other) {
    vector<vector<int>> res(dim, vector<int>(dim));
    for (int i = 0; i < dim; i++)
      for (int j = 0; j < dim; j++) res[i][j] = mat[i][j] + other.mat[i][j];
    return MatrixStrassen(res, dim);
  }

  MatrixStrassen operator-(MatrixStrassen other) {
    vector<vector<int>> res(dim, vector<int>(dim));
    for (int i = 0; i < dim; i++)
      for (int j = 0; j < dim; j++) res[i][j] = mat[i][j] - other.mat[i][j];
    return MatrixStrassen(res, dim);
  }

  MatrixStrassen operator*(MatrixStrassen other) {
    if (dim != other.dim) throw runtime_error("Invalid matrix multiplication");

    if (dim == 1) return MatrixStrassen({{mat[0][0] * other.mat[0][0]}}, 1);
    if (dim <= optimizer) return mul(other);
    int m = dim / 2;

    MatrixStrassen A11(m), A12(m), A21(m), A22(m), B11(m), B12(m), B21(m),
        B22(m);
    // division of matrix
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < m; j++) {
        A11.mat[i][j] = mat[i][j];
        A12.mat[i][j] = mat[i][j + m];
        A21.mat[i][j] = mat[i + m][j];
        A22.mat[i][j] = mat[i + m][j + m];
        B11.mat[i][j] = other.mat[i][j];
        B12.mat[i][j] = other.mat[i][j + m];
        B21.mat[i][j] = other.mat[i + m][j];
        B22.mat[i][j] = other.mat[i + m][j + m];
      }
    }

    // Formula calculation
    MatrixStrassen q1 = A11 + A22;
    MatrixStrassen q2 = B11 + B22;
    MatrixStrassen M1 = q1 * q2;
    MatrixStrassen M2 = (A21 + A22) * B11;
    MatrixStrassen M3 = A11 * (B12 - B22);
    MatrixStrassen M4 = A22 * (B21 - B11);
    MatrixStrassen M5 = (A11 + A12) * B22;
    MatrixStrassen M6 = (A21 - A11) * (B11 + B12);
    MatrixStrassen M7 = (A12 - A22) * (B21 + B22);
    MatrixStrassen C11 = (M1 + M4 - M5 + M7);
    MatrixStrassen C12 = (M3 + M5);
    MatrixStrassen C21 = (M2 + M4);
    MatrixStrassen C22 = (M1 - M2 + M3 + M6);

    // Merge the results to get the final matrix
    MatrixStrassen res(dim);
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < m; j++) {
        res.mat[i][j] = C11.mat[i][j];
        res.mat[i][j + m] = C12.mat[i][j];
        res.mat[i + m][j] = C21.mat[i][j];
        res.mat[i + m][j + m] = C22.mat[i][j];
      }
    }
    return res;
  }
};
int MatrixStrassen::optimizer = 0;
#endif