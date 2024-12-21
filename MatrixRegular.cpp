#include <vector>
#ifndef MATRIX_REGULAR_CPP
/*
A regular matrix implementation with O(n^3) complexity.
*/
struct MatrixRegular {
    std::vector<std::vector<int>> mat;
    int dim;
    MatrixRegular(std::vector<std::vector<int>> mat) : mat(mat), dim(mat.size()) {}
    MatrixRegular operator+(MatrixRegular other) {
        std::vector<std::vector<int>> res(dim, std::vector<int>(dim));
        for (int i = 0; i < dim; i++)
            for (int j = 0; j < dim; j++) res[i][j] = mat[i][j] + other.mat[i][j];
        return MatrixRegular(res);
    }
    MatrixRegular operator-(MatrixRegular other) {
        std::vector<std::vector<int>> res(dim, std::vector<int>(dim));
        for (int i = 0; i < dim; i++)
            for (int j = 0; j < dim; j++) res[i][j] = mat[i][j] - other.mat[i][j];
        return MatrixRegular(res);
    }
    MatrixRegular operator*(MatrixRegular other) {
        std::vector<std::vector<int>> res(dim, std::vector<int>(dim));
        for (int i = 0; i < dim; i++)
            for (int j = 0; j < dim; j++)
                for (int k = 0; k < dim; k++) res[i][j] += mat[i][k] * other.mat[k][j];
        return MatrixRegular(res);
    }
};
#define MATRIX_REGULAR_CPP
#endif