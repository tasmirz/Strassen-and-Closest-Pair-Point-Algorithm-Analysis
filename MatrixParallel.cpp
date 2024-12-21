#include <vector>
#include <stdexcept>
#include <future>
#include "MatrixStrassen.cpp"
using namespace std;

struct MatrixParallel {
    vector<vector<int>> mat;
    int dim = 0;
    static int optimizer;
    
    MatrixParallel() {}

    MatrixParallel(int dim) {
        this->dim = 1 << (32 - __builtin_clz(dim - 1));
        mat = vector<vector<int>>(this->dim, vector<int>(this->dim, 0));
    }

    MatrixParallel(vector<vector<int>> mat, int dim) : mat(mat), dim(dim) {
        int _dim = 1 << (32 - __builtin_clz(dim - 1));
        if (_dim != dim) {
            vector<vector<int>> _mat(_dim, vector<int>(_dim, 0));
            for (int i = 0; i < dim; i++) 
                for (int j = 0; j < dim; j++) 
                    _mat[i][j] = mat[i][j];
            mat = _mat;
            this->dim = _dim;
        }
    }

    MatrixParallel operator+(MatrixParallel other) {
        vector<vector<int>> res(dim, vector<int>(dim));
        for (int i = 0; i < dim; i++)
            for (int j = 0; j < dim; j++) res[i][j] = mat[i][j] + other.mat[i][j];
        return MatrixParallel(res, dim);
    }

    MatrixParallel operator-(MatrixParallel other) {
        vector<vector<int>> res(dim, vector<int>(dim));
        for (int i = 0; i < dim; i++)
            for (int j = 0; j < dim; j++) res[i][j] = mat[i][j] - other.mat[i][j];
        return MatrixParallel(res, dim);
    }

    MatrixParallel operator*(MatrixParallel other) {
        if (dim != other.dim) throw runtime_error("Invalid matrix multiplication");
        if (dim == 1) 
                return MatrixParallel({{mat[0][0] * other.mat[0][0]}}, 1);
        if (dim <= optimizer) return mul(other);
        int m = dim / 2;

        MatrixStrassen A11(m), A12(m), A21(m), A22(m);
        MatrixStrassen B11(m), B12(m), B21(m), B22(m);

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

        

        
            auto fut1 = async(launch::async, [&]() { return (A11 + A22) * (B11 + B22); });
            auto fut2 = async(launch::async, [&]() { return (A21 + A22) * B11; });
            auto fut3 = async(launch::async, [&]() { return A11 * (B12 - B22); });
            auto fut4 = async(launch::async, [&]() { return A22 * (B21 - B11); });
            auto fut5 = async(launch::async, [&]() { return (A11 + A12) * B22; });
            auto fut6 = async(launch::async, [&]() { return (A21 - A11) * (B11 + B12); });
            auto fut7 = async(launch::async, [&]() { return (A12 - A22) * (B21 + B22); });
            MatrixStrassen M1 = fut1.get();
            MatrixStrassen M2 = fut2.get();
            MatrixStrassen M3 = fut3.get();
            MatrixStrassen M4 = fut4.get();
            MatrixStrassen M5 = fut5.get();
            MatrixStrassen M6 = fut6.get();
            MatrixStrassen M7 = fut7.get();
            auto futC11 = async(launch::async, [&]() { return (M1 + M4 - M5 + M7); });
            auto futC12 = async(launch::async, [&]() { return (M3 + M5); });
            auto futC21 = async(launch::async, [&]() { return (M2 + M4); });
            auto futC22 = async(launch::async, [&]() { return (M1 - M2 + M3 + M6); });
            MatrixStrassen C11 = futC11.get();
            MatrixStrassen C12 = futC12.get();
            MatrixStrassen C21 = futC21.get();
            MatrixStrassen C22 = futC22.get();

            M1 = (A11 + A22) * (B11 + B22);
            M2 = (A21 + A22) * B11;
            M3 = A11 * (B12 - B22);
            M4 = A22 * (B21 - B11);
            M5 = (A11 + A12) * B22;
            M6 = (A21 - A11) * (B11 + B12);
            M7 = (A12 - A22) * (B21 + B22);
            C11 = (M1 + M4 - M5 + M7);
            C12 = (M3 + M5);
            C21 = (M2 + M4);
            C22 = (M1 - M2 + M3 + M6);

        MatrixParallel res(dim);
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
