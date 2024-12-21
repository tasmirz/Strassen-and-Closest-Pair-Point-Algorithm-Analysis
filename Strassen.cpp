/*
    `using namespace std` was not used to keep this code formal.
*/
#include <tuple>
template <typename T>
class MatrixInterface {
    public:
    T inline abstract operator()(int i, int j) const  = 0;
    std::tuple<int,int>inline abstract shape() const = 0;
    MatrixInterface<T> operator*(MatrixInterface& m) {
        
    }
    MatrixInterface<T> operator+(MatrixInterface& m) {
        Matrix res(rows, cols);
        for (int i = 0; i < rows; i++) 
            for (int j = 0; j < cols; j++) 
                res(i, j) = (*this)(i, j) + m(i, j);
    }
};
template <typename T, int rows, int cols>
class MatrixImmutable : public MatrixInterface<T> {
    T data[rows][cols];
    
    public:
    std::tuple<int,int> inline shape() const {
        return make_tuple(rows, cols);
    }
    MatrixImmutable(T data[rows][cols]) {
        this->data = data;
    }
    T inline const operator()(int i, int j) {
        return data[i][j];
    }
};
template <typename T>
class Matrix : public MatrixInterface<T> {
    vector<vector<T>> data;
    public:
    std::tuple<int,int> inline shape() const {
        return make_tuple(data.size(), data[0].size());
    }
    Matrix(vector<vector<T>> data) {
        this->data = data;
    }
    inline operator()(int i, int j) {
        return data[i][j];
    }
};