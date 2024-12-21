#include <chrono>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <vector>

#include "DCClosestPairPoint.cpp"
#include "MatrixParallel.cpp"
#include "MatrixRegular.cpp"
#include "MatrixStrassen.cpp"
#include "RClosestPairPoint.cpp"

// Function to benchmark the execution time of a given function
template <typename Func, typename... Args>
auto benchmark(Func func,
               Args&&... args) -> decltype(func(std::forward<Args>(args)...)) {
  auto start = std::chrono::high_resolution_clock::now();  // Start time
  auto result = func(std::forward<Args>(args)...);       // Execute the function
  auto end = std::chrono::high_resolution_clock::now();  // End time
  std::chrono::duration<double> elapsed =
      end - start;  // Calculate elapsed time
  std::cout << "Elapsed time: " << elapsed.count()
            << " seconds\n";  // Print elapsed time
  return result;              // Return the result of the function
}

using namespace std;

vector<vector<int>> generateRandomMatrix(int dim) {
  vector<vector<int>> mat(dim, vector<int>(dim));
  for (int i = 0; i < dim; ++i)
    for (int j = 0; j < dim; ++j) mat[i][j] = rand() % 100;
  return mat;
}

void benchmarkMatrixMultiplication() {
  vector<int> dimensions = {32, 64, 128, 256, 512, 1024};
  vector<int> optimizers = {0, 32, 64};

  for (int optimizer : optimizers) {
    MatrixStrassen::optimizer = optimizer;
    MatrixParallel::optimizer = optimizer;

    for (int dim : dimensions) {
      cout << "Benchmarking matrix multiplication with dimension " << dim
           << " and optimizer " << optimizer << "...\n";

      vector<vector<int>> matA = generateRandomMatrix(dim);
      vector<vector<int>> matB = generateRandomMatrix(dim);

    MatrixStrassen matrixA(matA, dim);
    MatrixStrassen matrixB(matB, dim);
    MatrixParallel matrixPA(matA, dim);
    MatrixParallel matrixPB(matB, dim);
    MatrixRegular matrixRA(matA);
    MatrixRegular matrixRB(matB);

    cout << "\tRegular matrix multiplication:\n";
    benchmark([&]() { return matrixRA * matrixRB; });

    cout << "\tStrassen matrix multiplication:\n";
    benchmark([&]() { return matrixA * matrixB; });

    cout << "\tParallel Strassen matrix multiplication:\n";
    benchmark([&]() { return matrixPA * matrixPB; });


    }
  }
}

void benchmarkClosestPairAlgorithms() {
    vector<int> numPointsList = {100, 1000, 10000, 1000000};  // Different number of points to test

    for (int numPoints : numPointsList) {
        vector<Point> points;       // Vector to store points
        points.reserve(numPoints);  // Reserve space for points

        // Generate random points
        for (int i = 0; i < numPoints; ++i) {
            double x = rand() % 1007;   // Generate random x coordinate
            double y = rand() % 1007;   // Generate random y coordinate
            points.emplace_back(x, y);  // Add point to vector
        }

        vector<Point> pointsCopy = points;  // Copy of points vector

        // Benchmark different closest pair algorithms
        cout << "Benchmarking closest pair algorithms with " << numPoints << " points...\n";
        cout << "\tDivide and Conquer closest pair algorithm:\n";
        PointPair closestPairResultDC = benchmark(DCClosestPairPoints, pointsCopy);
        pointsCopy = points;
        cout << "\tDivide and Conquer closest pair algorithm with pre-sorted y coordinates:\n";
        PointPair closestPairYResultYPreSorted = benchmark(DCClosestPairPointsY, pointsCopy);
        pointsCopy = points;
        cout << "\tDivide and Conquer closest pair algorithm with parallelization:\n";
        PointPair closestPairResultParallelized = benchmark(DCClosestPairPointsP, pointsCopy);
        pointsCopy = points;
        cout << "\tDivide and Conquer closest pair algorithm with all threads:\n";
        PointPair closestPairResultAllThreaded = benchmark(DCClosestPairPointsFP, pointsCopy);
        pointsCopy = points;
        cout << "\tO(n^2) closest pair algorithm:\n";
        PointPair closestPairResultNSquare = benchmark(RClosestPairPoint, pointsCopy);

        // Print results
        cout << closestPairResultDC << " " << closestPairYResultYPreSorted << " "
                 << closestPairResultParallelized << " " << closestPairResultAllThreaded
                 << " " << closestPairResultNSquare << endl;
    }
}

int main() {
  benchmarkClosestPairAlgorithms();
  benchmarkMatrixMultiplication();
  return 0;
}
