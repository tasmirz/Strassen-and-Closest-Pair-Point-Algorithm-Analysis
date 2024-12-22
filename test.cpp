#include <chrono>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <vector>
#include <fstream>

#include "DCClosestPairPoint.cpp"
#include "MatrixParallel.cpp"
#include "MatrixRegular.cpp"
#include "MatrixStrassen.cpp"
#include "RClosestPairPoint.cpp"

// Function to benchmark the execution time of a given function
template <typename Func, typename... Args>
 double benchmark(Func func, Args&&... args)  {
  auto start = std::chrono::high_resolution_clock::now();  // Start time
  auto result = func(std::forward<Args>(args)...);         // Execute the function
  auto end = std::chrono::high_resolution_clock::now();    // End time
  std::chrono::duration<double> elapsed = end - start;     // Calculate elapsed time
  std::cout << "Elapsed time: " << elapsed.count() << " seconds\n";  // Print elapsed time
  return elapsed.count();          // Return the result and elapsed time
}

using namespace std;

vector<vector<int>> generateRandomMatrix(int dim) {
  vector<vector<int>> mat(dim, vector<int>(dim));
  for (int i = 0; i < dim; ++i)
    for (int j = 0; j < dim; ++j) mat[i][j] = rand() % 1000000;
  return mat;
}

void benchmarkMatrixMultiplication() {
  ofstream csvFile("matrix_benchmark.csv");
  csvFile << "Dimension,Optimizer,Regular,Strassen,ParallelStrassen\n";

  vector<int> dimensions = {32, 64, 128, 256, 512, 1024};
  vector<int> optimizers = {0,32, 64,128,256,512};

  for (int optimizer : optimizers) {
    MatrixStrassen::optimizer = optimizer;
    MatrixParallel::optimizer = optimizer;

    for (int dim : dimensions) {
      cout << "Benchmarking matrix multiplication with dimension " << dim
           << " and optimizer " << optimizer << "...\n";
      if (dim>256 && optimizer==0) continue;
      vector<vector<int>> matA = generateRandomMatrix(dim);
      vector<vector<int>> matB = generateRandomMatrix(dim);

      MatrixStrassen matrixA(matA, dim);
      MatrixStrassen matrixB(matB, dim);
      MatrixParallel matrixPA(matA, dim);
      MatrixParallel matrixPB(matB, dim);
      MatrixRegular matrixRA(matA);
      MatrixRegular matrixRB(matB);

      cout << "\tRegular matrix multiplication:\n";
      auto regularResult = benchmark([&]() { matrixRA * matrixRB; return 0; });

      cout << "\tStrassen matrix multiplication:\n";
      auto strassenResult = benchmark([&]() { matrixA * matrixB; return 0; });

      cout << "\tParallel Strassen matrix multiplication:\n";
      auto parallelStrassenResult = benchmark([&]() { matrixPA * matrixPB; return 0; });

      csvFile << dim << "," << optimizer << "," << regularResult << ","
              << strassenResult << "," << parallelStrassenResult << "\n";
    }
  }

  csvFile.close();
}

void benchmarkClosestPairAlgorithms() {
  ofstream csvFile("closest_pair_benchmark.csv");
  csvFile << "NumPoints,DC,DCYPreSorted,DCParallelized,DCAllThreaded,NSquare\n";

  vector<int> numPointsList = {100, 1000, 10000, 100000};  // Different number of points to test

  for (int numPoints : numPointsList) {
    vector<Point> points;       // Vector to store points
    points.reserve(numPoints);  // Reserve space for points

    // Generate random points
    for (int i = 0; i < numPoints; ++i) {
      double x = rand() % 100007;   // Generate random x coordinate
      double y = rand() % 100007;   // Generate random y coordinate
      points.emplace_back(x, y);  // Add point to vector
    }

    vector<Point> pointsCopy = points;  // Copy of points vector

    // Benchmark different closest pair algorithms
    cout << "Benchmarking closest pair algorithms with " << numPoints << " points...\n";
    cout << "\tDivide and Conquer closest pair algorithm:\n";
    auto closestPairResultDC = benchmark(DCClosestPairPoints, pointsCopy);
    pointsCopy = points;
    cout << "\tDivide and Conquer closest pair algorithm with pre-sorted y coordinates:\n";
    auto closestPairYResultYPreSorted = benchmark(DCClosestPairPointsY, pointsCopy);
    pointsCopy = points;
    cout << "\tDivide and Conquer closest pair algorithm with parallelization:\n";
    auto closestPairResultParallelized = benchmark(DCClosestPairPointsP, pointsCopy);
    pointsCopy = points;
    cout << "\tDivide and Conquer closest pair algorithm with all threads:\n";
    auto closestPairResultAllThreaded = benchmark(DCClosestPairPointsFP, pointsCopy);
    pointsCopy = points;
    cout << "\tO(n^2) closest pair algorithm:\n";
    auto closestPairResultNSquare = benchmark(RClosestPairPoint, pointsCopy);

    // Write results to CSV
    csvFile << numPoints << "," << closestPairResultDC << ","
            << closestPairYResultYPreSorted << ","
            << closestPairResultParallelized << ","
            << closestPairResultAllThreaded << ","
            << closestPairResultNSquare << "\n";
  }

  csvFile.close();
}

int main() {
  benchmarkClosestPairAlgorithms();
  benchmarkMatrixMultiplication();
  return 0;
}
