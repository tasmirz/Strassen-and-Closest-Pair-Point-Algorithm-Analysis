#include <iostream>
#include "DCClosestPairPoint.cpp"
#include "RClosestPairPoint.cpp"

#include <chrono>
#include <functional>
#include <iostream>

template <typename Func, typename... Args>
auto benchmark(Func func, Args&&... args) -> decltype(func(std::forward<Args>(args)...)) {
    auto start = std::chrono::high_resolution_clock::now();
    auto result = func(std::forward<Args>(args)...);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Elapsed time: " << elapsed.count() << " seconds\n";
    return result;
}
using namespace std;
int main () {
    int kk = 661493;
     vector<Point>P;//={{2, 3}, {12, 30}, {40, 50}, {5, 1}, {12, 10}, {3, 4}};
    srand(kk);
    //cout<<"Test "<<kk<<endl;
     int k =45;
     P.reserve(k);
    for(int i = 0; i < k; ++i) {
        double x = rand() % 10000007;
        double y = rand() % 1000007;
        P.emplace_back(x, y);
    }
    PointPair p = benchmark(DCClosestPairPoints, P);
    PointPair q = benchmark(DCClosestPairPointsY, P);
    PointPair qq = benchmark(DCClosestPairPointsP,P);
    PointPair qqq = benchmark(DCClosestPairPointsFP,P);
    PointPair r = benchmark(RClosestPairPoint, P);
    cout << p << " " << q<<" "<<qq<<" "<<qqq<<" "<<r << endl;
}