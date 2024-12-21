#include "Point.cpp"
#include "PointPair.cpp"
#include <vector>
#include <utility>

PointPair RClosestPairPoint(std::vector<Point> & P) {
    double min = P[0] & P[1];
    PointPair res={P[0], P[1]};
    for (int i = 0; i < P.size(); i++) 
        for (int j = i+1; j < P.size(); j++) 
            if ((P[i] & P[j]) < min && i!=j) {
                min = P[i] & P[j];
                res = {P[i], P[j]};
            }
    return res;
}
