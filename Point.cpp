#ifndef POINT_CPP
#include <cmath>
struct Point {
    double x;
    double  y;
    Point() : x(0), y(0) {}
    Point(double x, double y) : x(x), y(y) {}
    double operator &(const Point& p) const {
        return sqrt((x-p.x)*(x-p.x)  + (y-p.y)*(y-p.y));
    }
    bool operator<(const Point& p) const {
        return x < p.x;
    }  
    bool operator()(const Point& p1, const Point& p2) const {
        return p1.y < p2.y;
    }
};

#define POINT_CPP
#endif