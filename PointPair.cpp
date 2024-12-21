#include "Point.cpp"
#ifndef POINT_PAIR_CPP
struct PointPair {
    Point first;
    Point second;
    static Point nillpoint;

    PointPair() : first(nillpoint), second(nillpoint) {}
    PointPair(Point p1, Point p2) : first(p1), second(p2) {}
    PointPair(PointPair& other) : first(other.first), second(other.second) {}
    PointPair(const PointPair& other) : first(other.first), second(other.second) {}
    
    PointPair& operator=(const PointPair& other) {
        if (this != &other) {
            first = other.first;
            second = other.second;
        }
        return *this;
    }
    operator double() const {
        return first & second;   
    }
    bool operator<(PointPair & p) {
        return (first & second) < (p.first & p.second);
    }
};
Point PointPair::nillpoint = Point(0,0);
#define POINT_PAIR_CPP
#endif