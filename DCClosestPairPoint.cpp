#include <algorithm>
#include <future>
#include <queue>
#include <stack>
#include <utility>
#include <vector>

#include "Point.cpp"
#include "PointPair.cpp"

PointPair dccpp(std::vector<Point>& points, int l, int r);
PointPair dccppP(std::vector<Point>& points, int l, int r);
PointPair dccpp_ysorted(std::vector<Point>& points, std::vector<Point>& pointY,
                        int l, int r);

PointPair DCClosestPairPoints(std::vector<Point>& P) {
  // Sort the points by x coordinate
  std::sort(P.begin(), P.end());
  return dccpp(P, 0, P.size() - 1);
}
PointPair DCClosestPairPointsFP(std::vector<Point>& P) {
  // Sort the points by x coordinate, all divisions to thread
  std::sort(P.begin(), P.end());
  return dccppP(P, 0, P.size() - 1);
}
PointPair DCClosestPairPointsY(std::vector<Point>& P) {
  // Presorted x & y
  std::sort(P.begin(), P.end());
  std::vector<Point> pointY = P;
  std::sort(pointY.begin(), pointY.end(), Point());
  return dccpp_ysorted(P, pointY, 0, P.size() - 1);
}
PointPair DCClosestPairPointsP(std::vector<Point>& points) {
  // CPU core aware parallel divide and conquer
  // Sort the points by x coordinate
  std::sort(points.begin(), points.end());
  int n = points.size();
  int num_threads = std::thread::hardware_concurrency();
  int chunk_size = n / num_threads;

  std::vector<std::future<PointPair>> futures;
  // broken to number of hardware threads
  for (int i = 0; i < num_threads; ++i) {
    int start = i * chunk_size;
    int end = (i == num_threads - 1) ? n - 1 : (i + 1) * chunk_size - 1;
    futures.push_back(
        std::async(std::launch::async, dccpp, std::ref(points), start, end));
  }

  // get smallest distance from all threads
  PointPair smallest = futures[0].get();
  for (int i = 1; i < num_threads; ++i) {
    PointPair result = futures[i].get();
    if (result < smallest) smallest = result;
  }

  double d = smallest;
  std::vector<Point> strip;

  // merge the results from each strip
  for (int i = 0; i < num_threads - 1; i++) {
    int l = i * chunk_size;
    int r = (i == num_threads - 2) ? n - 1 : (i + 2) * chunk_size - 1;
    int m = (l + r) / 2;
    auto start = points.begin() + l;
    auto end = points.begin() + r + 1;
    auto lower = std::lower_bound(start, end, Point(points[m].x - d, 0));
    start = lower;
    auto upper = std::upper_bound(start, end, Point(points[m].x + d, 0));
    std::vector<Point> strip(lower, upper);
    std::sort(strip.begin(), strip.end(), Point());
    for (int i = 0; i < (int)strip.size(); i++)
      for (int j = i + 1; j < 8+i && j < (int)strip.size(); j++)
        if ((strip[i] & strip[j]) < d) d = smallest = {strip[i], strip[j]};
  }

  return smallest;
}
PointPair dccpp(std::vector<Point>& points, int l, int r) {
  if (r - l == 1) return {points[l], points[r]};  // O(1)

  if (r - l == 2) {  // O(1)

    double d1 = points[l + 1] & points[l];
    double d2 = points[l] & points[r];
    double d3 = points[l + 1] & points[r];

    if (d1 < d2 && d1 < d3)
      return {points[l], points[l + 1]};
    else if (d2 < d3)
      return {points[l], points[r]};
    else
      return {points[l + 1], points[r]};
  }

  int m = (l + r) / 2;

  PointPair smallest = std::min(dccpp(points, l, m), dccpp(points, m + 1, r));
  double d = smallest;

  // Divide and search
  auto start = points.begin() + l;
  auto end = points.begin() + r;
  auto lower =
      std::lower_bound(start, end, Point(points[m].x - d, 0));  // O(log n)
  start = lower;
  auto upper =
      std::upper_bound(start, end, Point(points[m].x + d, 0));  // O(log n)
  if (upper == points.end()) upper = end;
  std::vector<Point> strip(lower, upper);
  // Sort the strip by y coordinate.
  std::sort(strip.begin(), strip.end(),
            Point());  // O(n log n) - n is the size of the strip: which can be
                       // at most (l-r)/2

  // Search the strip
  /*
   * The smallest distance is d
   *  In the X axis, the distance between two points is d, i.e. from the middle
   * point there can be shorterdistances in a d radius circle However moving
   * from up to down we do not need to consider all points, only the ones that
   * are below is enough -  as the points are sorted by y coordinate. For
   * simplicity, We search a rectangle instead of a circle. Now the dimention of
   * the rectangle is 2d x d. There
   */
  for (int i = 0; i < (int)strip.size(); i++)                     // O(n)
    for (int j = i + 1; j < i + 8 && j < (int)strip.size(); j++)  // O(1)
      if ((strip[i] & strip[j]) < d) d = smallest = {strip[i], strip[j]};

  return smallest;
}

PointPair dccppP(std::vector<Point>& points, int l, int r) {
  if (r - l == 1) return {points[l], points[r]};  // O(1)

  if (r - l == 2) {  // O(1)

    double d1 = points[l] & points[l + 1];
    double d2 = points[l] & points[r];
    double d3 = points[l + 1] & points[r];

    if (d1 < d2 && d1 < d3)
      return {points[l], points[l + 1]};
    else if (d2 < d3)
      return {points[l], points[r]};
    else
      return {points[l + 1], points[r]};
  }
  int m = (l + r) / 2;

auto future_left = std::async(std::launch::async, dccppP, std::ref(points), l, m);
auto future_right = std::async(std::launch::async, dccppP, std::ref(points), m + 1, r);
PointPair smallest = std::min(future_left.get(), future_right.get());
  double d = smallest;

  // Divide and search
  auto start = points.begin() + l;
  auto end = points.begin() + r + 1;
  auto lower =
      std::lower_bound(start, end, Point(points[m].x - d, 0));  // O(log n)
  start = lower;
  auto upper =
      std::upper_bound(start, end, Point(points[m].x + d, 0));  // O(log n)
  if (upper == points.end()) upper = end;
  std::vector<Point> strip(lower, upper);
  // Sort the strip by y coordinate.
  std::sort(strip.begin(), strip.end(),
            Point());  // O(n log n) - n is the size of the strip: which can be
                       // at most (l-r)/2

  // Search the strip
  /*
   * The smallest distance is d
   *  In the X axis, the distance between two points is d, i.e. from the middle
   * point there can be shorterdistances in a d radius circle However moving
   * from up to down we do not need to consider all points, only the ones that
   * are below is enough -  as the points are sorted by y coordinate. For
   * simplicity, We search a rectangle instead of a circle. Now the dimention of
   * the rectangle is 2d x d. There
   */
  for (int i = 0; i < (int)strip.size(); i++)                     // O(n)
    for (int j = i + 1; j < i + 8 && j < (int)strip.size(); j++)  // O(1)
      if ((strip[i] & strip[j]) < d) d = smallest = {strip[i], strip[j]};

  return smallest;
}

PointPair dccpp_ysorted(std::vector<Point>& points, std::vector<Point>& pointY,
                        int l, int r) {
  if (r - l == 1) return {points[l], points[r]};
  if (r - l == 2) {
    double d1 = points[l] & points[l + 1];
    double d2 = points[l] & points[r];
    double d3 = points[l + 1] & points[r];
    if (d1 < d2 && d1 < d3)
      return {points[l], points[l + 1]};
    else if (d2 < d3)
      return {points[l], points[r]};
    else
      return {points[l + 1], points[r]};
  }
  int m = (l + r) / 2;
  PointPair smallest = std::min(dccpp(points, l, m), dccpp(points, m + 1, r));
  double d = smallest;
  std::vector<Point> strip;
  for (int i = 0; i < pointY.size(); i++) {
    if (pointY[i].x >= points[m].x - d && pointY[i].x <= points[m].x + d)
      strip.push_back(pointY[i]);
  }
  for (int i = 0; i < (int)strip.size(); i++)
    for (int j = i + 1; j < 8 && j < (int)strip.size(); j++)
      if ((strip[i] & strip[j]) < d) d = smallest = {strip[i], strip[j]};
  return smallest;
}
