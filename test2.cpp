#include <iostream>
#include <algorithm>
#include <vector>

int main() {
    std::vector<int> sortedItems = {2, 4, 6, 8, 10, 12, 14, 16};
    int item = 16;

    auto it = std::upper_bound(sortedItems.begin(), sortedItems.end(), item);
    
    if (it != sortedItems.end()) {
        std::cout << "Upper bound of " << item << " is " << *it << std::endl;
    } else {
        std::cout << "No upper bound found for " << item << std::endl;
    }

    return 0;
}