#include <iostream>

int main() {
    unsigned int x = 1; // Example number
    int leading_zeros = (x == 0) ? 32 : 32 - __builtin_clz(x - 1);

    std::cout << "Number of leading zeros in " << x << " is: " << leading_zeros << std::endl;

    return 0;
}