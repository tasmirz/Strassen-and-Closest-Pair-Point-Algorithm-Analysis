test: *.cpp
	g++ -g -o test test.cpp -Wall -Wextra -std=c++23
test-run: test
	./test