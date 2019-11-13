#include <iostream>
#include <functional>
#include <algorithm>

#include "data.h"

bool test1() {
    MklAllocator<float> f;
    f.allocate(10);
    return true;
}

typedef std::function<bool()> TestFunc;
void check_test(std::string desc, const TestFunc& test_func) {
    bool success = false;
    try
    {
        success = test_func();
    }
    catch (const std::exception& e)
    {
        std::cout << "Failed test: " << desc.c_str() << std::endl << e.what();
        return;
    }
    if (success) {
        std::cout << "Success test: " << desc.c_str() << std::endl;
    }
    else {
        std::cout << "Failed test: " << desc.c_str() << std::endl;
    }
}
int main() {
    //check_test("check test", test1);
    return 0;
}