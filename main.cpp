#include <omp.h>

#include <cassert>
#include <chrono>
#include <cmath>
#include <functional>
#include <iostream>

constexpr int NUM_THREADS  = 8;
constexpr int NUM_POINTS   = 10'000'000;
constexpr double PRECISION = 1e-8;

class OutputPrinter {
public:
    explicit OutputPrinter(std::chrono::duration<double> duration, double value)
        : _duration(duration), _value(value) {}

    void display() const {
        std::cout << "Результат: " << _value << "\n";
        std::cout << "Время: " << _duration.count() << " секунд\n";
    }

    double getValue() const {
        return _value;
    }

private:
    double _value;
    std::chrono::duration<double> _duration;
};

std::chrono::duration<double> calculateTime(
    std::function<double()> function, double& output) {
    auto startTime = std::chrono::high_resolution_clock::now();
    output         = function();
    auto endTime   = std::chrono::high_resolution_clock::now();
    return endTime - startTime;
}

// Функция для интегрирования
inline double computeFunction(double x) {
    return x * std::sin(x);
}

// Последовательное вычисление интеграла
double computeIntegral(double lowerBound, double upperBound) {
    double result   = 0.0;
    double stepSize = (upperBound - lowerBound) / NUM_POINTS;
    for (int index = 1; index < NUM_POINTS; ++index) {
        result += computeFunction(lowerBound + (index - 0.5) * stepSize);
    }
    return result * stepSize;
}

// Параллельное вычисление интеграла
double computeIntegralParallel(double lowerBound, double upperBound) {
    double result   = 0.0;
    double stepSize = (upperBound - lowerBound) / NUM_POINTS;

#pragma omp parallel for reduction(+ : result) num_threads(NUM_THREADS)
    for (int index = 1; index < NUM_POINTS; ++index) {
        result += computeFunction(lowerBound + (index - 0.5) * stepSize);
    }
    return result * stepSize;
}

int main() {
    double sequentialResult, parallelResult;

    auto sequentialTime = calculateTime(
        []() { return computeIntegral(-M_PI, M_PI); }, sequentialResult);
    auto parallelTime = calculateTime(
        []() { return computeIntegralParallel(-M_PI, M_PI); }, parallelResult);

    assert(std::fabs(sequentialResult - parallelResult) < PRECISION);

    OutputPrinter sequentialPrinter(sequentialTime, sequentialResult);
    OutputPrinter parallelPrinter(parallelTime, parallelResult);

    std::cout << "=== Последовательное выполнение ===\n";
    sequentialPrinter.display();

    std::cout << "=== Параллельное выполнение ===\n";
    parallelPrinter.display();

    return 0;
}
