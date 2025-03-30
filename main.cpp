#include <iostream>

#include <iostream>
#include <cmath>
#include <functional>
#include <chrono>
#include <omp.h>
#include <cassert>

constexpr size_t THREADS = 8;
constexpr size_t N = 100'000'000;
constexpr double EPS = 1e-8;

class Printer {
public:
    explicit Printer(std::chrono::duration<double> time, double result)
        : _time(time), _result(result) {}

    void print() const {
        std::cout << "Результат: " << _result << "\n";
        std::cout << "Время: " << _time.count() << " секунд\n";
    }

    double get() const { return _result; }

private:
    double _result;
    std::chrono::duration<double> _time;
};

std::chrono::duration<double> measureTime(
    std::function<double()> func, double& result) {
    auto start = std::chrono::high_resolution_clock::now();
    result = func();
    auto end = std::chrono::high_resolution_clock::now();
    return end - start;
}

inline double func(double x) {
    return std::exp(-x * x);
}

double integrate(double a, double b) {
    double sum = 0.0;
    double step = (b - a) / N;
    for (size_t i = 1; i < N; ++i) {
        sum += func(a + (i - 0.5) * step);
    }
    return sum * step;
}

double integrateParallel(double a, double b) {
    double sum = 0.0;
    double step = (b - a) / N;

#pragma omp parallel for reduction(+:sum) num_threads(THREADS)
    for (size_t i = 1; i < N; ++i) {
        sum += func(a + (i - 0.5) * step);
    }
    return sum * step;
}

int main() {
    double resultSeq, resultPar;

    auto timeSeq = measureTime([]() { return integrate(-M_PI, M_PI); }, resultSeq);
    auto timePar = measureTime([]() { return integrateParallel(-M_PI, M_PI); }, resultPar);

    assert(std::fabs(resultSeq - resultPar) < EPS);

    Printer seqPrinter(timeSeq, resultSeq);
    Printer parPrinter(timePar, resultPar);

    std::cout << "=== Последовательное выполнение ===\n";
    seqPrinter.print();

    std::cout << "=== Параллельное выполнение ===\n";
    parPrinter.print();

    std::cout << "Ожидаемая точность: " << std::sqrt(M_PI) * std::erf(M_PI) << "\n";

    return 0;
}
