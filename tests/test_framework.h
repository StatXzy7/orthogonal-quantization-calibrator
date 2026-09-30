#ifndef OQC_TEST_FRAMEWORK_H
#define OQC_TEST_FRAMEWORK_H

#include <cmath>
#include <cstdio>
#include <exception>
#include <functional>
#include <string>
#include <vector>

namespace oqctest {

struct Case {
    std::string name;
    std::function<void()> fn;
};

inline std::vector<Case>& registry() {
    static std::vector<Case> cases;
    return cases;
}

struct Registrar {
    Registrar(const char* name, std::function<void()> fn) {
        registry().push_back({name, std::move(fn)});
    }
};

struct Failure : std::exception {
    std::string msg;
    explicit Failure(std::string m) : msg(std::move(m)) {}
    const char* what() const noexcept override { return msg.c_str(); }
};

inline void check(bool cond, const char* expr, const char* file, int line) {
    if (!cond) {
        throw Failure(std::string(file) + ":" + std::to_string(line) +
                      ": CHECK failed: " + expr);
    }
}

inline void check_close(double a, double b, double tol,
                        const char* sa, const char* sb,
                        const char* file, int line) {
    double diff = std::fabs(a - b);
    double scale = std::max({1.0, std::fabs(a), std::fabs(b)});
    if (!(diff <= tol * scale)) {
        throw Failure(std::string(file) + ":" + std::to_string(line) +
                      ": |" + sa + " - " + sb + "| = " +
                      std::to_string(diff) + " exceeds tol " +
                      std::to_string(tol) + " * scale");
    }
}

}  // namespace oqctest

#define TEST_CASE(name)                                                      \
    static void name();                                                      \
    static oqctest::Registrar reg_##name(#name, name);                       \
    static void name()

#define CHECK(cond) oqctest::check(static_cast<bool>(cond), #cond, __FILE__, __LINE__)
#define CHECK_THROWS(expr)                                                   \
    do {                                                                     \
        bool thrown = false;                                                 \
        try {                                                                \
            expr;                                                            \
        } catch (const std::exception&) {                                    \
            thrown = true;                                                  \
        }                                                                    \
        oqctest::check(thrown, "expected exception: " #expr, __FILE__, __LINE__); \
    } while (0)
#define CHECK_CLOSE(a, b, tol)                                               \
    oqctest::check_close((a), (b), (tol), #a, #b, __FILE__, __LINE__)

#endif
