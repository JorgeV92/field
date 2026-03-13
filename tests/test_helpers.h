#ifndef FIELD_TESTS_TEST_HELPERS_H_
#define FIELD_TESTS_TEST_HELPERS_H_

#include <cmath>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>

#define FIELD_EXPECT_TRUE(condition)                                                \
  do {                                                                              \
    if (!(condition)) {                                                             \
      std::cerr << "Expectation failed: " #condition << " at line " << __LINE__ \
                << "\n";                                                          \
      return EXIT_FAILURE;                                                          \
    }                                                                               \
  } while (false)

#define FIELD_EXPECT_NEAR(actual, expected, tolerance)                               \
  do {                                                                               \
    const auto field_actual = (actual);                                              \
    const auto field_expected = (expected);                                          \
    const auto field_tolerance = (tolerance);                                        \
    if (std::fabs(field_actual - field_expected) > field_tolerance) {                \
      std::cerr << "Expectation failed: |" #actual " - " #expected "| <= "      \
                << field_tolerance << " at line " << __LINE__ << " (actual="      \
                << field_actual << ", expected=" << field_expected << ")\n";     \
      return EXIT_FAILURE;                                                           \
    }                                                                                \
  } while (false)

#define FIELD_EXPECT_THROW(statement, exception_type)                               \
  do {                                                                              \
    bool field_threw_expected = false;                                              \
    try {                                                                           \
      statement;                                                                    \
    } catch (const exception_type&) {                                               \
      field_threw_expected = true;                                                  \
    } catch (const std::exception& ex) {                                            \
      std::cerr << "Expected exception " #exception_type                           \
                << ", but caught different exception at line " << __LINE__        \
                << ": " << ex.what() << "\n";                                  \
      return EXIT_FAILURE;                                                          \
    }                                                                               \
    if (!field_threw_expected) {                                                    \
      std::cerr << "Expected exception " #exception_type                           \
                << " was not thrown at line " << __LINE__ << "\n";              \
      return EXIT_FAILURE;                                                          \
    }                                                                               \
  } while (false)

#endif  // FIELD_TESTS_TEST_HELPERS_H_