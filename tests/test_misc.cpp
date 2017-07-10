#include "fmm_impl.hpp"
#include "doctest.h"
#include "test_helpers.hpp"

TEST_CASE("inscribe") {
    auto s =
        inscribe_surf<3>({{1, 1, 1}, 2 / std::sqrt(3)}, 0.5, {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}});
    REQUIRE(s.size() == size_t(3));
    REQUIRE_ARRAY_EQUAL(s[0], std::array<double,3>{2, 1, 1}, 3);
    REQUIRE_ARRAY_EQUAL(s[1], std::array<double,3>{1, 2, 1}, 3);
    REQUIRE_ARRAY_EQUAL(s[2], std::array<double,3>{1, 1, 2}, 3);
}

