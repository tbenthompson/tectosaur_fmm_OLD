#include "blas_wrapper.hpp"
#include "doctest.h"
#include "test_helpers.hpp"

#include <iostream>

TEST_CASE("non square psuedoinverse") {
// >>> A = np.array([[1,2,0],[1,1,1]])
// >>> np.linalg.pinv(A)
// array([[  8.32667268e-17,   3.33333333e-01],
//        [  5.00000000e-01,  -1.66666667e-01],
//        [ -5.00000000e-01,   8.33333333e-01]])
    std::vector<double> matrix{1,2,0,1,1,1};
    auto svd = svd_decompose(matrix.data(), 2, 3);
    auto pseudoinv = svd_pseudoinverse(svd);
    for (int i = 0 ;i < 6; i++) {
        std::cout << pseudoinv[i] << std::endl;
    }
    std::vector<double> correct{0, 1.0 / 3.0, 0.5, -5.0 / 3.0, -0.5, 5.0 / 6.0};
    REQUIRE_ARRAY_CLOSE(pseudoinv, correct, 6, 1e-14); 
}

TEST_CASE("DGEMM test") {
    std::vector<double> A{0,1,2,3};
    std::vector<double> B{9,8,7,6};
    std::vector<double> correct{7, 6, 39, 34};
    auto result = mat_mult(2, 2, false, A, false, B);
    REQUIRE_ARRAY_EQUAL(result, correct, 4);
}

TEST_CASE("LU solve") 
{
    std::vector<double> matrix{
        2, 1, -1, 0.5
    };
    auto lu = lu_decompose(matrix);
    auto soln = lu_solve(lu, {1,1});
    std::vector<double> correct{
        -0.25, 1.5
    };
    REQUIRE_ARRAY_CLOSE(soln, correct, 2, 1e-14);
}

TEST_CASE("SVD solve") 
{
    std::vector<double> matrix{
        2, 1, -1, 0.5
    };
    auto svd = svd_decompose(matrix.data(), 2, 2);
    auto soln = svd_solve(svd, {1,1});
    std::vector<double> correct{
        -0.25, 1.5
    };
    REQUIRE_ARRAY_CLOSE(soln, correct, 2, 1e-14);
}

TEST_CASE("Pseudoinverse") 
{
    std::vector<double> matrix{
        2, 1, -1, 0.5
    };
    auto svd = svd_decompose(matrix.data(), 2, 2);
    auto pseudoinv = svd_pseudoinverse(svd);
    std::vector<double> inv{
        0.25, -0.5, 0.5, 1.0
    };
    REQUIRE_ARRAY_CLOSE(pseudoinv, inv, 4, 1e-14);
}

TEST_CASE("Thresholded pseudoinverse") 
{
    // Matrix has two singular values: 1.0 and 1e-5
    std::vector<double> matrix{
        0.0238032718573239, 0.1524037864980028,
        0.1524037864980028, 0.9762067281426762
    };
    auto svd = svd_decompose(matrix.data(), 2, 2);
    auto no_threshold_pseudoinv = svd_pseudoinverse(svd);
    std::vector<double> correct_no_threshold{
        97620.6728142285282956, -15240.3786497941800917,
        -15240.3786497941782727, 2380.3271857314393856
    };
    REQUIRE_ARRAY_CLOSE(no_threshold_pseudoinv, correct_no_threshold, 4, 1e-4);
    set_threshold(svd, 1e-4);
    auto thresholded_pseudoinv = svd_pseudoinverse(svd);
    std::vector<double> correct_thresholded{
        0.0237935097924219, 0.1524053105511083,
        0.1524053105511085, 0.9762064902075779
    };
    REQUIRE_ARRAY_CLOSE(thresholded_pseudoinv, correct_thresholded, 4, 1e-12);
}

TEST_CASE("Condition number") 
{
    std::vector<double> matrix{
        2, 1, -1, 0.5
    };
    auto svd = svd_decompose(matrix.data(), 2, 2);
    double cond = condition_number(svd);
    REQUIRE(cond == doctest::Approx(2.7630857945186595).epsilon(1e-12));
}

TEST_CASE("matrix vector product")
{
    
    std::vector<double> matrix{
        2, 1, -1, 0.5
    };
    std::vector<double> vec{4, -2};
    auto result = matrix_vector_product(matrix.data(), 2, 2, vec.data());
    REQUIRE_ARRAY_CLOSE(result, std::vector<double>{6, -5}, 2, 1e-15);
}

TEST_CASE("matrix vector non-square")
{
    std::vector<double> matrix{
        2, 1, 1, -1, 0.5, 10
    };
    std::vector<double> vec{4,-2,0.5};
    auto result = matrix_vector_product(matrix.data(), 2, 3, vec.data());
    REQUIRE_ARRAY_CLOSE(result, std::vector<double>{6.5, 0}, 2, 1e-15);
}

TEST_CASE("matrix vector 0 columns")
{
    auto result = matrix_vector_product(nullptr, 0, 0, nullptr);
    REQUIRE(result.size() == size_t(0));
}

