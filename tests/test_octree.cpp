#include "doctest.h"
#include "test_helpers.hpp"
#include "octree.hpp"

TEST_CASE("yo") {
    REQUIRE(1 == 1);
}

// #include "util.h"
// #include "vec_ops.h"
// #include "geometry.h"

TEST_CASE("make child idx 3d") 
{
    REQUIRE(make_child_idx<3>(0) == (std::array<size_t,3>{0, 0, 0}));
    REQUIRE(make_child_idx<3>(2) == (std::array<size_t,3>{0, 1, 0}));
    REQUIRE(make_child_idx<3>(7) == (std::array<size_t,3>{1, 1, 1}));
}

TEST_CASE("make child idx 2d") 
{
    REQUIRE(make_child_idx<2>(1) == (std::array<size_t,2>{0, 1}));
    REQUIRE(make_child_idx<2>(3) == (std::array<size_t,2>{1, 1}));
}
// 
// TEST_CASE("check law of large numbers", "[octree]") 
// {
//     int n = 100000;
//     auto pts = random_pts<3>(n);
//     //TODO: Make a octree capacity test
//     auto tree = make_octree(pts, 100);
//     for (size_t i = 0; i < 8; i++) {
//         int n_pts = tree.children[i]->indices.size();
//         int diff = abs(n_pts - (n / 8));
//         CHECK(diff < (n / 16));
//     }
// }
// 
// TEST_CASE("one level octree", "[octree]") 
// {
//     auto es = random_balls<3>(3, 0.0);
//     auto oct = make_octree(es, 4);
//     REQUIRE(oct.level == 0);
//     for (size_t i = 0; i < 8; i++) {
//         REQUIRE(oct.children[i] == nullptr);
//     }
//     REQUIRE(oct.indices.size() == 3);
// }
// 
// TEST_CASE("many level octree", "[octree]") 
// {
//     auto pts = random_pts<3>(1000);
//     auto oct = make_octree(pts, 4);
//     REQUIRE(oct.indices.size() == 1000);
//     REQUIRE(oct.children[0]->level == 1);
// }
// 
// TEST_CASE("degenerate line octree in 2d", "[octree]") 
// {
//     std::vector<Vec<double,2>> pts;
//     for (size_t i = 0; i < 100; i++) {
//         pts.push_back({static_cast<double>(i), 0.0});
//     }
//     auto oct = make_octree(pts, 1);
//     REQUIRE(!oct.is_leaf());
// }
// 
// template <size_t dim>
// size_t n_pts(const Octree<dim>& cell) 
// {
//     if (cell.is_leaf()) {
//         return cell.indices.size();
//     }
//     size_t n_child_pts = 0;
//     for (size_t c = 0; c < Octree<dim>::split; c++) {
//         if (cell.children[c] == nullptr) {
//             continue;
//         }
//         n_child_pts += n_pts(*cell.children[c]);
//     }
//     return n_child_pts;
// }
// 
// template <size_t dim>
// void check_n_pts(const Octree<dim>& cell) 
// {
//     REQUIRE(n_pts(cell) == cell.indices.size());
//     for (size_t c = 0; c < Octree<dim>::split; c++) {
//         if (cell.children[c] == nullptr) {
//             continue;
//         }
//         if (cell.children[c]->is_leaf()) {
//             continue;
//         }
//         check_n_pts(*cell.children[c]);
//     }
// }
// 
// TEST_CASE("check octree cell counts", "[octree]") 
// {
//     auto pts = random_pts<3>(1000);
//     auto oct = make_octree(pts, 4);
//     check_n_pts(oct);
// }
// 
// TEST_CASE("check octree cell counts for degenerate line", "[octree]") 
// {
//     size_t n = 100;
//     std::vector<Vec<double,2>> pts;
//     for (size_t i = 0; i < n; i++) {
//         pts.push_back({static_cast<double>(i), 0.0});
//     }
//     auto oct = make_octree(pts, 26);
//     CHECK(n_pts(oct) == n);
// }
// 
// TEST_CASE("make octree with two identical points", "[octree]") 
// {
//     std::vector<Vec<double,3>> es{{1.0, 2.0, 0.0}, {1.0, 2.0, 0.0}};
//     auto oct = make_octree(es, 1);
// }
// 
// TEST_CASE("make octree with two very similar points", "[octree]") 
// {
//     std::vector<Vec<double,3>> es{
//         {1.0, 2.0, 0.0}, {1.0, 2.0 - 1e-20, 0.0}, {0.0, 0.0, 0.0}
//     };
//     auto oct = make_octree(es, 1);
// }
// 
// template <size_t dim>
// size_t count_children(const Octree<dim>& cell) 
// {
//     if (cell.is_leaf()) {
//         return 1;
//     }
// 
//     size_t n_c = 0;
//     for (auto& c: cell.children) {
//         if (c == nullptr) {
//             continue;
//         }
//         n_c += 1 + count_children(*c);
//     }
//     return n_c;
// }
// 
// TEST_CASE("count children for degenerate line octree", "[octree]") 
// {
//     size_t n = 10;
//     std::vector<Vec<double,2>> pts;     
//     for (size_t i = 0; i < n; i++) {
//         pts.push_back({static_cast<double>(i), 0});
//         pts.push_back({static_cast<double>(i + 1), 0});
//     }
// 
//     auto oct = make_octree(pts, 1);
//     REQUIRE(count_children(oct) == 31); 
// }
// 
// template <size_t dim>
// std::set<size_t> check_indices_unique(const Octree<dim>& oct,
//     const std::set<size_t>& indices)
// {
//     auto new_set = indices;
//     auto ret = new_set.emplace(oct.index);
//     CHECK(ret.second);
//     for (auto& c: oct.children) {
//         if (c == nullptr) {
//             continue;
//         }
//         new_set = check_indices_unique(*c, new_set);
//     }
//     return new_set;
// }
// 
// TEST_CASE("check cells have unique indices", "[octree]") 
// {
//     auto pts = random_pts<3>(1000);
//     auto oct = make_octree(pts, 4);
//     check_indices_unique(oct, std::set<size_t>{});
// }
// 
// TEST_CASE("non zero ball radius", "[octree]")
// {
//     auto balls = random_balls<3>(10, 0.1);
//     auto oct = make_octree(balls, 1);
// }
// 
// template <size_t dim>
// void check_true_bounds_contain_balls(const Octree<dim>& oct,
//     const std::vector<Ball<dim>>& balls)
// {
//     for (size_t i = 0; i < oct.indices.size(); i++) {
//         auto ball_idx = oct.indices[i];
//         REQUIRE(oct.true_bounds.in_box(balls[ball_idx]));
//     }
//     for (auto& c: oct.children) {
//         if (c == nullptr) {
//             continue;
//         }
//         check_true_bounds_contain_balls(*c, balls);
//     }
// }
// 
// TEST_CASE("test true bounds non-zero radii", "[octree]")
// {
//     auto balls = random_balls<3>(100, 0.05);
//     auto oct = make_octree(balls, 1);
//     check_true_bounds_contain_balls(oct, balls);
// }
// 
// TEST_CASE("impossible subdivision", "[octree]")
// {
//     std::vector<Ball<2>> bs{
//         {{0, 0}, 0.5},
//         {{0, 0}, 1.0}
//     };
//     auto oct = make_octree(bs, 1);
// }
// 
// TEST_CASE("find closest nonempty child", "[nearest_neighbors]")
// {
//     std::vector<Vec<double,3>> pts{
//         {1, 1, 1}, {-1, -1, -1}
//     };
//     auto oct = make_octree(pts, 1);  
//     auto idx = oct.find_closest_nonempty_child({0.1, 0.1, -1});
//     REQUIRE(idx == 0);
// }
