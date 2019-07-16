/*
 *       /\        Matteo Cicuttin (C) 2017,2018
 *      /__\       matteo.cicuttin@enpc.fr
 *     /_\/_\      École Nationale des Ponts et Chaussées - CERMICS
 *    /\    /\
 *   /__\  /__\    This is ProtoN, a library for fast Prototyping of
 *  /_\/_\/_\/_\   Numerical methods.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * If you use this code or parts of it for scientific publications, you
 * are required to cite it as following:
 *
 * Implementation of Discontinuous Skeletal methods on arbitrary-dimensional,
 * polytopal meshes using generic programming.
 * M. Cicuttin, D. A. Di Pietro, A. Ern.
 * Journal of Computational and Applied Mathematics.
 * DOI: 10.1016/j.cam.2017.09.017
 */

#pragma once

#include <iostream>
#include <vector>
#include <cassert>

#include "point.hpp"
#include "quadratures_dunavant.hpp"

template<typename T>
std::vector<std::pair<point<T,1>, T>>
golub_welsch(size_t degree)
{
    std::vector<std::pair<point<T,1>, T>> ret;

    using namespace Eigen;

    if ( degree % 2 == 0)
        degree++;

    size_t reqd_nodes = (degree+1)/2;

    if (reqd_nodes == 1)
    {
        auto qp = std::make_pair(point<T,1>({0.}), 2.0);
        ret.push_back(qp);
        return ret;
    }

    Matrix<T, Dynamic, Dynamic> M = Matrix<T, Dynamic, Dynamic>::Zero(reqd_nodes, reqd_nodes);
    for (size_t i = 1; i < reqd_nodes; i++)
    {
        T p = 4.0 - 1.0/(i*i);
        M(i, i-1) = sqrt(1.0/p);
    }

    SelfAdjointEigenSolver<Matrix<T, Dynamic, Dynamic>> solver;
    solver.compute(M);

    Matrix<T, Dynamic, Dynamic> weights = solver.eigenvectors().row(0);
                                weights = weights.array().square();
    Matrix<T, Dynamic, Dynamic> nodes = solver.eigenvalues();

    assert( weights.size() == nodes.size() );

    for (size_t i = 0; i < nodes.size(); i++)
    {
        auto qp = std::make_pair(point<T,1>({nodes(i)}), 2*weights(i));
        ret.push_back(qp);
    }

    return ret;
}


template<typename T>
std::vector<std::pair<point<T,1>, T>>
gauss_legendre(size_t degree)
{
    auto comp_degree = degree;

    if ( degree % 2 == 0 )
        comp_degree = degree + 1;

    size_t reqd_nodes = (comp_degree+1)/2;

    std::vector< std::pair<point<T,1>, T> > ret;
    ret.reserve(reqd_nodes);

    point<T,1>  qp;
    T           qw;
    T           a1, a2;

    switch(reqd_nodes)
    {
        case 1:
            qp = point<T,1>({0.0});
            qw = 2.0;
            ret.push_back( std::make_pair(qp, qw) );
            return ret;

        case 2:
            qp = point<T,1>({ 1.0/std::sqrt(3.0) });
            qw = 1.0;
            ret.push_back( std::make_pair(-qp, qw ) );
            ret.push_back( std::make_pair( qp, qw ) );
            return ret;

        case 3:
            qp = point<T,1>({ std::sqrt(3.0/5.0) });
            qw = 5.0/9.0;
            ret.push_back( std::make_pair(-qp, qw ) );
            ret.push_back( std::make_pair( qp, qw ) );
            qp = point<T,1>({0.0});
            qw = 8.0/9.0;
            ret.push_back( std::make_pair( qp, qw ) );
            return ret;

        case 4:
            a1 = 3.0/7.0;
            a2 = 2.0*std::sqrt(6.0/5.0)/7.0;
            qp = point<T,1>({ std::sqrt(a1 - a2) });
            qw = (18.0 + std::sqrt(30.0))/36.0;
            ret.push_back( std::make_pair(-qp, qw ) );
            ret.push_back( std::make_pair( qp, qw ) );
            qp = point<T,1>({ std::sqrt(a1 + a2) });
            qw = (18.0 - std::sqrt(30.0))/36.0;
            ret.push_back( std::make_pair(-qp, qw ) );
            ret.push_back( std::make_pair( qp, qw ) );
            return ret;

        case 5:
            qp = point<T,1>({ 0.0 });
            qw = 128.0/225.0;
            ret.push_back( std::make_pair( qp, qw ) );

            a1 = 5.0;
            a2 = 2.0*std::sqrt(10.0/7.0);
            qp = point<T,1>({ std::sqrt(a1 - a2)/3.0 });
            qw = (322 + 13.0*std::sqrt(70.0))/900.0;
            ret.push_back( std::make_pair(-qp, qw ) );
            ret.push_back( std::make_pair( qp, qw ) );

            qp = point<T,1>({ std::sqrt(a1 + a2)/3.0 });
            qw = (322 - 13.0*std::sqrt(70.0))/900.0;
            ret.push_back( std::make_pair(-qp, qw ) );
            ret.push_back( std::make_pair( qp, qw ) );
            return ret;

        default:
            return golub_welsch<T>(degree);

    }

    return ret;
}

template<typename T>
std::vector<std::pair<point<T,1>, T>>
edge_quadrature(size_t doe)
{
    return gauss_legendre<T>(doe);
}

/*
template<typename T>
std::vector<std::pair<point<T,2>, T>>
triangle_quadrature(const point<T,2>& p0, const point<T,2>& p1, const point<T,2>& p2, size_t deg)
{
    std::vector<std::pair<point<T,2>, T>>   ret;

    auto v0 = p1 - p0;
    auto v1 = p2 - p0;

    auto area = std::abs( (v0.x() * v1.y() - v0.y() * v1.x())/2.0 );

    point<T,2>      qp;
    T               qw;

    T               a1 = (6. - std::sqrt(15.)) / 21;
    T               a2 = (6. + std::sqrt(15.)) / 21;
    T               w1 = (155. - std::sqrt(15.)) / 1200;
    T               w2 = (155. + std::sqrt(15.)) / 1200;

    switch(deg)
    {
        case 0:
        case 1:
            qw = area;
            qp = (p0 + p1 + p2)/3;      ret.push_back( std::make_pair(qp, qw) );
            return ret;

        case 2:
            qw = area/3;
            qp = p0/6 + p1/6 + 2*p2/3;  ret.push_back( std::make_pair(qp, qw) );
            qp = p0/6 + 2*p1/3 + p2/6;  ret.push_back( std::make_pair(qp, qw) );
            qp = 2*p0/3 + p1/6 + p2/6;  ret.push_back( std::make_pair(qp, qw) );
            return ret;

        case 3:
            qw = 9*area/20;
            qp = (p0 + p1 + p2)/3;      ret.push_back( std::make_pair(qp, qw) );
            qw = 2*area/15;
            qp = (p0 + p1)/2;           ret.push_back( std::make_pair(qp, qw) );
            qp = (p0 + p2)/2;           ret.push_back( std::make_pair(qp, qw) );
            qp = (p1 + p2)/2;           ret.push_back( std::make_pair(qp, qw) );
            qw = area/20;
            qp = p0;                    ret.push_back( std::make_pair(qp, qw) );
            qp = p1;                    ret.push_back( std::make_pair(qp, qw) );
            qp = p2;                    ret.push_back( std::make_pair(qp, qw) );
            return ret;

        case 4:
        case 5:
            qw = 9*area/40;
            qp = (p0 + p1 + p2)/3;      ret.push_back( std::make_pair(qp, qw) );
            qw = w1 * area;
            qp = a1*p0 + a1*p1 + (1-2*a1)*p2;   ret.push_back( std::make_pair(qp, qw) );
            qp = a1*p0 + (1-2*a1)*p1 + a1*p2;   ret.push_back( std::make_pair(qp, qw) );
            qp = (1-2*a1)*p0 + a1*p1 + a1*p2;   ret.push_back( std::make_pair(qp, qw) );
            qw = w2 * area;
            qp = a2*p0 + a2*p1 + (1-2*a2)*p2;   ret.push_back( std::make_pair(qp, qw) );
            qp = a2*p0 + (1-2*a2)*p1 + a2*p2;   ret.push_back( std::make_pair(qp, qw) );
            qp = (1-2*a2)*p0 + a2*p1 + a2*p2;   ret.push_back( std::make_pair(qp, qw) );
            return ret;
            
        default:
            throw std::invalid_argument("Triangle quadrature: requested order too high");

    }

    return ret;
}
*/

template<typename T>
std::vector<std::pair<point<T,2>, T>>
triangle_quadrature(const point<T,2>& p0, const point<T,2>& p1, const point<T,2>& p2, size_t deg)
{
    if (deg == 0)
        deg = 1;

    if (deg > 8)
        throw std::invalid_argument("Quadrature order too high");

    auto v0 = p1 - p0;
    auto v1 = p2 - p0;

    auto area = (v0.x() * v1.y() - v0.y() * v1.x()) / 2.0;
    // the area is negative when the points are sorted clockwise
    if(area < 0)
        std::cout << "negative weights !!" << std::endl;

    using namespace dunavant_quadratures;

    std::vector<std::pair<point<T,2>, T>> ret;

    ret.reserve( rules[deg].num_points );

    for (size_t i = 0; i < rules[deg].num_points; i++)
    {
        point<T,2> qp = p0 * rules[deg].data[i][0] +
                        p1 * rules[deg].data[i][1] +
                        p2 * rules[deg].data[i][2];

        T qw          = area * rules[deg].data[i][3];

        ret.push_back( std::make_pair(qp, qw) );
    }

    return ret;
}


template<typename Mesh>
struct reference_transform;

template<typename T, typename CU, typename FU, typename NU>
struct reference_transform< quad_mesh<T, CU, FU, NU> >
{
    using mesh_type = quad_mesh<T, CU, FU, NU>;
    using cell_type = typename mesh_type::cell_type;
    using point_type = typename mesh_type::point_type;

    std::array<point_type, 4> pts;

    reference_transform(const mesh_type& msh, const cell_type& cl)
    {
        pts = points(msh, cl);
    }

    point_type ref_to_phys(const point_type& pt)
    {
        auto xi  = pt.x();
        auto eta = pt.y();

        return 0.25 * pts[0] * (1-xi)*(1-eta) +
               0.25 * pts[1] * (1+xi)*(1-eta) +
               0.25 * pts[2] * (1+xi)*(1+eta) +
               0.25 * pts[3] * (1-xi)*(1+eta);
    }

};

template<typename Mesh>
auto make_reference_transform(const Mesh& msh, const typename Mesh::cell_type& cl)
{
    return reference_transform<Mesh>(msh, cl);
}


template<typename T, typename CU, typename FU, typename NU>
std::vector< std::pair<point<T,2>, T> >
integrate(const mesh<T,4,CU,FU,NU>& msh,
          const typename mesh<T,4,CU,FU,NU>::cell_type& cl,
          size_t degree)
{
    typedef typename mesh<T,4,CU,FU,NU>::point_type    point_type;

    auto qps = edge_quadrature<T>(degree);
    auto pts = points(msh, cl);

    auto v0 = pts[1] - pts[0];
    auto v1 = pts[2] - pts[1];
    auto v2 = pts[3] - pts[2];
    auto v3 = pts[3] - pts[0];

    auto meas = 0.5*(v0.x()*v3.y() - v0.y()*v3.x()) + 0.5*(v1.x()*v2.y() - v1.y()*v2.x());

    std::vector<std::pair<point<T,2>, T>> ret;

    auto P = [&](T xi, T eta) -> T {
        return 0.25 * pts[0].x() * (1-xi)*(1-eta) +
               0.25 * pts[1].x() * (1+xi)*(1-eta) +
               0.25 * pts[2].x() * (1+xi)*(1+eta) +
               0.25 * pts[3].x() * (1-xi)*(1+eta);
    };

    auto Q = [&](T xi, T eta) -> T {
        return 0.25 * pts[0].y() * (1-xi)*(1-eta) +
               0.25 * pts[1].y() * (1+xi)*(1-eta) +
               0.25 * pts[2].y() * (1+xi)*(1+eta) +
               0.25 * pts[3].y() * (1-xi)*(1+eta);
    };

    auto J = [&](T xi, T eta) -> T {
        auto j11 = 0.25*((pts[1].x() - pts[0].x())*(1-eta) + (pts[2].x() - pts[3].x())*(1+eta));
        auto j12 = 0.25*((pts[1].y() - pts[0].y())*(1-eta) + (pts[2].y() - pts[3].y())*(1+eta));
        auto j21 = 0.25*((pts[3].x() - pts[0].x())*(1-xi) + (pts[2].x() - pts[1].x())*(1+xi));
        auto j22 = 0.25*((pts[3].y() - pts[0].y())*(1-xi) + (pts[2].y() - pts[1].y())*(1+xi));

        return std::abs(j11*j22 - j12*j21);
    };

    T sw = 0.0, swq = 0.0;
    for (auto jtor = qps.begin(); jtor != qps.end(); jtor++)
    {
        for (auto itor = qps.begin(); itor != qps.end(); itor++)
        {
            auto qp_x = *itor;
            auto qp_y = *jtor;

            auto xi = qp_x.first.x();
            auto eta = qp_y.first.x();

            auto px = P(xi, eta);
            auto py = Q(xi, eta);

            auto w = qp_x.second * qp_y.second * J(xi, eta);

            ret.push_back( std::make_pair( point_type(px, py), w ) );
        }
    }

    return ret;
}

template<typename T, typename CU, typename FU, typename NU>
std::vector< std::pair<point<T,2>, T> >
integrate(const poly_mesh<T,CU,FU,NU>& msh,
          const typename poly_mesh<T,CU,FU,NU>::cell_type& cl,
          size_t degree)
{
    auto bar = barycenter(msh, cl);
    auto pts = points(msh, cl);

    auto num_points = pts.size();

    std::vector< std::pair<point<T,2>, T> > ret;

    for (size_t i = 0; i < num_points; i++)
    {
        auto p0 = pts[i];
        auto p1 = pts[(i+1)%num_points];
        auto p2 = bar;

        auto qps = triangle_quadrature(p0, p1, p2, degree);

        ret.insert( ret.end(), qps.begin(), qps.end() );
    }

    return ret;
}

template<typename Mesh>
std::vector<std::pair<point<typename Mesh::coordinate_type,2>, typename Mesh::coordinate_type>>
integrate(const Mesh& msh, const typename Mesh::face_type& fc, size_t degree)
{
    using T = typename Mesh::coordinate_type;

    typedef typename Mesh::point_type    point_type;

    auto qps = edge_quadrature<T>(degree);
    auto pts = points(msh, fc);

    auto scale = (pts[1] - pts[0]);
    auto meas = scale.to_vector().norm();

    std::vector<std::pair<point<T,2>, T>>   ret;

    for (auto itor = qps.begin(); itor != qps.end(); itor++)
    {
        auto qp = *itor;
        //auto p = qp.first.x() * scale + pts[0];
        auto t = qp.first.x();
        auto p = 0.5*(1-t)*pts[0] + 0.5*(1+t)*pts[1];
        auto w = qp.second * meas * 0.5;

        ret.push_back( std::make_pair(p, w) );
    }

    return ret;
}


template<typename T, size_t N>
std::ostream&
operator<<(std::ostream& os, const std::pair<point<T,N>, T>& pw_pair)
{
    os << "[ " << pw_pair.first << ", " << pw_pair.second << " ]";
    return os;
}
