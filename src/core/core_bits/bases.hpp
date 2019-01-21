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

#include <Eigen/Dense>

template<typename T>
T iexp_pow(T x, size_t n)
{
    if (n == 0)
        return 1;

    T y = 1;
    while (n > 1)
    {
        if (n % 2 == 0)
        {
            x = x * x;
            n = n / 2;
        }
        else
        {
            y = x * y;
            x = x * x;
            n = (n-1)/2;
        }
    }

    return x*y;
}


size_t basis_size(size_t k, size_t d)
{
    size_t num = 1;
    size_t den = 1;

    for (size_t i = 1; i <= d; i++)
    {
        num *= k + i;
        den *= i;
    }

    return num/den;
}


//#define POWER_CACHE

template<typename Mesh, typename VT>
class cell_basis
{
    typedef typename Mesh::coordinate_type  coordinate_type;
    typedef typename Mesh::point_type       point_type;

    point_type          cell_bar;
    coordinate_type     cell_h;
    size_t              basis_degree, basis_size;

#ifdef POWER_CACHE
    std::vector<coordinate_type>  power_cache;
#endif

public:
    cell_basis(const Mesh& msh, const typename Mesh::cell_type& cl, size_t degree)
    {
        cell_bar        = barycenter(msh, cl);
        cell_h          = diameter(msh, cl);
        basis_degree    = degree;
        basis_size      = (basis_degree+2)*(basis_degree+1)/2;
    }

    Matrix<VT, Dynamic, 1>
    eval_basis(const point_type& pt)
    {
        Matrix<VT, Dynamic, 1> ret = Matrix<VT, Dynamic, 1>::Zero(basis_size);

        auto bx = (pt.x() - cell_bar.x()) / (0.5*cell_h);
        auto by = (pt.y() - cell_bar.y()) / (0.5*cell_h);

#ifdef POWER_CACHE
        if ( power_cache.size() != (basis_degree+1)*2 )
          power_cache.resize( (basis_degree+1)*2);

        power_cache[0] = 1.0;
        power_cache[1] = 1.0;
        for (size_t i = 1; i <= basis_degree; i++)
        {
            power_cache[2*i]    = iexp_pow(bx, i);
            power_cache[2*i+1]  = iexp_pow(by, i);
        }
#endif

        size_t pos = 0;
        for (size_t k = 0; k <= basis_degree; k++)
        {
            for (size_t i = 0; i <= k; i++)
            {
                auto pow_x = k-i;
                auto pow_y = i;
#ifdef POWER_CACHE
                auto bv = power_cache[2*pow_x] * power_cache[2*pow_y+1];
#else
                auto bv = iexp_pow(bx, pow_x) * iexp_pow(by, pow_y);
#endif
                ret(pos++) = bv;
            }
        }

        assert(pos == basis_size);

        return ret;
    }

    Matrix<VT, Dynamic, 2>
    eval_gradients(const point_type& pt)
    {
        Matrix<VT, Dynamic, 2> ret = Matrix<VT, Dynamic, 2>::Zero(basis_size, 2);

        auto bx = (pt.x() - cell_bar.x()) / (0.5*cell_h);
        auto by = (pt.y() - cell_bar.y()) / (0.5*cell_h);
        auto ih = 2.0/cell_h;

#ifdef POWER_CACHE
        if ( power_cache.size() != (basis_degree+1)*2 )
          power_cache.resize( (basis_degree+1)*2);

        power_cache[0] = 1.0;
        power_cache[1] = 1.0;
        for (size_t i = 1; i <= basis_degree; i++)
        {
            power_cache[2*i]    = iexp_pow(bx, i);
            power_cache[2*i+1]  = iexp_pow(by, i);
        }
#endif

        size_t pos = 0;
        for (size_t k = 0; k <= basis_degree; k++)
        {
            for (size_t i = 0; i <= k; i++)
            {
                auto pow_x = k-i;
                auto pow_y = i;
#ifdef POWER_CACHE
                auto px = power_cache[2*pow_x];
                auto py = power_cache[2*pow_y+1];
                auto dx = (pow_x == 0) ? 0 : pow_x*ih*power_cache[2*(pow_x-1)];
                auto dy = (pow_y == 0) ? 0 : pow_y*ih*power_cache[2*(pow_y-1)+1];
#else
                auto px = iexp_pow(bx, pow_x);
                auto py = iexp_pow(by, pow_y);
                auto dx = (pow_x == 0) ? 0 : pow_x*ih*iexp_pow(bx, pow_x-1);
                auto dy = (pow_y == 0) ? 0 : pow_y*ih*iexp_pow(by, pow_y-1);
#endif
                ret(pos,0) = dx*py;
                ret(pos,1) = px*dy;
                pos++;
            }
        }

        assert(pos == basis_size);

        return ret;
    }

    size_t size() const
    {
        return basis_size;
    }

    size_t degree() const
    {
        return basis_degree;
    }
    
    static size_t size(size_t degree)
    {
        return (degree+2)*(degree+1)/2;
    }
};

#if 0
template<template<typename, typename> class Basis, typename CT, typename VT>
class trial_function
{
    const Basis<CT, VT>& basis;

public:
    trial_function(const Basis<CT, VT>& b)
        : basis(b)
    {}
};

template<template<typename, typename> class Basis, typename CT, typename VT>
auto trial(const Basis<CT, VT>& b)
{
    return trial_function<Basis, CT, VT>(b);
}

template<template<typename, typename> class Basis, typename CT, typename VT>
class test_function
{
    const Basis<CT, VT>& basis;

public:
    test_function(const Basis<CT, VT>& b)
        : basis(b)
    {}
};

template<template<typename, typename> class Basis, typename CT, typename VT>
auto test(const Basis<CT, VT>& b)
{
    return test_function<Basis, CT, VT>(b);
}

template<template <typename, typename> class BasisL, template <typename, typename> class BasisR, typename CT, typename VT>
int
operator,(const trial_function<BasisL,CT,VT> trial, const test_function<BasisR,CT,VT>& test)
{
    std::cout << "\\o/  \\o/  \\o/" << std::endl;
    return 42;
}
#endif

template<typename Mesh, typename VT>
class face_basis
{
    typedef typename Mesh::coordinate_type  coordinate_type;
    typedef typename Mesh::point_type       point_type;

    point_type          face_bar;
    point_type          base;
    coordinate_type     face_h;
    size_t              basis_degree, basis_size;

public:
    face_basis(const Mesh& msh, const typename Mesh::face_type& fc, size_t degree)
    {
        face_bar        = barycenter(msh, fc);
        face_h          = diameter(msh, fc);
        basis_degree    = degree;
        basis_size      = degree+1;

        auto pts = points(msh, fc);
        base = face_bar - pts[0];
    }

    Matrix<VT, Dynamic, 1>
    eval_basis(const point_type& pt)
    {
        Matrix<VT, Dynamic, 1> ret = Matrix<VT, Dynamic, 1>::Zero(basis_size);

        auto v = base.to_vector();
        auto t = (pt - face_bar).to_vector();
        auto dot = v.dot(t);
        auto ep = 4.0*dot/(face_h*face_h);

        for (size_t i = 0; i <= basis_degree; i++)
        {
            auto bv = iexp_pow(ep, i);
            ret(i) = bv;
        }
        return ret;
    }

    size_t size() const
    {
        return basis_size;
    }

    size_t degree() const
    {
        return basis_degree;
    }

    static size_t size(size_t degree)
    {
        return degree+1;
    }
};



////////////////////  VECTOR BASIS  /////////////////////////

template<typename Mesh, typename VT>
class vector_cell_basis
{
    typedef typename Mesh::coordinate_type  coordinate_type;
    typedef typename Mesh::point_type       point_type;
    typedef Matrix<VT, 2, 2>                gradient_type;
    typedef Matrix<VT, Dynamic, 2>          function_type;

    point_type          cell_bar;
    coordinate_type     cell_h;
    size_t              basis_degree, basis_size;

    cell_basis<Mesh,VT>          scalar_basis;
    
#ifdef POWER_CACHE
    std::vector<coordinate_type>  power_cache;
#endif

public:
    vector_cell_basis(const Mesh& msh, const typename Mesh::cell_type& cl, size_t degree) :
        scalar_basis(msh, cl, degree)
    {
        cell_bar        = barycenter(msh, cl);
        cell_h          = diameter(msh, cl);
        basis_degree    = degree;
        basis_size      = 2*(basis_degree+2)*(basis_degree+1)/2;
    }

    Matrix<VT, Dynamic, 2>
    eval_basis(const point_type& pt)
    {
        Matrix<VT, Dynamic, 2> ret = Matrix<VT, Dynamic, 2>::Zero(basis_size, 2);
        // Matrix<VT, Dynamic, 2> ret;
        
        const auto phi = scalar_basis.eval_basis(pt);

        
        for(size_t i = 0; i < scalar_basis.size(); i++)
        {
            ret(2 * i, 0)     = phi(i);
            ret(2 * i + 1, 1) = phi(i);
        }
        
        assert(2 * scalar_basis.size() == basis_size);

        return ret;
    }

    std::vector<gradient_type>
    eval_gradients(const point_type& pt)
    {
        std::vector<gradient_type> ret;
        ret.reserve(basis_size);

        const function_type dphi = scalar_basis.eval_gradients(pt);

        for (size_t i = 0; i < scalar_basis.size(); i++)
        {
            const Matrix<VT, 1, 2> dphi_i = dphi.row(i);
            gradient_type                   g;

            g        = gradient_type::Zero();
            g.row(0) = dphi_i;
            ret.push_back(g);

            g        = gradient_type::Zero();
            g.row(1) = dphi_i;
            ret.push_back(g);
        }
        
        
        assert(ret.size() == basis_size);

        return ret;
    }

    size_t size() const
    {
        return basis_size;
    }

    size_t degree() const
    {
        return basis_degree;
    }

    static size_t size(size_t degree)
    {
        return 2*(degree+2)*(degree+1)/2;
    }
};


/////////

template<typename Mesh, typename VT>
class vector_face_basis
{
    typedef typename Mesh::coordinate_type  coordinate_type;
    typedef typename Mesh::point_type       point_type;

    point_type          face_bar;
    point_type          base;
    coordinate_type     face_h;
    size_t              basis_degree, basis_size;


    face_basis<Mesh,VT>          scalar_basis;
    
public:
    vector_face_basis(const Mesh& msh, const typename Mesh::face_type& fc, size_t degree) :
        scalar_basis(msh, fc, degree)
    {
        face_bar        = barycenter(msh, fc);
        face_h          = diameter(msh, fc);
        basis_degree    = degree;
        basis_size      = 2*(degree+1);

        auto pts = points(msh, fc);
        base = face_bar - pts[0];
    }

    Matrix<VT, Dynamic, 2>
    eval_basis(const point_type& pt)
    {
        Matrix<VT, Dynamic, 2> ret = Matrix<VT, Dynamic, 2>::Zero(basis_size, 2);

        const auto psi = scalar_basis.eval_basis(pt);

        for(size_t i = 0; i < scalar_basis.size(); i++)
        {
            ret(2 * i, 0)     = psi(i);
            ret(2 * i + 1, 1) = psi(i);
        }

        assert(2 * scalar_basis.size() == basis_size);
        
        return ret;
    }

    size_t size() const
    {
        return basis_size;
    }

    size_t degree() const
    {
        return basis_degree;
    }

    static size_t size(size_t degree)
    {
        return 2*(degree+1);
    }
};

////////////////////  MATRIX BASIS  /////////////////////////

template<typename Mesh, typename VT>
class matrix_cell_basis
{
    typedef typename Mesh::coordinate_type  coordinate_type;
    typedef typename Mesh::point_type       point_type;
    typedef Matrix<VT, 2, 2>                matrix_type;
    typedef Matrix<VT, Dynamic, 2>          function_type;

    point_type          cell_bar;
    coordinate_type     cell_h;
    size_t              basis_degree, basis_size;

    cell_basis<Mesh,VT>          scalar_basis;
    
#ifdef POWER_CACHE
    std::vector<coordinate_type>  power_cache;
#endif

public:
    matrix_cell_basis(const Mesh& msh, const typename Mesh::cell_type& cl, size_t degree) :
        scalar_basis(msh, cl, degree)
    {
        cell_bar        = barycenter(msh, cl);
        cell_h          = diameter(msh, cl);
        basis_degree    = degree;
        basis_size      = 2*2*(basis_degree+2)*(basis_degree+1)/2;
    }

    std::vector<function_type>
    eval_basis(const point_type& pt)
    {
        std::vector<function_type> ret;
        ret.reserve(basis_size);
        
        const auto phi = scalar_basis.eval_basis(pt);


        
        for (int k = 0; k < scalar_basis.size(); k++)
        {
            function_type fc;

            for (int j = 0; j < 2; j++)
            {
                for (int i = 0; i < 2; i++)
                {
                    fc       = function_type::Zero();
                    fc(i, j) = phi(k);
                    ret.push_back(fc);
                }
            }
        }
        
        assert(2 * 2 * scalar_basis.size() == basis_size);

        return ret;
    }

    size_t size() const
    {
        return basis_size;
    }

    size_t degree() const
    {
        return basis_degree;
    }

    static size_t size(size_t degree)
    {
        return 2*2*(degree+2)*(degree+1)/2;
    }
};
