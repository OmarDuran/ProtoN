/*
 *       /\        Matteo Cicuttin (C) 2017,2018; Guillaume Delay 2018,2019
 *      /__\       matteo.cicuttin@enpc.fr        guillaume.delay@enpc.fr
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

#include <iostream>
#include <fstream>
#include <vector>
#include <array>
#include <algorithm>
#include <numeric>
#include <cassert>
#include <cmath>
#include <memory>
#include <sstream>
#include <list>

#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <Eigen/SparseLU>
#include <unsupported/Eigen/SparseExtra>



using namespace Eigen;

#include "core/core"
#include "core/solvers"
#include "dataio/silo_io.hpp"

#include "methods/hho"
#include "methods/cuthho"

//#include "sol2/sol.hpp"

//////////////////////////    PRODUCTS    ////////////////////////////////

template<typename T, int N>
Matrix<T, Dynamic, N>
outer_product(const std::vector<Matrix<T, N, N>>& a, const Matrix<T, N, 1>& b)
{
    Matrix<T, Dynamic, N> ret(a.size(), N);
    for (size_t i = 0; i < a.size(); i++)
    {
        Matrix<T, N, 1> t = a[i] * b;
        ret.row(i)        = t.transpose();
    }
    return ret;
}

template<typename T, int N>
Matrix<T, N, N>
inner_product(const T& a, const Matrix<T, N, N>& b)
{
    return a * b;
}


template<typename T, int N>
T
inner_product(const Matrix<T, N, 1>& a, const Matrix<T, N, 1>& b)
{
    return a.dot(b);
}


template<typename T, int N>
T
inner_product(const Matrix<T, N, N>& b, const Matrix<T, N, N>& a)
{
    return a.cwiseProduct(b).sum();
}



template<typename T, int N>
Matrix<T, Dynamic, 1>
inner_product(const std::vector<Matrix<T, N, N>>& a, const Matrix<T, N, N>& b)
{

    //assert(b.cols() == b.rows() && b.cols() == N);
    
    Matrix<T, Dynamic, 1> ret(a.size(), 1);
    for (size_t i = 0; i < a.size(); i++)
    {
        ret[i]        = inner_product(a[i],b);
    }
    return ret;
}


/////////////////////////  LEVEL -- SETS  ///////////////////////////////

template<typename T>
struct circle_level_set
{
    T radius, alpha, beta;

    circle_level_set(T r, T a, T b)
        : radius(r), alpha(a), beta(b)
    {}

    T operator()(const point<T,2>& pt) const
    {
        auto x = pt.x();
        auto y = pt.y();

        return (x-alpha)*(x-alpha) + (y-beta)*(y-beta) - radius*radius;
    }

    Eigen::Matrix<T,2,1> gradient(const point<T,2>& pt) const
    {
        Eigen::Matrix<T,2,1> ret;
        ret(0) = 2*pt.x() - 2*alpha;
        ret(1) = 2*pt.y() - 2*beta;
        return ret;
    }

    Eigen::Matrix<T,2,1> normal(const point<T,2>& pt) const
    {
        Eigen::Matrix<T,2,1> ret;

        ret = gradient(pt);
        return ret/ret.norm();
    }

};

template<typename T>
struct line_level_set
{
    T cut_y;

    line_level_set(T cy)
        : cut_y(cy)
    {}

    T operator()(const point<T,2>& pt) const
    {
        auto x = pt.x();
        auto y = pt.y();

        return y - cut_y;
    }

    Eigen::Matrix<T,2,1> gradient(const point<T,2>& pt) const
    {
        Eigen::Matrix<T,2,1> ret;
        ret(0) = 0;
        ret(1) = 1;
        return ret;
    }

    Eigen::Matrix<T,2,1> normal(const point<T,2>& pt) const
    {
        Eigen::Matrix<T,2,1> ret;

        ret = gradient(pt);
        return ret/ret.norm();
    }

};


template<typename T>
struct carre_level_set
{
    T y_top, y_bot, x_left, x_right;

    carre_level_set(T yt, T yb, T xl, T xr)
        : y_top(yt), y_bot(yb), x_left(xl), x_right(xr)
    {}

    T operator()(const point<T,2>& pt) const
    {
        auto x = pt.x();
        auto y = pt.y();

        T in = 1;
        if(x > x_left && x < x_right && y > y_bot && y < y_top)
            in = 1;
        else
            in = -1;

        T dist_x = std::min( abs(x-x_left), abs(x-x_right));
        T dist_y = std::min( abs(y-y_bot), abs(y-y_top));

        
        return - in * std::min(dist_x , dist_y);
    }

    Eigen::Matrix<T,2,1> gradient(const point<T,2>& pt) const
    {
        Eigen::Matrix<T,2,1> ret;
        

        auto x = pt.x();
        auto y = pt.y();

        T dist = abs(x - x_left);
        ret(0) = -1;
        ret(1) = 0;
        
        if(abs(x - x_right) < dist )
        {
            dist = abs(x - x_right);
            ret(0) = 1;
            ret(1) = 0;
        }
        if(abs(y - y_bot) < dist )
        {
            dist = abs(y - y_bot);
            ret(0) = 0;
            ret(1) = -1;
        }
        if(abs(y - y_top) < dist)
        {
            ret(0) = 0;
            ret(1) = 1;
        }
        
        return ret;
    }

    Eigen::Matrix<T,2,1> normal(const point<T,2>& pt) const
    {
        Eigen::Matrix<T,2,1> ret;

        ret = gradient(pt);
        return ret/ret.norm();        
    }

};


/*****************************************************************************
 *   Test stuff
 *****************************************************************************/
template<typename Mesh>
void
plot_basis_functions(const Mesh& msh)
{
    using T = typename Mesh::coordinate_type;

    std::ofstream c_ofs("cell_basis_check.dat");

    for (auto cl : msh.cells)
    {
        cell_basis<Mesh, T> cb(msh, cl, 3);

        auto tps = make_test_points(msh, cl);

        for (auto& tp : tps)
        {
            c_ofs << tp.x() << " " << tp.y() << " ";

            auto vals = cb.eval_basis(tp);
            for(size_t i = 0; i < cb.size(); i++)
                c_ofs << vals(i) << " ";

            c_ofs << std::endl;
        }
    }

    c_ofs.close();

    std::ofstream f_ofs("face_basis_check.dat");

    for (auto fc : msh.faces)
    {
        face_basis<Mesh, T> fb(msh, fc, 2);

        auto tps = make_test_points(msh, fc);

        for (auto& tp : tps)
        {
            f_ofs << tp.x() << " " << tp.y() << " ";

            auto vals = fb.eval_basis(tp);
            for(size_t i = 0; i < fb.size(); i++)
                f_ofs << vals(i) << " ";

            f_ofs << std::endl;
        }
    }

    f_ofs.close();
}

template<typename Mesh>
void
plot_quadrature_points(const Mesh& msh, size_t degree)
{
    std::ofstream c_ofs("cell_quadrature_check.dat");

    for (auto& cl : msh.cells)
    {
        auto qps = integrate(msh, cl, degree);

        for (auto& qp : qps)
        {
            c_ofs << qp.first.x() << " " << qp.first.y();
            c_ofs << " " << qp.second << std::endl;
        }
    }

    c_ofs.close();

    std::ofstream f_ofs("face_quadrature_check.dat");

    for (auto& fc : msh.faces)
    {
        auto qps = integrate(msh, fc, degree);

        for (auto& qp : qps)
        {
            f_ofs << qp.first.x() << " " << qp.first.y();
            f_ofs << " " << qp.second << std::endl;
        }
    }

    f_ofs.close();
}


template<typename Mesh>
void
test_mass_matrices(const Mesh& msh, size_t degree)
{
    using T = typename Mesh::coordinate_type;

    auto rhs_fun = [](const typename Mesh::point_type& pt) -> T {
        return std::sin(M_PI*pt.x()) * std::sin(M_PI*pt.y());
    };

    std::ofstream c_ofs("cell_mass_check.dat");

    cell_basis<Mesh, T>::print_structure(degree);

    for (auto& cl : msh.cells)
    {
        Matrix<T, Dynamic, Dynamic> mass = make_mass_matrix(msh, cl, degree);
        Matrix<T, Dynamic, 1> rhs = make_rhs(msh, cl, degree, rhs_fun);
        Matrix<T, Dynamic, 1> sol = mass.llt().solve(rhs);

        cell_basis<T,T> cb(msh, cl, degree);

        auto tps = make_test_points(msh, cl);
        for (auto& tp : tps)
        {
            auto phi = cb.eval_basis(tp);
            auto val = sol.dot(phi);
            c_ofs << tp.x() << " " << tp.y() << " " << val << std::endl;
        }

    }

    c_ofs.close();


    std::ofstream f_ofs("face_mass_check.dat");

    for (auto& fc : msh.faces)
    {
        Matrix<T, Dynamic, Dynamic> mass = make_mass_matrix(msh, fc, degree);
        Matrix<T, Dynamic, 1> rhs = make_rhs(msh, fc, degree, rhs_fun);
        Matrix<T, Dynamic, 1> sol = mass.llt().solve(rhs);

        face_basis<T,T> fb(msh, fc, degree);

        auto tps = make_test_points(msh, fc);
        for (auto& tp : tps)
        {
            auto phi = fb.eval_basis(tp);
            auto val = sol.dot(phi);
            f_ofs << tp.x() << " " << tp.y() << " " << val << std::endl;
        }

    }

    f_ofs.close();
}

template<typename T, size_t ET>
void test_triangulation(const cuthho_mesh<T, ET>& msh)
{
    std::ofstream ofs("triangulation_dump.m");
    for (auto& cl : msh.cells)
    {
        if ( !is_cut(msh, cl) )
            continue;

        auto tris = triangulate(msh, cl, element_location::IN_NEGATIVE_SIDE);

        for (auto& tri : tris)
            ofs << tri << std::endl;
    }

    ofs.close();
}

template<typename T>
struct params
{
    T kappa_1, kappa_2, eta;

    params() : kappa_1(1.0), kappa_2(1.0), eta(5.0) {}
};

template<typename T, size_t ET>
T
cell_eta(const cuthho_mesh<T, ET>& msh, const typename cuthho_mesh<T, ET>::cell_type& cl)
{
    return 5.0;
}

template<typename T, size_t ET, typename Function>
std::pair<   Matrix<typename cuthho_mesh<T, ET>::coordinate_type, Dynamic, Dynamic>,
             Matrix<typename cuthho_mesh<T, ET>::coordinate_type, Dynamic, Dynamic>  >
make_hho_laplacian(const cuthho_mesh<T, ET>& msh, const typename cuthho_mesh<T, ET>::cell_type& cl,
                   const Function& level_set_function, hho_degree_info di,
                   element_location where)
{

    if ( !is_cut(msh, cl) )
        return make_hho_laplacian(msh, cl, di);

    auto recdeg = di.reconstruction_degree();
    auto celdeg = di.cell_degree();
    auto facdeg = di.face_degree();

    cell_basis<cuthho_mesh<T, ET>,T>     cb(msh, cl, recdeg);

    auto rbs = cell_basis<cuthho_mesh<T, ET>,T>::size(recdeg);
    auto cbs = cell_basis<cuthho_mesh<T, ET>,T>::size(celdeg);
    auto fbs = face_basis<cuthho_mesh<T, ET>,T>::size(facdeg);

    auto fcs = faces(msh, cl);
    auto num_faces = fcs.size();

    Matrix<T, Dynamic, Dynamic> stiff = Matrix<T, Dynamic, Dynamic>::Zero(rbs, rbs);
    Matrix<T, Dynamic, Dynamic> gr_lhs = Matrix<T, Dynamic, Dynamic>::Zero(rbs, rbs);
    Matrix<T, Dynamic, Dynamic> gr_rhs = Matrix<T, Dynamic, Dynamic>::Zero(rbs, cbs + num_faces*fbs);

    /* Cell term (cut) */
    auto qps = integrate(msh, cl, 2*recdeg, where);
    for (auto& qp : qps)
    {
        auto dphi = cb.eval_gradients(qp.first);
        stiff += qp.second * dphi * dphi.transpose();
    }

    auto hT = measure(msh, cl);

    /* Interface term */
    auto iqps = integrate_interface(msh, cl, 2*recdeg, where);
    for (auto& qp : iqps)
    {
        auto phi    = cb.eval_basis(qp.first);
        auto dphi   = cb.eval_gradients(qp.first);
        Matrix<T,2,1> n      = level_set_function.normal(qp.first);

        stiff -= qp.second * phi * (dphi * n).transpose();
        stiff -= qp.second * (dphi * n) * phi.transpose();
        stiff += qp.second * phi * phi.transpose() * cell_eta(msh, cl) / hT;
    }

    gr_lhs.block(0, 0, rbs, rbs) = stiff;
    gr_rhs.block(0, 0, rbs, cbs) = stiff.block(0, 0, rbs, cbs);

    auto ns = normals(msh, cl);
    for (size_t i = 0; i < fcs.size(); i++)
    {
        auto fc = fcs[i];
        auto n = ns[i];

        face_basis<cuthho_mesh<T, ET>,T> fb(msh, fc, facdeg);
        /* Terms on faces */
        auto qps = integrate(msh, fc, 2*recdeg, where);
        for (auto& qp : qps)
        {
            auto c_phi = cb.eval_basis(qp.first);
            auto f_phi = fb.eval_basis(qp.first);
            auto r_dphi_tmp = cb.eval_gradients(qp.first);
            auto r_dphi = r_dphi_tmp.block(0, 0, rbs, 2);
            gr_rhs.block(0, cbs+i*fbs, rbs, fbs) += qp.second * (r_dphi * n) * f_phi.transpose();
            gr_rhs.block(0, 0, rbs, cbs) -= qp.second * (r_dphi * n) * c_phi.transpose();
        }
    }

    Matrix<T, Dynamic, Dynamic> oper = gr_lhs.llt().solve(gr_rhs);
    Matrix<T, Dynamic, Dynamic> data = gr_rhs.transpose() * oper;
    return std::make_pair(oper, data);
}

template<typename T, size_t ET, typename Function>
std::pair<   Matrix<typename cuthho_mesh<T, ET>::coordinate_type, Dynamic, Dynamic>,
             Matrix<typename cuthho_mesh<T, ET>::coordinate_type, Dynamic, Dynamic>  >
make_hho_laplacian_interface(const cuthho_mesh<T, ET>& msh,
    const typename cuthho_mesh<T, ET>::cell_type& cl,
    const Function& level_set_function, hho_degree_info di, const params<T>& parms = params<T>())
{

    if ( !is_cut(msh, cl) )
        throw std::invalid_argument("The cell is not cut");

    auto recdeg = di.reconstruction_degree();
    auto celdeg = di.cell_degree();
    auto facdeg = di.face_degree();

    cell_basis<cuthho_mesh<T, ET>,T>     cb(msh, cl, recdeg);

    auto rbs = cell_basis<cuthho_mesh<T, ET>,T>::size(recdeg);
    auto cbs = cell_basis<cuthho_mesh<T, ET>,T>::size(celdeg);
    auto fbs = face_basis<cuthho_mesh<T, ET>,T>::size(facdeg);

    auto fcs = faces(msh, cl);
    auto num_faces = fcs.size();

    Matrix<T, Dynamic, Dynamic> stiff = Matrix<T, Dynamic, Dynamic>::Zero(2*rbs, 2*rbs);
    Matrix<T, Dynamic, Dynamic> gr_lhs = Matrix<T, Dynamic, Dynamic>::Zero(2*rbs, 2*rbs);
    Matrix<T, Dynamic, Dynamic> gr_rhs = Matrix<T, Dynamic, Dynamic>::Zero(2*rbs, 2*(cbs + num_faces*fbs));

    /* Cell term (cut) */

    auto qps_n = integrate(msh, cl, 2*recdeg, element_location::IN_NEGATIVE_SIDE);
    for (auto& qp : qps_n)
    {
        auto dphi = cb.eval_gradients(qp.first);
        stiff.block(0,0,rbs,rbs) += parms.kappa_1 * qp.second * dphi * dphi.transpose();
    }

    auto qps_p = integrate(msh, cl, 2*recdeg, element_location::IN_POSITIVE_SIDE);
    for (auto& qp : qps_p)
    {
        auto dphi = cb.eval_gradients(qp.first);
        stiff.block(rbs,rbs,rbs,rbs) += parms.kappa_2 * qp.second * dphi * dphi.transpose();
    }

    auto hT = measure(msh, cl);

    /* Interface term */
    auto iqps = integrate_interface(msh, cl, 2*recdeg, element_location::IN_NEGATIVE_SIDE);
    for (auto& qp : iqps)
    {
        auto phi        = cb.eval_basis(qp.first);
        auto dphi       = cb.eval_gradients(qp.first);
        Matrix<T,2,1> n = level_set_function.normal(qp.first);

        Matrix<T, Dynamic, Dynamic> a = parms.kappa_1 * qp.second * phi * (dphi * n).transpose();
        Matrix<T, Dynamic, Dynamic> b = parms.kappa_1 * qp.second * (dphi * n) * phi.transpose();
        Matrix<T, Dynamic, Dynamic> c = parms.kappa_1 * qp.second * phi * phi.transpose() * parms.eta / hT;

        stiff.block(  0,   0, rbs, rbs) -= a;
        stiff.block(rbs,   0, rbs, rbs) += a;

        stiff.block(  0,   0, rbs, rbs) -= b;
        stiff.block(  0, rbs, rbs, rbs) += b;

        stiff.block(  0,   0, rbs, rbs) += c;
        stiff.block(  0, rbs, rbs, rbs) -= c;
        stiff.block(rbs,   0, rbs, rbs) -= c;
        stiff.block(rbs, rbs, rbs, rbs) += c;

    }

    gr_lhs = stiff;
    gr_rhs.block(0,   0, 2*rbs, cbs) = stiff.block(0,   0, 2*rbs, cbs);
    gr_rhs.block(0, cbs, 2*rbs, cbs) = stiff.block(0, rbs, 2*rbs, cbs);

    auto ns = normals(msh, cl);
    for (size_t i = 0; i < fcs.size(); i++)
    {
        auto fc = fcs[i];
        auto n = ns[i];

        face_basis<cuthho_mesh<T, ET>,T> fb(msh, fc, facdeg);
        /* Terms on faces */
        auto qps_n = integrate(msh, fc, 2*recdeg, element_location::IN_NEGATIVE_SIDE);
        for (auto& qp : qps_n)
        {
            auto c_phi = cb.eval_basis(qp.first);
            auto f_phi = fb.eval_basis(qp.first);
            auto r_dphi = cb.eval_gradients(qp.first);

            gr_rhs.block(0, 0, rbs, cbs) -= parms.kappa_1 * qp.second * (r_dphi * n) * c_phi.transpose();
            size_t col_ofs = 2*cbs + i*fbs;
            gr_rhs.block(0, col_ofs, rbs, fbs) += parms.kappa_1 * qp.second * (r_dphi * n) * f_phi.transpose();
        }

        auto qps_p = integrate(msh, fc, 2*recdeg, element_location::IN_POSITIVE_SIDE);
        for (auto& qp : qps_p)
        {
            auto c_phi = cb.eval_basis(qp.first);
            auto f_phi = fb.eval_basis(qp.first);
            auto r_dphi = cb.eval_gradients(qp.first);

            gr_rhs.block(rbs, cbs, rbs, cbs) -= parms.kappa_2 * qp.second * (r_dphi * n) * c_phi.transpose();
            size_t col_ofs = 2*cbs + fbs*fcs.size() + i*fbs;
            gr_rhs.block(rbs, col_ofs, rbs, fbs) += parms.kappa_2 * qp.second * (r_dphi * n) * f_phi.transpose();
        }
    }

    Matrix<T, Dynamic, Dynamic> oper = gr_lhs.ldlt().solve(gr_rhs);
    Matrix<T, Dynamic, Dynamic> data = gr_rhs.transpose() * oper;

    return std::make_pair(oper, data);
}


//////////////////////  VECTOR GRADREC  /////////////////////////////


template<typename T, size_t ET>
std::pair<   Matrix<typename cuthho_mesh<T, ET>::coordinate_type, Dynamic, Dynamic>,
             Matrix<typename cuthho_mesh<T, ET>::coordinate_type, Dynamic, Dynamic>  >
make_hho_gradrec_vector(const cuthho_mesh<T, ET>& msh, const typename cuthho_mesh<T, ET>::cell_type& cl, const hho_degree_info& di)
{
    typedef Matrix<T, Dynamic, Dynamic> matrix_type;
    typedef Matrix<T, Dynamic, 1>       vector_type;

    const auto celdeg  = di.cell_degree();
    const auto facdeg  = di.face_degree();
    const auto graddeg = di.grad_degree();

    cell_basis<cuthho_mesh<T, ET>,T>            cb(msh, cl, celdeg);
    vector_cell_basis<cuthho_mesh<T, ET>,T>     gb(msh, cl, graddeg);
    
    auto cbs = cell_basis<cuthho_mesh<T, ET>,T>::size(celdeg);
    auto fbs = face_basis<cuthho_mesh<T, ET>,T>::size(facdeg);
    auto gbs = vector_cell_basis<cuthho_mesh<T, ET>,T>::size(graddeg);
    
    const auto num_faces = faces(msh, cl).size();

    matrix_type         gr_lhs = matrix_type::Zero(gbs, gbs);
    matrix_type         gr_rhs = matrix_type::Zero(gbs, cbs + num_faces * fbs);

    if(celdeg > 0)
    {
        const auto qps = integrate(msh, cl, celdeg - 1 + facdeg);
        for (auto& qp : qps)
        {
            const auto c_dphi = cb.eval_gradients(qp.first);
            const auto g_phi  = gb.eval_basis(qp.first);

            
            gr_lhs.block(0, 0, gbs, gbs) += qp.second * g_phi * g_phi.transpose();
            gr_rhs.block(0, 0, gbs, cbs) += qp.second * g_phi * c_dphi.transpose();
        }
    }

    const auto fcs = faces(msh, cl);
    const auto ns = normals(msh, cl);
    for (size_t i = 0; i < fcs.size(); i++)
    {
        const auto fc = fcs[i];
        const auto n  = ns[i];
        face_basis<cuthho_mesh<T, ET>,T> fb(msh, fc, facdeg);

        const auto qps_f = integrate(msh, fc, facdeg + std::max(facdeg, celdeg));
        for (auto& qp : qps_f)
        {
            const vector_type c_phi      = cb.eval_basis(qp.first);
            const vector_type f_phi      = fb.eval_basis(qp.first);
            const auto        g_phi      = gb.eval_basis(qp.first);
            const vector_type qp_g_phi_n = qp.second * g_phi * n;

            gr_rhs.block(0, cbs + i * fbs, gbs, fbs) += qp_g_phi_n * f_phi.transpose();
            gr_rhs.block(0, 0, gbs, cbs) -= qp_g_phi_n * c_phi.transpose();
        }
    }

    matrix_type oper = gr_lhs.ldlt().solve(gr_rhs);
    matrix_type data = gr_rhs.transpose() * oper;

    return std::make_pair(oper, data);
}


template<typename T, size_t ET, typename Function>
std::pair<   Matrix<typename cuthho_mesh<T, ET>::coordinate_type, Dynamic, Dynamic>,
             Matrix<typename cuthho_mesh<T, ET>::coordinate_type, Dynamic, Dynamic>  >
make_hho_gradrec_vector(const cuthho_mesh<T, ET>& msh, const typename cuthho_mesh<T, ET>::cell_type& cl, const Function& level_set_function, const hho_degree_info& di, element_location where)
{

    if ( !is_cut(msh, cl) )
        return make_hho_gradrec_vector(msh, cl, di);

    typedef Matrix<T, Dynamic, Dynamic> matrix_type;
    typedef Matrix<T, Dynamic, 1>       vector_type;

    const auto celdeg  = di.cell_degree();
    const auto facdeg  = di.face_degree();
    const auto graddeg = di.grad_degree();
    
    cell_basis<cuthho_mesh<T, ET>,T>            cb(msh, cl, celdeg);
    vector_cell_basis<cuthho_mesh<T, ET>,T>     gb(msh, cl, graddeg);


    auto cbs = cell_basis<cuthho_mesh<T, ET>,T>::size(celdeg);
    auto fbs = face_basis<cuthho_mesh<T, ET>,T>::size(facdeg);
    auto gbs = vector_cell_basis<cuthho_mesh<T, ET>,T>::size(graddeg);
   
    
    const auto num_faces = faces(msh, cl).size(); 
    
    matrix_type        gr_lhs = matrix_type::Zero(gbs, gbs);
    matrix_type        gr_rhs = matrix_type::Zero(gbs, cbs + num_faces * fbs);


    
    const auto qps = integrate(msh, cl, celdeg - 1 + facdeg, where);
    for (auto& qp : qps)
    {
        const auto c_dphi = cb.eval_gradients(qp.first);
        const auto g_phi  = gb.eval_basis(qp.first);

        gr_lhs.block(0, 0, gbs, gbs) += qp.second * g_phi * g_phi.transpose();
        gr_rhs.block(0, 0, gbs, cbs) += qp.second * g_phi * c_dphi.transpose();
    }
    


    const auto fcs = faces(msh, cl);
    const auto ns = normals(msh, cl);
    for (size_t i = 0; i < fcs.size(); i++)
    {
        const auto fc = fcs[i];
        const auto n  = ns[i];
        face_basis<cuthho_mesh<T, ET>,T> fb(msh, fc, facdeg);
        

        const auto qps_f = integrate(msh, fc, facdeg + std::max(facdeg, celdeg), where);
        for (auto& qp : qps_f)
        {
            const vector_type c_phi      = cb.eval_basis(qp.first);
            const vector_type f_phi      = fb.eval_basis(qp.first);
            const auto        g_phi      = gb.eval_basis(qp.first);
            const vector_type qp_g_phi_n = qp.second * g_phi * n;

            gr_rhs.block(0, cbs + i * fbs, gbs, fbs) += qp_g_phi_n * f_phi.transpose();
            gr_rhs.block(0, 0, gbs, cbs) -= qp_g_phi_n * c_phi.transpose();
        }
    }


    const auto iqps = integrate_interface(msh, cl, celdeg + graddeg, element_location::IN_NEGATIVE_SIDE);
    for (auto& qp : iqps)
    {
        const auto c_phi        = cb.eval_basis(qp.first);
        const auto g_phi        = gb.eval_basis(qp.first);

        Matrix<T,2,1> n = level_set_function.normal(qp.first);
        const vector_type qp_g_phi_n = qp.second * g_phi * n;
        
        gr_rhs.block(0 , 0, gbs, cbs) -= qp_g_phi_n * c_phi.transpose();
    }
    
    matrix_type oper = gr_lhs.ldlt().solve(gr_rhs);
    matrix_type data = gr_rhs.transpose() * oper;

    return std::make_pair(oper, data);
}


template<typename T, size_t ET, typename Function>
std::pair<   Matrix<typename cuthho_mesh<T, ET>::coordinate_type, Dynamic, Dynamic>,
             Matrix<typename cuthho_mesh<T, ET>::coordinate_type, Dynamic, Dynamic>  >
make_hho_gradrec_vector_interface(const cuthho_mesh<T, ET>& msh,
                                  const typename cuthho_mesh<T, ET>::cell_type& cl,
                                  const Function& level_set_function, const hho_degree_info& di,
                                  element_location where)
{

    if ( !is_cut(msh, cl) )
        throw std::invalid_argument("The cell is not cut");

    typedef Matrix<T, Dynamic, Dynamic> matrix_type;
    typedef Matrix<T, Dynamic, 1>       vector_type;

    const auto celdeg  = di.cell_degree();
    const auto facdeg  = di.face_degree();
    const auto graddeg = di.grad_degree();
    
    cell_basis<cuthho_mesh<T, ET>,T>            cb(msh, cl, celdeg);
    vector_cell_basis<cuthho_mesh<T, ET>,T>     gb(msh, cl, graddeg);


    auto cbs = cell_basis<cuthho_mesh<T, ET>,T>::size(celdeg);
    auto fbs = face_basis<cuthho_mesh<T, ET>,T>::size(facdeg);
    auto gbs = vector_cell_basis<cuthho_mesh<T, ET>,T>::size(graddeg);
   
    
    const auto num_faces = faces(msh, cl).size(); 

    matrix_type       rhs_tmp = matrix_type::Zero(gbs, cbs + num_faces * fbs);
    matrix_type        gr_lhs = matrix_type::Zero(gbs, gbs);
    matrix_type        gr_rhs = matrix_type::Zero(gbs, 2*cbs + 2*num_faces * fbs);


    
    const auto qps = integrate(msh, cl, celdeg - 1 + facdeg, where);
    for (auto& qp : qps)
    {
        const auto c_dphi = cb.eval_gradients(qp.first);
        const auto g_phi  = gb.eval_basis(qp.first);

        gr_lhs.block(0, 0, gbs, gbs) += qp.second * g_phi * g_phi.transpose();
        rhs_tmp.block(0, 0, gbs, cbs) += qp.second * g_phi * c_dphi.transpose();
    }
    


    const auto fcs = faces(msh, cl);
    const auto ns = normals(msh, cl);
    for (size_t i = 0; i < fcs.size(); i++)
    {
        const auto fc = fcs[i];
        const auto n  = ns[i];
        face_basis<cuthho_mesh<T, ET>,T> fb(msh, fc, facdeg);
        

        const auto qps_f = integrate(msh, fc, facdeg + std::max(facdeg, celdeg), where);
        for (auto& qp : qps_f)
        {
            const vector_type c_phi      = cb.eval_basis(qp.first);
            const vector_type f_phi      = fb.eval_basis(qp.first);
            const auto        g_phi      = gb.eval_basis(qp.first);
            const vector_type qp_g_phi_n = qp.second * g_phi * n;

            rhs_tmp.block(0, cbs + i * fbs, gbs, fbs) += qp_g_phi_n * f_phi.transpose();
            rhs_tmp.block(0, 0, gbs, cbs) -= qp_g_phi_n * c_phi.transpose();
        }
    }

    
    const auto iqps = integrate_interface(msh, cl, celdeg + graddeg, element_location::IN_NEGATIVE_SIDE);
    for (auto& qp : iqps)
    {
        const auto c_phi        = cb.eval_basis(qp.first);
        const auto g_phi        = gb.eval_basis(qp.first);

        Matrix<T,2,1> n = level_set_function.normal(qp.first);
        const vector_type qp_g_phi_n = qp.second * g_phi * n;
        
        gr_rhs.block(0 , 0, gbs, cbs) -= qp_g_phi_n * c_phi.transpose();
        gr_rhs.block(0 , cbs, gbs, cbs) += qp_g_phi_n * c_phi.transpose();
    }

    if(where == element_location::IN_NEGATIVE_SIDE)
    {
        gr_rhs.block(0, 0, gbs, cbs) += rhs_tmp.block(0, 0, gbs, cbs);
        gr_rhs.block(0, 2*cbs, gbs, num_faces*fbs)
            += rhs_tmp.block(0, cbs, gbs, num_faces*fbs);
    }
    else if( where == element_location::IN_POSITIVE_SIDE)
    {
        gr_rhs.block(0, cbs, gbs, cbs) += rhs_tmp.block(0, 0, gbs, cbs);
        gr_rhs.block(0, 2*cbs + num_faces*fbs, gbs, num_faces*fbs)
                     += rhs_tmp.block(0, cbs, gbs, num_faces*fbs);
    }
    
    matrix_type oper = gr_lhs.ldlt().solve(gr_rhs);
    matrix_type data = gr_rhs.transpose() * oper;

    return std::make_pair(oper, data);
}


//////////////////////  MATRIX GRADREC  /////////////////////////////
//
// The versions of gradrec_matrix written here can be optimized
// by taking into account the structure of the basis.
// We can then define it by using gradrec_vector
//

template<typename T, size_t ET>
std::pair<   Matrix<typename cuthho_mesh<T, ET>::coordinate_type, Dynamic, Dynamic>,
             Matrix<typename cuthho_mesh<T, ET>::coordinate_type, Dynamic, Dynamic>  >
make_hho_gradrec_matrix(const cuthho_mesh<T, ET>& msh, const typename cuthho_mesh<T, ET>::cell_type& cl, const hho_degree_info& di)
{   
    typedef Matrix<T, Dynamic, Dynamic> matrix_type;
    typedef Matrix<T, Dynamic, 1>       vector_type;

    const auto celdeg  = di.cell_degree();
    const auto facdeg  = di.face_degree();
    const auto graddeg = di.grad_degree();

    vector_cell_basis<cuthho_mesh<T, ET>,T>            cb(msh, cl, celdeg);
    matrix_cell_basis<cuthho_mesh<T, ET>,T>            gb(msh, cl, graddeg);
    
    auto cbs = vector_cell_basis<cuthho_mesh<T, ET>,T>::size(celdeg);
    auto fbs = vector_face_basis<cuthho_mesh<T, ET>,T>::size(facdeg);
    auto gbs = matrix_cell_basis<cuthho_mesh<T, ET>,T>::size(graddeg);
    
    const auto num_faces = faces(msh, cl).size();

    matrix_type         gr_lhs = matrix_type::Zero(gbs, gbs);
    matrix_type         gr_rhs = matrix_type::Zero(gbs, cbs + num_faces * fbs);
    

    // lhs
    const auto qps = integrate(msh, cl, celdeg - 1 + facdeg);
    for (auto& qp : qps)
    {   
        const auto g_phi  = gb.eval_basis(qp.first);

        // expensive computation -> can be improved
        for (size_t j = 0; j < gbs; j ++)
        {
            const auto qp_gphi_j = inner_product(qp.second, g_phi[j]);
            for (size_t i = j; i < gbs; i ++)
            {
                gr_lhs(i, j) += inner_product(g_phi[i], qp_gphi_j);
            }
        }        
    }

    // upper part
    for (size_t j = 0; j < gbs; j++)
    {
        for (size_t i = 0; i < j; i++)
        {
            gr_lhs(i, j) = gr_lhs(j, i);
        }
    }

    // rhs
    if(celdeg > 0)
    {
        const auto qps = integrate(msh, cl, celdeg - 1 + facdeg);
        for (auto& qp : qps)
        {
            const auto c_dphi = cb.eval_gradients(qp.first);
            const auto g_phi  = gb.eval_basis(qp.first);
            
            for (size_t j = 0; j < cbs; j++)
            {
                const auto qp_dphi_j = inner_product(qp.second, c_dphi[j]);
                for (size_t i = 0; i < gbs; i++)
                {
                    gr_rhs(i, j) += inner_product(g_phi[i], qp_dphi_j);
                }
            }
        }
    }
    
    const auto fcs = faces(msh, cl);
    const auto ns = normals(msh, cl);
    for (size_t i = 0; i < fcs.size(); i++)
    {
        const auto fc = fcs[i];
        const auto n  = ns[i];
        vector_face_basis<cuthho_mesh<T, ET>,T> fb(msh, fc, facdeg);

        const auto qps_f = integrate(msh, fc, facdeg + std::max(facdeg, celdeg));
        for (auto& qp : qps_f)
        {
            const auto     c_phi      = cb.eval_basis(qp.first);
            const auto     f_phi      = fb.eval_basis(qp.first);
            const auto     g_phi      = gb.eval_basis(qp.first);
            const matrix_type     qp_g_phi_n = qp.second * outer_product(g_phi, n);

            gr_rhs.block(0, cbs + i * fbs, gbs, fbs) += qp_g_phi_n * f_phi.transpose();
            gr_rhs.block(0, 0, gbs, cbs) -= qp_g_phi_n * c_phi.transpose();
        }
    }

    matrix_type oper = gr_lhs.ldlt().solve(gr_rhs);
    matrix_type data = gr_rhs.transpose() * oper;
    
    return std::make_pair(oper, data);    
}




template<typename T, size_t ET, typename Function>
std::pair<   Matrix<typename cuthho_mesh<T, ET>::coordinate_type, Dynamic, Dynamic>,
             Matrix<typename cuthho_mesh<T, ET>::coordinate_type, Dynamic, Dynamic>  >
make_hho_gradrec_matrix(const cuthho_mesh<T, ET>& msh, const typename cuthho_mesh<T, ET>::cell_type& cl, const Function& level_set_function, const hho_degree_info& di, element_location where)
{

    if ( !is_cut(msh, cl) )
        return make_hho_gradrec_matrix(msh, cl, di);
            
    typedef Matrix<T, Dynamic, Dynamic> matrix_type;
    typedef Matrix<T, Dynamic, 1>       vector_type;

    const auto celdeg  = di.cell_degree();
    const auto facdeg  = di.face_degree();
    const auto graddeg = di.grad_degree();
    
    vector_cell_basis<cuthho_mesh<T, ET>,T>            cb(msh, cl, celdeg);
    matrix_cell_basis<cuthho_mesh<T, ET>,T>            gb(msh, cl, graddeg);


    auto cbs = vector_cell_basis<cuthho_mesh<T, ET>,T>::size(celdeg);
    auto fbs = vector_face_basis<cuthho_mesh<T, ET>,T>::size(facdeg);
    auto gbs = matrix_cell_basis<cuthho_mesh<T, ET>,T>::size(graddeg);
   
    
    const auto num_faces = faces(msh, cl).size(); 
    
    matrix_type        gr_lhs = matrix_type::Zero(gbs, gbs);
    matrix_type        gr_rhs = matrix_type::Zero(gbs, cbs + num_faces * fbs);
    
    // lhs
    const auto qps = integrate(msh, cl, celdeg - 1 + facdeg, where);
    for (auto& qp : qps)
    {        
        const auto g_phi  = gb.eval_basis(qp.first);

        // expensive computation -> can be improved
        for (size_t j = 0; j < gbs; j++)
        {
            const auto qp_gphi_j = inner_product(qp.second, g_phi[j]);
            for (size_t i = j; i < gbs; i++)
            {
                gr_lhs(i, j) += inner_product(g_phi[i], qp_gphi_j);
            }
        }        
    }
    // upper part
    for (size_t j = 0; j < gbs; j++)
    {
        for (size_t i = 0; i < j; i++)
        {
            gr_lhs(i, j) = gr_lhs(j, i);
        }
    }

    
    // rhs
    if(celdeg > 0)
    {
        const auto qps = integrate(msh, cl, celdeg - 1 + facdeg, where);
        for (auto& qp : qps)
        {
            const auto c_dphi = cb.eval_gradients(qp.first);
            const auto g_phi  = gb.eval_basis(qp.first);
            
            for (size_t j = 0; j < cbs; j++)
            {
                const auto qp_dphi_j = inner_product(qp.second, c_dphi[j]);
                for (size_t i = 0; i < gbs; i++)
                {
                    gr_rhs(i, j) += inner_product(g_phi[i], qp_dphi_j);
                }
            }
        }
    }

    const auto fcs = faces(msh, cl);
    const auto ns = normals(msh, cl);
    for (size_t i = 0; i < fcs.size(); i++)
    {
        const auto fc = fcs[i];
        const auto n  = ns[i];
        vector_face_basis<cuthho_mesh<T, ET>,T> fb(msh, fc, facdeg);
        

        const auto qps_f = integrate(msh, fc, facdeg + std::max(facdeg, celdeg), where);
        for (auto& qp : qps_f)
        {
            const auto     c_phi      = cb.eval_basis(qp.first);
            const auto     f_phi      = fb.eval_basis(qp.first);
            const auto     g_phi      = gb.eval_basis(qp.first);
            const matrix_type     qp_g_phi_n = qp.second * outer_product(g_phi, n);

            gr_rhs.block(0, cbs + i * fbs, gbs, fbs) += qp_g_phi_n * f_phi.transpose();
            gr_rhs.block(0, 0, gbs, cbs) -= qp_g_phi_n * c_phi.transpose();
        }
    }


    const auto iqps = integrate_interface(msh, cl, celdeg + graddeg, element_location::IN_NEGATIVE_SIDE);
    for (auto& qp : iqps)
    {
        const auto c_phi        = cb.eval_basis(qp.first);
        const auto g_phi        = gb.eval_basis(qp.first);

        const Matrix<T,2,1>   n = level_set_function.normal(qp.first);
        const matrix_type  qp_g_phi_n = qp.second * outer_product(g_phi, n);
        
        gr_rhs.block(0 , 0, gbs, cbs) -= qp_g_phi_n * c_phi.transpose();
    }
    
    matrix_type oper = gr_lhs.ldlt().solve(gr_rhs);
    matrix_type data = gr_rhs.transpose() * oper;

    
    return std::make_pair(oper, data);
}


//////////////////  DIVERGENCE RECONSTRUCTION  //////////////////////////



template<typename T, size_t ET>
std::pair<   Matrix<typename cuthho_mesh<T, ET>::coordinate_type, Dynamic, Dynamic>,
             Matrix<typename cuthho_mesh<T, ET>::coordinate_type, Dynamic, Dynamic>  >
make_hho_divergence_reconstruction(const cuthho_mesh<T, ET>& msh, const typename cuthho_mesh<T, ET>::cell_type& cl, const hho_degree_info& di)
{
    typedef Matrix<T, Dynamic, Dynamic> matrix_type;

    const auto celdeg = di.cell_degree();
    const auto facdeg = di.face_degree();
    const auto recdeg = di.face_degree();

    cell_basis<cuthho_mesh<T, ET>,T>                   rb(msh, cl, recdeg);
    vector_cell_basis<cuthho_mesh<T, ET>,T>            cb(msh, cl, celdeg);

    auto rbs = cell_basis<cuthho_mesh<T, ET>,T>::size(recdeg);
    auto fbs = vector_face_basis<cuthho_mesh<T, ET>,T>::size(facdeg);
    auto cbs = vector_cell_basis<cuthho_mesh<T, ET>,T>::size(celdeg);

    const auto fcs = faces(msh, cl);
    const auto num_faces = fcs.size();
    const auto ns = normals(msh, cl);

    matrix_type dr_lhs = matrix_type::Zero(rbs, rbs);
    matrix_type dr_rhs = matrix_type::Zero(rbs, cbs + num_faces*fbs);


    const auto qps = integrate(msh, cl, celdeg + recdeg - 1);
    for (auto& qp : qps)
    {
        const auto s_phi  = rb.eval_basis(qp.first);
        const auto s_dphi = rb.eval_gradients(qp.first);
        const auto v_phi  = cb.eval_basis(qp.first);

        dr_lhs += qp.second * s_phi * s_phi.transpose();
        dr_rhs.block(0, 0, rbs, cbs) -= qp.second * s_dphi * v_phi.transpose();
    }


    for (size_t i = 0; i < fcs.size(); i++)
    {
        const auto fc     = fcs[i];
        const auto n      = ns[i];
        vector_face_basis<cuthho_mesh<T, ET>,T>            fb(msh, fc, facdeg);

        const auto qps_f = integrate(msh, fc, facdeg + recdeg);
        for (auto& qp : qps_f)
        {
            const auto s_phi = rb.eval_basis(qp.first);
            const auto f_phi = fb.eval_basis(qp.first);

            const Matrix<T, Dynamic, 2> s_phi_n = (s_phi * n.transpose());
            dr_rhs.block(0, cbs + i * fbs, rbs, fbs) += qp.second * s_phi_n * f_phi.transpose();
        }
    }

    
    assert(dr_lhs.rows() == rbs && dr_lhs.cols() == rbs);
    assert(dr_rhs.rows() == rbs && dr_rhs.cols() == cbs + num_faces * fbs);

    matrix_type oper = dr_lhs.ldlt().solve(dr_rhs);
    matrix_type data = dr_rhs;
    // matrix_type data = dr_rhs.transpose() * oper; used in diskpp -> wierd

    return std::make_pair(oper, data);
}


template<typename T, size_t ET, typename Function>
std::pair<   Matrix<typename cuthho_mesh<T, ET>::coordinate_type, Dynamic, Dynamic>,
             Matrix<typename cuthho_mesh<T, ET>::coordinate_type, Dynamic, Dynamic>  >
make_hho_divergence_reconstruction(const cuthho_mesh<T, ET>& msh, const typename cuthho_mesh<T, ET>::cell_type& cl, const Function& level_set_function, const hho_degree_info& di, element_location where)
{

    if ( !is_cut(msh, cl) )
        return make_hho_divergence_reconstruction(msh, cl, di);
    

    typedef Matrix<T, Dynamic, Dynamic> matrix_type;

    const auto celdeg = di.cell_degree();
    const auto facdeg = di.face_degree();
    const auto recdeg = di.face_degree();

    cell_basis<cuthho_mesh<T, ET>,T>                   rb(msh, cl, recdeg);
    vector_cell_basis<cuthho_mesh<T, ET>,T>            cb(msh, cl, celdeg);

    auto rbs = cell_basis<cuthho_mesh<T, ET>,T>::size(recdeg);
    auto fbs = vector_face_basis<cuthho_mesh<T, ET>,T>::size(facdeg);
    auto cbs = vector_cell_basis<cuthho_mesh<T, ET>,T>::size(celdeg);

    const auto fcs = faces(msh, cl);
    const auto num_faces = fcs.size();
    const auto ns = normals(msh, cl);

    matrix_type dr_lhs = matrix_type::Zero(rbs, rbs);
    matrix_type dr_rhs = matrix_type::Zero(rbs, cbs + num_faces*fbs);


    // Matrix<T, Dynamic, Dynamic> Ic = Matrix<T, Dynamic, Dynamic>::Identity(cbs, cbs);
    Matrix<T, 2, 2> Ic = Matrix<T, 2, 2>::Zero();
    Ic(0,0) = 1.0;    Ic(1,1) = 1.0;
    
    const auto qps = integrate(msh, cl, celdeg + recdeg - 1, where);
    for (auto& qp : qps)
    {
        const auto s_phi  = rb.eval_basis(qp.first);
        const auto s_dphi = rb.eval_gradients(qp.first);
        const auto v_phi  = cb.eval_basis(qp.first);
        const auto dv_phi = cb.eval_gradients(qp.first);

        dr_lhs += qp.second * s_phi * s_phi.transpose();
        dr_rhs.block(0, 0, rbs, cbs) -= qp.second * s_dphi * v_phi.transpose();

        // Matrix<T, Dynamic, 1> div = inner_product(dv_phi, Ic);
        
        // dr_rhs.block(0, 0, rbs, cbs) += qp.second * s_phi * div.transpose();
    }


    auto iqp = integrate_interface(msh, cl, celdeg + recdeg, element_location::IN_NEGATIVE_SIDE);
    for (auto& qp : iqp)
    {
        const auto v_phi = cb.eval_basis(qp.first);
        const auto s_phi = rb.eval_basis(qp.first);
        const auto n = level_set_function.normal(qp.first);

        const Matrix<T, Dynamic, 2> s_phi_n = (s_phi * n.transpose());
        
        // dr_rhs.block(0, 0, rbs, cbs) += qp.second * s_phi_n * v_phi.transpose();
    }
    
    

    for (size_t i = 0; i < fcs.size(); i++)
    {
        const auto fc     = fcs[i];
        const auto n      = ns[i];
        vector_face_basis<cuthho_mesh<T, ET>,T>            fb(msh, fc, facdeg);

        const auto qps_f = integrate(msh, fc, celdeg + recdeg, where);
        for (auto& qp : qps_f)
        {
            const auto v_phi = cb.eval_basis(qp.first);
            const auto s_phi = rb.eval_basis(qp.first);
            const auto f_phi = fb.eval_basis(qp.first);

            const Matrix<T, Dynamic, 2> s_phi_n = (s_phi * n.transpose());
            dr_rhs.block(0, cbs + i * fbs, rbs, fbs) += qp.second * s_phi_n * f_phi.transpose();
            // dr_rhs.block(0, 0, rbs, cbs) += qp.second * s_phi_n * v_phi.transpose();
        }
    }

    
    assert(dr_lhs.rows() == rbs && dr_lhs.cols() == rbs);
    assert(dr_rhs.rows() == rbs && dr_rhs.cols() == cbs + num_faces * fbs);

    matrix_type data = dr_rhs;
    matrix_type oper = dr_lhs.ldlt().solve(dr_rhs);
    
    // matrix_type data = dr_rhs.transpose() * oper; used in diskpp -> wierd

    return std::make_pair(oper, data);
}

////////////////////   TESTS   ///////////////////////


template<typename T, size_t ET, typename Function>
Matrix<T, Dynamic, 1>
check_eigs(const cuthho_mesh<T, ET>& msh, const typename cuthho_mesh<T, ET>::cell_type& cl,
           const Function& level_set_function, hho_degree_info di,
           element_location where)
{
    auto recdeg = di.reconstruction_degree();
    auto celdeg = di.cell_degree();
    auto facdeg = di.face_degree();

    cell_basis<cuthho_mesh<T, ET>,T>     cb(msh, cl, recdeg);

    auto rbs = cell_basis<cuthho_mesh<T, ET>,T>::size(recdeg);
    auto cbs = cell_basis<cuthho_mesh<T, ET>,T>::size(celdeg);
    auto fbs = face_basis<cuthho_mesh<T, ET>,T>::size(facdeg);

    Matrix<T, Dynamic, Dynamic> stiff = Matrix<T, Dynamic, Dynamic>::Zero(rbs, rbs);

    /* Cell term (cut) */
    auto qps = integrate(msh, cl, 2*recdeg, where);
    for (auto& qp : qps)
    {
        auto dphi = cb.eval_gradients(qp.first);
        stiff += qp.second * dphi * dphi.transpose();
    }

    if ( is_cut(msh, cl) )
    {

        auto hT = measure(msh, cl);

        /* Interface term */
        auto iqps = integrate_interface(msh, cl, 2*recdeg, where);
        for (auto& qp : iqps)
        {
            auto phi    = cb.eval_basis(qp.first);
            auto dphi   = cb.eval_gradients(qp.first);
            Matrix<T,2,1> n      = level_set_function.normal(qp.first);

            
            stiff -= qp.second * phi * (dphi * n).transpose();
            stiff -= qp.second * (dphi * n) * phi.transpose();
            stiff += qp.second * phi * phi.transpose() * cell_eta(msh, cl) / hT;
        }
    }

    SelfAdjointEigenSolver<Matrix<T, Dynamic, Dynamic>> solver;
    
    if ( is_cut(msh, cl) )
        solver.compute(stiff);
    else
        solver.compute(stiff.block(1, 1, rbs-1, rbs-1));

    return solver.eigenvalues();
}



///////////////////////   STABILIZATION   ///////////////////////////////

template<typename T, size_t ET>
Matrix<typename cuthho_mesh<T, ET>::coordinate_type, Dynamic, Dynamic>
make_hho_cut_stabilization(const cuthho_mesh<T, ET>& msh,
                           const typename cuthho_mesh<T, ET>::cell_type& cl,
                           const hho_degree_info& di, element_location where,
                           const params<T>& parms = params<T>())
{
    if ( !is_cut(msh, cl) )
        return make_hho_naive_stabilization(msh, cl, di);

    auto celdeg = di.cell_degree();
    auto facdeg = di.face_degree();

    auto cbs = cell_basis<cuthho_mesh<T, ET>,T>::size(celdeg);
    auto fbs = face_basis<cuthho_mesh<T, ET>,T>::size(facdeg);

    auto fcs = faces(msh, cl);
    auto num_faces = fcs.size();

    Matrix<T, Dynamic, Dynamic> data = Matrix<T, Dynamic, Dynamic>::Zero(cbs+num_faces*fbs, cbs+num_faces*fbs);
    Matrix<T, Dynamic, Dynamic> If = Matrix<T, Dynamic, Dynamic>::Identity(fbs, fbs);

    cell_basis<cuthho_mesh<T, ET>,T> cb(msh, cl, celdeg);

    auto hT = measure(msh, cl);


    for (size_t i = 0; i < num_faces; i++)
    {
        auto fc = fcs[i];
        face_basis<cuthho_mesh<T, ET>,T> fb(msh, fc, facdeg);

        Matrix<T, Dynamic, Dynamic> oper = Matrix<T, Dynamic, Dynamic>::Zero(fbs, cbs+num_faces*fbs);
        Matrix<T, Dynamic, Dynamic> mass = Matrix<T, Dynamic, Dynamic>::Zero(fbs, fbs);
        Matrix<T, Dynamic, Dynamic> trace = Matrix<T, Dynamic, Dynamic>::Zero(fbs, cbs);

        oper.block(0, cbs+i*fbs, fbs, fbs) = -If;

        auto qps = integrate(msh, fc, 2*facdeg, where);
        for (auto& qp : qps)
        {
            auto c_phi = cb.eval_basis(qp.first);
            auto f_phi = fb.eval_basis(qp.first);

            mass += qp.second * f_phi * f_phi.transpose();
            trace += qp.second * f_phi * c_phi.transpose();
        }

        if (qps.size() == 0) /* Avoid to invert a zero matrix */
            continue;

        oper.block(0, 0, fbs, cbs) = mass.llt().solve(trace);

        data += oper.transpose() * mass * oper * (1./hT);
    }


    auto iqps = integrate_interface(msh, cl, 2*celdeg, element_location::IN_NEGATIVE_SIDE);
    for (auto& qp : iqps)
    {
        const auto c_phi  = cb.eval_basis(qp.first);
        
        data.block(0, 0, cbs, cbs) += qp.second * c_phi * c_phi.transpose() * parms.eta / hT;
    }
    
    return data;
}




template<typename T, size_t ET, typename Function>
Matrix<typename cuthho_mesh<T, ET>::coordinate_type, Dynamic, Dynamic>
make_hho_stabilization_interface(const cuthho_mesh<T, ET>& msh,
                                 const typename cuthho_mesh<T, ET>::cell_type& cl,
                                 const Function& level_set_function,
                                 const hho_degree_info& di, const params<T>& parms = params<T>())
{
    if ( !is_cut(msh, cl) )
        throw std::invalid_argument("The cell is not cut ...");

    auto celdeg = di.cell_degree();
    auto facdeg = di.face_degree();

    auto cbs = cell_basis<cuthho_mesh<T, ET>,T>::size(celdeg);
    auto fbs = face_basis<cuthho_mesh<T, ET>,T>::size(facdeg);

    auto fcs = faces(msh, cl);
    auto num_faces = fcs.size();

    Matrix<T, Dynamic, Dynamic> data
        = Matrix<T, Dynamic, Dynamic>::Zero(2*cbs+2*num_faces*fbs, 2*cbs+2*num_faces*fbs);

    cell_basis<cuthho_mesh<T, ET>,T> cb(msh, cl, celdeg);

    auto hT = measure(msh, cl);


    const auto stab_n = make_hho_cut_stabilization(msh, cl, di,element_location::IN_NEGATIVE_SIDE);
    const auto stab_p = make_hho_cut_stabilization(msh, cl, di,element_location::IN_POSITIVE_SIDE);

    // cells--cells
    data.block(0, 0, cbs, cbs) += parms.kappa_1 * stab_n.block(0, 0, cbs, cbs);
    data.block(cbs, cbs, cbs, cbs) += parms.kappa_2 * stab_p.block(0, 0, cbs, cbs);
    // cells--faces
    data.block(0, 2*cbs, cbs, num_faces*fbs)
        += parms.kappa_1 * stab_n.block(0, cbs, cbs, num_faces*fbs);
    data.block(cbs, 2*cbs + num_faces*fbs, cbs, num_faces*fbs)
        += parms.kappa_2 * stab_p.block(0, cbs, cbs, num_faces*fbs);
    // faces--cells
    data.block(2*cbs, 0, num_faces*fbs, cbs)
        += parms.kappa_1 * stab_n.block(cbs, 0, num_faces*fbs, cbs);
    data.block(2*cbs + num_faces*fbs, cbs, num_faces*fbs, cbs)
        += parms.kappa_2 * stab_p.block(cbs, 0, num_faces*fbs, cbs);
    // faces--faces
    data.block(2*cbs, 2*cbs, num_faces*fbs, num_faces*fbs)
        += parms.kappa_1 * stab_n.block(cbs, cbs, num_faces*fbs, num_faces*fbs);
    data.block(2*cbs + num_faces*fbs, 2*cbs + num_faces*fbs, num_faces*fbs, num_faces*fbs)
        += parms.kappa_2 * stab_p.block(cbs, cbs, num_faces*fbs, num_faces*fbs);



    // complementary terms on the interface (cells--cells)
    Matrix<T, Dynamic, Dynamic> term_1 = Matrix<T, Dynamic, Dynamic>::Zero(cbs, cbs);
    Matrix<T, Dynamic, Dynamic> term_2 = Matrix<T, Dynamic, Dynamic>::Zero(cbs, cbs);
    
    auto iqps = integrate_interface(msh, cl, 2*celdeg, element_location::IN_NEGATIVE_SIDE);
    for (auto& qp : iqps)
    {
        const auto c_phi  = cb.eval_basis(qp.first);
        const auto dphi   = cb.eval_gradients(qp.first);
        const Matrix<T,2,1> n      = level_set_function.normal(qp.first);
        
        term_1 += qp.second * c_phi * c_phi.transpose() * parms.eta / hT;
        term_2 += qp.second * c_phi * (dphi * n).transpose();
        
    }    

    data.block(0, cbs, cbs, cbs) -= parms.kappa_1 * term_1;
    data.block(cbs, 0, cbs, cbs) -= parms.kappa_1 * term_1;
    data.block(cbs, cbs, cbs, cbs) += (parms.kappa_1 - parms.kappa_2) * term_1;

    data.block(0, cbs, cbs, cbs) += parms.kappa_2 * term_2;
    data.block(cbs, 0, cbs, cbs) += parms.kappa_2 * term_2.transpose();
    data.block(cbs, cbs, cbs, cbs) -= parms.kappa_2 * term_2;
    data.block(cbs, cbs, cbs, cbs) -= parms.kappa_2 * term_2.transpose();
    
    return data;
}

///////////////////   STABILIZATION VECT  ////////////////////
// possibility of optimization by using the scalar stabilization
// possibility to merge it with scalar case (not done in diskpp)


template<typename Mesh>
Matrix<typename Mesh::coordinate_type, Dynamic, Dynamic>
make_hho_vector_naive_stabilization(const Mesh& msh, const typename Mesh::cell_type& cl, const hho_degree_info& di)
{
    using T = typename Mesh::coordinate_type;

    auto celdeg = di.cell_degree();
    auto facdeg = di.face_degree();

    auto cbs = vector_cell_basis<Mesh,T>::size(celdeg);
    auto fbs = vector_face_basis<Mesh,T>::size(facdeg);

    auto fcs = faces(msh, cl);

    size_t msize = cbs+fcs.size()*fbs;
    Matrix<T, Dynamic, Dynamic> data = Matrix<T, Dynamic, Dynamic>::Zero(msize, msize);
    Matrix<T, Dynamic, Dynamic> If = Matrix<T, Dynamic, Dynamic>::Identity(fbs, fbs);

    vector_cell_basis<Mesh,T> cb(msh, cl, celdeg);

    // auto h = measure(msh, cl);

    for (size_t i = 0; i < fcs.size(); i++)
    {
        auto fc = fcs[i];
        vector_face_basis<Mesh,T> fb(msh, fc, facdeg);

        auto h = measure(msh, fc);
        
        Matrix<T, Dynamic, Dynamic> oper = Matrix<T, Dynamic, Dynamic>::Zero(fbs, msize);
        Matrix<T, Dynamic, Dynamic> mass = Matrix<T, Dynamic, Dynamic>::Zero(fbs, fbs);
        Matrix<T, Dynamic, Dynamic> trace = Matrix<T, Dynamic, Dynamic>::Zero(fbs, cbs);

        oper.block(0, cbs+i*fbs, fbs, fbs) = -If;

        auto qps = integrate(msh, fc, 2*facdeg);
        for (auto& qp : qps)
        {
            auto c_phi = cb.eval_basis(qp.first);
            auto f_phi = fb.eval_basis(qp.first);

            mass += qp.second * f_phi * f_phi.transpose();
            trace += qp.second * f_phi * c_phi.transpose();
        }

        oper.block(0, 0, fbs, cbs) = mass.llt().solve(trace);

        data += oper.transpose() * mass * oper * (1./h);
    }

    return data;
}



template<typename T, size_t ET, typename Function>
Matrix<typename cuthho_mesh<T, ET>::coordinate_type, Dynamic, Dynamic>
make_hho_vector_cut_stabilization(const cuthho_mesh<T, ET>& msh,
                           const typename cuthho_mesh<T, ET>::cell_type& cl,
                           const hho_degree_info& di, element_location where,
                                  const Function& level_set_function,
                           const params<T>& parms = params<T>())
{
    if ( !is_cut(msh, cl) )
        return make_hho_vector_naive_stabilization(msh, cl, di);

    auto celdeg = di.cell_degree();
    auto facdeg = di.face_degree();

    auto cbs = vector_cell_basis<cuthho_mesh<T, ET>,T>::size(celdeg);
    auto fbs = vector_face_basis<cuthho_mesh<T, ET>,T>::size(facdeg);

    auto fcs = faces(msh, cl);
    auto num_faces = fcs.size();

    Matrix<T, Dynamic, Dynamic> data = Matrix<T, Dynamic, Dynamic>::Zero(cbs+num_faces*fbs, cbs+num_faces*fbs);
    Matrix<T, Dynamic, Dynamic> If = Matrix<T, Dynamic, Dynamic>::Identity(fbs, fbs);

    vector_cell_basis<cuthho_mesh<T, ET>,T> cb(msh, cl, celdeg);

    auto hT = measure(msh, cl);


    for (size_t i = 0; i < num_faces; i++)
    {
        auto fc = fcs[i];
        vector_face_basis<cuthho_mesh<T, ET>,T> fb(msh, fc, facdeg);


        auto hF = measure(msh, fc);
        
        Matrix<T, Dynamic, Dynamic> oper = Matrix<T, Dynamic, Dynamic>::Zero(fbs, cbs+num_faces*fbs);
        Matrix<T, Dynamic, Dynamic> mass = Matrix<T, Dynamic, Dynamic>::Zero(fbs, fbs);
        Matrix<T, Dynamic, Dynamic> trace = Matrix<T, Dynamic, Dynamic>::Zero(fbs, cbs);

        oper.block(0, cbs+i*fbs, fbs, fbs) = -If;

        auto qps = integrate(msh, fc, 2*facdeg, where);
        for (auto& qp : qps)
        {
            auto c_phi = cb.eval_basis(qp.first);
            auto f_phi = fb.eval_basis(qp.first);

            mass += qp.second * f_phi * f_phi.transpose();
            trace += qp.second * f_phi * c_phi.transpose();
        }

        if (qps.size() == 0) /* Avoid to invert a zero matrix */
            continue;

        oper.block(0, 0, fbs, cbs) = mass.llt().solve(trace);

        // data += oper.transpose() * mass * oper * (1./hT);
        data += oper.transpose() * mass * oper * (1./hF);
    }


    auto iqps = integrate_interface(msh, cl, 2*celdeg, element_location::IN_NEGATIVE_SIDE);
    for (auto& qp : iqps)
    {
        const auto c_phi  = cb.eval_basis(qp.first);
        const auto d_phi  = cb.eval_gradients(qp.first);
        
        const Matrix<T,2,1> n      = level_set_function.normal(qp.first);
        
        const Matrix<T, Dynamic, Dynamic> d_phi_n = outer_product(d_phi, n);
        
        data.block(0, 0, cbs, cbs) +=
            sqrt(2) * qp.second * c_phi * c_phi.transpose() * parms.eta / hT;
        // data.block(0, 0, cbs, cbs) -= qp.second * d_phi_n * c_phi.transpose();
        // data.block(0, 0, cbs, cbs) -= qp.second * c_phi * d_phi_n.transpose();
    }
    
    return data;
}


///////////////////// VECTOR MASS MATRIX //////////////////////

template<typename Mesh, typename T = typename Mesh::coordinate_type>
Matrix<T, Dynamic, Dynamic>
make_vector_mass_matrix(const Mesh& msh, const typename Mesh::face_type& fc, size_t degree, size_t di = 0)
{
    vector_face_basis<Mesh,T> fb(msh, fc, degree);
    auto fbs = fb.size();

    Matrix<T, Dynamic, Dynamic> ret = Matrix<T, Dynamic, Dynamic>::Zero(fbs, fbs);

    auto qps = integrate(msh, fc, 2*(degree+di));

    for (auto& qp : qps)
    {
        auto phi = fb.eval_basis(qp.first);
        ret += qp.second * phi * phi.transpose();
    }

    return ret;
}

//////////////////////////  RHS  //////////////////////////////


template<typename T, size_t ET, typename F1, typename F2, typename F3>
Matrix<typename cuthho_mesh<T, ET>::coordinate_type, Dynamic, 1>
make_rhs(const cuthho_mesh<T, ET>& msh, const typename cuthho_mesh<T, ET>::cell_type& cl,
         size_t degree, const F1& f, const element_location where, const F2& level_set_function, const F3& bcs, Matrix<T, Dynamic, Dynamic> GR)
{
    if ( location(msh, cl) == where )
        return make_rhs(msh, cl, degree, f);
    else if ( location(msh, cl) == element_location::ON_INTERFACE )
    {
        
        cell_basis<cuthho_mesh<T, ET>,T> cb(msh, cl, degree);
        auto cbs = cb.size();

        vector_cell_basis<cuthho_mesh<T, ET>,T> gb(msh, cl, degree-1);
        auto gbs = gb.size();

        
        auto hT = measure(msh, cl);

        Matrix<T, Dynamic, 1> ret = Matrix<T, Dynamic, 1>::Zero(GR.cols());
        Matrix<T, Dynamic, 1> source_vect = Matrix<T, Dynamic, 1>::Zero(gbs);
        Matrix<T, Dynamic, 1> grad_term = Matrix<T, Dynamic, 1>::Zero(GR.cols());

        auto qps = integrate(msh, cl, 2*degree, where);
        for (auto& qp : qps)
        {
            auto phi = cb.eval_basis(qp.first);
            ret.block(0, 0, cbs, 1) += qp.second * phi * f(qp.first);
        }


        auto qpsi = integrate_interface(msh, cl, 2*degree, element_location::IN_NEGATIVE_SIDE);
        for (auto& qp : qpsi)
        {
            auto phi = cb.eval_basis(qp.first);
            auto dphi = cb.eval_gradients(qp.first);
            auto n = level_set_function.normal(qp.first);
            const auto g_phi  = gb.eval_basis(qp.first);
            
            ret.block(0, 0, cbs, 1)
                += qp.second * bcs(qp.first) * phi * cell_eta(msh, cl)/hT;
            
            source_vect += qp.second * bcs(qp.first) * g_phi * n;
        }


        grad_term = source_vect.transpose() * GR;

        ret -= grad_term;

        return ret;
    }
    else
    {
        auto cbs = cell_basis<cuthho_mesh<T, ET>,T>::size(degree);
        Matrix<T, Dynamic, 1> ret = Matrix<T, Dynamic, 1>::Zero(cbs);
        return ret;
    }
}


template<typename T, size_t ET, typename F1>
Matrix<typename cuthho_mesh<T, ET>::coordinate_type, Dynamic, 1>
make_vector_rhs(const cuthho_mesh<T, ET>& msh, const typename cuthho_mesh<T, ET>::cell_type& cl,
         size_t degree, const F1& f, size_t di = 0)
{
    vector_cell_basis<cuthho_mesh<T, ET>,T> cb(msh, cl, degree);
    auto cbs = cb.size();

    Matrix<T, Dynamic, 1> ret = Matrix<T, Dynamic, 1>::Zero(cbs);

    auto qps = integrate(msh, cl, 2*(degree+di));

    for (auto& qp : qps)
    {
        auto phi = cb.eval_basis(qp.first);
        ret += qp.second * phi * f(qp.first);
    }

    return ret;
}

template<typename Mesh, typename Function>
Matrix<typename Mesh::coordinate_type, Dynamic, 1>
make_vector_rhs(const Mesh& msh, const typename Mesh::face_type& fc,
         size_t degree, const Function& f, size_t di = 0)
{
    using T = typename Mesh::coordinate_type;

    vector_face_basis<Mesh,T> fb(msh, fc, degree);
    auto fbs = fb.size();

    Matrix<T, Dynamic, 1> ret = Matrix<T, Dynamic, 1>::Zero(fbs);

    auto qps = integrate(msh, fc, 2*(degree+di));

    for (auto& qp : qps)
    {
        auto phi = fb.eval_basis(qp.first);
        ret += qp.second * phi * f(qp.first);
    }

    return ret;
}

template<typename T, size_t ET, typename F1, typename F2, typename F3>
Matrix<typename cuthho_mesh<T, ET>::coordinate_type, Dynamic, 1>
make_vector_rhs(const cuthho_mesh<T, ET>& msh, const typename cuthho_mesh<T, ET>::cell_type& cl,
         size_t degree, const F1& f, const element_location where, const F2& level_set_function, const F3& bcs, Matrix<T, Dynamic, Dynamic> GR)
{
    if ( location(msh, cl) == where )
        return make_vector_rhs(msh, cl, degree, f);
    else if ( location(msh, cl) == element_location::ON_INTERFACE )
    {
        
        vector_cell_basis<cuthho_mesh<T, ET>,T> cb(msh, cl, degree);
        auto cbs = cb.size();

        matrix_cell_basis<cuthho_mesh<T, ET>,T> gb(msh, cl, degree-1);
        auto gbs = gb.size();

        
        auto hT = measure(msh, cl);

        Matrix<T, Dynamic, 1> ret = Matrix<T, Dynamic, 1>::Zero(GR.cols());
        Matrix<T, Dynamic, 1> source_vect = Matrix<T, Dynamic, 1>::Zero(gbs);
        Matrix<T, Dynamic, 1> grad_term = Matrix<T, Dynamic, 1>::Zero(GR.cols());

        auto qps = integrate(msh, cl, 2*degree, where);
        for (auto& qp : qps)
        {
            auto phi = cb.eval_basis(qp.first);
            ret.block(0, 0, cbs, 1) += qp.second * phi * f(qp.first);
        }

        
        auto qpsi = integrate_interface(msh, cl, 2*degree, element_location::IN_NEGATIVE_SIDE);
        for (auto& qp : qpsi)
        {
            auto phi = cb.eval_basis(qp.first);
            auto dphi = cb.eval_gradients(qp.first);
            auto n = level_set_function.normal(qp.first);
            const auto g_phi  = gb.eval_basis(qp.first);

            const Matrix<T, Dynamic, Dynamic> dphi_n = outer_product(dphi, n);
            
            ret.block(0, 0, cbs, 1)
                += qp.second * sqrt(2) * ( cell_eta(msh, cl)/hT * phi ) * bcs(qp.first);

            // ret.block(0, 0, cbs, 1)
            //     += qp.second  * ( sqrt(2) * cell_eta(msh, cl)/hT * phi - dphi_n ) * bcs(qp.first);

            // ret.block(0, 0, cbs, 1)
            //      += qp.second  * ( cell_eta(msh, cl)/hT * phi - dphi_n ) * bcs(qp.first);
            
            source_vect += qp.second * outer_product(g_phi, n) * bcs(qp.first);
        }
        
        grad_term = source_vect.transpose() * GR;

        ret -= grad_term;

        return ret;
    }
    else
    {
        auto cbs = vector_cell_basis<cuthho_mesh<T, ET>,T>::size(degree);
        Matrix<T, Dynamic, 1> ret = Matrix<T, Dynamic, 1>::Zero(cbs);
        return ret;
    }
}



template<typename T, size_t ET, typename F1, typename F2>
Matrix<typename cuthho_mesh<T, ET>::coordinate_type, Dynamic, 1>
make_pressure_rhs(const cuthho_mesh<T, ET>& msh, const typename cuthho_mesh<T, ET>::cell_type& cl,
         size_t degree, const element_location where, const F1& level_set_function, const F2& bcs)
{
    if( location(msh, cl) != element_location::ON_INTERFACE )
    {
        auto cbs = cell_basis<cuthho_mesh<T, ET>,T>::size(degree);
        Matrix<T, Dynamic, 1> ret = Matrix<T, Dynamic, 1>::Zero(cbs);
        return ret;
    }
    else
    {
        cell_basis<cuthho_mesh<T, ET>,T> cb(msh, cl, degree);
        auto cbs = cb.size();
        Matrix<T, Dynamic, 1> ret = Matrix<T, Dynamic, 1>::Zero(cbs);

        auto qpsi = integrate_interface(msh, cl, 2*degree, element_location::IN_NEGATIVE_SIDE );
        for (auto& qp : qpsi)
        {
            auto phi = cb.eval_basis(qp.first);
            auto n = level_set_function.normal(qp.first);
            
            ret -= qp.second * bcs(qp.first).dot(n) * phi;
        }
        
        return ret;
    }
}


template<typename T, size_t ET, typename F1>
Matrix<typename cuthho_mesh<T, ET>::coordinate_type, Dynamic, 1>
make_flux_jump(const cuthho_mesh<T, ET>& msh, const typename cuthho_mesh<T, ET>::cell_type& cl,
                   size_t degree, const element_location where, const F1& flux_jump)
{
    cell_basis<cuthho_mesh<T, ET>,T> cb(msh, cl, degree);
    auto cbs = cb.size();
    Matrix<T, Dynamic, 1> ret = Matrix<T, Dynamic, 1>::Zero(cbs);

    if( location(msh, cl) != element_location::ON_INTERFACE )
        return ret;
    
    if(where == element_location::IN_POSITIVE_SIDE) {
        auto qpsi = integrate_interface(msh, cl, 2*degree, element_location::IN_NEGATIVE_SIDE);
            
        for (auto& qp : qpsi)
        {
            auto phi = cb.eval_basis(qp.first);
            ret += qp.second * flux_jump(qp.first) * phi;
        }
    }
    return ret;
}


template<typename T, size_t ET, typename F1, typename F2>
Matrix<typename cuthho_mesh<T, ET>::coordinate_type, Dynamic, 1>
make_Dirichlet_jump(const cuthho_mesh<T, ET>& msh, const typename cuthho_mesh<T, ET>::cell_type& cl,
                   size_t degree, const element_location where, const F1& level_set_function, 
                    const F2& dir_jump, const params<T>& parms = params<T>())
{
    cell_basis<cuthho_mesh<T, ET>,T> cb(msh, cl, degree);
    auto cbs = cb.size();
    Matrix<T, Dynamic, 1> ret = Matrix<T, Dynamic, 1>::Zero(cbs);

    if( location(msh, cl) != element_location::ON_INTERFACE )
        return ret;

    auto hT = measure(msh, cl);
    
    if(where == element_location::IN_NEGATIVE_SIDE) {
        auto qpsi = integrate_interface(msh, cl, 2*degree, element_location::IN_NEGATIVE_SIDE );
	    
        for (auto& qp : qpsi)
        {
            auto phi = cb.eval_basis(qp.first);

            ret += qp.second * dir_jump(qp.first) * parms.kappa_1 * phi * cell_eta(msh, cl)/hT;
        }
    }
    else if(where == element_location::IN_POSITIVE_SIDE) {
        auto qpsi = integrate_interface(msh, cl, 2*degree, element_location::IN_NEGATIVE_SIDE );
            
        for (auto& qp : qpsi)
        {
            auto phi = cb.eval_basis(qp.first);
            auto dphi = cb.eval_gradients(qp.first);
            auto n = level_set_function.normal(qp.first);
            
            ret += qp.second * dir_jump(qp.first) * (parms.kappa_2 * dphi * n
                                                     - parms.kappa_1 * phi * cell_eta(msh, cl)/hT);
        }
    }
    return ret;
}


template<typename T>
std::string quiver(const point<T,2>& p, const Eigen::Matrix<T,2,1>& v)
{
    std::stringstream ss;

    ss << "quiver(" << p.x() << ", " << p.y() << ", ";
    ss << v(0) << ", " << v(1) << ", 0);";

    return ss.str();
}

template<typename T, size_t ET, typename Function1, typename Function2>
std::pair<T, T>
test_integration(const cuthho_mesh<T, ET>& msh, const Function1& f, const Function2& level_set_function)
{
    T surf_int_val = 0.0;
    T line_int_val = 0.0;

    std::ofstream ofs("normals.m");

    for (auto& cl : msh.cells)
    {
        bool in_negative_side = (location(msh, cl) == element_location::IN_NEGATIVE_SIDE);
        bool on_interface = (location(msh, cl) == element_location::ON_INTERFACE);
        if ( !in_negative_side && !on_interface )
            continue;

        auto qpts = integrate(msh, cl, 1, element_location::IN_NEGATIVE_SIDE);

        for (auto& qp : qpts)
            surf_int_val += qp.second * f(qp.first);

        if (on_interface)
        {
            auto iqpts = integrate_interface(msh, cl, 1, element_location::IN_NEGATIVE_SIDE);
            for (auto& qp : iqpts)
            {
                line_int_val += qp.second * f(qp.first);
                auto n = level_set_function.normal(qp.first);
                ofs << quiver(qp.first, n) << std::endl;
            }
        }

    }

    ofs.close();

    std::ofstream ofs_int("face_ints.m");

    for (auto& fc : msh.faces)
    {
        auto qpts = integrate(msh, fc, 2, element_location::IN_NEGATIVE_SIDE);
        for (auto& qp : qpts)
        {
            ofs_int << "hold on;" << std::endl;
            ofs_int << "plot(" << qp.first.x() << ", " << qp.first.y() << ", 'ko');" << std::endl;
        }
    }

    ofs_int.close();

    return std::make_pair(surf_int_val, line_int_val);
}




template<typename T>
class postprocess_output_object {

public:
    postprocess_output_object()
    {}

    virtual bool write() = 0;
};

template<typename T>
class silo_output_object : public postprocess_output_object<T>
{

};

template<typename T>
class gnuplot_output_object : public postprocess_output_object<T>
{
    std::string                                 output_filename;
    std::vector< std::pair< point<T,2>, T > >   data;

public:
    gnuplot_output_object(const std::string& filename)
        : output_filename(filename)
    {}

    void add_data(const point<T,2>& pt, const T& val)
    {
        data.push_back( std::make_pair(pt, val) );
    }

    bool write()
    {
        std::ofstream ofs(output_filename);

        for (auto& d : data)
            ofs << d.first.x() << " " << d.first.y() << " " << d.second << std::endl;

        ofs.close();

        return true;
    }
};


template<typename T>
class postprocess_output
{
    std::list< std::shared_ptr< postprocess_output_object<T>> >     postprocess_objects;

public:
    postprocess_output()
    {}

    void add_object( std::shared_ptr<postprocess_output_object<T>> obj )
    {
        postprocess_objects.push_back( obj );
    }

    bool write(void) const
    {
        for (auto& obj : postprocess_objects)
            obj->write();

        return true;
    }
};

template<typename Mesh, typename Function>
void
run_cuthho_fictdom(const Mesh& msh, const Function& level_set_function, size_t degree)
{
    using RealType = typename Mesh::coordinate_type;


    
    /************** OPEN SILO DATABASE **************/
    silo_database silo;
    silo.create("cuthho_stokes.silo");
    silo.add_mesh(msh, "mesh");

    /************** MAKE A SILO VARIABLE FOR CELL POSITIONING **************/
    std::vector<RealType> cut_cell_markers;
    for (auto& cl : msh.cells)
    {
        if ( location(msh, cl) == element_location::IN_POSITIVE_SIDE )
            cut_cell_markers.push_back(1.0);
        else if ( location(msh, cl) == element_location::IN_NEGATIVE_SIDE )
            cut_cell_markers.push_back(-1.0);
        else if ( location(msh, cl) == element_location::ON_INTERFACE )
            cut_cell_markers.push_back(0.0);
        else
            throw std::logic_error("shouldn't have arrived here...");
    }
    silo.add_variable("mesh", "cut_cells", cut_cell_markers.data(), cut_cell_markers.size(), zonal_variable_t);

    /************** MAKE A SILO VARIABLE FOR LEVEL SET FUNCTION **************/
    std::vector<RealType> level_set_vals;
    for (auto& pt : msh.points)
        level_set_vals.push_back( level_set_function(pt) );
    silo.add_variable("mesh", "level_set", level_set_vals.data(), level_set_vals.size(), nodal_variable_t);

    /************** MAKE A SILO VARIABLE FOR NODE POSITIONING **************/
    std::vector<RealType> node_pos;
    for (auto& n : msh.nodes)
        node_pos.push_back( location(msh, n) == element_location::IN_POSITIVE_SIDE ? +1.0 : -1.0 );
    silo.add_variable("mesh", "node_pos", node_pos.data(), node_pos.size(), nodal_variable_t);

    
    /************** DEFINE PROBLEM RHS, SOLUTION AND BCS **************/
#if 0  // null velocity on the boundary
    auto rhs_fun = [](const typename cuthho_poly_mesh<RealType>::point_type& pt) -> auto {
        Matrix<RealType, 2, 1> ret;

        RealType x1 = pt.x() - 0.5;
        RealType x2 = x1 * x1;
        RealType y1 = pt.y() - 0.5;
        RealType y2 = y1 * y1;

        RealType r = sqrt(x2 + y2);
        
        RealType A = 32 - 18.0 / (3.0 * r);
        
        ret(0) = A * y1 + 5.* x2 * x2;
        ret(1) = - A * x1 + 5.* y2 * y2;

        return ret;
    };

    auto sol_vel = [](const typename cuthho_poly_mesh<RealType>::point_type& pt) -> auto {
        Matrix<RealType, 2, 1> ret;

        RealType x1 = pt.x() - 0.5;
        RealType x2 = x1 * x1;
        RealType y1 = pt.y() - 0.5;
        RealType y2 = y1 * y1;

        RealType r = sqrt(x2 + y2);
        
        RealType B = 2. * (1./3. - r) * ( 1./3. - 2.*r);
        
        ret(0) =  - B * y1;
        ret(1) = B * x1;

        return ret;
    };

    auto sol_grad = [](const typename cuthho_poly_mesh<RealType>::point_type& pt) -> auto {
        Matrix<RealType, 2, 2> ret;

        RealType x1 = pt.x() - 0.5;
        RealType x2 = x1 * x1;
        RealType y1 = pt.y() - 0.5;
        RealType y2 = y1 * y1;

        
        RealType r = sqrt(x2 + y2);
        
        RealType B = 2. * (1./3. - r) * ( 1./3. - 2.*r);
        RealType C = 8.0 - 6.0 / (3.0 * r);

        
        
        ret(0,0) = - C * x1 * y1;
        ret(0,1) = - B - C * y2;
        ret(1,0) = B + C * x2;
        ret(1,1) = C * x1 * y1;
        
        return ret;
    };

    auto bcs_fun = [&](const typename cuthho_poly_mesh<RealType>::point_type& pt) -> auto {
        return sol_vel(pt);
    };

    auto pressure =  [](const typename cuthho_poly_mesh<RealType>::point_type& pt) -> RealType {
        return std::pow(pt.x() - 0.5, 5.)  +  std::pow(pt.y() - 0.5, 5.);
    };
#elif 1  // non null velocity on the boundary
    auto rhs_fun = [](const typename cuthho_poly_mesh<RealType>::point_type& pt) -> auto {
        Matrix<RealType, 2, 1> ret;

        RealType x1 = pt.x() - 0.5;
        RealType x2 = x1 * x1;
        RealType y1 = pt.y() - 0.5;
        RealType y2 = y1 * y1;

        RealType ax =  x2 * (x2 - 2. * x1 + 1.);
        RealType ay =  y2 * (y2 - 2. * y1 + 1.);
        RealType bx =  x1 * (4. * x2 - 6. * x1 + 2.);
        RealType by =  y1 * (4. * y2 - 6. * y1 + 2.);
        RealType cx = 12. * x2 - 12.* x1 + 2.;
        RealType cy = 12. * y2 - 12.* y1 + 2.;
        RealType dx = 24. * x1 - 12.;
        RealType dy = 24. * y1 - 12.;

        ret(0) = - cx * by - ax * dy + 5.* x2 * x2;
        ret(1) = + cy * bx + ay * dx + 5.* y2 * y2;

        return ret;
    };

    auto sol_vel = [](const typename cuthho_poly_mesh<RealType>::point_type& pt) -> auto {
        Matrix<RealType, 2, 1> ret;

        RealType x1 = pt.x() - 0.5;
        RealType x2 = x1 * x1;
        RealType y1 = pt.y() - 0.5;
        RealType y2 = y1 * y1;

        ret(0) =  x2 * (x2 - 2. * x1 + 1.)  * y1 * (4. * y2 - 6. * y1 + 2.);
        ret(1) = -y2 * (y2 - 2. * y1 + 1. ) * x1 * (4. * x2 - 6. * x1 + 2.);

        return ret;
    };

    auto sol_grad = [](const typename cuthho_poly_mesh<RealType>::point_type& pt) -> auto {
        Matrix<RealType, 2, 2> ret;

        RealType x1 = pt.x() - 0.5;
        RealType x2 = x1 * x1;
        RealType y1 = pt.y() - 0.5;
        RealType y2 = y1 * y1;
        
        ret(0,0) = x1 * (4. * x2 - 6. * x1 + 2.) * y1 * (4. * y2 - 6. * y1 + 2.);
        ret(0,1) = x2 * ( x2 - 2. * x1 + 1.) * (12. * y2 - 12. * y1 + 2.);
        ret(1,0) = - y2 * ( y2 - 2. * y1 + 1.) * (12. * x2 - 12. * x1 + 2.);
        ret(1,1) = - ret(0,0);
        
        return ret;
    };

    auto bcs_fun = [&](const typename cuthho_poly_mesh<RealType>::point_type& pt) -> auto {
        return sol_vel(pt);
    };

    auto pressure =  [](const typename cuthho_poly_mesh<RealType>::point_type& pt) -> RealType {
        return std::pow(pt.x() - 0.5, 5.)  +  std::pow(pt.y() - 0.5, 5.);
    };
#elif 0  // test on a rectangle -> adapt the level-set
    auto rhs_fun = [](const typename cuthho_poly_mesh<RealType>::point_type& pt) -> auto {
        Matrix<RealType, 2, 1> ret;

        RealType mid_y = (0. + 1.0) / 2.;
        
        RealType x1 = pt.x() - 0.5;
        RealType x2 = x1 * x1;
        RealType y1 = pt.y() - mid_y;
        RealType y2 = y1 * y1;

        RealType ax =  x2 * (x2 - 2. * x1 + 1.);
        RealType ay =  y2 * (y2 - 2. * y1 + 1.);
        RealType bx =  x1 * (4. * x2 - 6. * x1 + 2.);
        RealType by =  y1 * (4. * y2 - 6. * y1 + 2.);
        RealType cx = 12. * x2 - 12.* x1 + 2.;
        RealType cy = 12. * y2 - 12.* y1 + 2.;
        RealType dx = 24. * x1 - 12.;
        RealType dy = 24. * y1 - 12.;

        ret(0) = - cx * by - ax * dy + 5.* x2 * x2;
        ret(1) = + cy * bx + ay * dx + 5.* y2 * y2;

        return ret;
    };

    auto sol_vel = [](const typename cuthho_poly_mesh<RealType>::point_type& pt) -> auto {
        Matrix<RealType, 2, 1> ret;

        RealType mid_y = (0. + 1.0) / 2.;
        
        RealType x1 = pt.x() - 0.5;
        RealType x2 = x1 * x1;
        RealType y1 = pt.y() - mid_y;
        RealType y2 = y1 * y1;

        ret(0) =  x2 * (x2 - 2. * x1 + 1.)  * y1 * (4. * y2 - 6. * y1 + 2.);
        ret(1) = -y2 * (y2 - 2. * y1 + 1. ) * x1 * (4. * x2 - 6. * x1 + 2.);

        return ret;
    };

    auto sol_grad = [](const typename cuthho_poly_mesh<RealType>::point_type& pt) -> auto {
        Matrix<RealType, 2, 2> ret;

        RealType mid_y = (0. + 1.0) / 2.;
        
        RealType x1 = pt.x() - 0.5;
        RealType x2 = x1 * x1;
        RealType y1 = pt.y() - mid_y;
        RealType y2 = y1 * y1;
        
        ret(0,0) = x1 * (4. * x2 - 6. * x1 + 2.) * y1 * (4. * y2 - 6. * y1 + 2.);
        ret(0,1) = x2 * ( x2 - 2. * x1 + 1.) * (12. * y2 - 12. * y1 + 2.);
        ret(1,0) = - y2 * ( y2 - 2. * y1 + 1.) * (12. * x2 - 12. * x1 + 2.);
        ret(1,1) = - ret(0,0);
        
        return ret;
    };

    auto bcs_fun = [&](const typename cuthho_poly_mesh<RealType>::point_type& pt) -> auto {
        return sol_vel(pt);
    };

    auto pressure =  [](const typename cuthho_poly_mesh<RealType>::point_type& pt) -> RealType {

        RealType mid_y = (0. + 1.0) / 2.;
        
        return std::pow(pt.x() - 0.5, 5.)  +  std::pow(pt.y() - mid_y, 5.);
    };
#elif 0  // test on an immersed square -> adapt the level-set
    //          + homogeneous BC
    auto rhs_fun = [](const typename cuthho_poly_mesh<RealType>::point_type& pt) -> auto {
        Matrix<RealType, 2, 1> ret;

        RealType x_max = 1.0;
        RealType x_min = 0.0;
        RealType coeff = 2.*M_PI/(x_max - x_min);
        
        RealType X = (pt.x() - x_min) / (x_max - x_min);
        RealType Y = (pt.y() - x_min) / (x_max - x_min);

        RealType sin_x = std::sin(2.*M_PI*X);
        RealType sin_y = std::sin(2.*M_PI*Y);
        RealType cos_x = std::cos(2.*M_PI*X);
        RealType cos_y = std::cos(2.*M_PI*Y);

        ret(0) = - coeff*coeff*(2.*sin_y * cos_y * cos_x * cos_x
                              - 6. * sin_y * cos_y * sin_x * sin_x)
        + 5. * std::pow(pt.x() - 0.5, 4.);
        
        ret(1) = - coeff * coeff * ( 6. * sin_y * sin_y * cos_x * sin_x
                                   - 2. * sin_x * cos_x * cos_y * cos_y)
        + 5. * std::pow(pt.y() - 0.5, 4.);


        // ret(0) = - coeff*coeff*(2.*sin_y * cos_y * cos_x * cos_x
        //                       - 6. * sin_y * cos_y * sin_x * sin_x)
        // + 5. * std::pow(pt.x(), 4.);
        
        // ret(1) = - coeff * coeff * ( 6. * sin_y * sin_y * cos_x * sin_x
        //                            - 2. * sin_x * cos_x * cos_y * cos_y)
        // + 5. * std::pow(pt.y(), 4.);
        
        
        return ret;
    };

    auto sol_vel = [](const typename cuthho_poly_mesh<RealType>::point_type& pt) -> auto {
        Matrix<RealType, 2, 1> ret;


        RealType x_max = 1.0;
        RealType x_min = 0.0;
        
        RealType X = (pt.x() - x_min) / (x_max - x_min);
        RealType Y = (pt.y() - x_min) / (x_max - x_min);

        RealType sin_x = std::sin(2.*M_PI*X);
        RealType sin_y = std::sin(2.*M_PI*Y);
        RealType cos_x = std::cos(2.*M_PI*X);
        RealType cos_y = std::cos(2.*M_PI*Y);

        
        ret(0) =  sin_x * sin_x * sin_y * cos_y;
        ret(1) = - sin_x * cos_x * sin_y * sin_y;

        return ret;
    };

    auto sol_grad = [](const typename cuthho_poly_mesh<RealType>::point_type& pt) -> auto {
        Matrix<RealType, 2, 2> ret;


        
        RealType x_max = 1.0;
        RealType x_min = 0.0;
        RealType coeff = 2.*M_PI/(x_max - x_min);
        
        RealType X = (pt.x() - x_min) / (x_max - x_min);
        RealType Y = (pt.y() - x_min) / (x_max - x_min);

        RealType sin_x = std::sin(2.*M_PI*X);
        RealType sin_y = std::sin(2.*M_PI*Y);
        RealType cos_x = std::cos(2.*M_PI*X);
        RealType cos_y = std::cos(2.*M_PI*Y);

        
        ret(0,0) = coeff * 2. * sin_x * cos_x * sin_y * cos_y;
        ret(0,1) = coeff * sin_x * sin_x * (cos_y * cos_y - sin_y * sin_y);
        ret(1,0) = - coeff * sin_y * sin_y * (cos_x * cos_x - sin_x * sin_x);
        ret(1,1) = - 2. * coeff * sin_x * cos_x * cos_y * sin_y;
        
        return ret;
    };

    auto bcs_fun = [&](const typename cuthho_poly_mesh<RealType>::point_type& pt) -> auto {
        return sol_vel(pt);
    };

    auto pressure =  [](const typename cuthho_poly_mesh<RealType>::point_type& pt) -> RealType {

        RealType mid_y = (0. + 1.0) / 2.;
        
        return std::pow(pt.x() - 0.5, 5.)  +  std::pow(pt.y() - mid_y, 5.);
        // return std::pow(pt.x(), 5.)  +  std::pow(pt.y(), 5.) - 1./3.;
    };
#elif 0  // test in diskpp (on a square) (homogeneous BC)
    auto rhs_fun = [](const typename cuthho_poly_mesh<RealType>::point_type& pt) -> auto {
        Matrix<RealType, 2, 1> ret;

        RealType x1 = pt.x();
        RealType x2 = x1 * x1;
        RealType y1 = pt.y();
        RealType y2 = y1 * y1;

        RealType ax =  x2 * (x2 - 2. * x1 + 1.);
        RealType ay =  y2 * (y2 - 2. * y1 + 1.);
        RealType bx =  x1 * (4. * x2 - 6. * x1 + 2.);
        RealType by =  y1 * (4. * y2 - 6. * y1 + 2.);
        RealType cx = 12. * x2 - 12.* x1 + 2.;
        RealType cy = 12. * y2 - 12.* y1 + 2.;
        RealType dx = 24. * x1 - 12.;
        RealType dy = 24. * y1 - 12.;

        ret(0) = - cx * by - ax * dy + 5.* x2 * x2;
        ret(1) = + cy * bx + ay * dx + 5.* y2 * y2;
        
        
        return ret;
    };

    auto sol_vel = [](const typename cuthho_poly_mesh<RealType>::point_type& pt) -> auto {
        Matrix<RealType, 2, 1> ret;
 
        RealType x1 = pt.x();
        RealType x2 = x1 * x1;
        RealType y1 = pt.y();
        RealType y2 = y1 * y1;

        ret(0) =  x2 * (x2 - 2. * x1 + 1.)  * y1 * (4. * y2 - 6. * y1 + 2.);
        ret(1) = -y2 * (y2 - 2. * y1 + 1. ) * x1 * (4. * x2 - 6. * x1 + 2.);

        return ret;
    };

    auto sol_grad = [](const typename cuthho_poly_mesh<RealType>::point_type& pt) -> auto {
        Matrix<RealType, 2, 2> ret;
        
        RealType x1 = pt.x();
        RealType x2 = x1 * x1;
        RealType y1 = pt.y();
        RealType y2 = y1 * y1;
        
        ret(0,0) = x1 * (4. * x2 - 6. * x1 + 2.) * y1 * (4. * y2 - 6. * y1 + 2.);
        ret(0,1) = x2 * ( x2 - 2. * x1 + 1.) * (12. * y2 - 12. * y1 + 2.);
        ret(1,0) = - y2 * ( y2 - 2. * y1 + 1.) * (12. * x2 - 12. * x1 + 2.);
        ret(1,1) = - ret(0,0);
        
        return ret;
    };

    auto bcs_fun = [&](const typename cuthho_poly_mesh<RealType>::point_type& pt) -> auto {
        return sol_vel(pt);
    };

    auto pressure =  [](const typename cuthho_poly_mesh<RealType>::point_type& pt) -> RealType {
        return std::pow(pt.x(), 5.)  +  std::pow(pt.y(), 5.) - 1./3.;
    };
#endif

    timecounter tc;

    /************** ASSEMBLE PROBLEM **************/
    hho_degree_info hdi(degree+1, degree);

    element_location where = element_location::IN_NEGATIVE_SIDE;

    /* reconstruction and stabilization template for square cells.
     * BEWARE of the fact that I'm using cell 0 to compute it! */
    auto gr_template = make_hho_laplacian(msh, msh.cells[0], level_set_function, hdi, where);
    Matrix<RealType, Dynamic, Dynamic> stab_template = make_hho_cut_stabilization(msh, msh.cells[0], hdi, where);
    Matrix<RealType, Dynamic, Dynamic> lc_template = gr_template.second + stab_template;

    tc.tic();

    auto celdeg = hdi.cell_degree();
    auto facdeg = hdi.face_degree();

    auto cbs = vector_cell_basis<Mesh,RealType>::size(celdeg);
    auto fbs = vector_face_basis<Mesh,RealType>::size(facdeg);
    auto cbs_B = cell_basis<Mesh,RealType>::size(facdeg);
    
    
    bool sc = false;
    
    auto assembler = make_stokes_assembler(msh, hdi);
    auto assembler_sc = make_stokes_condensed_assembler(msh, hdi);
    
    for (auto& cl : msh.cells)
    {
        if ( false && !cl.user_data.distorted && location(msh, cl) != element_location::ON_INTERFACE )
        {
            // Matrix<RealType, Dynamic, 1> f = Matrix<RealType, Dynamic, 1>::Zero(lc_template.rows());
            // f = make_rhs(msh, cl, hdi.cell_degree(), rhs_fun, where, level_set_function, bcs_fun);
            // assembler.assemble(msh, cl, lc_template, f, bcs_fun);
        }
        else
        {
            auto gr = make_hho_gradrec_matrix(msh, cl, level_set_function, hdi, where);   
            Matrix<RealType, Dynamic, Dynamic> stab = make_hho_vector_cut_stabilization(msh, cl, hdi, where, level_set_function);
            auto dr = make_hho_divergence_reconstruction(msh, cl, level_set_function, hdi, where);
            Matrix<RealType, Dynamic, Dynamic> lc = gr.second + stab;
            Matrix<RealType, Dynamic, 1> f = Matrix<RealType, Dynamic, 1>::Zero(lc.rows());

            if(location(msh, cl) == element_location::ON_INTERFACE)
                f = make_vector_rhs(msh, cl, hdi.cell_degree(), rhs_fun, where, level_set_function, bcs_fun, gr.first);
            else
                f.block(0, 0, cbs, 1) = make_vector_rhs(msh, cl, hdi.cell_degree(), rhs_fun, where, level_set_function, bcs_fun, gr.first);
            
            Matrix<RealType, Dynamic, 1> p_rhs = make_pressure_rhs(msh, cl, hdi.face_degree(), where, level_set_function, bcs_fun);




            ////////////////   TEST   //////////////////
            // if( location(msh,cl) == element_location::ON_INTERFACE )
            // {
            //     cell_basis<cuthho_poly_mesh<RealType>, RealType> s_cb(msh, cl, facdeg);
            
            //     vector_cell_basis<cuthho_poly_mesh<RealType>, RealType> cb(msh, cl, celdeg);


            //     auto fcs = faces(msh, cl);
            //     auto num_faces = fcs.size();

            //     size_t P_offset = cbs + num_faces * fbs;
            
            //     auto iqp = integrate_interface(msh, cl, celdeg + facdeg, element_location::IN_NEGATIVE_SIDE);
            //     for( auto& qp : iqp )
            //     {
            //         const auto v_phi = cb.eval_basis(qp.first);
            //         const auto s_phi = s_cb.eval_basis(qp.first);
            //         const auto n = level_set_function.normal(qp.first);
                
            //         const Matrix<RealType, Dynamic, 2> s_phi_n = (s_phi * n.transpose());
                
            //         lc.block(0, P_offset, cbs, cbs_B) += qp.second * v_phi * s_phi_n.transpose();
            //     }
            // }
            /////////////   END  TEST   ////////////////
            

            // Matrix<RealType, Dynamic, 1> p_rhs = Matrix<RealType, Dynamic, 1>::Zero(cbs_B);

            if( sc )
            {
                assembler_sc.assemble(msh, cl, lc, dr.second, f, p_rhs, bcs_fun, where);
            }
            else 
                assembler.assemble(msh, cl, lc, -dr.second, f, -p_rhs, bcs_fun, where);
        }
    }

    if( sc )
        assembler_sc.finalize();
    else 
        assembler.finalize();

    tc.toc();
    std::cout << bold << yellow << "Matrix assembly: " << tc << " seconds" << reset << std::endl;

    if( sc )
        std::cout << "System unknowns: " << assembler_sc.LHS.rows() << std::endl;
    else
        std::cout << "System unknowns: " << assembler.LHS.rows() << std::endl;
    std::cout << "Cells: " << msh.cells.size() << std::endl;
    std::cout << "Faces: " << msh.faces.size() << std::endl;

    /************** SOLVE **************/
    tc.tic();
#if 1
    SparseLU<SparseMatrix<RealType>>  solver;

    Matrix<RealType, Dynamic, 1> sol;
        
    if( sc ) {
        solver.analyzePattern(assembler_sc.LHS);
        solver.factorize(assembler_sc.LHS);
        sol = solver.solve(assembler_sc.RHS);
    }
    else
    {
        solver.analyzePattern(assembler.LHS);
        solver.factorize(assembler.LHS);
        sol = solver.solve(assembler.RHS);
    }
#endif
#if 0
    Matrix<RealType, Dynamic, 1> sol;

    if( sc )
    {
        sol = Matrix<RealType, Dynamic, 1>::Zero(assembler_sc.RHS.rows());
        cg_params<RealType> cgp;
        cgp.max_iter = assembler_sc.LHS.rows();
        cgp.histfile = "cuthho_cg_hist.dat";
        cgp.verbose = true;
        cgp.apply_preconditioner = true;
        conjugated_gradient(assembler_sc.LHS, assembler_sc.RHS, sol, cgp);
    }
    else
    {
        sol = Matrix<RealType, Dynamic, 1>::Zero(assembler.RHS.rows());
        cg_params<RealType> cgp;
        cgp.max_iter = assembler.LHS.rows();
        cgp.histfile = "cuthho_cg_hist.dat";
        cgp.verbose = true;
        cgp.apply_preconditioner = true;
        conjugated_gradient(assembler.LHS, assembler.RHS, sol, cgp);
    }
#endif
    tc.toc();
    std::cout << bold << yellow << "Linear solver: " << tc << " seconds" << reset << std::endl;

    /************** POSTPROCESS **************/


    // std::vector<RealType>   mesh;


    postprocess_output<RealType>  postoutput;

    auto uT1_gp  = std::make_shared< gnuplot_output_object<RealType> >("fictdom_uT1.dat");
    auto uT2_gp  = std::make_shared< gnuplot_output_object<RealType> >("fictdom_uT2.dat");
    auto p_gp    = std::make_shared< gnuplot_output_object<RealType> >("fictdom_p.dat");
    auto sol_p_gp    = std::make_shared< gnuplot_output_object<RealType> >("fictdom_sol_p.dat");
    auto diff_p_gp    = std::make_shared< gnuplot_output_object<RealType> >("fictdom_diff_p.dat");
    auto diff_p_gp2    = std::make_shared< gnuplot_output_object<RealType> >("fictdom_diff_p2.dat");
    
    
    auto int_gp  = std::make_shared< gnuplot_output_object<RealType> >("ficdom_int.dat");


    auto tests_p_gp = std::make_shared< gnuplot_output_object<RealType> >("test_p.dat");


    RealType mean_pressure = 0.0;
    
    std::vector< Matrix<RealType, 2, 1> >   solution_uT;

    tc.tic();
    RealType    L2_error = 0.0;
    RealType    H1_error = 0.0;
    RealType    H1_sol_norm = 0.0;
    RealType    L2_pressure_error = 0.0;
    size_t      cell_i   = 0;
    for (auto& cl : msh.cells)
    {
        bool hide_fict_dom = true; // hide the fictitious domain in the gnuplot outputs
        if (hide_fict_dom && location(msh,cl) == element_location::IN_POSITIVE_SIDE)
            continue;
        
        cell_basis<cuthho_poly_mesh<RealType>, RealType> s_cb(msh, cl, hdi.face_degree());

        vector_cell_basis<cuthho_poly_mesh<RealType>, RealType> cb(msh, cl, hdi.cell_degree());
        auto cbs = cb.size();
        
        vector_cell_basis<cuthho_poly_mesh<RealType>, RealType> rb(msh, cl, hdi.reconstruction_degree());
        auto rbs = rb.size();


        Matrix<RealType, Dynamic, 1> locdata_vel;
        Matrix<RealType, Dynamic, 1> loc_pressure;
        Matrix<RealType, Dynamic, 1> cell_dofs;
        if( sc )
        {
            locdata_vel = assembler_sc.take_velocity(msh, cl, sol, bcs_fun);

            RealType pressure_zero = assembler_sc.take_pressure(msh, cl, sol);

            auto bar = barycenter(msh, cl, element_location::IN_NEGATIVE_SIDE);
            
            tests_p_gp->add_data( bar, pressure_zero );
            
            
            auto gr = make_hho_gradrec_matrix(msh, cl, level_set_function, hdi, where);   
            Matrix<RealType, Dynamic, Dynamic> stab = make_hho_vector_cut_stabilization(msh, cl, hdi, where, level_set_function);
            auto dr = make_hho_divergence_reconstruction(msh, cl, level_set_function, hdi, where);
            Matrix<RealType, Dynamic, Dynamic> lc = gr.second + stab;
            Matrix<RealType, Dynamic, 1> f = Matrix<RealType, Dynamic, 1>::Zero(lc.rows());

            if(location(msh, cl) == element_location::ON_INTERFACE)
                f = make_vector_rhs(msh, cl, hdi.cell_degree(), rhs_fun, where, level_set_function, bcs_fun, gr.first);
            else
                f.block(0, 0, cbs, 1) = make_vector_rhs(msh, cl, hdi.cell_degree(), rhs_fun, where, level_set_function, bcs_fun, gr.first);

           
            Matrix<RealType, Dynamic, 1> p_rhs = make_pressure_rhs(msh, cl, hdi.face_degree(), where, level_set_function, bcs_fun);

            Matrix<RealType, Dynamic, 1> cell_rhs = f.head(cbs);

            
            Matrix<RealType, Dynamic, 1> sol_rec
                = stokes_static_condensation_recover(msh, cl, hdi, lc, dr.second, cell_rhs,
                                                     p_rhs, locdata_vel, pressure_zero);
                
            loc_pressure = sol_rec.tail(cbs_B);
            // loc_pressure[0] = pressure_zero;
            
            cell_dofs = sol_rec.head(cbs);
        }
        else
        {
            locdata_vel = assembler.take_velocity(msh, cl, sol, bcs_fun);
            loc_pressure = assembler.take_pressure(msh, cl, sol);
            cell_dofs = locdata_vel.head(cbs);
        }
        
        auto bar = barycenter(msh, cl, element_location::IN_NEGATIVE_SIDE);
        
        Matrix<RealType, Dynamic, 2> c_phi = cb.eval_basis(bar);
        auto c_val = c_phi.transpose() * cell_dofs;
        solution_uT.push_back(c_val);
        
        
        auto qps = integrate(msh, cl, 5, element_location::IN_NEGATIVE_SIDE);
        if ( !hide_fict_dom ) qps = integrate(msh, cl, 5/*, element_location::IN_NEGATIVE_SIDE*/);
        
        for (auto& qp : qps)
        {
            auto tp = qp.first;

            auto c_phi_bis = cb.eval_basis(tp);
            auto c_val_bis = c_phi_bis.transpose() * cell_dofs;

            
            RealType p_val = s_cb.eval_basis(tp).dot(loc_pressure);

            RealType diff_p_val = pressure(tp) - p_val ;
            RealType diff_p_val2 = diff_p_val / pressure(tp);

            uT1_gp->add_data( tp, c_val_bis(0,0) );
            uT2_gp->add_data( tp, c_val_bis(1,0) ); 
            p_gp->add_data( tp, p_val );
            diff_p_gp->add_data( tp, diff_p_val );
            diff_p_gp2->add_data( tp, diff_p_val2 );
            sol_p_gp->add_data( tp, pressure(tp) );
        }

        if ( location(msh, cl) == element_location::IN_NEGATIVE_SIDE ||
             location(msh, cl) == element_location::ON_INTERFACE )
        {
            Matrix<RealType, 1, 2> real_grad_int = Matrix<RealType, 1, 2>::Zero();
            Matrix<RealType, 1, 2> comp_grad_int = Matrix<RealType, 1, 2>::Zero();
            auto qps = integrate(msh, cl, 2*hdi.cell_degree(), element_location::IN_NEGATIVE_SIDE);
            for (auto& qp : qps)
            {
                /* Compute L2-error */
                auto cphi = cb.eval_basis( qp.first );
                Matrix<RealType, 2, 1> sol_num = Matrix<RealType, 2, 1>::Zero();

                sol_num += cphi.transpose() * cell_dofs;

                Matrix<RealType, 2, 1> sol_diff = sol_vel(qp.first) - sol_num;
                L2_error += qp.second * sol_diff.dot(sol_diff);

                /* Compute H1-error */
                auto d_cphi = cb.eval_gradients( qp.first );
                Matrix<RealType, 2, 2> grad = Matrix<RealType, 2, 2>::Zero();

                for (size_t i = 0; i < cbs; i++ )
                    grad += cell_dofs(i) * d_cphi[i].block(0, 0, 2, 2);

                Matrix<RealType, 2, 2> grad_diff = sol_grad(qp.first) - grad;
                
                H1_error += qp.second * inner_product(grad_diff , grad_diff);


                /* Compute pressure L2-error */
                auto s_cphi = s_cb.eval_basis( qp.first );
                RealType p_num = s_cphi.dot(loc_pressure);
                RealType p_diff = pressure( qp.first ) - p_num;

                L2_pressure_error += qp.second * p_diff * p_diff;


                /* mean pressure */
                mean_pressure += qp.second * p_num;

                
                int_gp->add_data( qp.first, 1.0 );

            }
        }

        cell_i++;
    }

    std::cout << bold << green << "Energy-norm absolute error:           " << std::sqrt(H1_error) << std::endl;

    std::cout << bold << green << "L2-norm absolute error:           " << std::sqrt(L2_error) << std::endl;

    std::cout << bold << green << "L2-norm pressure absolute error:           " << std::sqrt(L2_pressure_error) << std::endl;

    std::cout << bold << blue << "mean pressure :           " << mean_pressure << std::endl;

    postoutput.add_object(uT1_gp);
    postoutput.add_object(uT2_gp);
    postoutput.add_object(p_gp);
    postoutput.add_object(diff_p_gp);
    postoutput.add_object(diff_p_gp2);
    postoutput.add_object(sol_p_gp);
    postoutput.add_object(int_gp);
    if( sc ) postoutput.add_object(tests_p_gp); 
    postoutput.write();

    tc.toc();
    std::cout << bold << yellow << "Postprocessing: " << tc << " seconds" << reset << std::endl;

}

//////////////////////   STATIC CONDENSATION   //////////////////////////////

template<typename Mesh, typename T>
auto
stokes_static_condensation_compute(const Mesh& msh,
                                   const typename Mesh::cell_type& cl, const hho_degree_info hdi,
                                   const typename Eigen::Matrix<T, Dynamic, Dynamic>& lhs_A,
                                   const typename Eigen::Matrix<T, Dynamic, Dynamic>& lhs_B,
                                   const typename Eigen::Matrix<T, Dynamic, 1>& rhs_A,
                                   const typename Eigen::Matrix<T, Dynamic, 1>& rhs_B)
{
    using matrix_type = Matrix<T, Dynamic, Dynamic>;
    using vector_type = Matrix<T, Dynamic, 1>;

    auto celdeg = hdi.cell_degree();
    auto facdeg = hdi.face_degree();
    auto cbs_B = cell_basis<Mesh,T>::size(facdeg);
    auto cbs_A = vector_cell_basis<Mesh,T>::size(celdeg);
    auto fbs_A = vector_face_basis<Mesh,T>::size(facdeg);

    auto fcs = faces(msh, cl);
    auto num_faces = fcs.size();

    auto face_size = num_faces * fbs_A;
    
    assert(lhs_A.rows() == lhs_A.cols());
    assert(lhs_A.cols() == cbs_A + face_size);
    
    assert(lhs_B.rows() == cbs_B);
    assert(lhs_B.cols() == lhs_A.cols());
    
    assert(rhs_A.rows() == lhs_A.rows() );
    assert(rhs_B.rows() == lhs_B.rows() );

    
    // l : local    g : global

    size_t l_size = cbs_A + cbs_B - 1;
    size_t g_size = face_size + 1;

    // K_ll
    matrix_type K_ll = matrix_type::Zero(l_size, l_size);
    K_ll.block(0, 0, cbs_A, cbs_A) = lhs_A.block(0, 0, cbs_A, cbs_A);
    if(facdeg > 0)
    {
        K_ll.block(cbs_A, 0, cbs_B - 1, cbs_A) = lhs_B.block(1, 0, cbs_B - 1, cbs_A);
        K_ll.block(0, cbs_A, cbs_A, cbs_B - 1) = lhs_B.transpose().block(0, 1, cbs_A, cbs_B - 1);
        // for( size_t i = 0; i < cbs_A; i++ )
        // {
        //     for( size_t j = 1; j < cbs_B; j++ )
        //     {
        //         K_ll(i, cbs_A + j - 1) = lhs_B(j,i);
        //         K_ll(cbs_A + j - 1, i) = lhs_B(j,i);
        //     }
        // }
    }
    
    // K_lg
    matrix_type K_lg = matrix_type::Zero(l_size, g_size);
    K_lg.block(0, 0, cbs_A, face_size) = lhs_A.block(0, cbs_A, cbs_A, face_size);
    if(facdeg > 0)
        K_lg.block(cbs_A, 0, cbs_B - 1, face_size) = lhs_B.block(1, cbs_A, cbs_B - 1, face_size);

    // K_gl
    matrix_type K_gl = matrix_type::Zero(g_size, l_size);
    K_gl = K_lg.transpose();
    
    // K_gg
    matrix_type K_gg = matrix_type::Zero(g_size, g_size);
    K_gg.block(0, 0, face_size, face_size) = lhs_A.block(cbs_A, cbs_A, face_size, face_size);
    K_gg.block(face_size, 0, 1, face_size) = lhs_B.block(0, cbs_A, 1, face_size);
    K_gg.block(0, face_size, face_size, 1) = lhs_B.transpose().block(cbs_A, 0, face_size, 1);
    
    // F_l
    vector_type F_l = vector_type::Zero(l_size);
    F_l.block(0, 0, cbs_A, 1) = rhs_A.block(0, 0, cbs_A, 1);
    if(facdeg > 0) F_l.block(cbs_A, 0, cbs_B - 1, 1) = rhs_B.block(1, 0, cbs_B - 1, 1);
    
    // F_g
    vector_type F_g = vector_type::Zero(g_size);
    F_g.block(0, 0, face_size, 1) = rhs_A.block(cbs_A, 0, face_size, 1);
    F_g[face_size] = rhs_B[0];


    assert(K_ll.cols() == l_size);
    assert(K_ll.cols() + K_lg.cols() == l_size + g_size);
    assert(K_ll.rows() + K_gl.rows() == l_size + g_size);
    assert(K_lg.rows() + K_gg.rows() == l_size + g_size);
    assert(K_gl.cols() + K_gg.cols() == l_size + g_size);

    
    
    auto K_ll_ldlt = K_ll.ldlt();
    matrix_type AL = K_ll_ldlt.solve(K_lg);
    vector_type bL = K_ll_ldlt.solve(F_l);
    
    matrix_type AC = K_gg - K_gl * AL;
    vector_type bC = F_g - K_gl * bL;

    return std::make_pair(AC, bC);
}


template<typename Mesh, typename T>
Eigen::Matrix<T, Eigen::Dynamic, 1>
stokes_static_condensation_recover(const Mesh& msh,
    const typename Mesh::cell_type& cl, const hho_degree_info hdi,
    const typename Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& lhs_A,
    const typename Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& lhs_B,
    const typename Eigen::Matrix<T, Eigen::Dynamic, 1>& cell_rhs,
    const typename Eigen::Matrix<T, Eigen::Dynamic, 1>& B_rhs,
    const typename Eigen::Matrix<T, Eigen::Dynamic, 1>& sol_F,
    const T mean_pressure)
{
    using matrix_type = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    using vector_type = Eigen::Matrix<T, Eigen::Dynamic, 1>;

    auto celdeg = hdi.cell_degree();
    auto facdeg = hdi.face_degree();
    auto cbs_A = vector_cell_basis<Mesh,T>::size(celdeg);
    auto fbs_A = vector_face_basis<Mesh,T>::size(facdeg);
    auto cbs_B = cell_basis<Mesh,T>::size(facdeg);
    
    auto fcs = faces(msh, cl);
    auto num_faces = fcs.size();

    vector_type ret( cbs_A + num_faces * fbs_A + cbs_B);

    size_t face_size = num_faces * fbs_A;
    
    // l : local      g : global
    size_t l_size = cbs_A + cbs_B - 1;
    size_t g_size = face_size + 1;
    
    // K_ll
    matrix_type K_ll = matrix_type::Zero(l_size, l_size);
    K_ll.block(0, 0, cbs_A, cbs_A) = lhs_A.block(0, 0, cbs_A, cbs_A);
    if(facdeg > 0)
    {
        K_ll.block(cbs_A, 0, cbs_B - 1, cbs_A) = lhs_B.block(1, 0, cbs_B - 1, cbs_A);
        K_ll.block(0, cbs_A, cbs_A, cbs_B - 1) = lhs_B.transpose().block(0, 1, cbs_A, cbs_B - 1);
    }
    
    // K_lg
    matrix_type K_lg = matrix_type::Zero(l_size, g_size);
    K_lg.block(0, 0, cbs_A, face_size) = lhs_A.block(0, cbs_A, cbs_A, face_size);
    if(facdeg > 0)
        K_lg.block(cbs_A, 0, cbs_B - 1, face_size) = lhs_B.block(1, cbs_A, cbs_B - 1, face_size);


    // assert( sol_F.cols() ==  )
    
    // sol_g
    vector_type sol_g = vector_type::Zero(g_size);
    sol_g.head( face_size ) = sol_F;
    sol_g[ face_size ] = mean_pressure;


    // F_l
    vector_type F_l = vector_type::Zero(l_size);
    F_l.head(cbs_A) = cell_rhs;
    if(facdeg > 0) F_l.tail(cbs_B - 1) = B_rhs.tail(cbs_B - 1);
    
    // sol_l
    vector_type sol_l = K_ll.ldlt().solve(F_l - K_lg*sol_g);


    

    assert(K_ll.cols() == l_size);
    assert(K_ll.cols() + K_lg.cols() == l_size + g_size);
    assert(sol_g.rows() == g_size);
    assert(sol_l.rows() == l_size);
    
    
    ret.head(cbs_A)                         = sol_l.head(cbs_A);
    ret.block(cbs_A, 0, face_size, 1)       = sol_F;
    ret[cbs_A + face_size]                  = mean_pressure;
    if(facdeg > 0) ret.tail(cbs_B - 1)      = sol_l.tail(cbs_B - 1);

    return ret;
}


//////////////////////////   ASSEMBLERS   //////////////////////////////



template<typename Mesh>
class vector_assembler
{
    using T = typename Mesh::coordinate_type;
    std::vector<size_t>                 compress_table;
    std::vector<size_t>                 expand_table;

    hho_degree_info                     di;

    std::vector< Triplet<T> >           triplets;

    class assembly_index
    {
        size_t  idx;
        bool    assem;

    public:
        assembly_index(size_t i, bool as)
            : idx(i), assem(as)
        {}

        operator size_t() const
        {
            if (!assem)
                throw std::logic_error("Invalid assembly_index");

            return idx;
        }

        bool assemble() const
        {
            return assem;
        }

        friend std::ostream& operator<<(std::ostream& os, const assembly_index& as)
        {
            os << "(" << as.idx << "," << as.assem << ")";
            return os;
        }
    };

public:

    SparseMatrix<T>         LHS;
    Matrix<T, Dynamic, 1>   RHS;

    vector_assembler(const Mesh& msh, hho_degree_info hdi)
        : di(hdi)
    {
        auto is_dirichlet = [&](const typename Mesh::face_type& fc) -> bool {
            return fc.is_boundary && fc.bndtype == boundary::DIRICHLET;
        };

        auto num_all_faces = msh.faces.size();
        auto num_dirichlet_faces = std::count_if(msh.faces.begin(), msh.faces.end(), is_dirichlet);
        auto num_other_faces = num_all_faces - num_dirichlet_faces;

        compress_table.resize( num_all_faces );
        expand_table.resize( num_other_faces );

        size_t compressed_offset = 0;
        for (size_t i = 0; i < num_all_faces; i++)
        {
            auto fc = msh.faces[i];
            if ( !is_dirichlet(fc) )
            {
                compress_table.at(i) = compressed_offset;
                expand_table.at(compressed_offset) = i;
                compressed_offset++;
            }
        }

        auto celdeg = di.cell_degree();
        auto facdeg = di.face_degree();

        auto cbs = vector_cell_basis<Mesh,T>::size(celdeg);
        auto fbs = vector_face_basis<Mesh,T>::size(facdeg);

        auto system_size = cbs * msh.cells.size() + fbs * num_other_faces;

        LHS = SparseMatrix<T>( system_size, system_size );
        RHS = Matrix<T, Dynamic, 1>::Zero( system_size );
    }

    void dump_tables() const
    {
        std::cout << "Compress table: " << std::endl;
        for (size_t i = 0; i < compress_table.size(); i++)
            std::cout << i << " -> " << compress_table.at(i) << std::endl;
    }

    template<typename Function>
    void
    assemble(const Mesh& msh, const typename Mesh::cell_type& cl,
             const Matrix<T, Dynamic, Dynamic>& lhs, const Matrix<T, Dynamic, 1>& rhs,
             const Function& dirichlet_bf)
    {
        auto celdeg = di.cell_degree();
        auto facdeg = di.face_degree();

        auto cbs = vector_cell_basis<Mesh,T>::size(celdeg);
        auto fbs = vector_face_basis<Mesh,T>::size(facdeg);

        auto fcs = faces(msh, cl);
        auto num_faces = fcs.size();

        std::vector<assembly_index> asm_map;
        asm_map.reserve(cbs + num_faces*fbs);

        auto cell_offset        = offset(msh, cl);
        auto cell_LHS_offset    = cell_offset * cbs;

        for (size_t i = 0; i < cbs; i++)
            asm_map.push_back( assembly_index(cell_LHS_offset+i, true) );

        Matrix<T, Dynamic, 1> dirichlet_data = Matrix<T, Dynamic, 1>::Zero(cbs + num_faces*fbs);

        for (size_t face_i = 0; face_i < num_faces; face_i++)
        {
            auto fc = fcs[face_i];
            auto face_offset = offset(msh, fc);
            auto face_LHS_offset = cbs * msh.cells.size() + compress_table.at(face_offset)*fbs;

            bool dirichlet = fc.is_boundary && fc.bndtype == boundary::DIRICHLET;

            for (size_t i = 0; i < fbs; i++)
                asm_map.push_back( assembly_index(face_LHS_offset+i, !dirichlet) );

            if (dirichlet)
            {
                Matrix<T, Dynamic, Dynamic> mass = make_vector_mass_matrix(msh, fc, facdeg);
                Matrix<T, Dynamic, 1> rhs = make_vector_rhs(msh, fc, facdeg, dirichlet_bf);
                dirichlet_data.block(cbs+face_i*fbs, 0, fbs, 1) = mass.llt().solve(rhs);
            }
        }
        
        assert( asm_map.size() == lhs.rows() && asm_map.size() == lhs.cols() );
        
        for (size_t i = 0; i < lhs.rows(); i++)
        {
            if (!asm_map[i].assemble())
                continue;

            for (size_t j = 0; j < lhs.cols(); j++)
            {   
                if ( asm_map[j].assemble() )
                    triplets.push_back( Triplet<T>(asm_map[i], asm_map[j], lhs(i,j)) );
                else
                    RHS(asm_map[i]) -= lhs(i,j)*dirichlet_data(j);
            }
        }
        
        RHS.block(cell_LHS_offset, 0, cbs, 1) += rhs.block(0, 0, cbs, 1);
        if ( rhs.rows() > cbs )
        {
            for (size_t face_i = 0; face_i < num_faces; face_i++)
            {
                auto fc = fcs[face_i];
                auto face_offset = offset(msh, fc);
                auto face_LHS_offset = cbs * msh.cells.size() + compress_table.at(face_offset)*fbs;
                
                RHS.block(face_LHS_offset, 0, fbs, 1) += rhs.block(cbs+face_i*fbs, 0, fbs, 1);
            }
        }
    } // assemble()

    template<typename Function>
    Matrix<T, Dynamic, 1>
    take_local_data(const Mesh& msh, const typename Mesh::cell_type& cl,
    const Matrix<T, Dynamic, 1>& solution, const Function& dirichlet_bf)
    {
        auto celdeg = di.cell_degree();
        auto facdeg = di.face_degree();

        auto cbs = vector_cell_basis<Mesh,T>::size(celdeg);
        auto fbs = vector_face_basis<Mesh,T>::size(facdeg);

        auto cell_offset        = offset(msh, cl);
        auto cell_SOL_offset    = cell_offset * cbs;

        auto fcs = faces(msh, cl);
        auto num_faces = fcs.size();

        Matrix<T, Dynamic, 1> ret = Matrix<T, Dynamic, 1>::Zero(cbs + num_faces*fbs);
        ret.block(0, 0, cbs, 1) = solution.block(cell_SOL_offset, 0, cbs, 1);

        for (size_t face_i = 0; face_i < num_faces; face_i++)
        {
            auto fc = fcs[face_i];

            bool dirichlet = fc.is_boundary && fc.bndtype == boundary::DIRICHLET;

            if (dirichlet)
            {
                Matrix<T, Dynamic, Dynamic> mass = make_vector_mass_matrix(msh, fc, facdeg);
                Matrix<T, Dynamic, 1> rhs = make_vector_rhs(msh, fc, facdeg, dirichlet_bf);
                ret.block(cbs+face_i*fbs, 0, fbs, 1) = mass.llt().solve(rhs);
            }
            else
            {
                auto face_offset = offset(msh, fc);
                auto face_SOL_offset = cbs * msh.cells.size() + compress_table.at(face_offset)*fbs;
                ret.block(cbs+face_i*fbs, 0, fbs, 1) = solution.block(face_SOL_offset, 0, fbs, 1);
            }
        }

        return ret;
    }

    void finalize(void)
    {
        LHS.setFromTriplets( triplets.begin(), triplets.end() );
        triplets.clear();
    }
};


template<typename Mesh>
auto make_vector_assembler(const Mesh& msh, hho_degree_info hdi)
{
    return vector_assembler<Mesh>(msh, hdi);
}



//////////////

template<typename Mesh>
class stokes_assembler
{
    using T = typename Mesh::coordinate_type;
    std::vector<size_t>                 compress_table;
    std::vector<size_t>                 expand_table;

    hho_degree_info                     di;

    std::vector< Triplet<T> >           triplets;

    size_t num_other_faces;
    
    class assembly_index
    {
        size_t  idx;
        bool    assem;

    public:
        assembly_index(size_t i, bool as)
            : idx(i), assem(as)
        {}

        operator size_t() const
        {
            if (!assem)
                throw std::logic_error("Invalid assembly_index");

            return idx;
        }

        bool assemble() const
        {
            return assem;
        }

        friend std::ostream& operator<<(std::ostream& os, const assembly_index& as)
        {
            os << "(" << as.idx << "," << as.assem << ")";
            return os;
        }
    };

public:

    SparseMatrix<T>         LHS;
    Matrix<T, Dynamic, 1>   RHS;

    stokes_assembler(const Mesh& msh, hho_degree_info hdi)
        : di(hdi)
    {
        auto is_dirichlet = [&](const typename Mesh::face_type& fc) -> bool {
            return fc.is_boundary && fc.bndtype == boundary::DIRICHLET;
        };

        auto num_all_faces = msh.faces.size();
        auto num_dirichlet_faces = std::count_if(msh.faces.begin(), msh.faces.end(), is_dirichlet);

        num_other_faces = num_all_faces - num_dirichlet_faces;

        compress_table.resize( num_all_faces );
        expand_table.resize( num_other_faces );

        size_t compressed_offset = 0;
        for (size_t i = 0; i < num_all_faces; i++)
        {
            auto fc = msh.faces[i];
            if ( !is_dirichlet(fc) )
            {
                compress_table.at(i) = compressed_offset;
                expand_table.at(compressed_offset) = i;
                compressed_offset++;
            }
        }

        auto celdeg = di.cell_degree();
        auto facdeg = di.face_degree();

        
        auto cbs_B = cell_basis<Mesh,T>::size(facdeg);
        auto cbs_A = vector_cell_basis<Mesh,T>::size(celdeg);
        auto fbs = vector_face_basis<Mesh,T>::size(facdeg);

        auto system_size = (cbs_A + cbs_B) * msh.cells.size() + fbs * num_other_faces + 1;

        LHS = SparseMatrix<T>( system_size, system_size );
        RHS = Matrix<T, Dynamic, 1>::Zero( system_size );
    }

    void dump_tables() const
    {
        std::cout << "Compress table: " << std::endl;
        for (size_t i = 0; i < compress_table.size(); i++)
            std::cout << i << " -> " << compress_table.at(i) << std::endl;
    }

    template<typename Function>
    void
    assemble(const Mesh& msh, const typename Mesh::cell_type& cl,
             const Matrix<T, Dynamic, Dynamic>& lhs_A, const Matrix<T, Dynamic, Dynamic>& lhs_B,
             const Matrix<T, Dynamic, 1>& rhs_A, const Matrix<T, Dynamic, 1>& rhs_B,
             const Function& dirichlet_bf, const element_location where)
    {
        auto celdeg = di.cell_degree();
        auto facdeg = di.face_degree();

        auto cbs_B = cell_basis<Mesh,T>::size(facdeg);
        auto cbs_A = vector_cell_basis<Mesh,T>::size(celdeg);
        auto fbs_A = vector_face_basis<Mesh,T>::size(facdeg);

        auto fcs = faces(msh, cl);
        auto num_faces = fcs.size();

        std::vector<assembly_index> asm_map;
        asm_map.reserve(cbs_A + num_faces*fbs_A);

        auto cell_offset        = offset(msh, cl);
        auto cell_LHS_offset    = cell_offset * cbs_A;
        

        auto B_offset = cbs_A * msh.cells.size() + fbs_A * num_other_faces + cbs_B * cell_offset;
        
        for (size_t i = 0; i < cbs_A; i++)
            asm_map.push_back( assembly_index(cell_LHS_offset+i, true) );

        Matrix<T, Dynamic, 1> dirichlet_data = Matrix<T, Dynamic, 1>::Zero(cbs_A + num_faces*fbs_A);

        for (size_t face_i = 0; face_i < num_faces; face_i++)
        {
            auto fc = fcs[face_i];
            auto face_offset = offset(msh, fc);
            auto face_LHS_offset = cbs_A * msh.cells.size() + compress_table.at(face_offset)*fbs_A;

            bool dirichlet = fc.is_boundary && fc.bndtype == boundary::DIRICHLET;

            for (size_t i = 0; i < fbs_A; i++)
                asm_map.push_back( assembly_index(face_LHS_offset+i, !dirichlet) );

            if (dirichlet)
            {
                Matrix<T, Dynamic, Dynamic> mass = make_vector_mass_matrix(msh, fc, facdeg);
                Matrix<T, Dynamic, 1> rhs = make_vector_rhs(msh, fc, facdeg, dirichlet_bf);
                dirichlet_data.block(cbs_A + face_i*fbs_A, 0, fbs_A, 1) = mass.llt().solve(rhs);
            }
        }
        
        assert( asm_map.size() == lhs_A.rows() && asm_map.size() == lhs_A.cols() );
        
        for (size_t i = 0; i < lhs_A.rows(); i++)
        {
            if (!asm_map[i].assemble())
                continue;

            for (size_t j = 0; j < lhs_A.cols(); j++)
            {   
                if ( asm_map[j].assemble() )
                    triplets.push_back( Triplet<T>(asm_map[i], asm_map[j], lhs_A(i,j)) );
                else
                    RHS(asm_map[i]) -= lhs_A(i,j) * dirichlet_data(j);
            }
        }

        for (size_t i = 0; i < lhs_B.rows(); i++)
        {
            for (size_t j = 0; j < lhs_B.cols(); j++)
            {
                auto global_i = B_offset + i;
                auto global_j = asm_map[j];
                if ( asm_map[j].assemble() )
                {
                    triplets.push_back( Triplet<T>(global_i, global_j, lhs_B(i,j)) );
                    triplets.push_back( Triplet<T>(global_j, global_i, lhs_B(i,j)) );
                }
                else
                    RHS(global_i) -= lhs_B(i,j)*dirichlet_data(j);
            }
        }

        // null pressure mean condition
        cell_basis<cuthho_poly_mesh<T>, T> cb(msh, cl, di.face_degree());
        auto qpsi = integrate(msh, cl, di.face_degree(), where);
        Matrix<T, Dynamic, 1> mult = Matrix<T, Dynamic, 1>::Zero( cbs_B );
        for (auto& qp : qpsi)
        {
            auto phi = cb.eval_basis(qp.first);
            mult += qp.second * phi;
        }
        auto mult_offset = cbs_A * msh.cells.size() + fbs_A * num_other_faces + cbs_B * msh.cells.size();

        for (size_t i = 0; i < mult.rows(); i++)
        {
            triplets.push_back( Triplet<T>(B_offset+i, mult_offset, mult(i)) );
            triplets.push_back( Triplet<T>(mult_offset, B_offset+i, mult(i)) );
        }

        
        // handling of rhs terms
        RHS.block(cell_LHS_offset, 0, cbs_A, 1) += rhs_A.block(0, 0, cbs_A, 1);
        if ( rhs_A.rows() > cbs_A )
        {
            for (size_t face_i = 0; face_i < num_faces; face_i++)
            {
                auto fc = fcs[face_i];
                auto face_offset = offset(msh, fc);
                auto face_LHS_offset = cbs_A * msh.cells.size() + compress_table.at(face_offset)*fbs_A;
                
                RHS.block(face_LHS_offset, 0, fbs_A, 1) += rhs_A.block(cbs_A+face_i*fbs_A, 0, fbs_A, 1);
            }
            // pressure rhs
            RHS.block(B_offset, 0, cbs_B, 1) += rhs_B.block(0, 0, cbs_B, 1);
        }
    } // assemble()

    template<typename Function>
    Matrix<T, Dynamic, 1>
    take_velocity(const Mesh& msh, const typename Mesh::cell_type& cl,
    const Matrix<T, Dynamic, 1>& solution, const Function& dirichlet_bf)
    {
        auto celdeg = di.cell_degree();
        auto facdeg = di.face_degree();

        auto cbs = vector_cell_basis<Mesh,T>::size(celdeg);
        auto fbs = vector_face_basis<Mesh,T>::size(facdeg);

        auto cell_offset        = offset(msh, cl);
        auto cell_SOL_offset    = cell_offset * cbs;

        auto fcs = faces(msh, cl);
        auto num_faces = fcs.size();

        Matrix<T, Dynamic, 1> ret = Matrix<T, Dynamic, 1>::Zero(cbs + num_faces*fbs);
        ret.block(0, 0, cbs, 1) = solution.block(cell_SOL_offset, 0, cbs, 1);

        for (size_t face_i = 0; face_i < num_faces; face_i++)
        {
            auto fc = fcs[face_i];

            bool dirichlet = fc.is_boundary && fc.bndtype == boundary::DIRICHLET;

            if (dirichlet)
            {
                Matrix<T, Dynamic, Dynamic> mass = make_vector_mass_matrix(msh, fc, facdeg);
                Matrix<T, Dynamic, 1> rhs = make_vector_rhs(msh, fc, facdeg, dirichlet_bf);
                ret.block(cbs+face_i*fbs, 0, fbs, 1) = mass.llt().solve(rhs);
            }
            else
            {
                auto face_offset = offset(msh, fc);
                auto face_SOL_offset = cbs * msh.cells.size() + compress_table.at(face_offset)*fbs;
                ret.block(cbs+face_i*fbs, 0, fbs, 1) = solution.block(face_SOL_offset, 0, fbs, 1);
            }
        }

        return ret;
    }


    Matrix<T, Dynamic, 1>
    take_pressure(const Mesh& msh, const typename Mesh::cell_type& cl,
                  const Matrix<T, Dynamic, 1>& sol) const
    {
        auto celdeg = di.cell_degree();
        auto facdeg = di.face_degree();
        
        auto cbs_A = vector_cell_basis<Mesh,T>::size(celdeg);
        auto fbs_A = vector_face_basis<Mesh,T>::size(facdeg);
        auto cbs_B = cell_basis<Mesh,T>::size(facdeg);
        
        auto cell_offset    = offset(msh, cl);
        auto pres_offset    = cbs_A * msh.cells.size() + fbs_A * num_other_faces
                                                            + cbs_B * cell_offset;

        Matrix<T, Dynamic, 1> spres = sol.block(pres_offset, 0, cbs_B, 1);
        return spres;
    }
    
    void finalize(void)
    {
        LHS.setFromTriplets( triplets.begin(), triplets.end() );
        triplets.clear();
    }
};


template<typename Mesh>
auto make_stokes_assembler(const Mesh& msh, hho_degree_info hdi)
{
    return stokes_assembler<Mesh>(msh, hdi);
}


//////////////

template<typename Mesh>
class stokes_condensed_assembler
{
    using T = typename Mesh::coordinate_type;
    std::vector<size_t>                 compress_table;
    std::vector<size_t>                 expand_table;

    hho_degree_info                     di;

    std::vector< Triplet<T> >           triplets;

    size_t num_other_faces;
    
    class assembly_index
    {
        size_t  idx;
        bool    assem;

    public:
        assembly_index(size_t i, bool as)
            : idx(i), assem(as)
        {}

        operator size_t() const
        {
            if (!assem)
                throw std::logic_error("Invalid assembly_index");

            return idx;
        }

        bool assemble() const
        {
            return assem;
        }

        friend std::ostream& operator<<(std::ostream& os, const assembly_index& as)
        {
            os << "(" << as.idx << "," << as.assem << ")";
            return os;
        }
    };

public:

    SparseMatrix<T>         LHS;
    Matrix<T, Dynamic, 1>   RHS;

    stokes_condensed_assembler(const Mesh& msh, hho_degree_info hdi)
        : di(hdi)
    {
        auto is_dirichlet = [&](const typename Mesh::face_type& fc) -> bool {
            return fc.is_boundary && fc.bndtype == boundary::DIRICHLET;
        };

        auto num_all_faces = msh.faces.size();
        auto num_dirichlet_faces = std::count_if(msh.faces.begin(), msh.faces.end(), is_dirichlet);

        num_other_faces = num_all_faces - num_dirichlet_faces;

        compress_table.resize( num_all_faces );
        expand_table.resize( num_other_faces );

        size_t compressed_offset = 0;
        for (size_t i = 0; i < num_all_faces; i++)
        {
            auto fc = msh.faces[i];
            if ( !is_dirichlet(fc) )
            {
                compress_table.at(i) = compressed_offset;
                expand_table.at(compressed_offset) = i;
                compressed_offset++;
            }
        }

        auto facdeg = di.face_degree();

        auto fbs = vector_face_basis<Mesh,T>::size(facdeg);

        auto system_size = fbs * num_other_faces + msh.cells.size() + 1;

        LHS = SparseMatrix<T>( system_size, system_size );
        RHS = Matrix<T, Dynamic, 1>::Zero( system_size );
    }

    void dump_tables() const
    {
        std::cout << "Compress table: " << std::endl;
        for (size_t i = 0; i < compress_table.size(); i++)
            std::cout << i << " -> " << compress_table.at(i) << std::endl;
    }

    template<typename Function>
    void
    assemble(const Mesh& msh, const typename Mesh::cell_type& cl,
             const Matrix<T, Dynamic, Dynamic>& lhs_A, const Matrix<T, Dynamic, Dynamic>& lhs_B,
             const Matrix<T, Dynamic, 1>& rhs_A, const Matrix<T, Dynamic, 1>& rhs_B,
             const Function& dirichlet_bf, const element_location where)
    {
        auto facdeg = di.face_degree();
        auto celdeg = di.cell_degree();

        auto cbs_A = vector_cell_basis<Mesh,T>::size(celdeg);
        auto cbs_B = cell_basis<Mesh,T>::size(facdeg);
        auto fbs_A = vector_face_basis<Mesh,T>::size(facdeg);

        auto fcs = faces(msh, cl);
        auto num_faces = fcs.size();
        size_t f_dofs = num_faces * fbs_A;
        
        // static condensation
        auto mat_sc = stokes_static_condensation_compute(msh, cl, di, lhs_A, lhs_B, rhs_A, rhs_B);

        // condensed matrices and rhs
        Matrix<T, Dynamic, Dynamic> lhs_A_sc = mat_sc.first.block(0, 0, f_dofs, f_dofs);
        Matrix<T, Dynamic, Dynamic> lhs_B_sc = mat_sc.first.block(f_dofs, 0, 1, f_dofs);
        Matrix<T, Dynamic, 1> rhs_A_sc = mat_sc.second.head(f_dofs);
        Matrix<T, Dynamic, 1> rhs_B_sc = mat_sc.second.tail(1);


        
        std::vector<assembly_index> asm_map;
        asm_map.reserve(num_faces*fbs_A); 

        auto cell_offset        = offset(msh, cl);
        

        auto B_offset = fbs_A * num_other_faces + cell_offset;
        
        Matrix<T, Dynamic, 1> dirichlet_data = Matrix<T, Dynamic, 1>::Zero(num_faces*fbs_A);

        for (size_t face_i = 0; face_i < num_faces; face_i++)
        {
            auto fc = fcs[face_i];
            auto face_offset = offset(msh, fc);
            auto face_LHS_offset = compress_table.at(face_offset)*fbs_A;

            bool dirichlet = fc.is_boundary && fc.bndtype == boundary::DIRICHLET;

            for (size_t i = 0; i < fbs_A; i++)
                asm_map.push_back( assembly_index(face_LHS_offset+i, !dirichlet) );

            if (dirichlet)
            {
                Matrix<T, Dynamic, Dynamic> mass = make_vector_mass_matrix(msh, fc, facdeg);
                Matrix<T, Dynamic, 1> rhs = make_vector_rhs(msh, fc, facdeg, dirichlet_bf);
                dirichlet_data.block(face_i*fbs_A, 0, fbs_A, 1) = mass.llt().solve(rhs);
            }
        }
        
        assert( asm_map.size() == lhs_A_sc.rows() && asm_map.size() == lhs_A_sc.cols() );
        
        for (size_t i = 0; i < lhs_A_sc.rows(); i++)
        {
            if (!asm_map[i].assemble())
                continue;

            for (size_t j = 0; j < lhs_A_sc.cols(); j++)
            {   
                if ( asm_map[j].assemble() )
                    triplets.push_back( Triplet<T>(asm_map[i], asm_map[j], lhs_A_sc(i,j)) );
                else
                    RHS(asm_map[i]) -= lhs_A_sc(i,j) * dirichlet_data(j);
            }
        }

        assert( lhs_B_sc.rows() == 1 && lhs_B_sc.cols() == num_faces * fbs_A);
        
        for (size_t i = 0; i < lhs_B_sc.rows(); i++)
        {
            for (size_t j = 0; j < lhs_B_sc.cols(); j++)
            {
                auto global_i = B_offset + i;
                auto global_j = asm_map[j];
                if ( asm_map[j].assemble() )
                {
                    triplets.push_back( Triplet<T>(global_i, global_j, lhs_B_sc(i,j)) );
                    triplets.push_back( Triplet<T>(global_j, global_i, lhs_B_sc(i,j)) );
                }
                else
                    RHS(global_i) -= lhs_B_sc(i,j)*dirichlet_data(j);
            }
        }

        /////////////////////////
        // null mean pressure condition
        /////////////////////////
        
        Matrix<T, Dynamic, 1> p_rec_lhs = Matrix<T, Dynamic, 1>::Zero(f_dofs);
        T p_rec_rhs = 0;

        cell_basis<cuthho_poly_mesh<T>, T> cb(msh, cl, facdeg);

        
        Matrix<T, Dynamic, 1> mult = Matrix<T, Dynamic, 1>::Zero( 1 );
        Matrix<T, Dynamic, 1> vect_phi= Matrix<T, Dynamic, 1>::Zero(cbs_B-1);
        
        auto qps = integrate(msh, cl, facdeg, where );
        for (auto& qp : qps)
        {
            auto phi = cb.eval_basis(qp.first);
            mult += qp.second * phi.head(1);
            vect_phi += phi.tail(cbs_B-1);
        }

        for(size_t i = 0; i < f_dofs; i++)
        {
            Matrix<T, Dynamic, 1> p_rec=Matrix<T, Dynamic, 1>::Zero(cbs_B-1);

            // lhs
            Matrix<T, Dynamic, 1> sol_F= Matrix<T, Dynamic, 1>::Zero(f_dofs);
            sol_F(i) = 1.0;

            Matrix<T, Dynamic, 1> cell_rhs= Matrix<T, Dynamic, 1>::Zero(cbs_A);
            Matrix<T, Dynamic, 1> B_rhs= Matrix<T, Dynamic, 1>::Zero(cbs_B);
                    
            p_rec = stokes_static_condensation_recover(msh, cl, di, lhs_A, lhs_B, cell_rhs, B_rhs, sol_F, 0.0).tail(cbs_B-1);
                    
            p_rec_lhs(i) = p_rec.dot(vect_phi);
                    
            // rhs
            sol_F(i) = 0.0;
            cell_rhs = rhs_A.head(cbs_A);
            B_rhs.tail(cbs_B-1) = rhs_B.tail(cbs_B-1);

            p_rec = stokes_static_condensation_recover(msh, cl, di, lhs_A, lhs_B, cell_rhs, B_rhs, sol_F, 0.0).tail(cbs_B-1);

            p_rec_rhs += p_rec.dot(vect_phi);
        }
                
        auto mult_offset = fbs_A * num_other_faces + msh.cells.size();

        triplets.push_back( Triplet<T>(B_offset, mult_offset, mult(0)) );
        triplets.push_back( Triplet<T>(mult_offset, B_offset, mult(0)) );        
        
        for (size_t i = 0; i < p_rec_lhs.rows(); i++)
        {   
            if ( asm_map[i].assemble() )
            {
                triplets.push_back( Triplet<T>(asm_map[i], mult_offset, p_rec_lhs(i) ) );
                triplets.push_back( Triplet<T>(mult_offset, asm_map[i], p_rec_lhs(i) ) );
            }
            else
                RHS(mult_offset) -= p_rec_lhs(i) * dirichlet_data(i);
        }
        
        RHS(mult_offset) -= p_rec_rhs;

        ////////////////////////
        // handling of rhs terms
        ////////////////////////
        for (size_t face_i = 0; face_i < num_faces; face_i++)
        {
            auto fc = fcs[face_i];
            auto face_offset = offset(msh, fc);
            auto face_LHS_offset = compress_table.at(face_offset)*fbs_A;
            
            RHS.block(face_LHS_offset, 0, fbs_A, 1) += rhs_A_sc.block(face_i*fbs_A, 0, fbs_A, 1);
        }
        // pressure rhs
        RHS.block(B_offset, 0, 1, 1) += rhs_B_sc.block(0, 0, 1, 1);
        
    } // assemble()

    template<typename Function>
    Matrix<T, Dynamic, 1>
    take_velocity(const Mesh& msh, const typename Mesh::cell_type& cl,
    const Matrix<T, Dynamic, 1>& solution, const Function& dirichlet_bf)
    {
        auto facdeg = di.face_degree();

        auto fbs = vector_face_basis<Mesh,T>::size(facdeg);

        auto fcs = faces(msh, cl);
        auto num_faces = fcs.size();

        Matrix<T, Dynamic, 1> ret = Matrix<T, Dynamic, 1>::Zero(num_faces * fbs);

        for (size_t face_i = 0; face_i < num_faces; face_i++)
        {
            auto fc = fcs[face_i];

            bool dirichlet = fc.is_boundary && fc.bndtype == boundary::DIRICHLET;

            if (dirichlet)
            {
                Matrix<T, Dynamic, Dynamic> mass = make_vector_mass_matrix(msh, fc, facdeg);
                Matrix<T, Dynamic, 1> rhs = make_vector_rhs(msh, fc, facdeg, dirichlet_bf);
                ret.block(face_i * fbs, 0, fbs, 1) = mass.llt().solve(rhs);
            }
            else
            {
                auto face_offset = offset(msh, fc);
                auto face_SOL_offset = compress_table.at(face_offset) * fbs;
                ret.block(face_i * fbs, 0, fbs, 1) = solution.block(face_SOL_offset, 0, fbs, 1);
            }
        }

        return ret;
    }


    T
    take_pressure(const Mesh& msh, const typename Mesh::cell_type& cl,
                  const Matrix<T, Dynamic, 1>& sol) const
    {
        auto facdeg = di.face_degree();
        
        auto fbs_A = vector_face_basis<Mesh,T>::size(facdeg);
        
        auto cell_offset    = offset(msh, cl);
        auto pres_offset    = fbs_A * num_other_faces + cell_offset;

        T pres = sol[pres_offset];
        return pres;
    }
    
    void finalize(void)
    {
        LHS.setFromTriplets( triplets.begin(), triplets.end() );
        triplets.clear();
    }
};


template<typename Mesh>
auto make_stokes_condensed_assembler(const Mesh& msh, hho_degree_info hdi)
{
    return stokes_condensed_assembler<Mesh>(msh, hdi);
}




//////////////



template<typename Mesh>
class interface_assembler
{
    using T = typename Mesh::coordinate_type;
    std::vector<size_t>                 cell_table, face_table;
    size_t num_all_cells, num_all_faces;

    hho_degree_info                     di;

    std::vector< Triplet<T> >           triplets;

    class assembly_index
    {
        size_t  idx;
        bool    assem;

    public:
        assembly_index(size_t i, bool as)
            : idx(i), assem(as)
        {}

        operator size_t() const
        {
            if (!assem)
                throw std::logic_error("Invalid assembly_index");

            return idx;
        }

        bool assemble() const
        {
            return assem;
        }

        friend std::ostream& operator<<(std::ostream& os, const assembly_index& as)
        {
            os << "(" << as.idx << "," << as.assem << ")";
            return os;
        }
    };

public:

    SparseMatrix<T>         LHS;
    Matrix<T, Dynamic, 1>   RHS;

    interface_assembler(const Mesh& msh, hho_degree_info hdi)
        : di(hdi)
    {
        auto is_dirichlet = [&](const typename Mesh::face_type& fc) -> bool {
            return fc.is_boundary && fc.bndtype == boundary::DIRICHLET;
        };

        num_all_cells = 0; /* counts cells with dup. unknowns */
        for (auto& cl : msh.cells)
        {
            cell_table.push_back( num_all_cells );
            if (location(msh, cl) == element_location::ON_INTERFACE)
                num_all_cells += 2;
            else
                num_all_cells += 1;
        }
        assert(cell_table.size() == msh.cells.size());

        num_all_faces = 0; /* counts faces with dup. unknowns */
        for (auto& fc : msh.faces)
        {
            if (location(msh, fc) == element_location::ON_INTERFACE)
                num_all_faces += 2;
            else
                num_all_faces += 1;
        }

        /* We assume that cut cells can not have dirichlet faces */
        auto num_dirichlet_faces = std::count_if(msh.faces.begin(), msh.faces.end(), is_dirichlet);
        auto num_other_faces = num_all_faces - num_dirichlet_faces;

        face_table.resize( msh.faces.size() );

        size_t compressed_offset = 0;
        for (size_t i = 0; i < msh.faces.size(); i++)
        {
            auto fc = msh.faces.at(i);
            if ( !is_dirichlet(fc) )
            {
                face_table.at(i) = compressed_offset;
                if ( location(msh, fc) == element_location::ON_INTERFACE )
                    compressed_offset += 2;
                else
                    compressed_offset += 1;
            }
        }

        auto celdeg = di.cell_degree();
        auto facdeg = di.face_degree();

        auto cbs = cell_basis<Mesh,T>::size(celdeg);
        auto fbs = face_basis<Mesh,T>::size(facdeg);

        auto system_size = cbs * num_all_cells + fbs * num_other_faces;

        LHS = SparseMatrix<T>( system_size, system_size );
        RHS = Matrix<T, Dynamic, 1>::Zero( system_size );
    }

    void dump_tables() const
    {
        //std::cout << "Compress table: " << std::endl;
        //for (size_t i = 0; i < compress_table.size(); i++)
        //    std::cout << i << " -> " << compress_table.at(i) << std::endl;
    }

    template<typename Function>
    void
    assemble(const Mesh& msh, const typename Mesh::cell_type& cl,
             const Matrix<T, Dynamic, Dynamic>& lhs, const Matrix<T, Dynamic, 1>& rhs,
             const Function& dirichlet_bf)
    {
        if (location(msh, cl) == element_location::ON_INTERFACE)
            throw std::invalid_argument("UNcut cell expected.");

        auto celdeg = di.cell_degree();
        auto facdeg = di.face_degree();

        auto cbs = cell_basis<Mesh,T>::size(celdeg);
        auto fbs = face_basis<Mesh,T>::size(facdeg);

        auto fcs = faces(msh, cl);
        auto num_faces = fcs.size();

        std::vector<assembly_index> asm_map;
        asm_map.reserve(cbs + num_faces*fbs);

        auto cell_offset        = offset(msh, cl);
        auto cell_LHS_offset = cell_table.at(cell_offset) * cbs;

        for (size_t i = 0; i < cbs; i++)
            asm_map.push_back( assembly_index(cell_LHS_offset+i, true) );

        Matrix<T, Dynamic, 1> dirichlet_data = Matrix<T, Dynamic, 1>::Zero(cbs + num_faces*fbs);

        for (size_t face_i = 0; face_i < num_faces; face_i++)
        {
            auto fc = fcs[face_i];
            auto face_offset = offset(msh, fc);
            auto face_LHS_offset = num_all_cells * cbs + face_table.at(face_offset) * fbs;

            bool dirichlet = fc.is_boundary && fc.bndtype == boundary::DIRICHLET;

            for (size_t i = 0; i < fbs; i++)
                asm_map.push_back( assembly_index(face_LHS_offset+i, !dirichlet) );

            if (dirichlet)
            {
                Matrix<T, Dynamic, Dynamic> mass = make_mass_matrix(msh, fc, facdeg);
                Matrix<T, Dynamic, 1> rhs = make_rhs(msh, fc, facdeg, dirichlet_bf);
                dirichlet_data.block(cbs+face_i*fbs, 0, fbs, 1) = mass.llt().solve(rhs);
            }
        }

        assert( asm_map.size() == lhs.rows() && asm_map.size() == lhs.cols() );

        for (size_t i = 0; i < lhs.rows(); i++)
        {
            if (!asm_map[i].assemble())
                continue;

            for (size_t j = 0; j < lhs.cols(); j++)
            {
                if ( asm_map[j].assemble() )
                    triplets.push_back( Triplet<T>(asm_map[i], asm_map[j], lhs(i,j)) );
                else
                    RHS(asm_map[i]) -= lhs(i,j)*dirichlet_data(j);
            }
        }

        RHS.block(cell_LHS_offset, 0, cbs, 1) += rhs.block(0, 0, cbs, 1);

        //for (auto& am : asm_map)
        //    std::cout << am << " ";
        //std::cout << std::endl;
    } // assemble()

    void
    assemble_cut(const Mesh& msh, const typename Mesh::cell_type& cl,
             const Matrix<T, Dynamic, Dynamic>& lhs, const Matrix<T, Dynamic, 1>& rhs)
    {
        if (location(msh, cl) != element_location::ON_INTERFACE)
            throw std::invalid_argument("Cut cell expected.");

        auto celdeg = di.cell_degree();
        auto facdeg = di.face_degree();

        auto cbs = cell_basis<Mesh,T>::size(celdeg);
        auto fbs = face_basis<Mesh,T>::size(facdeg);

        auto fcs = faces(msh, cl);
        auto num_faces = fcs.size();

        std::vector<assembly_index> asm_map;
        asm_map.reserve( 2*(cbs + num_faces*fbs) );

        auto cell_offset = offset(msh, cl);
        auto cell_LHS_offset = cell_table.at(cell_offset) * cbs;

        for (size_t i = 0; i < 2*cbs; i++)
            asm_map.push_back( assembly_index(cell_LHS_offset+i, true) );

        for (size_t face_i = 0; face_i < num_faces; face_i++)
        {
            auto fc = fcs[face_i];
            auto face_offset = offset(msh, fc);
            auto face_LHS_offset = num_all_cells * cbs + face_table.at(face_offset) * fbs;

            bool dirichlet = fc.is_boundary && fc.bndtype == boundary::DIRICHLET;
            if ( dirichlet )
                throw std::invalid_argument("Dirichlet boundary on cut cell not supported.");

            bool c1 = location(msh, fc) == element_location::IN_NEGATIVE_SIDE;
            bool c2 = location(msh, fc) == element_location::ON_INTERFACE;
            for (size_t i = 0; i < fbs; i++)
                asm_map.push_back( assembly_index(face_LHS_offset+i, true) );
        }

        for (size_t face_i = 0; face_i < num_faces; face_i++)
        {
            auto fc = fcs[face_i];
            auto face_offset = offset(msh, fc);

            auto d = (location(msh, fc) == element_location::ON_INTERFACE) ? fbs : 0;

            auto face_LHS_offset = num_all_cells * cbs + face_table.at(face_offset) * fbs + d;

            bool dirichlet = fc.is_boundary && fc.bndtype == boundary::DIRICHLET;
            if ( dirichlet )
                throw std::invalid_argument("Dirichlet boundary on cut cell not supported.");

            bool c1 = location(msh, fc) == element_location::IN_POSITIVE_SIDE;
            bool c2 = location(msh, fc) == element_location::ON_INTERFACE;
            for (size_t i = 0; i < fbs; i++)
                asm_map.push_back( assembly_index(face_LHS_offset+i, true) );
        }

        assert( asm_map.size() == lhs.rows() && asm_map.size() == lhs.cols() );

        for (size_t i = 0; i < lhs.rows(); i++)
        {
            if (!asm_map[i].assemble())
                continue;

            for (size_t j = 0; j < lhs.cols(); j++)
            {
                if ( asm_map[j].assemble() )
                    triplets.push_back( Triplet<T>(asm_map[i], asm_map[j], lhs(i,j)) );
            }
        }

        RHS.block(cell_LHS_offset, 0, 2*cbs, 1) += rhs.block(0, 0, 2*cbs, 1);

        size_t face_offset_loc = 0;
        for (size_t face_i = 0; face_i < num_faces; face_i++)
        {
            auto fc = fcs[face_i];
            auto face_offset = offset(msh, fc);
            
            auto face_LHS_offset = num_all_cells * cbs + face_table.at(face_offset) * fbs;

            if( location(msh, fc) == element_location::ON_INTERFACE )
            {
                RHS.block(face_LHS_offset , 0 , fbs , 1)
                    += rhs.block(2*cbs + face_i * fbs , 0 , fbs , 1);
                
                RHS.block(face_LHS_offset + fbs , 0 , fbs , 1)
                    += rhs.block(2*cbs + (num_faces + face_i) * fbs , 0 , fbs , 1);
            }
            else if( location(msh, fc) == element_location::IN_NEGATIVE_SIDE )
            {
                RHS.block(face_LHS_offset , 0 , fbs , 1)
                    += rhs.block(2*cbs + face_i * fbs , 0 , fbs , 1);
            }
            else if( location(msh, fc) == element_location::IN_POSITIVE_SIDE )
            {
                RHS.block(face_LHS_offset , 0 , fbs , 1)
                    += rhs.block(2*cbs + (num_faces + face_i) * fbs , 0 , fbs , 1);
            }
            else
                throw std::logic_error("shouldn't have arrived here...");
        }
    } // assemble_cut()


    template<typename Function>
    Matrix<T, Dynamic, 1>
    take_local_data(const Mesh& msh, const typename Mesh::cell_type& cl,
                    const Matrix<T, Dynamic, 1>& solution,
                    const Function& dirichlet_bf,
                    element_location where)
    {
        auto celdeg = di.cell_degree();
        auto facdeg = di.face_degree();

        auto cbs = cell_basis<Mesh,T>::size(celdeg);
        auto fbs = face_basis<Mesh,T>::size(facdeg);

        auto cell_offset        = offset(msh, cl);
        size_t cell_SOL_offset;
        if ( location(msh, cl) == element_location::ON_INTERFACE )
        {
            if (where == element_location::IN_NEGATIVE_SIDE)
                cell_SOL_offset = cell_table.at(cell_offset) * cbs;
            else if (where == element_location::IN_POSITIVE_SIDE)
                cell_SOL_offset = cell_table.at(cell_offset) * cbs + cbs;
            else
                throw std::invalid_argument("Invalid location");
        }
        else
        {
            cell_SOL_offset = cell_table.at(cell_offset) * cbs;
        }

        auto fcs = faces(msh, cl);
        auto num_faces = fcs.size();

        Matrix<T, Dynamic, 1> ret = Matrix<T, Dynamic, 1>::Zero(cbs + num_faces*fbs);
        ret.block(0, 0, cbs, 1) = solution.block(cell_SOL_offset, 0, cbs, 1);

        for (size_t face_i = 0; face_i < num_faces; face_i++)
        {
            auto fc = fcs[face_i];

            auto face_offset = offset(msh, fc);
            size_t face_SOL_offset;
            if ( location(msh, fc) == element_location::ON_INTERFACE )
            {
                if (where == element_location::IN_NEGATIVE_SIDE)
                    face_SOL_offset = num_all_cells * cbs + face_table.at(face_offset) * fbs;
                else if (where == element_location::IN_POSITIVE_SIDE)
                    face_SOL_offset = num_all_cells * cbs + face_table.at(face_offset) * fbs + fbs;
                else
                    throw std::invalid_argument("Invalid location");
            }
            else
            {
                face_SOL_offset = num_all_cells * cbs + face_table.at(face_offset) * fbs;
            }

            bool dirichlet = fc.is_boundary && fc.bndtype == boundary::DIRICHLET;

            if (dirichlet)
            {
                Matrix<T, Dynamic, Dynamic> mass = make_mass_matrix(msh, fc, facdeg);
                Matrix<T, Dynamic, 1> rhs = make_rhs(msh, fc, facdeg, dirichlet_bf);
                ret.block(cbs+face_i*fbs, 0, fbs, 1) = mass.llt().solve(rhs);
            }
            else
            {
                ret.block(cbs+face_i*fbs, 0, fbs, 1) = solution.block(face_SOL_offset, 0, fbs, 1);
            }
        }

        return ret;
    }

    void finalize(void)
    {
        LHS.setFromTriplets( triplets.begin(), triplets.end() );
        triplets.clear();
    }
};


template<typename Mesh>
auto make_interface_assembler(const Mesh& msh, hho_degree_info hdi)
{
    return interface_assembler<Mesh>(msh, hdi);
}







template<typename Mesh, typename Function>
void
output_mesh_info(const Mesh& msh, const Function& level_set_function)
{
    using RealType = typename Mesh::coordinate_type;

    /************** OPEN SILO DATABASE **************/
    silo_database silo;
    silo.create("cuthho_meshinfo.silo");
    silo.add_mesh(msh, "mesh");

    /************** MAKE A SILO VARIABLE FOR CELL POSITIONING **************/
    std::vector<RealType> cut_cell_markers;
    for (auto& cl : msh.cells)
    {
        if ( location(msh, cl) == element_location::IN_POSITIVE_SIDE )
            cut_cell_markers.push_back(1.0);
        else if ( location(msh, cl) == element_location::IN_NEGATIVE_SIDE )
            cut_cell_markers.push_back(-1.0);
        else if ( location(msh, cl) == element_location::ON_INTERFACE )
            cut_cell_markers.push_back(0.0);
        else
            throw std::logic_error("shouldn't have arrived here...");
    }
    silo.add_variable("mesh", "cut_cells", cut_cell_markers.data(), cut_cell_markers.size(), zonal_variable_t);

    /************** MAKE A SILO VARIABLE FOR LEVEL SET FUNCTION **************/
    std::vector<RealType> level_set_vals;
    for (auto& pt : msh.points)
        level_set_vals.push_back( level_set_function(pt) );
    silo.add_variable("mesh", "level_set", level_set_vals.data(), level_set_vals.size(), nodal_variable_t);

    /************** MAKE A SILO VARIABLE FOR NODE POSITIONING **************/
    std::vector<RealType> node_pos;
    for (auto& n : msh.nodes)
        node_pos.push_back( location(msh, n) == element_location::IN_POSITIVE_SIDE ? +1.0 : -1.0 );
    silo.add_variable("mesh", "node_pos", node_pos.data(), node_pos.size(), nodal_variable_t);

    std::vector<RealType> cell_set;
    for (auto& cl : msh.cells)
    {
        RealType r;

        switch ( cl.user_data.agglo_set )
        {
            case cell_agglo_set::UNDEF:
                r = 0.0;
                break;

            case cell_agglo_set::T_OK:
                r = 1.0;
                break;

            case cell_agglo_set::T_KO_NEG:
                r = 2.0;
                break;

            case cell_agglo_set::T_KO_POS:
                r = 3.0;
                break;

        }

        cell_set.push_back( r );
    }
    silo.add_variable("mesh", "agglo_set", cell_set.data(), cell_set.size(), zonal_variable_t);

    
    silo.close();

    
    /*************  MAKE AN OUTPUT FOR THE INTERSECTION POINTS *************/
    std::vector<RealType> int_pts_x;
    std::vector<RealType> int_pts_y;
    
    for (auto& fc : msh.faces)
    {
        if( fc.user_data.location != element_location::ON_INTERFACE ) continue;

        RealType x = fc.user_data.intersection_point.x();
        RealType y = fc.user_data.intersection_point.y();
        
        int_pts_x.push_back(x);
        int_pts_y.push_back(y);
        
    }
    
    
    std::ofstream points_file("int_points.3D", std::ios::out | std::ios::trunc); 
 

    if(points_file) 
    {       
        // instructions
        points_file << "X   Y   Z   val" << std::endl;

        for( size_t i = 0; i<int_pts_x.size(); i++)
        {
            points_file << int_pts_x[i] << "   " <<  int_pts_y[i]
                        << "   0.0     0.0" << std::endl;
        }
            
        points_file.close(); 
    }

    else 
        std::cerr << "Points_file has not been opened" << std::endl;

}



template<typename Mesh>
std::vector<size_t>
agglomerate_cells(const Mesh& msh, const typename Mesh::cell_type& cl_tgt,
                  const typename Mesh::cell_type& cl_other)
{
    auto ofs_tgt    = offset(msh, cl_tgt);
    auto ofs_other  = offset(msh, cl_other);

    auto pts_tgt    = points(msh, cl_tgt);
    auto pts_other  = points(msh, cl_other);

    size_t Nx = 0;

    std::vector<size_t> ret;

    if (ofs_other == ofs_tgt - Nx - 1)
    {
        ret.push_back( pts_tgt[0] );
        ret.push_back( pts_tgt[1] );
        ret.push_back( pts_tgt[2] );
        ret.push_back( pts_tgt[3] );
        ret.push_back( pts_other[2] );
        ret.push_back( pts_other[3] );
        ret.push_back( pts_other[0] );
        ret.push_back( pts_other[1] );
    }
    else if (ofs_other == ofs_tgt - 1)
    {
        ret.push_back( pts_tgt[0] );
        ret.push_back( pts_tgt[1] );
        ret.push_back( pts_tgt[2] );
        ret.push_back( pts_tgt[3] );
        ret.push_back( pts_other[3] );
        ret.push_back( pts_other[0] );
    }
    else if (ofs_other == ofs_tgt + Nx - 1)
    {
        ret.push_back( pts_tgt[0] );
        ret.push_back( pts_tgt[1] );
        ret.push_back( pts_tgt[2] );
        ret.push_back( pts_tgt[3] );
        ret.push_back( pts_other[2] );
        ret.push_back( pts_other[3] );
        ret.push_back( pts_other[0] );
        ret.push_back( pts_other[1] );
    }
    else if (ofs_other == ofs_tgt + Nx)
    {
        ret.push_back( pts_tgt[0] );
        ret.push_back( pts_tgt[1] );
        ret.push_back( pts_tgt[2] );
        ret.push_back( pts_other[2] );
        ret.push_back( pts_other[3] );
        ret.push_back( pts_other[0] );
    }
    else if (ofs_other == ofs_tgt + Nx + 1)
    {   
        ret.push_back( pts_tgt[0] );
        ret.push_back( pts_tgt[1] );
        ret.push_back( pts_tgt[2] );
        ret.push_back( pts_other[1] );
        ret.push_back( pts_other[2] );
        ret.push_back( pts_other[3] );
        ret.push_back( pts_other[0] );
        ret.push_back( pts_other[3] );
    }
    else if (ofs_other == ofs_tgt + 1)
    {
        ret.push_back( pts_tgt[0] );
        ret.push_back( pts_tgt[1] );
        ret.push_back( pts_other[1] );
        ret.push_back( pts_other[2] );
        ret.push_back( pts_other[3] );
        ret.push_back( pts_tgt[3] );
    }
    else if (ofs_other == ofs_tgt - Nx + 1)
    {
        ret.push_back( pts_tgt[0] );
        ret.push_back( pts_tgt[1] );
        ret.push_back( pts_other[0] );
        ret.push_back( pts_other[1] );
        ret.push_back( pts_other[2] );
        ret.push_back( pts_other[3] );
        ret.push_back( pts_tgt[2] );
        ret.push_back( pts_tgt[3] );
    }
    else if (ofs_other == ofs_tgt - Nx)
    {
        ret.push_back( pts_tgt[0] );
        ret.push_back( pts_other[0] );
        ret.push_back( pts_other[1] );
        ret.push_back( pts_other[2] );
        ret.push_back( pts_tgt[2] );
        ret.push_back( pts_tgt[3] );
    }
    else throw std::invalid_argument("The specified cells are not neighbors");

    return ret;
}



template<typename Mesh, typename Function>
void
run_cuthho_interface(const Mesh& msh, const Function& level_set_function, size_t degree)
{
    using RealType = typename Mesh::coordinate_type;

    /************** DEFINE PROBLEM RHS, SOLUTION AND BCS **************/
#if 0 // test case 1 : a domain decomposition
    auto rhs_fun = [](const typename cuthho_poly_mesh<RealType>::point_type& pt) -> RealType {
        return 2.0 * M_PI * M_PI * std::sin(M_PI*pt.x()) * std::sin(M_PI*pt.y());
    };
    auto sol_fun = [](const typename cuthho_poly_mesh<RealType>::point_type& pt) -> RealType {
        return std::sin(M_PI*pt.x()) * std::sin(M_PI*pt.y());
        //auto v = (pt.y() - 0.5) * 2.0;
        //return pt.y();
    };

    auto sol_grad = [](const typename cuthho_poly_mesh<RealType>::point_type& pt) -> auto {
        Matrix<RealType, 1, 2> ret;

        ret(0) = M_PI * std::cos(M_PI*pt.x()) * std::sin(M_PI*pt.y());
        ret(1) = M_PI * std::sin(M_PI*pt.x()) * std::cos(M_PI*pt.y());

        return ret;
    };

    auto bcs_fun = [&](const typename cuthho_poly_mesh<RealType>::point_type& pt) -> RealType {
        return sol_fun(pt);
    };


    auto dirichlet_jump = [](const typename cuthho_poly_mesh<RealType>::point_type& pt) -> RealType {
        return 0.0;
    };

    auto neumann_jump = [](const typename cuthho_poly_mesh<RealType>::point_type& pt) -> RealType {
        return 0.0;
    };

    
    struct params<RealType> parms;

    parms.kappa_1 = 1.0;
    parms.kappa_2 = 1.0;

#elif 1 // test case 2 : a constrast problem

    struct params<RealType> parms;

    parms.kappa_1 = 1.0;
    parms.kappa_2 = 10000.0;
    
    auto rhs_fun = [](const typename cuthho_poly_mesh<RealType>::point_type& pt) -> RealType {
        return -4.0;
    };
    auto sol_fun = [](const typename cuthho_poly_mesh<RealType>::point_type& pt) -> RealType {
        RealType r2;
        RealType kappa1 = 1.0;
        RealType kappa2 = 10000.0;
        
        r2 = (pt.x() - 0.5) * (pt.x() - 0.5) + (pt.y() - 0.5) * (pt.y() - 0.5);
        if( r2 < 1.0/9 )
            return r2 / kappa1;
        
        else
            return r2 / kappa2 + 1.0/9 * ( 1.0 / kappa1 - 1.0 / kappa2 );
    };

    auto sol_grad = [](const typename cuthho_poly_mesh<RealType>::point_type& pt) -> auto {
        Matrix<RealType, 1, 2> ret;

        RealType kappa1 = 1.0;
        RealType kappa2 = 10000.0;

        RealType r2 = (pt.x() - 0.5) * (pt.x() - 0.5) + (pt.y() - 0.5) * (pt.y() - 0.5);

        if( r2 < 1.0/9 )
        {
            ret(0) = 2 * ( pt.x() - 0.5 ) / kappa1 ;
            ret(1) = 2 * ( pt.y() - 0.5 ) / kappa1 ;
        }
        else
        {
            ret(0) = 2 * ( pt.x() - 0.5 ) / kappa2 ;
            ret(1) = 2 * ( pt.y() - 0.5 ) / kappa2 ;
        }
        
        return ret;
    };

    auto bcs_fun = [&](const typename cuthho_poly_mesh<RealType>::point_type& pt) -> RealType {
        return sol_fun(pt);
    };


    auto dirichlet_jump = [](const typename cuthho_poly_mesh<RealType>::point_type& pt) -> RealType {
        return 0.0;
    };

    auto neumann_jump = [](const typename cuthho_poly_mesh<RealType>::point_type& pt) -> RealType {
        return 0.0;
    };

#elif 0 // test case 3 : a jump problem
    auto rhs_fun = [](const typename cuthho_poly_mesh<RealType>::point_type& pt) -> RealType {
        RealType r2 = (pt.x() - 0.5) * (pt.x() - 0.5) + (pt.y() - 0.5) * (pt.y() - 0.5);
        if(r2 < 1.0/9) {
            return 2.0 * M_PI * M_PI * std::sin(M_PI*pt.x()) * std::sin(M_PI*pt.y());
        }
        else
        {
            return 0.0;
        }
    };
    auto sol_fun = [](const typename cuthho_poly_mesh<RealType>::point_type& pt) -> RealType {
        //return std::sin(M_PI*pt.x()) * std::sin(M_PI*pt.y());
        RealType r2 = (pt.x() - 0.5) * (pt.x() - 0.5) + (pt.y() - 0.5) * (pt.y() - 0.5);
        if(r2 < 1.0/9) {
            return std::sin(M_PI*pt.x()) * std::sin(M_PI*pt.y());
        }
        else
        {
            return exp(pt.x()) * std::cos(pt.y());
        }
    };

    auto sol_grad = [](const typename cuthho_poly_mesh<RealType>::point_type& pt) -> auto {
        Matrix<RealType, 1, 2> ret;

        RealType r2 = (pt.x() - 0.5) * (pt.x() - 0.5) + (pt.y() - 0.5) * (pt.y() - 0.5);
        if(r2 < 1.0/9) {
            ret(0) = M_PI * std::cos(M_PI*pt.x()) * std::sin(M_PI*pt.y());
            ret(1) = M_PI * std::sin(M_PI*pt.x()) * std::cos(M_PI*pt.y());
        }
        else
        {
            ret(0) = exp(pt.x()) * std::cos(pt.y());
            ret(1) = - exp(pt.x()) * std::sin(pt.y());
        }
        
        return ret;
    };

    auto bcs_fun = [&](const typename cuthho_poly_mesh<RealType>::point_type& pt) -> RealType {
        return sol_fun(pt);
    };


    auto dirichlet_jump = [](const typename cuthho_poly_mesh<RealType>::point_type& pt) -> RealType {
        return std::sin(M_PI*pt.x()) * std::sin(M_PI*pt.y()) - exp(pt.x()) * std::cos(pt.y());
    };

    auto neumann_jump = [](const typename cuthho_poly_mesh<RealType>::point_type& pt) -> RealType {
        Matrix<RealType, 1, 2> normal;
        normal(0) = 2*pt.x() - 1.0;
        normal(1) = 2*pt.y() - 1.0;

        normal = normal/normal.norm();

        
        return (M_PI * std::cos(M_PI*pt.x()) * std::sin(M_PI*pt.y()) - exp(pt.x()) * std::cos(pt.y())) * normal(0) + ( M_PI * std::sin(M_PI*pt.x()) * std::cos(M_PI*pt.y()) + exp(pt.x()) * std::sin(pt.y()) ) * normal(1);
    };

    
    struct params<RealType> parms;

    parms.kappa_1 = 1.0;
    parms.kappa_2 = 1.0;


#endif

    timecounter tc;

    /************** ASSEMBLE PROBLEM **************/
    hho_degree_info hdi(degree+1, degree);

    tc.tic();
    auto assembler = make_interface_assembler(msh, hdi);
    for (auto& cl : msh.cells)
    {
        if (location(msh, cl) != element_location::ON_INTERFACE)
        {
            RealType kappa;
            if ( location(msh, cl) == element_location::IN_NEGATIVE_SIDE )
                kappa = parms.kappa_1;
            else
                kappa = parms.kappa_2;

            auto gr = make_hho_gradrec_vector(msh, cl, hdi);
            Matrix<RealType, Dynamic, Dynamic> stab = make_hho_naive_stabilization(msh, cl, hdi);
            Matrix<RealType, Dynamic, Dynamic> lc = kappa * ( gr.second + stab );
            Matrix<RealType, Dynamic, 1> f = Matrix<RealType, Dynamic, 1>::Zero(lc.rows());
            f = make_rhs(msh, cl, hdi.cell_degree(), rhs_fun);
            assembler.assemble(msh, cl, lc, f, bcs_fun);
        }
        else
        {
            auto cbs = cell_basis<cuthho_poly_mesh<RealType>,RealType>::size(hdi.cell_degree());
            auto gbs = vector_cell_basis<cuthho_poly_mesh<RealType>,RealType>::size(hdi.grad_degree());
            auto fbs = face_basis<cuthho_poly_mesh<RealType>,RealType>::size(hdi.face_degree());
            auto fcs = faces(msh, cl);
            auto nfdofs = fcs.size()*fbs;

            
            auto gr_n = make_hho_gradrec_vector_interface(msh, cl, level_set_function, hdi,
                                                          element_location::IN_NEGATIVE_SIDE);
            auto gr_p = make_hho_gradrec_vector_interface(msh, cl, level_set_function, hdi,
                                                          element_location::IN_POSITIVE_SIDE);

            auto stab = make_hho_stabilization_interface(msh, cl, level_set_function, hdi, parms);
                        
            Matrix<RealType, Dynamic, Dynamic> lc = stab + parms.kappa_1 * gr_n.second
                + parms.kappa_2 * gr_p.second;




            Matrix<RealType, Dynamic, 1> f = Matrix<RealType, Dynamic, 1>::Zero( lc.rows() );

            f.head(cbs) = make_rhs(msh, cl, hdi.cell_degree(), element_location::IN_NEGATIVE_SIDE, rhs_fun);
            f.head(cbs) += make_Dirichlet_jump(msh, cl, hdi.cell_degree(), element_location::IN_NEGATIVE_SIDE, level_set_function, dirichlet_jump, parms);
            f.head(cbs) += make_flux_jump(msh, cl, hdi.cell_degree(), element_location::IN_NEGATIVE_SIDE, neumann_jump);

            
            f.block(cbs, 0, cbs, 1) = make_rhs(msh, cl, hdi.cell_degree(), element_location::IN_POSITIVE_SIDE, rhs_fun);
            f.block(cbs, 0, cbs, 1) += make_Dirichlet_jump(msh, cl, hdi.cell_degree(), element_location::IN_POSITIVE_SIDE, level_set_function, dirichlet_jump, parms);
            f.block(cbs, 0, cbs, 1) += make_flux_jump(msh, cl, hdi.cell_degree(), element_location::IN_POSITIVE_SIDE, neumann_jump);


            // rhs term with GR
            vector_cell_basis<cuthho_poly_mesh<RealType>, RealType> gb( msh, cl, hdi.grad_degree() );
            Matrix<RealType, Dynamic, 1> F2 = Matrix<RealType, Dynamic, 1>::Zero( gbs );
            auto iqps = integrate_interface(msh, cl, 2*hdi.grad_degree(),
                                            element_location::IN_NEGATIVE_SIDE);
            for (auto& qp : iqps)
            {
                const auto g_phi    = gb.eval_basis(qp.first);
                const Matrix<RealType,2,1> n      = level_set_function.normal(qp.first);

                F2 += qp.second * dirichlet_jump(qp.first) * g_phi * n;
            }
            f -= F2.transpose() * (parms.kappa_1 * gr_n.first + parms.kappa_2 * gr_p.first);
            
            assembler.assemble_cut(msh, cl, lc, f);
        }
    }

    assembler.finalize();

    //dump_sparse_matrix(assembler.LHS, "matrix.dat");

    tc.toc();
    std::cout << bold << yellow << "Matrix assembly: " << tc << " seconds" << reset << std::endl;

    std::cout << "System unknowns: " << assembler.LHS.rows() << std::endl;
    std::cout << "Cells: " << msh.cells.size() << std::endl;
    std::cout << "Faces: " << msh.faces.size() << std::endl;

    /************** SOLVE **************/
    tc.tic();
#if 0
    SparseLU<SparseMatrix<RealType>>  solver;

    solver.analyzePattern(assembler.LHS);
    solver.factorize(assembler.LHS);
    Matrix<RealType, Dynamic, 1> sol = solver.solve(assembler.RHS);
#endif
//#if 0
    Matrix<RealType, Dynamic, 1> sol = Matrix<RealType, Dynamic, 1>::Zero(assembler.RHS.rows());
    cg_params<RealType> cgp;
    cgp.max_iter = assembler.LHS.rows();
    cgp.histfile = "cuthho_cg_hist.dat";
    cgp.verbose = true;
    cgp.apply_preconditioner = true;
    conjugated_gradient(assembler.LHS, assembler.RHS, sol, cgp);
//#endif
    tc.toc();
    std::cout << bold << yellow << "Linear solver: " << tc << " seconds" << reset << std::endl;

    /************** POSTPROCESS **************/


    postprocess_output<RealType>  postoutput;

    auto uT_gp  = std::make_shared< gnuplot_output_object<RealType> >("interface_uT.dat");
    auto Ru_gp  = std::make_shared< gnuplot_output_object<RealType> >("interface_Ru.dat");
    auto diff_gp  = std::make_shared< gnuplot_output_object<RealType> >("interface_diff.dat");


    std::vector<RealType>   solution_uT, solution_Ru, eigval_data;

    tc.tic();
    RealType    H1_error = 0.0;
    size_t      cell_i   = 0;
    for (auto& cl : msh.cells)
    {
        cell_basis<cuthho_poly_mesh<RealType>, RealType> cb(msh, cl, hdi.cell_degree());
        auto cbs = cb.size();
        auto fcs = faces(msh, cl);
        auto num_faces = fcs.size();
        auto fbs = face_basis<cuthho_poly_mesh<RealType>,RealType>::size(hdi.face_degree());
        
        Matrix<RealType, Dynamic, 1> locdata_n, locdata_p, locdata;
        Matrix<RealType, Dynamic, 1> cell_dofs_n, cell_dofs_p, cell_dofs;

        if (location(msh, cl) == element_location::ON_INTERFACE)
        {
            locdata_n = assembler.take_local_data(msh, cl, sol, bcs_fun, element_location::IN_NEGATIVE_SIDE);
            locdata_p = assembler.take_local_data(msh, cl, sol, bcs_fun, element_location::IN_POSITIVE_SIDE);
            
            Matrix<RealType, Dynamic, 1> locdata_tot = Matrix<RealType, Dynamic, 1>::Zero(2*cbs + 2*num_faces*fbs);
            locdata_tot.head(cbs) = locdata_n.head(cbs);
            locdata_tot.block(cbs, 0 , cbs, 1) = locdata_p.head(cbs);
            locdata_tot.block(2 * cbs, 0, num_faces*fbs, 1) = locdata_n.tail(num_faces*fbs);
            locdata_tot.tail(num_faces*fbs) = locdata_p.tail(num_faces*fbs);
            
            auto gr = make_hho_laplacian_interface(msh, cl, level_set_function, hdi);
            Matrix<RealType, Dynamic, 1> rec_dofs = gr.first * locdata_tot;
            
            // mean value of the reconstruction chosen as the same as the one of the cell component
            RealType mean_cell = 0.0;
            RealType meas_n = 0.0;
            RealType meas_p = 0.0;
            RealType mean_rec = 0.0;
            cell_dofs_n = locdata_n.head(cbs);
            auto qps_n = integrate(msh, cl, 2*hdi.cell_degree(), element_location::IN_NEGATIVE_SIDE);
            for (auto& qp : qps_n)
            {
                auto t_phi = cb.eval_basis( qp.first );
                meas_n += qp.second;
                mean_cell += qp.second * cell_dofs_n.dot( t_phi );
                mean_rec += qp.second * rec_dofs.head( cbs ).dot ( t_phi );
            }
            
            cell_dofs_p = locdata_p.head(cbs);
            auto qps_p = integrate(msh, cl, 2*hdi.cell_degree(), element_location::IN_POSITIVE_SIDE);
            for (auto& qp : qps_p)
            {
                auto t_phi = cb.eval_basis( qp.first );
                meas_p += qp.second;
                mean_cell += qp.second * cell_dofs_p.dot( t_phi );
                mean_rec += qp.second * rec_dofs.tail( cbs ).dot ( t_phi );
            }
            
            mean_cell /= ( meas_n + meas_p );
            mean_rec /= ( meas_n + meas_p );
            
            RealType mean_diff = mean_cell - mean_rec;
            rec_dofs[0] += mean_diff; 
            rec_dofs[cbs] += mean_diff; 
            
            
            for (auto& qp : qps_n)
            {
                /* Compute H1-error */
                auto t_dphi = cb.eval_gradients( qp.first );
                Matrix<RealType, 1, 2> grad = Matrix<RealType, 1, 2>::Zero();

                for (size_t i = 1; i < cbs; i++ )
                    grad += cell_dofs_n(i) * t_dphi.block(i, 0, 1, 2);

                H1_error += qp.second * (sol_grad(qp.first) - grad).dot(sol_grad(qp.first) - grad);

                auto t_phi = cb.eval_basis( qp.first );
                auto v = cell_dofs_n.dot(t_phi);
                uT_gp->add_data(qp.first, v);
                
                RealType Ru_val = rec_dofs.head(cbs).dot( t_phi );
                Ru_gp->add_data( qp.first, Ru_val );
            }
            
            
            for (auto& qp : qps_p)
            {
                /* Compute H1-error */
                auto t_dphi = cb.eval_gradients( qp.first );
                Matrix<RealType, 1, 2> grad = Matrix<RealType, 1, 2>::Zero();

                for (size_t i = 1; i < cbs; i++ )
                    grad += cell_dofs_p(i) * t_dphi.block(i, 0, 1, 2);

                H1_error += qp.second * (sol_grad(qp.first) - grad).dot(sol_grad(qp.first) - grad);

                auto t_phi = cb.eval_basis( qp.first );
                auto v = cell_dofs_p.dot(t_phi);
                uT_gp->add_data(qp.first, v);

                RealType Ru_val = rec_dofs.tail(cbs).dot( t_phi );
                Ru_gp->add_data( qp.first, Ru_val );
            }
        }
        else
        {
            locdata = assembler.take_local_data(msh, cl, sol, bcs_fun, element_location::IN_POSITIVE_SIDE);
            cell_dofs = locdata.head(cbs);

            auto gr = make_hho_laplacian(msh, cl, hdi);
            Matrix<RealType, Dynamic, 1> rec_dofs = gr.first * locdata;
            
            auto qps = integrate(msh, cl, 2*hdi.cell_degree());
            for (auto& qp : qps)
            {
                /* Compute H1-error */
                auto t_dphi = cb.eval_gradients( qp.first );
                Matrix<RealType, 1, 2> grad = Matrix<RealType, 1, 2>::Zero();

                for (size_t i = 1; i < cbs; i++ )
                    grad += cell_dofs(i) * t_dphi.block(i, 0, 1, 2);

                H1_error += qp.second * (sol_grad(qp.first) - grad).dot(sol_grad(qp.first) - grad);

                auto t_phi = cb.eval_basis( qp.first );
                auto v = cell_dofs.dot(t_phi);
                uT_gp->add_data(qp.first, v);

                RealType Ru_val = rec_dofs.dot( t_phi.tail(cbs-1) ) + locdata(0);
                Ru_gp->add_data( qp.first, Ru_val );
            }
        }

        cell_i++;
    }

    std::cout << bold << green << "Energy-norm absolute error:           " << std::sqrt(H1_error) << std::endl;

    postoutput.add_object(uT_gp);
    postoutput.add_object(Ru_gp);
    postoutput.add_object(diff_gp);
    postoutput.write();

    tc.toc();
    std::cout << bold << yellow << "Postprocessing: " << tc << " seconds" << reset << std::endl;

}






template<typename Mesh, typename Function>
void
test_interface_gr(const Mesh& msh, const Function& level_set_function, size_t degree)
{
    using RealType = typename Mesh::coordinate_type;

    auto test_fun_neg = [](const typename cuthho_poly_mesh<RealType>::point_type& pt) -> RealType {
        return pt.x();
    };

    auto test_fun_pos = [](const typename cuthho_poly_mesh<RealType>::point_type& pt) -> RealType {
        return 2*pt.x()*pt.y();
    };

    timecounter tc;

    hho_degree_info hdi(degree+1, degree);

    std::ofstream ofs1("gr1.dat");
    std::ofstream ofs2("gr2.dat");

    for (auto& cl : msh.cells)
    {
        if (location(msh, cl) == element_location::ON_INTERFACE)
        {
            auto gr = make_hho_laplacian_interface(msh, cl, level_set_function, hdi);
            Matrix<RealType, Dynamic, Dynamic> lc = gr.second;

            Matrix<RealType, Dynamic, Dynamic> stab_n = make_hho_cut_stabilization(msh, cl, hdi, element_location::IN_NEGATIVE_SIDE);
            Matrix<RealType, Dynamic, Dynamic> stab_p = make_hho_cut_stabilization(msh, cl, hdi, element_location::IN_POSITIVE_SIDE);

            Matrix<RealType, Dynamic, 1> proj_n = project_function(msh, cl, hdi, element_location::IN_NEGATIVE_SIDE, test_fun_neg);
            Matrix<RealType, Dynamic, 1> proj_p = project_function(msh, cl, hdi, element_location::IN_POSITIVE_SIDE, test_fun_pos);


            auto fcs = faces(msh, cl);
            auto num_faces = fcs.size();
            auto cbs = cell_basis<cuthho_poly_mesh<RealType>,RealType>::size(hdi.cell_degree());
            auto fbs = face_basis<cuthho_poly_mesh<RealType>,RealType>::size(hdi.face_degree());

            Matrix<RealType, Dynamic, 1> proj = Matrix<RealType, Dynamic, 1>::Zero( 2*cbs + 2*num_faces*fbs );

            proj.block(  0, 0, cbs, 1) = proj_n.head(cbs);
            proj.block(cbs, 0, cbs, 1) = proj_p.head(cbs);
            proj.block( 2*cbs, 0, num_faces*fbs, 1) = proj_n.tail(num_faces*fbs);
            proj.block( 2*cbs + num_faces*fbs, 0, num_faces*fbs, 1) = proj_p.tail(num_faces*fbs);

            Matrix<RealType, Dynamic, 1> rec = gr.first * proj;

            cell_basis<cuthho_poly_mesh<RealType>,RealType> rb(msh, cl, hdi.reconstruction_degree());

            auto qps_n = integrate(msh, cl, 5, element_location::IN_NEGATIVE_SIDE);
            for (auto& qp : qps_n)
            {
                auto tp = qp.first;
                auto t_phi = rb.eval_basis( tp );

                RealType val = rec.block(0,0,cbs,1).dot(t_phi);

                ofs1 << tp.x() << " " << tp.y() << " " << val << std::endl;
            }

            auto qps_p = integrate(msh, cl, 5, element_location::IN_POSITIVE_SIDE);
            for (auto& qp : qps_p)
            {
                auto tp = qp.first;
                auto t_phi = rb.eval_basis( tp );

                RealType val = rec.block(cbs,0,cbs,1).dot(t_phi);

                ofs2 << tp.x() << " " << tp.y() << " " << val << std::endl;
            }
        }
    }
    ofs1.close();
    ofs2.close();
}










int main(int argc, char **argv)
{
    using RealType = double;
    
    size_t degree           = 0;
    size_t int_refsteps     = 4;

    bool dump_debug         = false;
    bool solve_interface    = false;
    bool solve_fictdom      = false;
    bool agglomeration      = false;

    mesh_init_params<RealType> mip;
    mip.Nx = 5;
    mip.Ny = 5;

    
    /* k <deg>:     method degree
     * M <num>:     number of cells in x direction
     * N <num>:     number of cells in y direction
     * r <num>:     number of interface refinement steps
     *
     * i:           solve interface problem
     * f:           solve fictitious domain problem
     *
     * D:           use node displacement to solve bad cuts (default)
     * A:           use agglomeration to solve bad cuts 
     *
     * d:           dump debug data
     */

    int ch;
    while ( (ch = getopt(argc, argv, "k:M:N:r:ifDAd")) != -1 )
    {
        switch(ch)
        {
            case 'k':
                degree = atoi(optarg);
                break;

            case 'M':
                mip.Nx = atoi(optarg);
                break;

            case 'N':
                mip.Ny = atoi(optarg);
                break;

            case 'r':
                int_refsteps = atoi(optarg);
                break;

            case 'i':
                solve_interface = true;
                break;

            case 'f':
                solve_fictdom = true;
                break;

            case 'D':
                agglomeration = false;
                break;

            case 'A':
                agglomeration = true;
                break;

            case 'd':
                dump_debug = true;
                break;

            case '?':
            default:
                std::cout << "wrong arguments" << std::endl;
                exit(1);
        }
    }

    argc -= optind;
    argv += optind;


    timecounter tc;

    /************** BUILD MESH **************/
    tc.tic();
    cuthho_poly_mesh<RealType> msh(mip);
    tc.toc();
    std::cout << bold << yellow << "Mesh generation: " << tc << " seconds" << reset << std::endl;
    /************** LEVEL SET FUNCTION **************/
    RealType radius = 1.0/3.0;
    auto level_set_function = circle_level_set<RealType>(radius, 0.5, 0.5);
    // auto level_set_function = line_level_set<RealType>(1.2);
    // auto level_set_function = carre_level_set<RealType>(1.0, 0.0, 0.0, 1.0);
    /************** DO cutHHO MESH PROCESSING **************/

    tc.tic();
    detect_node_position(msh, level_set_function);
    detect_cut_faces(msh, level_set_function);

    if (agglomeration)
    {
        detect_cut_cells(msh, level_set_function);
        detect_cell_agglo_set(msh, level_set_function);
        make_neighbors_info(msh);
    }
    else
    {
        move_nodes(msh, level_set_function);
        detect_cut_faces(msh, level_set_function); //do it again to update intersection points
        detect_cut_cells(msh, level_set_function);
    }

    refine_interface(msh, level_set_function, int_refsteps);
    
    tc.toc();
    std::cout << bold << yellow << "cutHHO-specific mesh preprocessing: " << tc << " seconds" << reset << std::endl;

    if (dump_debug)
    {
        dump_mesh(msh);
        test_triangulation(msh);
        output_mesh_info(msh, level_set_function);
    }

    if (solve_interface)
        run_cuthho_interface(msh, level_set_function, degree);
    
    if (solve_fictdom)
        run_cuthho_fictdom(msh, level_set_function, degree);


#if 0









    auto intfunc = [](const point<RealType,2>& pt) -> RealType {
        return 1;
    };
    auto ints = test_integration(msh, intfunc, level_set_function);


    auto expval = radius*radius*M_PI;
    std::cout << "Integral relative error: " << 100*std::abs(ints.first-expval)/expval << "%" <<std::endl;
    expval = 2*M_PI*radius;
    std::cout << "Integral relative error: " << 100*std::abs(ints.second-expval)/expval << "%" <<std::endl;



    hho_degree_info hdi(degree+1, degree);

    size_t cell_i = 0;
    for (auto& cl : msh.cells)
    {
        if (!is_cut(msh, cl))
        {
            cell_i++;
            continue;
        }

        std::cout << red << bold << " --- .oO CELL " << cell_i << " BEGIN Oo. ---\x1b[0m" << reset << std::endl;

        std::cout << bold << "NEGATIVE SIDE" << reset << std::endl;
        auto gr1 = make_hho_laplacian(msh, cl, level_set_function, hdi, element_location::IN_NEGATIVE_SIDE);
        std::cout << yellow << gr1.first << nocolor << std::endl << std::endl;

        std::cout << bold << "POSITIVE SIDE" << reset << std::endl;
        auto gr2 = make_hho_laplacian(msh, cl, level_set_function, hdi, element_location::IN_POSITIVE_SIDE);
        std::cout << yellow << gr2.first << nocolor << std::endl << std::endl;

        std::cout << bold << "WHOLE" << reset << std::endl;
        auto gr3 = make_hho_laplacian(msh, cl, hdi);
        std::cout << yellow << gr3.first << nocolor << std::endl << std::endl;

        cell_basis<cuthho_quad_mesh<RealType>, RealType> cb(msh, cl, hdi.cell_degree());
        auto bar = barycenter(msh, cl);

        Matrix<RealType, Dynamic, Dynamic> dphi = cb.eval_gradients(bar);
        std::cout << green << dphi.transpose() << nocolor << std::endl;

        cell_i++;
    }



#endif



    return 0;
}
