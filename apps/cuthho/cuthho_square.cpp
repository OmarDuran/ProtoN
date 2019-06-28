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

    auto hT = diameter(msh, cl);

    /* Interface term */
    auto iqps = integrate_interface(msh, cl, 2*recdeg, where);
    for (auto& qp : iqps)
    {
        auto phi    = cb.eval_basis(qp.first);
        auto dphi   = cb.eval_gradients(qp.first);
        Matrix<T,2,1> n      = level_set_function.normal(qp.first);

        //if (where == element_location::IN_POSITIVE_SIDE)
        //    n = -n;

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

    auto hT = diameter(msh, cl);

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

        auto hT = diameter(msh, cl);

        /* Interface term */
        auto iqps = integrate_interface(msh, cl, 2*recdeg, where);
        for (auto& qp : iqps)
        {
            auto phi    = cb.eval_basis(qp.first);
            auto dphi   = cb.eval_gradients(qp.first);
            Matrix<T,2,1> n      = level_set_function.normal(qp.first);

            //if (where == element_location::IN_POSITIVE_SIDE)
            //    n = -n;

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





template<typename T, size_t ET>
Matrix<typename cuthho_mesh<T, ET>::coordinate_type, Dynamic, Dynamic>
make_hho_cut_stabilization(const cuthho_mesh<T, ET>& msh,
                           const typename cuthho_mesh<T, ET>::cell_type& cl,
                           const hho_degree_info& di, element_location where)
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

    auto hT = diameter(msh, cl);

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

    return data;
}

template<typename T, size_t ET, typename F1, typename F2, typename F3>
Matrix<typename cuthho_mesh<T, ET>::coordinate_type, Dynamic, 1>
make_rhs(const cuthho_mesh<T, ET>& msh, const typename cuthho_mesh<T, ET>::cell_type& cl,
         size_t degree, const F1& f, const element_location where, const F2& level_set_function, const F3& bcs)
{
    if ( location(msh, cl) == where )
        return make_rhs(msh, cl, degree, f);
    else if ( location(msh, cl) == element_location::ON_INTERFACE )
    {
        cell_basis<cuthho_mesh<T, ET>,T> cb(msh, cl, degree);
        auto cbs = cb.size();

        auto hT = diameter(msh, cl);

        Matrix<T, Dynamic, 1> ret = Matrix<T, Dynamic, 1>::Zero(cbs);

        auto qps = integrate(msh, cl, 2*degree, where);
        for (auto& qp : qps)
        {
            auto phi = cb.eval_basis(qp.first);
            ret += qp.second * phi * f(qp.first);
        }


        auto qpsi = integrate_interface(msh, cl, degree, where);
        for (auto& qp : qpsi)
        {
            auto phi = cb.eval_basis(qp.first);
            auto dphi = cb.eval_gradients(qp.first);
            auto n = level_set_function.normal(qp.first);

            ret += qp.second * bcs(qp.first) * ( phi * cell_eta(msh, cl)/hT - dphi*n);
        }


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
                   const F2& dir_jump)
{
    cell_basis<cuthho_mesh<T, ET>,T> cb(msh, cl, degree);
    auto cbs = cb.size();
    Matrix<T, Dynamic, 1> ret = Matrix<T, Dynamic, 1>::Zero(cbs);

    if( location(msh, cl) != element_location::ON_INTERFACE )
        return ret;

    auto hT = diameter(msh, cl);
    
    if(where == element_location::IN_NEGATIVE_SIDE) {
        auto qpsi = integrate_interface(msh, cl, 2*degree, element_location::IN_NEGATIVE_SIDE );
	    
        for (auto& qp : qpsi)
        {
            auto phi = cb.eval_basis(qp.first);
            auto dphi = cb.eval_gradients(qp.first);
            auto n = level_set_function.normal(qp.first);

            ret += qp.second * dir_jump(qp.first) * ( phi * cell_eta(msh, cl)/hT - dphi*n);
        }
    }
    else if(where == element_location::IN_POSITIVE_SIDE) {
        auto qpsi = integrate_interface(msh, cl, 2*degree, element_location::IN_NEGATIVE_SIDE );
            
        for (auto& qp : qpsi)
        {
            auto phi = cb.eval_basis(qp.first);
            ret -= qp.second * dir_jump(qp.first) * phi * cell_eta(msh, cl)/hT;
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
    silo.create("cuthho_fictdom.silo");
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
    auto rhs_fun = [](const typename cuthho_poly_mesh<RealType>::point_type& pt) -> RealType {
        return 2.0 * M_PI * M_PI * std::sin(M_PI*pt.x()) * std::sin(M_PI*pt.y());
    };

    auto sol_fun = [](const typename cuthho_poly_mesh<RealType>::point_type& pt) -> RealType {
        return std::sin(M_PI*pt.x()) * std::sin(M_PI*pt.y());
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
    auto assembler = make_assembler(msh, hdi);
    for (auto& cl : msh.cells)
    {
        if ( false && !cl.user_data.distorted && location(msh, cl) != element_location::ON_INTERFACE )
        {
            Matrix<RealType, Dynamic, 1> f = Matrix<RealType, Dynamic, 1>::Zero(lc_template.rows());
            f = make_rhs(msh, cl, hdi.cell_degree(), rhs_fun, where, level_set_function, bcs_fun);
            assembler.assemble(msh, cl, lc_template, f, bcs_fun);
        }
        else
        {
            auto gr = make_hho_laplacian(msh, cl, level_set_function, hdi, where);
            Matrix<RealType, Dynamic, Dynamic> stab = make_hho_cut_stabilization(msh, cl, hdi, where);
            Matrix<RealType, Dynamic, Dynamic> lc = gr.second + stab;
            Matrix<RealType, Dynamic, 1> f = Matrix<RealType, Dynamic, 1>::Zero(lc.rows());
            f = make_rhs(msh, cl, hdi.cell_degree(), rhs_fun, where, level_set_function, bcs_fun);
            assembler.assemble(msh, cl, lc, f, bcs_fun);
        }
    }

    assembler.finalize();

    tc.toc();
    std::cout << bold << yellow << "Matrix assembly: " << tc << " seconds" << reset << std::endl;

    std::cout << "System unknowns: " << assembler.LHS.rows() << std::endl;
    std::cout << "Cells: " << msh.cells.size() << std::endl;
    std::cout << "Faces: " << msh.faces.size() << std::endl;


    /************** SOLVE **************/
    tc.tic();
//#if 0
    SparseLU<SparseMatrix<RealType>>  solver;

    solver.analyzePattern(assembler.LHS);
    solver.factorize(assembler.LHS);
    Matrix<RealType, Dynamic, 1> sol = solver.solve(assembler.RHS);
//#endif
#if 0
    Matrix<RealType, Dynamic, 1> sol = Matrix<RealType, Dynamic, 1>::Zero(assembler.RHS.rows());
    cg_params<RealType> cgp;
    cgp.max_iter = assembler.LHS.rows();
    cgp.histfile = "cuthho_cg_hist.dat";
    cgp.verbose = true;
    cgp.apply_preconditioner = true;
    conjugated_gradient(assembler.LHS, assembler.RHS, sol, cgp);
#endif
    tc.toc();
    std::cout << bold << yellow << "Linear solver: " << tc << " seconds" << reset << std::endl;

    /************** POSTPROCESS **************/



    postprocess_output<RealType>  postoutput;

    auto uT_gp  = std::make_shared< gnuplot_output_object<RealType> >("fictdom_uT.dat");
    auto Ru_gp  = std::make_shared< gnuplot_output_object<RealType> >("fictdom_Ru.dat");
    auto int_gp  = std::make_shared< gnuplot_output_object<RealType> >("ficdom_int.dat");
    auto diff_gp  = std::make_shared< gnuplot_output_object<RealType> >("fictdom_diff.dat");


    std::vector<RealType>   solution_uT, solution_Ru, eigval_data;

    tc.tic();
    RealType    H1_error = 0.0;
    RealType    H1_sol_norm = 0.0;
    size_t      cell_i   = 0;
    for (auto& cl : msh.cells)
    {
        bool hide_fict_dom = true; // hide the fictitious domain in the gnuplot outputs
        if (hide_fict_dom && location(msh,cl) == element_location::IN_POSITIVE_SIDE)
            continue;
        
        cell_basis<cuthho_poly_mesh<RealType>, RealType> cb(msh, cl, hdi.cell_degree());
        auto cbs = cb.size();
        
        cell_basis<cuthho_poly_mesh<RealType>, RealType> rb(msh, cl, hdi.reconstruction_degree());
        auto rbs = rb.size();
        
        auto gr = make_hho_laplacian(msh, cl, level_set_function, hdi, where);
        Matrix<RealType, Dynamic, 1> locdata = assembler.take_local_data(msh, cl, sol, sol_fun);
        Matrix<RealType, Dynamic, 1> cell_dofs = locdata.head(cbs);
        Matrix<RealType, Dynamic, 1> rec_dofs = gr.first * locdata;
        
        auto bar = barycenter(msh, cl, element_location::IN_NEGATIVE_SIDE);
        
        Matrix<RealType, Dynamic, 1> c_phi = cb.eval_basis(bar);
        auto c_val = cell_dofs.dot( c_phi );
        solution_uT.push_back(c_val);
        
        Matrix<RealType, Dynamic, 1> r_phi = rb.eval_basis(bar);
        RealType r_val;
        
        if ( is_cut(msh, cl) )
            r_val = rec_dofs.head(rbs).dot( r_phi );
        else
            r_val = rec_dofs.dot( r_phi.tail(rbs-1) ) + locdata(0);
        
        solution_Ru.push_back(r_val);
        
        /*
        if ( location(msh, cl) != element_location::IN_POSITIVE_SIDE )
        {
            Matrix<RealType, Dynamic, 1> proj = project_function(msh, cl, hdi, element_location::IN_NEGATIVE_SIDE, sol_fun);

            Matrix<RealType, Dynamic, 1> comp = Matrix<RealType, Dynamic, 1>::Zero(rbs);

            if ( is_cut(msh, cl) )
            {
                comp = rec_dofs.transpose().head(rbs);
                std::cout << green << "CUT CELL" << std::endl;
            }
            else
            {
                comp.tail(rbs-1) = rec_dofs.transpose();
                comp(0) = locdata(0);
                std::cout << green << "REGULAR CELL" << std::endl;
            }

            Matrix<RealType, Dynamic, 1> abserr = Matrix<RealType, Dynamic, 1>::Zero(rbs);

            for (size_t i = 0; i < rbs; i++)
                abserr(i) = 100*std::abs(proj.transpose().head(cbs)(i) - comp(i))/std::abs(proj.transpose().head(cbs)(i));

            std::cout << magenta << proj.transpose().head(cbs) << nocolor << std::endl;
            std::cout << red << comp.transpose() << nocolor << std::endl;
            std::cout << yellow << abserr.transpose() << nocolor << std::endl;
        }
        */

        //auto tps = make_test_points(msh, cl, level_set_function, element_location::IN_NEGATIVE_SIDE);

        
        auto qps = integrate(msh, cl, 5, element_location::IN_NEGATIVE_SIDE);
        if ( !hide_fict_dom ) qps = integrate(msh, cl, 5/*, element_location::IN_NEGATIVE_SIDE*/);
        
        for (auto& qp : qps)
        {
            auto tp = qp.first;
            auto t_phi = rb.eval_basis( tp );

            uT_gp->add_data( tp, cell_dofs.dot(t_phi) );

            RealType Ru_val;

            if ( is_cut(msh, cl) )
                Ru_val = rec_dofs.head(rbs).dot( t_phi );
            else
                Ru_val = rec_dofs.dot( t_phi.tail(rbs-1) ) + locdata(0);

            Ru_gp->add_data( tp, Ru_val );

            diff_gp->add_data( tp, std::abs(Ru_val - sol_fun(tp))*100.0/sol_fun(tp) );
        }

        if ( location(msh, cl) == element_location::IN_NEGATIVE_SIDE ||
             location(msh, cl) == element_location::ON_INTERFACE )
        {
            Matrix<RealType, 1, 2> real_grad_int = Matrix<RealType, 1, 2>::Zero();
            Matrix<RealType, 1, 2> comp_grad_int = Matrix<RealType, 1, 2>::Zero();
            auto qps = integrate(msh, cl, 2*hdi.cell_degree(), element_location::IN_NEGATIVE_SIDE);
            for (auto& qp : qps)
            {
                /* Compute H1-error */
                auto t_dphi = rb.eval_gradients( qp.first );
                Matrix<RealType, 1, 2> grad = Matrix<RealType, 1, 2>::Zero();

                for (size_t i = 1; i < rbs; i++ )
                    grad += cell_dofs(i) * t_dphi.block(i, 0, 1, 2);

                H1_error += qp.second * (sol_grad(qp.first) - grad).dot(sol_grad(qp.first) - grad);

                int_gp->add_data( qp.first, 1.0 );
            }
        }


         Matrix<RealType, Dynamic, 1> eigs = check_eigs(msh, cl, level_set_function, hdi, where);
         RealType min_eig = eigs(0);
         for (size_t i = 1; i < eigs.size(); i++)
            min_eig = std::min(min_eig, eigs(i));

        eigval_data.push_back(min_eig);


        cell_i++;
    }

    std::cout << bold << green << "Energy-norm absolute error:           " << std::sqrt(H1_error) << std::endl;

    postoutput.add_object(uT_gp);
    postoutput.add_object(Ru_gp);
    postoutput.add_object(diff_gp);
    postoutput.add_object(int_gp);
    postoutput.write();


    silo.add_variable("mesh", "min_eig", eigval_data.data(), eigval_data.size(), zonal_variable_t);
    silo.add_variable("mesh", "uT", solution_uT.data(), solution_uT.size(), zonal_variable_t);
    silo.add_variable("mesh", "Ru", solution_Ru.data(), solution_Ru.size(), zonal_variable_t);

    tc.toc();
    std::cout << bold << yellow << "Postprocessing: " << tc << " seconds" << reset << std::endl;

}










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

        //for (auto& am : asm_map)
        //    std::cout << am << " ";
        //std::cout << std::endl;

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

#elif 0 // test case 2 : a constrast problem

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

#elif 1 // test case 3 : a jump problem
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

            auto gr = make_hho_laplacian(msh, cl, hdi);
            Matrix<RealType, Dynamic, Dynamic> stab = make_hho_naive_stabilization(msh, cl, hdi);
            Matrix<RealType, Dynamic, Dynamic> lc = kappa * ( gr.second + stab );
            Matrix<RealType, Dynamic, 1> f = Matrix<RealType, Dynamic, 1>::Zero(lc.rows());
            f = make_rhs(msh, cl, hdi.cell_degree(), rhs_fun);
            assembler.assemble(msh, cl, lc, f, bcs_fun);
        }
        else
        {
            auto cbs = cell_basis<cuthho_poly_mesh<RealType>,RealType>::size(hdi.cell_degree());
            auto fbs = face_basis<cuthho_poly_mesh<RealType>,RealType>::size(hdi.face_degree());
            auto fcs = faces(msh, cl);
            auto nfdofs = fcs.size()*fbs;

            auto gr = make_hho_laplacian_interface(msh, cl, level_set_function, hdi, parms);
            Matrix<RealType, Dynamic, Dynamic> lc = gr.second;

            Matrix<RealType, Dynamic, Dynamic> stab_n = parms.kappa_1 * make_hho_cut_stabilization(msh, cl, hdi, element_location::IN_NEGATIVE_SIDE);
            Matrix<RealType, Dynamic, Dynamic> stab_p = parms.kappa_2 * make_hho_cut_stabilization(msh, cl, hdi, element_location::IN_POSITIVE_SIDE);

            lc.block(    0,     0,    cbs,    cbs) += stab_n.block(  0,   0,    cbs,    cbs);
            lc.block(    0, 2*cbs,    cbs, nfdofs) += stab_n.block(  0, cbs,    cbs, nfdofs);
            lc.block(2*cbs,     0, nfdofs,    cbs) += stab_n.block(cbs,   0, nfdofs,    cbs);
            lc.block(2*cbs, 2*cbs, nfdofs, nfdofs) += stab_n.block(cbs, cbs, nfdofs, nfdofs);

            lc.block(         cbs,          cbs,    cbs,    cbs) += stab_p.block(  0,   0,    cbs,    cbs);
            lc.block(         cbs, 2*cbs+nfdofs,    cbs, nfdofs) += stab_p.block(  0, cbs,    cbs, nfdofs);
            lc.block(2*cbs+nfdofs,          cbs, nfdofs,    cbs) += stab_p.block(cbs,   0, nfdofs,    cbs);
            lc.block(2*cbs+nfdofs, 2*cbs+nfdofs, nfdofs, nfdofs) += stab_p.block(cbs, cbs, nfdofs, nfdofs);



            Matrix<RealType, Dynamic, 1> f = Matrix<RealType, Dynamic, 1>::Zero(2*cbs);

            f.head(cbs) = make_rhs(msh, cl, hdi.cell_degree(), element_location::IN_NEGATIVE_SIDE, rhs_fun);
            f.head(cbs) += parms.kappa_1 * make_Dirichlet_jump(msh, cl, hdi.cell_degree(), element_location::IN_NEGATIVE_SIDE, level_set_function, dirichlet_jump);
            f.head(cbs) += make_flux_jump(msh, cl, hdi.cell_degree(), element_location::IN_NEGATIVE_SIDE, neumann_jump);

            f.tail(cbs) = make_rhs(msh, cl, hdi.cell_degree(), element_location::IN_POSITIVE_SIDE, rhs_fun);
            f.tail(cbs) += parms.kappa_1 * make_Dirichlet_jump(msh, cl, hdi.cell_degree(), element_location::IN_POSITIVE_SIDE, level_set_function, dirichlet_jump);
            f.tail(cbs) += make_flux_jump(msh, cl, hdi.cell_degree(), element_location::IN_POSITIVE_SIDE, neumann_jump);

            
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
    //auto level_set_function = line_level_set<RealType>(0.5);
    /************** DO cutHHO MESH PROCESSING **************/

    tc.tic();
    detect_node_position(msh, level_set_function);
    detect_cut_faces(msh, level_set_function);

    if (agglomeration)
    {
        detect_cut_cells(msh, level_set_function);
        detect_cell_agglo_set(msh, level_set_function);
        make_neighbors_info_cartesian(msh);
        // make_neighbors_info(msh);
        refine_interface(msh, level_set_function, int_refsteps);
        make_agglomeration(msh, level_set_function);
    }
    else
    {
        move_nodes(msh, level_set_function);
        detect_cut_faces(msh, level_set_function); //do it again to update intersection points
        detect_cut_cells(msh, level_set_function);
        refine_interface(msh, level_set_function, int_refsteps);
    }


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
