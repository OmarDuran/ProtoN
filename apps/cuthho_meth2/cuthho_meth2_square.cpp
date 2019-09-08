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
#include <Spectra/SymEigsSolver.h>
#include <Spectra/MatOp/SparseSymMatProd.h>

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



template<typename Mesh, typename Function>
void
test_quadrature_bis(const Mesh& msh, const Function& level_set_function, size_t degree)
{
    using T = typename Mesh::coordinate_type;

    // x^k
    auto ref_x = [](const typename cuthho_poly_mesh<T>::point_type& pt, const size_t k) -> T {
        return iexp_pow(pt.x(), k);
    };

    // y^k
    auto ref_y = [](const typename cuthho_poly_mesh<T>::point_type& pt, const size_t k) -> T {
        return iexp_pow(pt.y(), k);
    };

    T integral_x = 0.0;
    T integral_y = 0.0;

    auto cbs = cell_basis<Mesh, T>::size(degree);
    for (auto cl : msh.cells)
    {
        // basis functions
        cell_basis<Mesh, T> cb(msh, cl, degree);

        if( location(msh, cl) == element_location::ON_INTERFACE )
        {
            // local mass matrix
            Matrix<T, Dynamic, Dynamic> mass1 = Matrix<T, Dynamic, Dynamic>::Zero(cbs,cbs);
            Matrix<T, Dynamic, Dynamic> mass2 = Matrix<T, Dynamic, Dynamic>::Zero(cbs,cbs);

            // projection : right-hand side
            Matrix<T, Dynamic, 1> RHS_1x = Matrix<T, Dynamic, 1>::Zero(cbs);
            Matrix<T, Dynamic, 1> RHS_2x = Matrix<T, Dynamic, 1>::Zero(cbs);
            Matrix<T, Dynamic, 1> RHS_1y = Matrix<T, Dynamic, 1>::Zero(cbs);
            Matrix<T, Dynamic, 1> RHS_2y = Matrix<T, Dynamic, 1>::Zero(cbs);

            auto qps_n = integrate(msh, cl, 2*degree, element_location::IN_NEGATIVE_SIDE);
            for (auto& qp : qps_n)
            {
                auto phi = cb.eval_basis(qp.first);
                RHS_1x += qp.second * ref_x(qp.first, degree) * phi;
                RHS_1y += qp.second * ref_y(qp.first, degree) * phi;

                mass1 += qp.second * phi * phi.transpose();
            }
            auto qps_p = integrate(msh, cl, 2*degree, element_location::IN_POSITIVE_SIDE);
            for (auto& qp : qps_p)
            {
                auto phi = cb.eval_basis(qp.first);
                RHS_2x += qp.second * ref_x(qp.first, degree) * phi;
                RHS_2y += qp.second * ref_y(qp.first, degree) * phi;

                mass2 += qp.second * phi * phi.transpose();
            }
            // computation of degrees of projection coefficients
            auto M1_ldlt = mass1.ldlt();
            auto M2_ldlt = mass2.ldlt();
            Matrix<T, Dynamic, 1> U_1x = M1_ldlt.solve(RHS_1x);
            Matrix<T, Dynamic, 1> U_1y = M1_ldlt.solve(RHS_1y);
            Matrix<T, Dynamic, 1> U_2x = M2_ldlt.solve(RHS_2x);
            Matrix<T, Dynamic, 1> U_2y = M2_ldlt.solve(RHS_2y);

            // computation of the integral value
            for (auto& qp : qps_n)
            {
                auto t_phi = cb.eval_basis( qp.first );
                integral_x += qp.second * U_1x.dot(t_phi);
                integral_y += qp.second * U_1y.dot(t_phi);
            }
            for (auto& qp : qps_p)
            {
                auto t_phi = cb.eval_basis( qp.first );
                integral_x += qp.second * U_2x.dot(t_phi);
                integral_y += qp.second * U_2y.dot(t_phi);
            }
        }
        else
        {
            // local mass matrix
            Matrix<T, Dynamic, Dynamic> mass = make_mass_matrix(msh, cl, degree);

            // projection : right-hand side
            Matrix<T, Dynamic, 1> RHS_x = Matrix<T, Dynamic, 1>::Zero(cbs);
            Matrix<T, Dynamic, 1> RHS_y = Matrix<T, Dynamic, 1>::Zero(cbs);
            auto qps = integrate(msh, cl, degree);
            for (auto& qp : qps)
            {
                auto phi = cb.eval_basis(qp.first);
                RHS_x += qp.second * ref_x(qp.first, degree) * phi;
                RHS_y += qp.second * ref_y(qp.first, degree) * phi;
            }

            // computation of degrees of projection coefficients
            auto M_ldlt = mass.ldlt();
            Matrix<T, Dynamic, 1> U_x = M_ldlt.solve(RHS_x);
            Matrix<T, Dynamic, 1> U_y = M_ldlt.solve(RHS_y);

            // computation of the integral value
            for (auto& qp : qps)
            {
                auto t_phi = cb.eval_basis( qp.first );
                integral_x += qp.second * U_x.dot(t_phi);
                integral_y += qp.second * U_y.dot(t_phi);
            }
        }
    }

    // final error
    std::cout << "error x^k = " << 1.0/(degree+1) - integral_x << std::endl;
    std::cout << "error y^k = " << 1.0/(degree+1) - integral_y << std::endl;
}




template<typename Mesh, typename Function>
void test_projection(const Mesh& msh, const Function& level_set_function, size_t degree)
{
    using T = typename Mesh::coordinate_type;

    // reference function (to be projected)
    auto ref_fun = [](const typename cuthho_poly_mesh<T>::point_type& pt) -> T {
        return std::sin(M_PI*pt.x()) * std::sin(M_PI*pt.y());
    };

    auto grad_ref_fun = [](const typename cuthho_poly_mesh<T>::point_type& pt) -> auto {
        Matrix<T, 1, 2> ret;

        ret(0) = M_PI * std::cos(M_PI*pt.x()) * std::sin(M_PI*pt.y());
        ret(1) = M_PI * std::sin(M_PI*pt.x()) * std::cos(M_PI*pt.y());

        return ret;
    };

    T H1_error_uncut = 0.0;
    T H1_error_cut = 0.0;
    T L2_error_uncut = 0.0;
    T L2_error_cut = 0.0;
    T interface_L2_error = 0.0;

    size_t proj_degree = degree + 1;
    auto cbs = cell_basis<Mesh, T>::size(proj_degree);
    for (auto cl : msh.cells)
    {
        // basis functions
        cell_basis<Mesh, T> cb(msh, cl, proj_degree);

        // local mass matrix
        Matrix<T, Dynamic, Dynamic> mass = make_mass_matrix(msh, cl, proj_degree);

        // projection : right-hand side
        Matrix<T, Dynamic, 1> RHS = Matrix<T, Dynamic, 1>::Zero(cbs);
        auto qps = integrate(msh, cl, 2*proj_degree);
        for (auto& qp : qps)
        {
            auto phi = cb.eval_basis(qp.first);
            RHS += qp.second * ref_fun(qp.first) * phi;
        }

        // computation of projection coefficients
        auto M_ldlt = mass.ldlt();
        Matrix<T, Dynamic, 1> U = M_ldlt.solve(RHS);

        // computation of H1 and L2 errors
        if( location(msh, cl) == element_location::ON_INTERFACE )
        {
            auto qps_n = integrate(msh, cl, 2*proj_degree, element_location::IN_NEGATIVE_SIDE);
            for (auto& qp : qps_n)
            {
                // H1_error
                auto t_dphi = cb.eval_gradients( qp.first );
                Matrix<T, 1, 2> grad = Matrix<T, 1, 2>::Zero();
                for (size_t i = 1; i < cbs; i++ )
                    grad += U(i) * t_dphi.block(i, 0, 1, 2);

                Matrix<T, Dynamic, 1> delta_grad = grad_ref_fun(qp.first) - grad;
                H1_error_cut += qp.second * delta_grad.dot(delta_grad);

                // L2_error
                auto t_phi = cb.eval_basis( qp.first );
                auto v = U.dot(t_phi);
                L2_error_cut += qp.second * (ref_fun(qp.first) - v) * (ref_fun(qp.first) - v);
            }

            auto qps_p = integrate(msh, cl, 2*proj_degree, element_location::IN_POSITIVE_SIDE);
            for (auto& qp : qps_p)
            {
                // H1_error
                auto t_dphi = cb.eval_gradients( qp.first );
                Matrix<T, 1, 2> grad = Matrix<T, 1, 2>::Zero();
                for (size_t i = 1; i < cbs; i++ )
                    grad += U(i) * t_dphi.block(i, 0, 1, 2);

                Matrix<T, Dynamic, 1> delta_grad = grad_ref_fun(qp.first) - grad;
                H1_error_cut += qp.second * delta_grad.dot(delta_grad);

                // L2_error
                auto t_phi = cb.eval_basis( qp.first );
                auto v = U.dot(t_phi);
                L2_error_cut += qp.second * (ref_fun(qp.first) - v) * (ref_fun(qp.first) - v);
            }

            auto qps_int = integrate_interface(msh, cl, 2*proj_degree,
                                               element_location::IN_NEGATIVE_SIDE);
            for (auto& qp : qps_int)
            {
                // L2_error
                auto t_phi = cb.eval_basis( qp.first );
                auto v = U.dot(t_phi);
                interface_L2_error += qp.second * (ref_fun(qp.first) - v) * (ref_fun(qp.first) - v);
            }
        }
        else
        {
            auto qps = integrate(msh, cl, 2*proj_degree);
            for (auto& qp : qps)
            {
                // H1_error
                auto t_dphi = cb.eval_gradients( qp.first );
                Matrix<T, 1, 2> grad = Matrix<T, 1, 2>::Zero();
                for (size_t i = 1; i < cbs; i++ )
                    grad += U(i) * t_dphi.block(i, 0, 1, 2);

                Matrix<T, Dynamic, 1> delta_grad = grad_ref_fun(qp.first) - grad;
                H1_error_uncut += qp.second * delta_grad.dot(delta_grad);

                // L2_error
                auto t_phi = cb.eval_basis( qp.first );
                auto v = U.dot(t_phi);
                L2_error_uncut += qp.second * (ref_fun(qp.first) - v) * (ref_fun(qp.first) - v);
            }
        }
    }

    // write the results
    std::cout << bold << green << "UNcut cells: " << std::endl;
    std::cout << bold << green << "Energy-norm absolute error:           " << std::sqrt(H1_error_uncut) << std::endl;
    std::cout << bold << green << "L2-norm absolute error:           " << std::sqrt(L2_error_uncut) << std::endl;

    std::cout << bold << green << "CUT cells: " << std::endl;
    std::cout << bold << green << "Energy-norm absolute error:           " << std::sqrt(H1_error_cut) << std::endl;
    std::cout << bold << green << "L2-norm absolute error:           " << std::sqrt(L2_error_cut) << std::endl;
    std::cout << bold << red << "L2-norm absolute error on interface :           " << std::sqrt(interface_L2_error) << std::endl;
}




void tests_stabilization()
{
    using T = double;
    using Mesh = cuthho_poly_mesh<T>;

    // reference function (to be projected)
    auto ref_fun = [](const typename cuthho_poly_mesh<T>::point_type& pt) -> T {
        return std::sin(M_PI*pt.x()) * std::sin(M_PI*pt.y());
    };


    std::vector<size_t> mesh_sizes, pol_orders;

    // meshes
    mesh_sizes.push_back(8);
    mesh_sizes.push_back(16);
    mesh_sizes.push_back(32);
    mesh_sizes.push_back(64);
    mesh_sizes.push_back(128);

    // polynomial orders
    pol_orders.push_back(0);
    pol_orders.push_back(1);
    pol_orders.push_back(2);
    pol_orders.push_back(3);

    for (std::vector<size_t>::iterator it = pol_orders.begin(); it != pol_orders.end(); it++)
    {
        size_t k = *it;

        std::cout << bold << blue << "!!!!!!!! start tests for k = " << k << std::endl;

        for (std::vector<size_t>::iterator it_msh = mesh_sizes.begin();
             it_msh != mesh_sizes.end(); it_msh++)
        {
            // mesh
            size_t N = *it_msh;
            mesh_init_params<T> mip;
            mip.Nx = N;
            mip.Ny = N;
            cuthho_poly_mesh<T> msh(mip);


            T error_T = 0.0;
            T error_F = 0.0;
            T error_stab = 0.0;

            auto cbs = cell_basis<Mesh, T>::size(k+1);
            auto fbs = face_basis<Mesh, T>::size(k);
            for (auto cl : msh.cells)
            {
                T hT = diameter(msh, cl);

                /////////  CELL PROJECTION
                // basis functions
                cell_basis<Mesh, T> cb(msh, cl, k+1);

                // local mass matrix
                Matrix<T, Dynamic, Dynamic> mass_T = make_mass_matrix(msh, cl, k+1);

                // projection : right-hand side
                Matrix<T, Dynamic, 1> RHS_T = Matrix<T, Dynamic, 1>::Zero(cbs);
                auto qps_T = integrate(msh, cl, 2*(k+1));
                for (auto& qp : qps_T)
                {
                    auto phi = cb.eval_basis(qp.first);
                    RHS_T += qp.second * ref_fun(qp.first) * phi;
                }

                // computation of projection coefficients (cell)
                auto M_ldlt = mass_T.ldlt();
                Matrix<T, Dynamic, 1> U_T = M_ldlt.solve(RHS_T);

                // computation of cell projection error
                for (auto& qp : qps_T)
                {
                    auto phi = cb.eval_basis(qp.first);
                    auto delta = ref_fun(qp.first) - phi.dot(U_T);
                    error_T += qp.second * delta * delta;
                }

                // stabilization matrix
                hho_degree_info hdi(k+1, k);
                auto hho_stab = make_hho_naive_stabilization(msh, cl, hdi);

                auto fcs = faces(msh, cl);
                auto num_faces = fcs.size();
                Matrix<T, Dynamic, 1> loc_vect
                    = Matrix<T, Dynamic, 1>::Zero( cbs + num_faces * fbs );
                loc_vect.head(cbs) = U_T;

                ////////// FACE PROJECTION
                for (size_t i = 0; i < fcs.size(); i++)
                {
                    auto fc = fcs[i];
                    face_basis<Mesh, T> fb(msh, fc, k);

                    Matrix<T, Dynamic, Dynamic> mass_F = make_mass_matrix(msh, fc, k);
                    Matrix<T, Dynamic, 1> RHS_F = Matrix<T, Dynamic, 1>::Zero(fbs);
                    auto qps_F = integrate(msh, fc, 2*k);
                    for (auto& qp : qps_F)
                    {
                        auto phi_F = fb.eval_basis(qp.first);
                        RHS_F += qp.second * ref_fun(qp.first) * phi_F;
                    }

                    // computation of projection coefficients (face)
                    auto M_ldlt_F = mass_F.ldlt();
                    Matrix<T, Dynamic, 1> U_F = M_ldlt_F.solve(RHS_F);


                    ///////// Computation of errors
                    // computation of face projection error
                    auto qps_F_bis = integrate(msh, fc, 2*(k+1));
                    for (auto& qp : qps_F_bis)
                    {
                        auto phi_F = fb.eval_basis(qp.first);
                        auto delta = ref_fun(qp.first) - phi_F.dot(U_F);
                        error_F += hT * qp.second * delta * delta;
                    }
                    // computation of the stabilization error

                    loc_vect.block(cbs+i*fbs, 0, fbs, 1) = U_F;
                }
                auto loc_error = (hho_stab * loc_vect).dot(loc_vect);
                if( loc_error < 0)
                {
                    // std::cout << bold << green << "!!!!! loc_error < 0 !!!!! : "
                    //           << loc_error << std::endl;
                    error_stab -= loc_error;
                }
                else
                    error_stab += loc_error;
            }

            std::cout << bold << red
                      << "mesh_size = " << 1.0/N << std::endl;
            std::cout << bold << yellow
                      << "Errors : proj_cells = " << sqrt(error_T) << std::endl;
            std::cout << "         proj_faces = " << sqrt(error_F) << std::endl;
            std::cout << "         stab_error = " << sqrt(error_stab) << std::endl;
        }
    }
}


//////////////////////////   END  TESTS   /////////////////////////////

template<typename T, size_t ET>
T
cell_eta(const cuthho_mesh<T, ET>& msh, const typename cuthho_mesh<T, ET>::cell_type& cl)
{
    return 1.0;
}








////////////////////////  STATIC CONDENSATION  //////////////////////////
template<typename T>
std::pair<   Matrix<T, Dynamic, Dynamic>, Matrix<T, Dynamic, 1>  >
static_condensation_compute(const Matrix<T, Dynamic, Dynamic> lhs, const Matrix<T, Dynamic, 1> rhs,
                            const size_t cell_size, const size_t face_size)
{
    size_t size_tot = cell_size + face_size;
    assert(lhs.cols() == size_tot && lhs.rows() == size_tot);
    assert(rhs.rows() == size_tot || rhs.rows() == cell_size);

    Matrix<T, Dynamic, Dynamic> lhs_sc = Matrix<T, Dynamic, Dynamic>::Zero(face_size, face_size);
    Matrix<T, Dynamic, 1> rhs_sc = Matrix<T, Dynamic, 1>::Zero(face_size);


    // sub--lhs
    Matrix<T, Dynamic, Dynamic> K_TT = lhs.topLeftCorner(cell_size, cell_size);
    Matrix<T, Dynamic, Dynamic> K_TF = lhs.topRightCorner(cell_size, face_size);
    Matrix<T, Dynamic, Dynamic> K_FT = lhs.bottomLeftCorner(face_size, cell_size);
    Matrix<T, Dynamic, Dynamic> K_FF = lhs.bottomRightCorner(face_size, face_size);

    // sub--rhs
    Matrix<T, Dynamic, 1> cell_rhs = Matrix<T, Dynamic, 1>::Zero(cell_size);
    Matrix<T, Dynamic, 1> face_rhs = Matrix<T, Dynamic, 1>::Zero(face_size);
    if(rhs.rows() == cell_size)
        cell_rhs = rhs;
    else
    {
        cell_rhs = rhs.head(cell_size);
        face_rhs = rhs.tail(face_size);
    }

    // static condensation
    auto K_TT_ldlt = K_TT.ldlt();
    Matrix<T, Dynamic, Dynamic> AL = K_TT_ldlt.solve(K_TF);
    Matrix<T, Dynamic, 1> bL = K_TT_ldlt.solve(cell_rhs);

    lhs_sc = K_FF - K_FT * AL;
    rhs_sc = face_rhs - K_FT * bL;

    return std::make_pair(lhs_sc, rhs_sc);
}

template<typename T>
Matrix<T, Dynamic, 1>
static_condensation_recover(const Matrix<T, Dynamic, Dynamic> lhs, const Matrix<T, Dynamic, 1> rhs,
                            const size_t cell_size, const size_t face_size,
                            const Matrix<T, Dynamic, 1> solF)
{
    size_t size_tot = cell_size + face_size;
    assert(lhs.cols() == size_tot && lhs.rows() == size_tot);
    assert(rhs.rows() == size_tot || rhs.rows() == cell_size);
    assert(solF.rows() == face_size);

    // sub--lhs
    Matrix<T, Dynamic, Dynamic> K_TT = lhs.topLeftCorner(cell_size, cell_size);
    Matrix<T, Dynamic, Dynamic> K_TF = lhs.topRightCorner(cell_size, face_size);

    // sub--rhs
    Matrix<T, Dynamic, 1> cell_rhs = Matrix<T, Dynamic, 1>::Zero(cell_size);
    cell_rhs = rhs.head(cell_size);

    // recover cell solution
    Matrix<T, Dynamic, 1> ret = Matrix<T, Dynamic, 1>::Zero(size_tot);

    ret.head(cell_size) = K_TT.ldlt().solve(cell_rhs - K_TF*solF);
    ret.tail(face_size) = solF;

    return ret;
}


///////////////////////////////   TESTS    //////////////////////////////
template<typename T>
class test_info {
public:
    test_info()
        {
            H1 = 0.0;
            L2 = 0.0;
            cond = 0.0;
        }
    T H1; // H1-error
    T L2; // L2-error
    T cond; // condition number
};


/////////////////  END TESTS  //////////////////////////////



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








template<typename Mesh>
class fict_condensed_assembler
{
    using T = typename Mesh::coordinate_type;
    std::vector<size_t>                 compress_table;
    std::vector<size_t>                 expand_table;
    std::vector<size_t>                 compress_cells_table;

    hho_degree_info                     di;

    std::vector< Triplet<T> >           triplets;

    std::vector< Matrix<T, Dynamic, Dynamic> > loc_LHS;
    std::vector< Matrix<T, Dynamic, 1> > loc_RHS;

    element_location loc_zone;

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

    fict_condensed_assembler(const Mesh& msh, hho_degree_info hdi, element_location where)
        : di(hdi), loc_zone(where)
    {
        auto is_removed = [&](const typename Mesh::face_type& fc) -> bool {
            bool is_dirichlet = fc.is_boundary && fc.bndtype == boundary::DIRICHLET;
            auto loc = location(msh,fc);
            bool is_where = (loc == where || loc == element_location::ON_INTERFACE);
            return is_dirichlet || (!is_where);
        };

        auto num_all_faces = msh.faces.size();
        auto num_removed_faces = std::count_if(msh.faces.begin(), msh.faces.end(), is_removed);
        auto num_other_faces = num_all_faces - num_removed_faces;

        compress_table.resize( num_all_faces );
        expand_table.resize( num_other_faces );

        size_t compressed_offset = 0;
        for (size_t i = 0; i < num_all_faces; i++)
        {
            auto fc = msh.faces[i];
            if ( !is_removed(fc) )
            {
                compress_table.at(i) = compressed_offset;
                expand_table.at(compressed_offset) = i;
                compressed_offset++;
            }
        }

        compress_cells_table.resize( msh.cells.size() );
        compressed_offset = 0;
        for (size_t i = 0; i < msh.cells.size(); i++)
        {
            auto cl = msh.cells[i];
            if (location(msh, cl) == where || location(msh, cl) == element_location::ON_INTERFACE)
            {
                compress_cells_table.at(i) = compressed_offset;
                compressed_offset++;
            }
        }
        loc_LHS.resize( compressed_offset );
        loc_RHS.resize( compressed_offset );

        auto facdeg = di.face_degree();
        auto fbs = face_basis<Mesh,T>::size(facdeg);

        auto system_size = fbs * num_other_faces;

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
        if( !(location(msh, cl) == loc_zone ||
              location(msh, cl) == element_location::ON_INTERFACE) )
            return;

        // save local matrices
        size_t cell_offset = compress_cells_table.at( offset(msh, cl) );
        loc_LHS.at( cell_offset ) = lhs;
        loc_RHS.at( cell_offset ) = rhs;

        auto celdeg = di.cell_degree();
        auto facdeg = di.face_degree();

        auto cbs = cell_basis<Mesh,T>::size(celdeg);
        auto fbs = face_basis<Mesh,T>::size(facdeg);

        auto fcs = faces(msh, cl);
        auto num_faces = fcs.size();
        size_t f_dofs = num_faces * fbs;

        // static condensation
        auto mat_sc = static_condensation_compute(lhs, rhs, cbs, f_dofs);
        Matrix<T, Dynamic, Dynamic> lhs_sc = mat_sc.first;
        Matrix<T, Dynamic, 1> rhs_sc = mat_sc.second;

        std::vector<assembly_index> asm_map;
        asm_map.reserve(f_dofs);

        Matrix<T, Dynamic, 1> dirichlet_data = Matrix<T, Dynamic, 1>::Zero(f_dofs);

        for (size_t face_i = 0; face_i < num_faces; face_i++)
        {
            auto fc = fcs[face_i];
            auto face_offset = offset(msh, fc);
            auto face_LHS_offset = compress_table.at(face_offset) * fbs;

            element_location loc_fc = location(msh, fc);
            bool in_dom = (loc_fc == element_location::ON_INTERFACE ||
                           loc_fc == loc_zone);

            bool dirichlet = fc.is_boundary && fc.bndtype == boundary::DIRICHLET
                && in_dom;

            for (size_t i = 0; i < fbs; i++)
                asm_map.push_back( assembly_index(face_LHS_offset+i, !dirichlet) );

            if (dirichlet)
            {
                Matrix<T, Dynamic, Dynamic> mass = make_mass_matrix(msh, fc, facdeg, loc_zone);
                Matrix<T, Dynamic, 1> loc_rhs = make_rhs(msh, fc, facdeg, loc_zone, dirichlet_bf);
                dirichlet_data.block(face_i*fbs, 0, fbs, 1) = mass.ldlt().solve(loc_rhs);
            }
        }

        assert( asm_map.size() == lhs_sc.rows() && asm_map.size() == lhs_sc.cols() );

        for (size_t i = 0; i < lhs_sc.rows(); i++)
        {
            if (!asm_map[i].assemble())
                continue;

            for (size_t j = 0; j < lhs_sc.cols(); j++)
            {
                if ( asm_map[j].assemble() )
                    triplets.push_back( Triplet<T>(asm_map[i], asm_map[j], lhs_sc(i,j)) );
                else
                    RHS(asm_map[i]) -= lhs_sc(i,j)*dirichlet_data(j);
            }
        }

        //RHS
        for (size_t i = 0; i < rhs_sc.rows(); i++)
        {
            if (!asm_map[i].assemble())
                continue;

            RHS(asm_map[i]) += rhs_sc(i);
        }

    } // assemble()

    //// take_local_data
    // loc_mat is the full loc matrix (cell + faces dofs)
    // loc_rhs can be either cell rhs or full rhs
    template<typename Function>
    Matrix<T, Dynamic, 1>
    take_local_data(const Mesh& msh, const typename Mesh::cell_type& cl,
                    const Matrix<T, Dynamic, 1>& solution, const Function& dirichlet_bf,
                    const Matrix<T, Dynamic, Dynamic> loc_mat, const Matrix<T, Dynamic, 1> loc_rhs)
    {
        auto celdeg = di.cell_degree();
        auto facdeg = di.face_degree();

        auto cbs = cell_basis<Mesh,T>::size(celdeg);
        auto fbs = face_basis<Mesh,T>::size(facdeg);

        auto fcs = faces(msh, cl);
        auto num_faces = fcs.size();
        size_t f_dofs = num_faces*fbs;


        Matrix<T, Dynamic, 1> solF = Matrix<T, Dynamic, 1>::Zero(f_dofs);

        for (size_t face_i = 0; face_i < num_faces; face_i++)
        {
            auto fc = fcs[face_i];

            bool dirichlet = fc.is_boundary && fc.bndtype == boundary::DIRICHLET;

            if (dirichlet)
            {
                Matrix<T, Dynamic, Dynamic> mass = make_mass_matrix(msh, fc, facdeg);
                Matrix<T, Dynamic, 1> rhs = make_rhs(msh, fc, facdeg, dirichlet_bf);
                solF.block(face_i*fbs, 0, fbs, 1) = mass.ldlt().solve(rhs);
            }
            else
            {
                auto face_offset = offset(msh, fc);
                auto face_SOL_offset = compress_table.at(face_offset)*fbs;
                solF.block(face_i*fbs, 0, fbs, 1) = solution.block(face_SOL_offset, 0, fbs, 1);
            }
        }

        // Recover the full solution
        return static_condensation_recover(loc_mat, loc_rhs, cbs, f_dofs, solF);
    }


    //// take_local_data
    // uses the local matrices stored previously
    template<typename Function>
    Matrix<T, Dynamic, 1>
    take_local_data(const Mesh& msh, const typename Mesh::cell_type& cl,
                    const Matrix<T, Dynamic, 1>& solution,
                    const Function& dirichlet_bf)
    {
        size_t offset_cl = compress_cells_table.at( offset(msh, cl) );

        return take_local_data(msh, cl, solution, dirichlet_bf, loc_LHS.at(offset_cl),
                               loc_RHS.at(offset_cl));
    }

    void finalize(void)
    {
        LHS.setFromTriplets( triplets.begin(), triplets.end() );
        triplets.clear();
    }
};


template<typename Mesh>
auto make_fict_condensed_assembler(const Mesh& msh, hho_degree_info hdi, element_location where)
{
    return fict_condensed_assembler<Mesh>(msh, hdi, where);
}




template<typename T, size_t ET, typename F1, typename F2, typename F3>
std::pair<   Matrix<T, Dynamic, Dynamic>, Matrix<T, Dynamic, 1>  >
compute_fictdom_contrib(const cuthho_mesh<T, ET>& msh,
                        const typename cuthho_mesh<T, ET>::cell_type& cl,
                        const F1& level_set_function, const F2& bcs_fun,
                        const F3& rhs_fun, const size_t method, const hho_degree_info hdi,
                        const element_location where = element_location::IN_NEGATIVE_SIDE,
                        const params<T>& parms = params<T>())
{

    if( location(msh, cl) != element_location::ON_INTERFACE )
    {
        auto gr = make_hho_gradrec_vector(msh, cl, hdi);
        Matrix<T, Dynamic, Dynamic> stab = make_hho_naive_stabilization(msh, cl, hdi);
        Matrix<T, Dynamic, Dynamic> lc = gr.second + stab;
        Matrix<T, Dynamic, 1> f = make_rhs(msh, cl, hdi.cell_degree(), rhs_fun);
        return std::make_pair(lc, f);
    }
    /////////   METHOD 1 : USE NON CONSISTENT GRADIENT RECONSTRUCTION
    if(method == 1)
    {
        auto gr = make_hho_gradrec_vector(msh, cl, level_set_function, hdi, where, 1);
        Matrix<T, Dynamic, Dynamic> stab = make_hho_cut_stabilization(msh, cl, hdi, where)
            + make_hho_cut_interface_penalty(msh, cl, hdi, cell_eta(msh, cl));
        Matrix<T, Dynamic, Dynamic> lc = gr.second + stab;
        Matrix<T, Dynamic, 1> f = Matrix<T, Dynamic, 1>::Zero(lc.rows());
        f = make_rhs(msh, cl, hdi.cell_degree(), rhs_fun, where, level_set_function, bcs_fun, gr.first);
        return std::make_pair(lc, f);
    }
    /////////    METHOD 2 : USE NITSCHE'S METHOD
    else if(method == 2)
    {
        auto gr = make_hho_gradrec_vector(msh, cl, level_set_function, hdi, where, 2);
        Matrix<T, Dynamic, Dynamic> stab = make_hho_cut_stabilization(msh, cl, hdi, where)
            + make_hho_cut_interface_penalty(msh, cl, hdi, cell_eta(msh, cl));
        Matrix<T, Dynamic, Dynamic> Nitsche = make_Nitsche(msh, cl, level_set_function, hdi);
        Matrix<T, Dynamic, Dynamic> lc = gr.second + stab + Nitsche;
        // Matrix<T, Dynamic, Dynamic> lc = gr.second + stab;
        Matrix<T, Dynamic, 1> f = Matrix<T, Dynamic, 1>::Zero(lc.rows());
        f = make_rhs(msh, cl, hdi.cell_degree(), rhs_fun, where, level_set_function, bcs_fun, gr.first, 2);
        return std::make_pair(lc, f);
    }
}





template<typename Mesh, typename testType>
test_info<typename Mesh::coordinate_type>
run_cuthho_fictdom(const Mesh& msh, size_t degree, testType test_case)
{
    using RealType = typename Mesh::coordinate_type;

    auto level_set_function = test_case.level_set_;

    auto rhs_fun = test_case.rhs_fun;    
    auto sol_fun = test_case.sol_fun;
    auto sol_grad = test_case.sol_grad;
    auto bcs_fun = test_case.bcs_fun;
    
    
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
    
    
    timecounter tc;

    bool sc = true; // static condensation
    size_t method = 1; // 1 : non consistent gradient // 2 : Nitsche

    /************** ASSEMBLE PROBLEM **************/
    hho_degree_info hdi(degree+1, degree);

    element_location where = element_location::IN_NEGATIVE_SIDE;

    /* reconstruction and stabilization template for square cells.
     * BEWARE of the fact that I'm using cell 0 to compute it! */

    tc.tic();
    auto assembler = make_cut_assembler(msh, hdi, where);
    auto assembler_sc = make_fict_condensed_assembler(msh, hdi, where);

    for (auto& cl : msh.cells)
    {
        auto contrib = compute_fictdom_contrib(msh, cl, level_set_function, bcs_fun,
                                               rhs_fun, method, hdi);
        auto lc = contrib.first;
        auto f = contrib.second;


        if( sc )
            assembler_sc.assemble(msh, cl, lc, f, bcs_fun);
        else
            assembler.assemble(msh, cl, lc, f, bcs_fun);
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
#if 0
    SparseLU<SparseMatrix<RealType>>  solver;
    Matrix<RealType, Dynamic, 1> sol;

    if( sc )
    {
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
#if 1
    Matrix<RealType, Dynamic, 1> sol;
    cg_params<RealType> cgp;
    cgp.histfile = "cuthho_cg_hist.dat";
    cgp.verbose = true;
    cgp.apply_preconditioner = true;
    if( sc )
    {
        sol = Matrix<RealType, Dynamic, 1>::Zero(assembler_sc.RHS.rows());
        cgp.max_iter = assembler_sc.LHS.rows();
        conjugated_gradient(assembler_sc.LHS, assembler_sc.RHS, sol, cgp);
    }
    else
    {
        sol = Matrix<RealType, Dynamic, 1>::Zero(assembler.RHS.rows());
        cgp.max_iter = assembler.LHS.rows();
        conjugated_gradient(assembler.LHS, assembler.RHS, sol, cgp);
    }
#endif
    tc.toc();
    std::cout << bold << yellow << "Linear solver: " << tc << " seconds" << reset << std::endl;

    /************** POSTPROCESS **************/



    postprocess_output<RealType>  postoutput;

    auto uT_gp  = std::make_shared< gnuplot_output_object<RealType> >("fictdom_uT.dat");
    auto Ru_gp  = std::make_shared< gnuplot_output_object<RealType> >("fictdom_Ru.dat");
    auto int_gp  = std::make_shared< gnuplot_output_object<RealType> >("ficdom_int.dat");
    auto diff_gp  = std::make_shared< gnuplot_output_object<RealType> >("fictdom_diff.dat");


    std::vector<RealType>   solution_uT, eigval_data;

    tc.tic();
    RealType    H1_error = 0.0;
    RealType    L2_error = 0.0;
    size_t      cell_i   = 0;
    for (auto& cl : msh.cells)
    {
        bool hide_fict_dom = true; // hide the fictitious domain in the gnuplot outputs
        if (hide_fict_dom && location(msh,cl) == element_location::IN_POSITIVE_SIDE)
            continue;
        
        cell_basis<cuthho_poly_mesh<RealType>, RealType> cb(msh, cl, hdi.cell_degree());
        auto cbs = cb.size();

        Matrix<RealType, Dynamic, 1> locdata;
        if( sc )
        {
            locdata = assembler_sc.take_local_data(msh, cl, sol, sol_fun);
        }
        else
            locdata = assembler.take_local_data(msh, cl, sol, sol_fun);
        
        Matrix<RealType, Dynamic, 1> cell_dofs = locdata.head(cbs);
        
        auto bar = barycenter(msh, cl, element_location::IN_NEGATIVE_SIDE);
        
        Matrix<RealType, Dynamic, 1> c_phi = cb.eval_basis(bar);
        auto c_val = cell_dofs.dot( c_phi );
        solution_uT.push_back(c_val);
        

        if ( location(msh, cl) == element_location::IN_NEGATIVE_SIDE ||
             location(msh, cl) == element_location::ON_INTERFACE )
        {
            Matrix<RealType, 1, 2> real_grad_int = Matrix<RealType, 1, 2>::Zero();
            Matrix<RealType, 1, 2> comp_grad_int = Matrix<RealType, 1, 2>::Zero();
            auto qps = integrate(msh, cl, 2*hdi.cell_degree(), element_location::IN_NEGATIVE_SIDE);
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
                L2_error += qp.second * (sol_fun(qp.first) - v) * (sol_fun(qp.first) - v);

                int_gp->add_data( qp.first, 1.0 );
                uT_gp->add_data( qp.first, cell_dofs.dot(t_phi) );
            }
        }

        cell_i++;
    }

    std::cout << bold << green << "Energy-norm absolute error:           " << std::sqrt(H1_error) << std::endl;

    postoutput.add_object(uT_gp);
    postoutput.add_object(Ru_gp);
    postoutput.add_object(diff_gp);
    postoutput.add_object(int_gp);
    postoutput.write();

    silo.add_variable("mesh", "uT", solution_uT.data(), solution_uT.size(), zonal_variable_t);

    test_info<RealType> TI;
    TI.H1 = std::sqrt(H1_error);
    TI.L2 = std::sqrt(L2_error);

    tc.toc();
    std::cout << bold << yellow << "Postprocessing: " << tc << " seconds" << reset << std::endl;

    return TI;
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
                dirichlet_data.block(cbs+face_i*fbs, 0, fbs, 1) = mass.ldlt().solve(rhs);
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
                ret.block(cbs+face_i*fbs, 0, fbs, 1) = mass.ldlt().solve(rhs);
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

////////////////////////////////////////////

template<typename Mesh>
class interface_condensed_assembler
{
    using T = typename Mesh::coordinate_type;
    std::vector<size_t>                 face_table;
    size_t num_all_faces;

    hho_degree_info                     di;

    std::vector< Triplet<T> >           triplets;

    std::vector< Matrix<T, Dynamic, Dynamic> > loc_LHS;
    std::vector< Matrix<T, Dynamic, 1> > loc_RHS;

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

    interface_condensed_assembler(const Mesh& msh, hho_degree_info hdi)
        : di(hdi)
    {
        auto is_dirichlet = [&](const typename Mesh::face_type& fc) -> bool {
            return fc.is_boundary && fc.bndtype == boundary::DIRICHLET;
        };

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

        auto facdeg = di.face_degree();
        auto fbs = face_basis<Mesh,T>::size(facdeg);

        auto system_size = fbs * num_other_faces;

        LHS = SparseMatrix<T>( system_size, system_size );
        RHS = Matrix<T, Dynamic, 1>::Zero( system_size );

        loc_LHS.resize( msh.cells.size() );
        loc_RHS.resize( msh.cells.size() );
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

        // save local matrices
        size_t cell_offset = offset(msh, cl);
        loc_LHS.at( cell_offset ) = lhs;
        loc_RHS.at( cell_offset ) = rhs;

        auto celdeg = di.cell_degree();
        auto facdeg = di.face_degree();

        auto cbs = cell_basis<Mesh,T>::size(celdeg);
        auto fbs = face_basis<Mesh,T>::size(facdeg);

        auto fcs = faces(msh, cl);
        auto num_faces = fcs.size();
        size_t f_dofs = num_faces * fbs;

        // static condensation
        auto mat_sc = static_condensation_compute(lhs, rhs, cbs, f_dofs);
        Matrix<T, Dynamic, Dynamic> lhs_sc = mat_sc.first;
        Matrix<T, Dynamic, 1> rhs_sc = mat_sc.second;

        std::vector<assembly_index> asm_map;
        asm_map.reserve(f_dofs);

        Matrix<T, Dynamic, 1> dirichlet_data = Matrix<T, Dynamic, 1>::Zero(num_faces*fbs);

        for (size_t face_i = 0; face_i < num_faces; face_i++)
        {
            auto fc = fcs[face_i];
            auto face_offset = offset(msh, fc);
            auto face_LHS_offset = face_table.at(face_offset) * fbs;

            bool dirichlet = fc.is_boundary && fc.bndtype == boundary::DIRICHLET;

            for (size_t i = 0; i < fbs; i++)
                asm_map.push_back( assembly_index(face_LHS_offset+i, !dirichlet) );

            if (dirichlet)
            {
                Matrix<T, Dynamic, Dynamic> mass = make_mass_matrix(msh, fc, facdeg);
                Matrix<T, Dynamic, 1> rhs_loc = make_rhs(msh, fc, facdeg, dirichlet_bf);
                dirichlet_data.block(face_i*fbs, 0, fbs, 1) = mass.ldlt().solve(rhs_loc);
            }
        }

        assert( asm_map.size() == lhs_sc.rows() && asm_map.size() == lhs_sc.cols() );

        for (size_t i = 0; i < lhs_sc.rows(); i++)
        {
            if (!asm_map[i].assemble())
                continue;

            for (size_t j = 0; j < lhs_sc.cols(); j++)
            {
                if ( asm_map[j].assemble() )
                    triplets.push_back( Triplet<T>(asm_map[i], asm_map[j], lhs_sc(i,j)) );
                else
                    RHS(asm_map[i]) -= lhs_sc(i,j)*dirichlet_data(j);
            }
        }


        // RHS
        for (size_t i = 0; i < lhs_sc.rows(); i++)
        {
            if (!asm_map[i].assemble())
                continue;

            RHS(asm_map[i]) += rhs_sc(i);
        }
    } // assemble()

    void
    assemble_cut(const Mesh& msh, const typename Mesh::cell_type& cl,
             const Matrix<T, Dynamic, Dynamic>& lhs, const Matrix<T, Dynamic, 1>& rhs)
    {
        if (location(msh, cl) != element_location::ON_INTERFACE)
            throw std::invalid_argument("Cut cell expected.");

        // save local matrices
        size_t cell_offset = offset(msh, cl);
        loc_LHS.at( cell_offset ) = lhs;
        loc_RHS.at( cell_offset ) = rhs;

        auto celdeg = di.cell_degree();
        auto facdeg = di.face_degree();

        auto cbs = cell_basis<Mesh,T>::size(celdeg);
        auto fbs = face_basis<Mesh,T>::size(facdeg);

        auto fcs = faces(msh, cl);
        auto num_faces = fcs.size();

        // static condensation
        auto mat_sc = static_condensation_compute(lhs, rhs, 2*cbs, 2*num_faces*fbs);
        Matrix<T, Dynamic, Dynamic> lhs_sc = mat_sc.first;
        Matrix<T, Dynamic, 1> rhs_sc = mat_sc.second;

        std::vector<assembly_index> asm_map;
        asm_map.reserve( 2*num_faces*fbs );


        for (size_t face_i = 0; face_i < num_faces; face_i++)
        {
            auto fc = fcs[face_i];
            auto face_offset = offset(msh, fc);
            auto face_LHS_offset = face_table.at(face_offset) * fbs;

            bool dirichlet = fc.is_boundary && fc.bndtype == boundary::DIRICHLET;
            if ( dirichlet )
                throw std::invalid_argument("Dirichlet boundary on cut cell not supported.");

            for (size_t i = 0; i < fbs; i++)
                asm_map.push_back( assembly_index(face_LHS_offset+i, true) );
        }

        for (size_t face_i = 0; face_i < num_faces; face_i++)
        {
            auto fc = fcs[face_i];
            auto face_offset = offset(msh, fc);

            auto d = (location(msh, fc) == element_location::ON_INTERFACE) ? fbs : 0;

            auto face_LHS_offset = face_table.at(face_offset) * fbs + d;

            bool dirichlet = fc.is_boundary && fc.bndtype == boundary::DIRICHLET;
            if ( dirichlet )
                throw std::invalid_argument("Dirichlet boundary on cut cell not supported.");

            for (size_t i = 0; i < fbs; i++)
                asm_map.push_back( assembly_index(face_LHS_offset+i, true) );
        }

        assert( asm_map.size() == lhs_sc.rows() && asm_map.size() == lhs_sc.cols() );

        for (size_t i = 0; i < lhs_sc.rows(); i++)
        {
            if (!asm_map[i].assemble())
                continue;

            for (size_t j = 0; j < lhs_sc.cols(); j++)
            {
                if ( asm_map[j].assemble() )
                    triplets.push_back( Triplet<T>(asm_map[i], asm_map[j], lhs_sc(i,j)) );
            }
        }

        size_t face_offset_loc = 0;
        for (size_t face_i = 0; face_i < num_faces; face_i++)
        {
            auto fc = fcs[face_i];
            auto face_offset = offset(msh, fc);

            auto face_LHS_offset = face_table.at(face_offset) * fbs;

            if( location(msh, fc) == element_location::ON_INTERFACE )
            {
                RHS.block(face_LHS_offset , 0 , fbs , 1)
                    += rhs_sc.block(face_i * fbs , 0 , fbs , 1);

                RHS.block(face_LHS_offset + fbs , 0 , fbs , 1)
                    += rhs_sc.block((num_faces + face_i) * fbs , 0 , fbs , 1);
            }
            else if( location(msh, fc) == element_location::IN_NEGATIVE_SIDE )
            {
                RHS.block(face_LHS_offset , 0 , fbs , 1)
                    += rhs_sc.block(face_i * fbs , 0 , fbs , 1);
            }
            else if( location(msh, fc) == element_location::IN_POSITIVE_SIDE )
            {
                RHS.block(face_LHS_offset , 0 , fbs , 1)
                    += rhs_sc.block((num_faces + face_i) * fbs , 0 , fbs , 1);
            }
            else
                throw std::logic_error("shouldn't have arrived here...");
        }
    } // assemble_cut()


    //// take_local_data
    // loc_mat is the full loc matrix (cell + faces dofs)
    // loc_rhs can be either cell rhs or full rhs
    template<typename Function>
    Matrix<T, Dynamic, 1>
    take_local_data(const Mesh& msh, const typename Mesh::cell_type& cl,
                    const Matrix<T, Dynamic, 1>& solution,
                    const Function& dirichlet_bf,
                    const element_location where, const Matrix<T, Dynamic, Dynamic> loc_mat,
                    const Matrix<T, Dynamic, 1> loc_rhs)
    {
        auto celdeg = di.cell_degree();
        auto facdeg = di.face_degree();

        auto cbs = cell_basis<Mesh,T>::size(celdeg);
        auto fbs = face_basis<Mesh,T>::size(facdeg);

        auto fcs = faces(msh, cl);
        auto num_faces = fcs.size();

        Matrix<T, Dynamic, 1> solF;

        if( location(msh, cl) == element_location::ON_INTERFACE )
            solF = Matrix<T, Dynamic, 1>::Zero(2*num_faces*fbs);
        else
            solF = Matrix<T, Dynamic, 1>::Zero(num_faces*fbs);

        for (size_t face_i = 0; face_i < num_faces; face_i++)
        {
            auto fc = fcs[face_i];

            auto face_offset = offset(msh, fc);
            size_t face_SOL_offset = face_table.at(face_offset) * fbs;
            if ( location(msh, fc) == element_location::ON_INTERFACE )
            {
                // we assume that there is not boundary condition on cut cells
                solF.block(face_i*fbs, 0, fbs, 1) = solution.block(face_SOL_offset, 0, fbs, 1);
                solF.block( (num_faces+face_i)*fbs, 0, fbs, 1)
                    = solution.block(face_SOL_offset + fbs, 0, fbs, 1);
                continue;
            }
            // else

            bool dirichlet = fc.is_boundary && fc.bndtype == boundary::DIRICHLET;

            if (dirichlet)  // we assume that a cut cell has no Dirichlet faces
            {
                Matrix<T, Dynamic, Dynamic> mass = make_mass_matrix(msh, fc, facdeg);
                Matrix<T, Dynamic, 1> rhs = make_rhs(msh, fc, facdeg, dirichlet_bf);
                solF.block(face_i*fbs, 0, fbs, 1) = mass.ldlt().solve(rhs);
                continue;
            }

            if( location(msh, cl) == element_location::ON_INTERFACE &&
                location(msh, fc) == element_location::IN_POSITIVE_SIDE )
            {
                solF.block((num_faces+face_i)*fbs, 0, fbs, 1)
                    = solution.block(face_SOL_offset, 0, fbs, 1);
                continue;
            }
            // else
            solF.block(face_i*fbs, 0, fbs, 1) = solution.block(face_SOL_offset, 0, fbs, 1);
        }

        // Recover the full solution
        Matrix<T, Dynamic, 1> ret = Matrix<T, Dynamic, 1>::Zero(cbs + num_faces*fbs);

        if( location(msh, cl) == element_location::ON_INTERFACE )
        {
            if( where == element_location::IN_NEGATIVE_SIDE )
            {
                ret.head(cbs) = static_condensation_recover(loc_mat, loc_rhs, 2*cbs, 2*num_faces*fbs, solF).head(cbs);
                ret.tail(num_faces*fbs) = solF.head(num_faces*fbs);
            }

            if( where == element_location::IN_POSITIVE_SIDE )
            {
                ret.head(cbs) = static_condensation_recover(loc_mat, loc_rhs, 2*cbs, 2*num_faces*fbs, solF).block(cbs, 0, cbs, 1);
                ret.tail(num_faces*fbs) = solF.tail(num_faces*fbs);
            }
        }
        else
        {
            ret.head(cbs) = static_condensation_recover(loc_mat, loc_rhs, cbs, num_faces*fbs, solF).head(cbs);
            ret.tail(num_faces*fbs) = solF;
        }

        return ret;
    }


    //// take_local_data
    // uses the local matrices stored previously
    template<typename Function>
    Matrix<T, Dynamic, 1>
    take_local_data(const Mesh& msh, const typename Mesh::cell_type& cl,
                    const Matrix<T, Dynamic, 1>& solution,
                    const Function& dirichlet_bf, const element_location where)
    {
        size_t offset_cl = offset(msh, cl);
        return take_local_data(msh, cl, solution, dirichlet_bf, where, loc_LHS.at(offset_cl),
                               loc_RHS.at(offset_cl));
    }

    void finalize(void)
    {
        LHS.setFromTriplets( triplets.begin(), triplets.end() );
        triplets.clear();
    }
};


template<typename Mesh>
auto make_interface_condensed_assembler(const Mesh& msh, hho_degree_info hdi)
{
    return interface_condensed_assembler<Mesh>(msh, hdi);
}


//////////////////////////






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



template<typename T, size_t ET, typename F1, typename F2, typename F3, typename F4>
std::pair<   Matrix<T, Dynamic, Dynamic>, Matrix<T, Dynamic, 1>  >
compute_interface_contrib(const cuthho_mesh<T, ET>& msh,
                          const typename cuthho_mesh<T, ET>::cell_type& cl,
                          const F1& level_set_function, const F2& dir_jump,
                          const F3& neumann_jump, const F4& rhs_fun,
                          const size_t method, const hho_degree_info hdi,
                          const params<T>& parms = params<T>())
{
    auto cbs = cell_basis<cuthho_poly_mesh<T>,T>::size(hdi.cell_degree());
    auto gbs = vector_cell_basis<cuthho_poly_mesh<T>,T>::size(hdi.grad_degree());
    auto fbs = face_basis<cuthho_poly_mesh<T>,T>::size(hdi.face_degree());
    auto fcs = faces(msh, cl);
    auto nfdofs = fcs.size()*fbs;

    Matrix<T, Dynamic, Dynamic> lc;
    Matrix<T, Dynamic, 1> f;

    if( location(msh, cl) != element_location::ON_INTERFACE )
    {
        T kappa;
        if ( location(msh, cl) == element_location::IN_NEGATIVE_SIDE )
            kappa = parms.kappa_1;
        else
            kappa = parms.kappa_2;

        auto gr = make_hho_gradrec_vector(msh, cl, hdi);
        Matrix<T, Dynamic, Dynamic> stab = make_hho_naive_stabilization(msh, cl, hdi);
        lc = kappa * ( gr.second + stab );
        f = make_rhs(msh, cl, hdi.cell_degree(), rhs_fun);
        return std::make_pair(lc, f);
    }
    // else

    f = Matrix<T, Dynamic, 1>::Zero( 2*(cbs+nfdofs) );
    f.head(cbs) = make_rhs(msh, cl, hdi.cell_degree(), element_location::IN_NEGATIVE_SIDE, rhs_fun);
    f.block(cbs, 0, cbs, 1) = make_rhs(msh, cl, hdi.cell_degree(), element_location::IN_POSITIVE_SIDE, rhs_fun);

///////////////////////////////////////////////////////////////////////////////////////////
    if (method == 1) // bold G -- bold G + negative Nitsche's terms
    {
        auto gr_n = make_hho_gradrec_vector_interface(msh, cl, level_set_function, hdi,
                                                      element_location::IN_NEGATIVE_SIDE, 1);
        auto gr_p = make_hho_gradrec_vector_interface(msh, cl, level_set_function, hdi,
                                                      element_location::IN_POSITIVE_SIDE, 1);
        auto stab = make_hho_stabilization_interface(msh, cl, level_set_function, hdi, 1, parms);

        // LHS
        lc = stab + parms.kappa_1 * gr_n.second + parms.kappa_2 * gr_p.second;

        // complete RHS
        // neg part
        f.head(cbs) += make_Dirichlet_jump(msh, cl, hdi.cell_degree(), element_location::IN_NEGATIVE_SIDE, level_set_function, dir_jump, 1, parms);
        f.head(cbs) += make_flux_jump(msh, cl, hdi.cell_degree(), element_location::IN_NEGATIVE_SIDE, neumann_jump);

        // pos part
        f.block(cbs, 0, cbs, 1) += make_Dirichlet_jump(msh, cl, hdi.cell_degree(), element_location::IN_POSITIVE_SIDE, level_set_function, dir_jump, 1, parms);

    }
///////////////////////////////////////////////////////////////////////////////////////////
    else if (method == 2) // tilde bold G -- tilde bold G
    {
        auto gr_n = make_hho_gradrec_vector_interface(msh, cl, level_set_function, hdi,
                                                      element_location::IN_NEGATIVE_SIDE, 2);
        auto gr_p = make_hho_gradrec_vector_interface(msh, cl, level_set_function, hdi,
                                                      element_location::IN_POSITIVE_SIDE, 2);
        auto stab = make_hho_stabilization_interface(msh, cl, level_set_function, hdi, 2, parms);

        // LHS
        lc = stab + parms.kappa_1 * gr_n.second + parms.kappa_2 * gr_p.second;

        // complete RHS
        // neg part
        f.head(cbs) += make_Dirichlet_jump(msh, cl, hdi.cell_degree(), element_location::IN_NEGATIVE_SIDE, level_set_function, dir_jump, 2, parms);
        f.head(cbs) += 0.5 * make_flux_jump(msh, cl, hdi.cell_degree(), element_location::IN_NEGATIVE_SIDE, neumann_jump);

        // pos part
        f.block(cbs, 0, cbs, 1) += make_Dirichlet_jump(msh, cl, hdi.cell_degree(), element_location::IN_POSITIVE_SIDE, level_set_function, dir_jump, 2, parms);
        f.block(cbs, 0, cbs, 1) += 0.5 * make_flux_jump(msh, cl, hdi.cell_degree(), element_location::IN_POSITIVE_SIDE, neumann_jump);

        // rhs term with GR
        vector_cell_basis<cuthho_poly_mesh<T>, T> gb( msh, cl, hdi.grad_degree() );
        Matrix<T, Dynamic, 1> F_bis = Matrix<T, Dynamic, 1>::Zero( gbs );
        auto iqps = integrate_interface(msh, cl, 2*hdi.grad_degree(),
                                        element_location::IN_NEGATIVE_SIDE);
        for (auto& qp : iqps)
        {
            const auto g_phi    = gb.eval_basis(qp.first);
            const Matrix<T,2,1> n      = level_set_function.normal(qp.first);

            F_bis += qp.second * dir_jump(qp.first) * g_phi * n;
        }
        f -= 0.5 * F_bis.transpose() * (parms.kappa_1 * gr_n.first + parms.kappa_2 * gr_p.first);
    }
///////////////////////////////////////////////////////////////////////////////////////////
    else if (method == 3) // hat bold G -- bold G
    {
        auto gr_n = make_hho_gradrec_vector_interface(msh, cl, level_set_function, hdi,
                                                      element_location::IN_NEGATIVE_SIDE, 3);
        auto gr_p = make_hho_gradrec_vector_interface(msh, cl, level_set_function, hdi,
                                                      element_location::IN_POSITIVE_SIDE, 1);
        auto stab = make_hho_stabilization_interface(msh, cl, level_set_function, hdi, 3, parms);

        // LHS
        lc = stab + parms.kappa_1 * gr_n.second + parms.kappa_2 * gr_p.second;

        // complete RHS
        // neg part
        f.head(cbs) += make_Dirichlet_jump(msh, cl, hdi.cell_degree(), element_location::IN_NEGATIVE_SIDE, level_set_function, dir_jump, 3, parms);

        // pos part
        f.block(cbs, 0, cbs, 1) += make_Dirichlet_jump(msh, cl, hdi.cell_degree(), element_location::IN_POSITIVE_SIDE, level_set_function, dir_jump, 3, parms);
        f.block(cbs, 0, cbs, 1) += make_flux_jump(msh, cl, hdi.cell_degree(), element_location::IN_POSITIVE_SIDE, neumann_jump);

        // rhs term with GR
        vector_cell_basis<cuthho_poly_mesh<T>, T> gb( msh, cl, hdi.grad_degree() );
        Matrix<T, Dynamic, 1> F_bis = Matrix<T, Dynamic, 1>::Zero( gbs );
        auto iqps = integrate_interface(msh, cl, 2*hdi.grad_degree(),
                                        element_location::IN_NEGATIVE_SIDE);
        for (auto& qp : iqps)
        {
            const auto g_phi    = gb.eval_basis(qp.first);
            const Matrix<T,2,1> n      = level_set_function.normal(qp.first);

            F_bis += qp.second * dir_jump(qp.first) * g_phi * n;
        }
        f -= F_bis.transpose() * (parms.kappa_1 * gr_n.first );

    }
///////////////////////////////////////////////////////////////////////////////////////////
    else if (method == 4) // hat bold G -- hat bold G + positive Nitsche's terms
    {
        auto gr_n = make_hho_gradrec_vector_interface(msh, cl, level_set_function, hdi,
                                                      element_location::IN_NEGATIVE_SIDE, 3);
        auto gr_p = make_hho_gradrec_vector_interface(msh, cl, level_set_function, hdi,
                                                      element_location::IN_POSITIVE_SIDE, 3);
        auto stab = make_hho_stabilization_interface(msh, cl, level_set_function, hdi, 4, parms);

        // LHS
        lc = stab + parms.kappa_1 * gr_n.second + parms.kappa_2 * gr_p.second;

        // complete RHS
        // neg part
        f.head(cbs) += make_Dirichlet_jump(msh, cl, hdi.cell_degree(), element_location::IN_NEGATIVE_SIDE, level_set_function, dir_jump, 4, parms);

        // pos part
        f.block(cbs, 0, cbs, 1) += make_Dirichlet_jump(msh, cl, hdi.cell_degree(), element_location::IN_POSITIVE_SIDE, level_set_function, dir_jump, 4, parms);
        f.block(cbs, 0, cbs, 1) += make_flux_jump(msh, cl, hdi.cell_degree(), element_location::IN_POSITIVE_SIDE, neumann_jump);

        // rhs term with GR
        vector_cell_basis<cuthho_poly_mesh<T>, T> gb( msh, cl, hdi.grad_degree() );
        Matrix<T, Dynamic, 1> F_bis = Matrix<T, Dynamic, 1>::Zero( gbs );
        auto iqps = integrate_interface(msh, cl, 2*hdi.grad_degree(),
                                        element_location::IN_NEGATIVE_SIDE);
        for (auto& qp : iqps)
        {
            const auto g_phi    = gb.eval_basis(qp.first);
            const Matrix<T,2,1> n      = level_set_function.normal(qp.first);

            F_bis += qp.second * dir_jump(qp.first) * g_phi * n;
        }
        f -= F_bis.transpose() * (parms.kappa_1 * gr_n.first + parms.kappa_2 * gr_p.first);
    }
////////////////////////////////////////////////////////////////////////////////////////////

    return std::make_pair(lc, f);
}


/////////////////////  TESTS   ////////////////////////////////

//////  interface_residus
// to test the interface problem
void interface_residus(void)
{
    using T = double;

    ///// TEST CASE
        auto rhs_fun = [](const typename cuthho_poly_mesh<T>::point_type& pt) -> T {
        return 2.0 * M_PI * M_PI * std::sin(M_PI*pt.x()) * std::sin(M_PI*pt.y());
    };
    auto sol_fun = [](const typename cuthho_poly_mesh<T>::point_type& pt) -> T {
        return std::sin(M_PI*pt.x()) * std::sin(M_PI*pt.y());
        //auto v = (pt.y() - 0.5) * 2.0;
        //return pt.y();
    };

    auto sol_grad = [](const typename cuthho_poly_mesh<T>::point_type& pt) -> auto {
        Matrix<T, 1, 2> ret;

        ret(0) = M_PI * std::cos(M_PI*pt.x()) * std::sin(M_PI*pt.y());
        ret(1) = M_PI * std::sin(M_PI*pt.x()) * std::cos(M_PI*pt.y());

        return ret;
    };

    auto bcs_fun = [&](const typename cuthho_poly_mesh<T>::point_type& pt) -> T {
        return sol_fun(pt);
    };


    auto dirichlet_jump = [](const typename cuthho_poly_mesh<T>::point_type& pt) -> T {
        return 0.0;
    };

    auto neumann_jump = [](const typename cuthho_poly_mesh<T>::point_type& pt) -> T {
        return 0.0;
    };


    struct params<T> parms;

    parms.kappa_1 = 1.0;
    parms.kappa_2 = 1.0;


    timecounter tc;

    //////// FIRST STEP : compute the solution on a fitted domain
    tc.tic();
    size_t N = 20;
    size_t degree = 3;
    size_t method = 3;
    mesh_init_params<T> mip;
    mip.Nx = N;
    mip.Ny = N;
    cuthho_poly_mesh<T> msh(mip);
    size_t int_refsteps = 1;
    auto level_set_function1 = square_level_set<T>(1.05, -0.05, -0.05, 1.05);
    detect_node_position(msh, level_set_function1);
    detect_cut_faces(msh, level_set_function1);
    detect_cut_cells(msh, level_set_function1);
    detect_cell_agglo_set(msh, level_set_function1);
    make_neighbors_info_cartesian(msh);
    refine_interface(msh, level_set_function1, int_refsteps);
    make_agglomeration(msh, level_set_function1);


    hho_degree_info hdi(degree+1, degree);
    auto assembler = make_interface_assembler(msh, hdi);
    auto assembler_sc = make_interface_condensed_assembler(msh, hdi);

    bool sc1 = true;   // static condensation for the first stage
    bool sc2 = false;  // static condensation for the other stages

    // compute the matrix
    for (auto& cl : msh.cells)
    {
        auto contrib = compute_interface_contrib(msh, cl, level_set_function1, dirichlet_jump,
                                                 neumann_jump, rhs_fun, method, hdi, parms);
        auto lc = contrib.first;
        auto f = contrib.second;


        if (location(msh, cl) != element_location::ON_INTERFACE)
        {
            if( sc1 )
                assembler_sc.assemble(msh, cl, lc, f, bcs_fun);
            else
                assembler.assemble(msh, cl, lc, f, bcs_fun);
        }
        else
        {
            if( sc1 )
                assembler_sc.assemble_cut(msh, cl, lc, f);
            else
                assembler.assemble_cut(msh, cl, lc, f);
        }
    }

    if( sc1 )
        assembler_sc.finalize();
    else
        assembler.finalize();

#if 1
    SparseLU<SparseMatrix<T>>  solver;
    Matrix<T, Dynamic, 1> sol1;

    if( sc1 )
    {
        solver.analyzePattern(assembler_sc.LHS);
        solver.factorize(assembler_sc.LHS);
        sol1 = solver.solve(assembler_sc.RHS);
    }
    else
    {
        solver.analyzePattern(assembler.LHS);
        solver.factorize(assembler.LHS);
        sol1 = solver.solve(assembler.RHS);
    }
#endif
#if 0
    Matrix<T, Dynamic, 1> sol1;
    cg_params<T> cgp;
    cgp.histfile = "cuthho_cg_hist.dat";
    cgp.verbose = true;
    cgp.apply_preconditioner = true;
    if( sc1 )
    {
        sol1 = Matrix<T, Dynamic, 1>::Zero(assembler_sc.RHS.rows());
        cgp.max_iter = assembler_sc.LHS.rows();
        conjugated_gradient(assembler_sc.LHS, assembler_sc.RHS, sol1, cgp);
    }
    else
    {
        sol1 = Matrix<T, Dynamic, 1>::Zero(assembler.RHS.rows());
        cgp.max_iter = assembler.LHS.rows();
        conjugated_gradient(assembler.LHS, assembler.RHS, sol1, cgp);
    }
#endif


    // if sc1 is used, then decondensate sol1 -> sol1_tot
    Matrix<T, Dynamic, 1> sol1_tot = Matrix<T, Dynamic, 1>::Zero( assembler.RHS.rows() );
    if (sc1)
    {
        auto cbs = cell_basis<cuthho_poly_mesh<T>,T>::size(hdi.cell_degree());
        auto fbs = face_basis<cuthho_poly_mesh<T>,T>::size(hdi.face_degree());

        size_t loc_offset = 0;
        for (auto& cl : msh.cells)
        {
            cell_basis<cuthho_poly_mesh<T>, T> cb(msh, cl, hdi.cell_degree());
            auto fcs = faces(msh, cl);
            auto num_faces = fcs.size();

            Matrix<T, Dynamic, 1> locdata, locdata_n, locdata_p;

            if (location(msh, cl) == element_location::ON_INTERFACE)
            {
                auto contrib = compute_interface_contrib(
                    msh, cl, level_set_function1, dirichlet_jump,
                    neumann_jump, rhs_fun, method, hdi, parms);
                auto lc = contrib.first;
                auto f = contrib.second;

                locdata_n = assembler_sc.take_local_data(msh, cl, sol1, bcs_fun, element_location::IN_NEGATIVE_SIDE, lc, f);
                locdata_p = assembler_sc.take_local_data(msh, cl, sol1, bcs_fun, element_location::IN_POSITIVE_SIDE, lc, f);

                sol1_tot.block(loc_offset * cbs, 0, cbs, 1) = locdata_n.head(cbs);
                loc_offset++;
                sol1_tot.block(loc_offset * cbs, 0, cbs, 1) = locdata_p.head(cbs);
                loc_offset++;
            }
            else
            {
                auto contrib = compute_interface_contrib(
                    msh, cl, level_set_function1, dirichlet_jump,
                    neumann_jump, rhs_fun, method, hdi, parms);
                auto lc = contrib.first;
                auto f = contrib.second;

                locdata = assembler_sc.take_local_data(msh, cl, sol1, bcs_fun, element_location::IN_POSITIVE_SIDE, lc, f);

                sol1_tot.block(loc_offset * cbs, 0, cbs, 1) = locdata.head(cbs);
                loc_offset++;
            }
        }

        size_t FACES_OFFSET = loc_offset * cbs;
        loc_offset = 0;

        for (auto& fc : msh.faces)
        {
            if (fc.is_boundary && fc.bndtype == boundary::DIRICHLET)
                continue;

            if (location(msh, fc) == element_location::ON_INTERFACE)
            {
                sol1_tot.block(FACES_OFFSET + loc_offset*fbs, 0, 2*fbs, 1)
                    = sol1.block(loc_offset*fbs, 0, 2*fbs, 1);

                loc_offset = loc_offset + 2;
            }
            else
            {
                sol1_tot.block(FACES_OFFSET + loc_offset*fbs, 0, fbs, 1)
                    = sol1.block(loc_offset*fbs, 0, fbs, 1);

                loc_offset++;
            }
        }
    }
    else
        sol1_tot = sol1;

    tc.toc();
    std::cout << bold << yellow << "Step One : " << tc << " seconds" << reset << std::endl;
    //////// SECOND STEP : double the unknowns on the cut cells
    ///  No agglomeration at the moment
    tc.tic();

    // new mesh
    cuthho_poly_mesh<T> msh2(mip);
    auto level_set_function2 = square_level_set<T>(0.77, 0.23, 0.23, 0.77);
    detect_node_position(msh2, level_set_function2);
    detect_cut_faces(msh2, level_set_function2);
    detect_cut_cells(msh2, level_set_function2);
    detect_cell_agglo_set(msh2, level_set_function2);
    make_neighbors_info_cartesian(msh2);
    refine_interface(msh2, level_set_function2, int_refsteps);
    // make_agglomeration(msh2, level_set_function2);

    // new solution -> transfert sol1_tot in sol2_tot
    auto assembler2 = make_interface_assembler(msh2, hdi);
    auto assembler2_sc = make_interface_condensed_assembler(msh2, hdi);

    Matrix<T, Dynamic, 1> sol2_tot = Matrix<T, Dynamic, 1>::Zero( assembler2.RHS.rows() );

    auto celdeg = hdi.cell_degree();
    auto facdeg = hdi.face_degree();
    auto cbs = cell_basis<cuthho_poly_mesh<T>,T>::size(celdeg);
    auto fbs = face_basis<cuthho_poly_mesh<T>,T>::size(facdeg);

    size_t offset1 = 0;
    size_t offset2 = 0;
    for (auto& cl : msh2.cells)
    {

        if (location(msh2, cl) == element_location::ON_INTERFACE)
        {
            sol2_tot.block(offset2*cbs, 0, cbs, 1) = sol1_tot.block(offset1*cbs, 0, cbs, 1);
            sol2_tot.block((offset2+1)*cbs, 0, cbs, 1) = sol1_tot.block(offset1*cbs, 0, cbs, 1);

            offset2 = offset2 + 2;
        }
        else
        {
            sol2_tot.block(offset2*cbs, 0, cbs, 1) = sol1_tot.block(offset1*cbs, 0, cbs, 1);

            offset2++;
        }
        offset1++;
    }

    size_t FACE_OFFSET1 = offset1 * cbs;
    size_t FACE_OFFSET2 = offset2 * cbs;
    offset1 = 0;
    offset2 = 0;
    for (auto& fc : msh2.faces)
    {
        if (fc.is_boundary && fc.bndtype == boundary::DIRICHLET)
            continue;

        if (location(msh2, fc) == element_location::ON_INTERFACE)
        {
            sol2_tot.block(FACE_OFFSET2 + offset2*fbs, 0, fbs, 1)
                = sol1_tot.block(FACE_OFFSET1 + offset1*fbs, 0, fbs, 1);
            sol2_tot.block(FACE_OFFSET2 + (offset2+1)*fbs, 0, fbs, 1)
                = sol1_tot.block(FACE_OFFSET1 + offset1*fbs, 0, fbs, 1);

            offset2 = offset2 + 2;
        }
        else
        {
            sol2_tot.block(FACE_OFFSET2 + offset2*fbs, 0, fbs, 1)
                = sol1_tot.block(FACE_OFFSET1 + offset1*fbs, 0, fbs, 1);

            offset2++;
        }
        offset1++;
    }


    Matrix<T, Dynamic, 1> sol2;
    if (sc2)
    {
        sol2 = Matrix<T, Dynamic, 1>::Zero(assembler2_sc.RHS.rows());

        throw std::logic_error("sc2 = true not available yet");
    }
    else
        sol2 = sol2_tot;

    tc.toc();
    std::cout << bold << yellow << "Step Two : " << tc << " seconds" << reset << std::endl;

    //////// THIRD STEP : compute the matrix associated to the cut mesh
    tc.tic();
    for (auto& cl : msh2.cells)
    {
        auto contrib = compute_interface_contrib(msh2, cl, level_set_function2, dirichlet_jump,
                                                 neumann_jump, rhs_fun, method, hdi, parms);
        auto lc = contrib.first;
        auto f = contrib.second;


        if (location(msh2, cl) != element_location::ON_INTERFACE)
        {
            if( sc2 )
                assembler2_sc.assemble(msh2, cl, lc, f, bcs_fun);
            else
                assembler2.assemble(msh2, cl, lc, f, bcs_fun);
        }
        else
        {
            if( sc2 )
                assembler2_sc.assemble_cut(msh2, cl, lc, f);
            else
                assembler2.assemble_cut(msh2, cl, lc, f);
        }
    }

    if( sc2 )
        assembler2_sc.finalize();
    else
        assembler2.finalize();

    SparseMatrix<T> Mat2;
    Matrix<T, Dynamic, 1> RHS2;
    if( sc2 )
    {
        Mat2 = assembler2_sc.LHS;
        RHS2 = assembler2_sc.RHS;
    }
    else
    {
        Mat2 = assembler2.LHS;
        RHS2 = assembler2.RHS;
    }

    tc.toc();
    std::cout << bold << yellow << "Step Three : " << tc << " seconds" << reset << std::endl;

    ///////////  FOURTH STEP : compute the residual
    tc.tic();
    auto res = Mat2 * sol2 - RHS2;

    tc.toc();
    std::cout << bold << yellow << "Step Four : " << tc << " seconds" << reset << std::endl;

    ///////////  FIFTH STEP : output the residual
    tc.tic();

    postprocess_output<T>  postoutput;

    auto res_gp  = std::make_shared< gnuplot_output_object<T> >("test_res.dat");
    auto sol2_gp  = std::make_shared< gnuplot_output_object<T> >("test_sol2.dat");


    for (auto& cl : msh2.cells)
    {
        cell_basis<cuthho_poly_mesh<T>, T> cb(msh2, cl, hdi.cell_degree());
        auto cbs = cb.size();
        auto fcs = faces(msh2, cl);
        auto num_faces = fcs.size();
        auto fbs = face_basis<cuthho_poly_mesh<T>,T>::size(hdi.face_degree());

        Matrix<T, Dynamic, 1> locdata_n, locdata_p, locdata;
        Matrix<T, Dynamic, 1> locdata2_n, locdata2_p, locdata2;
        Matrix<T, Dynamic, 1> cell_dofs_n, cell_dofs_p, cell_dofs;
        Matrix<T, Dynamic, 1> cell_dofs2_n, cell_dofs2_p, cell_dofs2;

        if (location(msh2, cl) == element_location::ON_INTERFACE)
        {
            if( sc2 )
            {
                auto contrib = compute_interface_contrib(
                    msh2, cl, level_set_function2, dirichlet_jump,
                    neumann_jump, rhs_fun, method, hdi, parms);
                auto lc = contrib.first;
                auto f_bis = contrib.second;
                Matrix<T, Dynamic, 1> f = Matrix<T, Dynamic, 1>::Zero(contrib.second.rows());

                locdata_n = assembler2_sc.take_local_data(msh2, cl, res, bcs_fun, element_location::IN_NEGATIVE_SIDE, lc, f);
                locdata_p = assembler2_sc.take_local_data(msh2, cl, res, bcs_fun, element_location::IN_POSITIVE_SIDE, lc, f);

                locdata2_n = assembler2_sc.take_local_data(msh2, cl, sol2, bcs_fun, element_location::IN_NEGATIVE_SIDE, lc, f_bis);
                locdata2_p = assembler2_sc.take_local_data(msh2, cl, sol2, bcs_fun, element_location::IN_POSITIVE_SIDE, lc, f_bis);
            }
            else
            {
                locdata_n = assembler2.take_local_data(msh2, cl, res, bcs_fun, element_location::IN_NEGATIVE_SIDE);
                locdata_p = assembler2.take_local_data(msh2, cl, res, bcs_fun, element_location::IN_POSITIVE_SIDE);

                locdata2_n = assembler2.take_local_data(msh2, cl, sol2, bcs_fun, element_location::IN_NEGATIVE_SIDE);
                locdata2_p = assembler2.take_local_data(msh2, cl, sol2, bcs_fun, element_location::IN_POSITIVE_SIDE);
            }

            cell_dofs_n = locdata_n.head(cbs);
            cell_dofs_p = locdata_p.head(cbs);

            cell_dofs2_n = locdata2_n.head(cbs);
            cell_dofs2_p = locdata2_p.head(cbs);

            auto qps_n = integrate(msh2, cl, hdi.cell_degree(), element_location::IN_NEGATIVE_SIDE);
            for (auto& qp : qps_n)
            {
                auto t_phi = cb.eval_basis( qp.first );
                auto v = cell_dofs_n.dot(t_phi);
                res_gp->add_data(qp.first, v);

                auto v2 = cell_dofs2_n.dot(t_phi);
                sol2_gp->add_data(qp.first, v2);
            }

            auto qps_p = integrate(msh2, cl, hdi.cell_degree(), element_location::IN_POSITIVE_SIDE);
            for (auto& qp : qps_p)
            {
                auto t_phi = cb.eval_basis( qp.first );
                auto v = cell_dofs_p.dot(t_phi);
                res_gp->add_data(qp.first, v);

                auto v2 = cell_dofs2_p.dot(t_phi);
                sol2_gp->add_data(qp.first, v2);
            }
        }
        else
        {
            if( sc2 )
            {
                auto contrib = compute_interface_contrib(
                    msh2, cl, level_set_function2, dirichlet_jump,
                    neumann_jump, rhs_fun, method, hdi, parms);
                auto lc = contrib.first;
                auto f_bis = contrib.second;
                Matrix<T, Dynamic, 1> f = Matrix<T, Dynamic, 1>::Zero(contrib.second.rows());

                locdata = assembler2_sc.take_local_data(msh2, cl, res, bcs_fun, element_location::IN_POSITIVE_SIDE, lc, f);

                locdata2 = assembler2_sc.take_local_data(msh2, cl, sol2, bcs_fun, element_location::IN_POSITIVE_SIDE, lc, f_bis);
            }
            else
            {
                locdata = assembler2.take_local_data(msh2, cl, res, bcs_fun, element_location::IN_POSITIVE_SIDE);
                locdata2 = assembler2.take_local_data(msh2, cl, sol2, bcs_fun, element_location::IN_POSITIVE_SIDE);
            }
            cell_dofs = locdata.head(cbs);
            cell_dofs2 = locdata2.head(cbs);

            auto qps = integrate(msh2, cl, hdi.cell_degree());
            for (auto& qp : qps)
            {
                auto t_phi = cb.eval_basis( qp.first );
                auto v = cell_dofs.dot(t_phi);
                res_gp->add_data(qp.first, v);

                auto v2 = cell_dofs2.dot(t_phi);
                sol2_gp->add_data(qp.first, v2);
            }
        }
    }

    postoutput.add_object(sol2_gp);
    postoutput.add_object(res_gp);
    postoutput.write();

    tc.toc();
    std::cout << bold << yellow << "Step Five : " << tc << " seconds" << reset << std::endl;
}

////////////////////  END TESTS  //////////////////////////////



template<typename Mesh, typename testType>
test_info<typename Mesh::coordinate_type>
run_cuthho_interface(const Mesh& msh, size_t degree, size_t method, testType test_case)
{
    using RealType = typename Mesh::coordinate_type;

    auto level_set_function = test_case.level_set_;

    auto rhs_fun = test_case.rhs_fun;
    auto sol_fun = test_case.sol_fun;
    auto sol_grad = test_case.sol_grad;
    auto bcs_fun = test_case.bcs_fun;
    auto dirichlet_jump = test_case.dirichlet_jump;
    auto neumann_jump = test_case.neumann_jump;
    struct params<RealType> parms = test_case.parms;

    timecounter tc;

    bool sc = true; // static condensation

    /************** ASSEMBLE PROBLEM **************/
    hho_degree_info hdi(degree+1, degree);

    tc.tic();
    auto assembler = make_interface_assembler(msh, hdi);
    auto assembler_sc = make_interface_condensed_assembler(msh, hdi);
    for (auto& cl : msh.cells)
    {
        auto contrib = compute_interface_contrib(msh, cl, level_set_function, dirichlet_jump,
                                                 neumann_jump, rhs_fun, method, hdi, parms);
        auto lc = contrib.first;
        auto f = contrib.second;

        if (location(msh, cl) != element_location::ON_INTERFACE)
        {
            if( sc )
                assembler_sc.assemble(msh, cl, lc, f, bcs_fun);
            else
                assembler.assemble(msh, cl, lc, f, bcs_fun);
        }
        else
        {
            if( sc )
                assembler_sc.assemble_cut(msh, cl, lc, f);
            else
                assembler.assemble_cut(msh, cl, lc, f);
        }
    }

    if( sc )
        assembler_sc.finalize();
    else
        assembler.finalize();

    //dump_sparse_matrix(assembler.LHS, "matrix.dat");

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
#if 0
    SparseLU<SparseMatrix<RealType>>  solver;
    Matrix<RealType, Dynamic, 1> sol;

    if( sc )
    {
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
//#if 0
    Matrix<RealType, Dynamic, 1> sol;
    cg_params<RealType> cgp;
    cgp.histfile = "cuthho_cg_hist.dat";
    cgp.verbose = true;
    cgp.apply_preconditioner = true;
    if( sc )
    {
        sol = Matrix<RealType, Dynamic, 1>::Zero(assembler_sc.RHS.rows());
        cgp.max_iter = assembler_sc.LHS.rows();
        conjugated_gradient(assembler_sc.LHS, assembler_sc.RHS, sol, cgp);
    }
    else
    {
        sol = Matrix<RealType, Dynamic, 1>::Zero(assembler.RHS.rows());
        cgp.max_iter = assembler.LHS.rows();
        conjugated_gradient(assembler.LHS, assembler.RHS, sol, cgp);
    }
//#endif
    tc.toc();
    std::cout << bold << yellow << "Linear solver: " << tc << " seconds" << reset << std::endl;

    /************** POSTPROCESS **************/


    postprocess_output<RealType>  postoutput;

    auto uT_gp  = std::make_shared< gnuplot_output_object<RealType> >("interface_uT.dat");
    auto diff_gp  = std::make_shared< gnuplot_output_object<RealType> >("interface_diff.dat");


    std::vector<RealType>   solution_uT;

    tc.tic();
    RealType    H1_error = 0.0;
    RealType    L2_error = 0.0;
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
            if( sc )
            {
                locdata_n = assembler_sc.take_local_data(msh, cl, sol, bcs_fun, element_location::IN_NEGATIVE_SIDE);
                locdata_p = assembler_sc.take_local_data(msh, cl, sol, bcs_fun, element_location::IN_POSITIVE_SIDE);
            }
            else
            {
                locdata_n = assembler.take_local_data(msh, cl, sol, bcs_fun, element_location::IN_NEGATIVE_SIDE);
                locdata_p = assembler.take_local_data(msh, cl, sol, bcs_fun, element_location::IN_POSITIVE_SIDE);
            }

            cell_dofs_n = locdata_n.head(cbs);
            cell_dofs_p = locdata_p.head(cbs);


            auto qps_n = integrate(msh, cl, 2*hdi.cell_degree(), element_location::IN_NEGATIVE_SIDE);
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
                
                /* Compute L2-error */
                L2_error += qp.second * (sol_fun(qp.first) - v) * (sol_fun(qp.first) - v);
            }
            
            
            auto qps_p = integrate(msh, cl, 2*hdi.cell_degree(), element_location::IN_POSITIVE_SIDE);
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

                /* Compute L2-error */
                L2_error += qp.second * (sol_fun(qp.first) - v) * (sol_fun(qp.first) - v);
            }
        }
        else
        {
            if( sc )
            {
                locdata = assembler_sc.take_local_data(msh, cl, sol, bcs_fun, element_location::IN_POSITIVE_SIDE);
            }
            else
                locdata = assembler.take_local_data(msh, cl, sol, bcs_fun, element_location::IN_POSITIVE_SIDE);
            cell_dofs = locdata.head(cbs);

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

                /* Compute L2-error */
                L2_error += qp.second * (sol_fun(qp.first) - v) * (sol_fun(qp.first) - v);
            }
        }

        cell_i++;
    }

    std::cout << bold << green << "Energy-norm absolute error:           " << std::sqrt(H1_error) << std::endl;
    std::cout << bold << green << "L2-norm absolute error:           " << std::sqrt(L2_error) << std::endl;

    postoutput.add_object(uT_gp);
    postoutput.add_object(diff_gp);
    postoutput.write();



    test_info<RealType> TI;
    TI.H1 = std::sqrt(H1_error);
    TI.L2 = std::sqrt(L2_error);

    if (false)
    {
        /////////////// compute condition number
        SparseMatrix<RealType> Mat;
        // Matrix<RealType, Dynamic, Dynamic> Mat;
        if (sc)
            Mat = assembler_sc.LHS;
        else
            Mat = assembler.LHS;


        RealType sigma_max, sigma_min;

        // Construct matrix operation object using the wrapper class SparseSymMatProd
        Spectra::SparseSymMatProd<RealType> op(Mat);
        // Construct eigen solver object, requesting the largest eigenvalue
        Spectra::SymEigsSolver< RealType, Spectra::LARGEST_MAGN,
                                Spectra::SparseSymMatProd<RealType> > max_eigs(&op, 1, 10);
        max_eigs.init();
        max_eigs.compute();
        if(max_eigs.info() == Spectra::SUCCESSFUL)
            sigma_max = max_eigs.eigenvalues()(0);


        // Construct eigen solver object, requesting the smallest eigenvalue
        Spectra::SymEigsSolver< RealType, Spectra::SMALLEST_MAGN,
                                Spectra::SparseSymMatProd<RealType> > min_eigs(&op, 1, 10);

        min_eigs.init();
        min_eigs.compute();
        if(min_eigs.info() == Spectra::SUCCESSFUL)
            sigma_min = min_eigs.eigenvalues()(0);

        // compute condition number
        RealType cond = sigma_max / sigma_min;
        TI.cond = cond;
        std::cout << "sigma_max = " << sigma_max << "   sigma_min = "
                  << sigma_min << "  cond = " << cond
                  << std::endl;
    }
    else
        TI.cond = 0.0;

    tc.toc();
    std::cout << bold << yellow << "Postprocessing: " << tc << " seconds" << reset << std::endl;


    return TI;
}




/////////////////////////   AUTOMATIC TESTS  //////////////////////////


void convergence_test(void)
{
    using T = double;

    std::vector<size_t> mesh_sizes, pol_orders;

    // meshes
    mesh_sizes.push_back(8);
    mesh_sizes.push_back(16);
    mesh_sizes.push_back(32);
    mesh_sizes.push_back(64);
    mesh_sizes.push_back(128);
    // mesh_sizes.push_back(256);

    // polynomial orders
    pol_orders.push_back(0);
    pol_orders.push_back(1);
    pol_orders.push_back(2);
    pol_orders.push_back(3);


    // export to files ...
    std::vector<std::string> files;
    files.push_back("./output/test_k0.txt");
    files.push_back("./output/test_k1.txt");
    files.push_back("./output/test_k2.txt");
    files.push_back("./output/test_k3.txt");

    for (std::vector<size_t>::iterator it = pol_orders.begin(); it != pol_orders.end(); it++)
    {
        size_t k = *it;

        std::cout << "start tests for k = " << k << std::endl;

        // init the file
        std::ofstream output;
        output.open (files.at(*it), std::ios::in | std::ios::trunc);
        if (!output.is_open())
            throw std::logic_error("file not open");

        // output << "N\th\tH1\tordre1\tL2\tordre2" << std::endl;
        output << "N\th\tH1\tordre1\tL2\tordre2\tcond" << std::endl;

        // convergence tests
        T previous_H1 = 0.0;
        T previous_L2 = 0.0;
        T previous_h = 0.0;
        for (std::vector<size_t>::iterator it_msh = mesh_sizes.begin();
             it_msh != mesh_sizes.end(); it_msh++)
        {
            size_t N = *it_msh;

            // init mesh (with agglomeration)
            mesh_init_params<T> mip;
            mip.Nx = N;
            mip.Ny = N;
            cuthho_poly_mesh<T> msh(mip);
            size_t int_refsteps = 1;
            T radius = 1.0/3.0;
            auto circle_level_set_function = circle_level_set<T>(radius, 0.5, 0.5);

            // auto level_set_function = flower_level_set<T>(0.31, 0.5, 0.5, 4, 0.04);
            // auto level_set_function = circle_level_set<T>(radius, 0.5, 0.5);
            // auto level_set_function = square_level_set<T>(1.05, -0.05, -0.05, 1.05);
            // auto level_set_function = square_level_set<T>(1.0, -0.0, -0.0, 1.0);
            // auto level_set_function = square_level_set<T>(0.76, 0.24, 0.24, 0.76);
            auto level_set_function = square_level_set<T>(0.751, 0.249, 0.249, 0.751);
            detect_node_position(msh, level_set_function);
            detect_cut_faces(msh, level_set_function);
            if(1)  // AGGLOMERATION
            {
                detect_cut_cells(msh, level_set_function);
                detect_cell_agglo_set(msh, level_set_function);
                make_neighbors_info_cartesian(msh);
                refine_interface(msh, level_set_function, int_refsteps);
                make_agglomeration(msh, level_set_function);
            }
            else  // NODE DISPLACEMENT
            {
                move_nodes(msh, level_set_function);
                detect_cut_faces(msh, level_set_function); //do it again to update intersection points
                detect_cut_cells(msh, level_set_function);
                refine_interface(msh, level_set_function, int_refsteps);
            }

            // compute solution/errors
            test_info<T> TI;
            // auto TI = run_cuthho_interface(msh, level_set_function, k, 3);

            // auto TI = run_cuthho_interface(msh, level_set_function, k, 3, test_case);
            if(0) // sin(\pi x) * sin(\pi y)
            {
                auto test_case = make_test_case_laplacian_sin_sin(msh, level_set_function);
                TI = run_cuthho_interface(msh, k, 3, test_case);
                // TI = run_cuthho_fictdom(msh, k, test_case);
            }
            if(0) // 1 + sin(\pi x) * sin(\pi y)
            {
                auto test_case = make_test_case_laplacian_sin_sin_bis(msh, level_set_function);
                // TI = run_cuthho_interface(msh, k, 3, test_case);
                TI = run_cuthho_fictdom(msh, k, test_case);
            }
            if(0) // sin(\pi (x-a)/(b-a)) * sin(\pi (y-c)/(d-c))
            {
                T a = 0.23;
                T b = 0.77;
                T c = 0.23;
                T d = 0.77;
                auto test_case = make_test_case_laplacian_sin_sin_gen(msh, level_set_function,
                                                                      a, b, c, d);
                // TI = run_cuthho_interface(msh, k, 3, test_case);
                TI = run_cuthho_fictdom(msh, k, test_case);
            }
            if(0) // exp(x) * cos(y)
            {
                auto test_case = make_test_case_laplacian_exp_cos(msh, level_set_function);
                // TI = run_cuthho_interface(msh, k, 3, test_case);
                TI = run_cuthho_fictdom(msh, k, test_case);
            }
            if(1) // jumps sin_sin -> exp_cos
            {
                auto test_case = make_test_case_laplacian_jumps_1(msh, level_set_function);
                TI = run_cuthho_interface(msh, k, 3, test_case);
            }
            if(0) // jumps2 exp_cos -> sin_sin
            {
                auto test_case = make_test_case_laplacian_jumps_2(msh, level_set_function);
                TI = run_cuthho_interface(msh, k, 3, test_case);
            }
            if(0) // jumps3 sin_sin -> sin_sin + pol
            {
                auto test_case = make_test_case_laplacian_jumps_3(msh, level_set_function);
                TI = run_cuthho_interface(msh, k, 3, test_case);
            }
            if(0) // contrast deg 2
            {
                auto parms = params<T>();
                parms.kappa_1 = 1.0;
                parms.kappa_2 = 1000.0;

                auto test_case = make_test_case_laplacian_contrast_2(msh, circle_level_set_function, parms);
                TI = run_cuthho_interface(msh, k, 3, test_case);
            }
            if(0) // contrast deg 6
            {
                auto parms = params<T>();
                parms.kappa_1 = 1.0;
                parms.kappa_2 = 1.0;

                auto test_case = make_test_case_laplacian_contrast_6(msh, circle_level_set_function, parms);
                TI = run_cuthho_interface(msh, k, 3, test_case);
            }

            if(0) // homogeneous test case on a circle
            {
                auto test_case = make_test_case_laplacian_circle_hom(msh, circle_level_set_function);
                // TI = run_cuthho_interface(msh, k, 3, test_case);
                TI = run_cuthho_fictdom(msh, k, test_case);
            }

            // auto TI = run_cuthho_fictdom(msh, level_set_function, k);


            // report info in the file
            T h = 1.0/N;
            if (it_msh == mesh_sizes.begin())
            {
                // output << N << "\t" << h << "\t" << TI.H1 << "\t" << "."
                //        << "\t" << TI.L2 << "\t" << "." << "\t" << "0.0"
                //        << std::endl;
                output << N << "\t" << h << "\t" << TI.H1 << "\t" << "."
                       << "\t" << TI.L2 << "\t" << "." << "\t" << TI.cond
                       << std::endl;
            }
            else
            {
                T orderH = log(previous_H1 / TI.H1) / log(previous_h / h);
                T orderL = log(previous_L2 / TI.L2) / log(previous_h / h);
                // output << N << "\t" << h << "\t" << TI.H1 << "\t" << orderH
                //        << "\t" << TI.L2 << "\t" << orderL << "\t" << "0.0"
                //        << std::endl;
                output << N << "\t" << h << "\t" << TI.H1 << "\t" << orderH
                       << "\t" << TI.L2 << "\t" << orderL << "\t" << TI.cond
                       << std::endl;
            }
            previous_H1 = TI.H1;
            previous_L2 = TI.L2;
            previous_h = h;
        }
        // close the file
        output.close();
    }

    // update the gnuplot curves
    system("gnuplot './output/gnuplot_script.txt'");

    // update the .pdf file
    system("pdflatex ./output/autom_tests.tex");

    // open the .pdf file
    system("xdg-open ./autom_tests.pdf");
}

//////////////////////////     MAIN        ////////////////////////////
#if 1
int main(int argc, char **argv)
{
    convergence_test();
    // tests_stabilization();
    // interface_residus();
    return 1;
}
#endif

#if 0
int main(int argc, char **argv)
{
    using RealType = double;
    
    size_t degree           = 0;
    size_t int_refsteps     = 4;
    size_t method           = 4;

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
    while ( (ch = getopt(argc, argv, "k:M:N:r:m:ifDAd")) != -1 )
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

            case 'm':
                method = atoi(optarg);
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
    // auto level_set_function = circle_level_set<RealType>(radius, 0.5, 0.5);
    // auto level_set_function = line_level_set<RealType>(0.5);
    auto level_set_function = flower_level_set<RealType>(0.31, 0.5, 0.5, 4, 0.04);
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
        output_mesh_info(msh, level_set_function);
        test_projection(msh, level_set_function, degree);
    }

    output_mesh_info(msh, level_set_function);

    // jumps sin_sin -> exp_cos
    // auto test_case = make_test_case_laplacian_jumps_1(msh, level_set_function);
    // jumps3 sin_sin -> sin_sin + pol
    auto test_case = make_test_case_laplacian_jumps_3(msh, level_set_function);

    if (solve_interface)
        run_cuthho_interface(msh, degree, method, test_case);
    
    if (solve_fictdom)
        run_cuthho_fictdom(msh, degree, test_case);


    return 0;
}
#endif
