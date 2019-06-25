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

#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <Eigen/SparseLU>
#include <unsupported/Eigen/SparseExtra>

using namespace Eigen;

/*****************************************************************************
 *   Conjugated gradient solver
 *****************************************************************************/
enum class cg_exit_reason
{
    CONVERGED,
    DIVERGED,
    MAX_ITER_REACHED,
};

template<typename T>
struct cg_params
{
    T               convergence_threshold;
    T               divergence_threshold;
    size_t          max_iter;
    bool            verbose;
    bool            apply_preconditioner;
    std::string     histfile;

    cg_params() :
        convergence_threshold(1e-14),
        divergence_threshold(100),
        max_iter(1000),
        verbose(false),
        apply_preconditioner(false)
    {}
};

template<typename T>
cg_exit_reason
conjugated_gradient(const Eigen::SparseMatrix<T>& A,
                    const Eigen::Matrix<T, Eigen::Dynamic, 1>& b,
                    Eigen::Matrix<T, Eigen::Dynamic, 1>& x,
                    const cg_params<T>& parms = cg_params<T>())
{
    size_t                      N = A.cols();
    size_t                      iter = 0;
    T                           nr, nr0;
    T                           alpha, beta, rho;

    Eigen::Matrix<T, Eigen::Dynamic, 1> d(N), r(N), r0(N), y(N), iMr(N);
    x = Eigen::Matrix<T, Eigen::Dynamic, 1>::Zero(N);

    Eigen::Matrix<T, Eigen::Dynamic, 1> iA = A.diagonal();

    for (size_t i = 0; i < iA.rows(); i++)
      iA(i) = 1./iA(i);

    r0 = r = b - A*x;
    d = parms.apply_preconditioner ? iA.cwiseProduct(r0) : r0;
    nr = nr0 = r.norm();

    std::ofstream ofs;

    if ( parms.histfile.size() != 0 )
        ofs.open(parms.histfile);

    cg_exit_reason exit_reason;

    while ( 1 )
    {
        if ( parms.verbose && (iter%100 == 0) ) {
            std::cout << erase_line << " -> Iteration " << iter << ", rr = ";
            std::cout << nr/nr0 << "\r";
            std::cout.flush();
        }

        if ( parms.histfile.size() != 0 )
            ofs << nr/nr0 << std::endl;

        y = A*d;
        iMr = parms.apply_preconditioner ? iA.cwiseProduct(r) : r;
        rho = r.dot(iMr);
        alpha = rho/d.dot(y);
        x = x + alpha * d;
        r = r - alpha * y;
        nr = r.norm();

        if ( nr/nr0 < parms.convergence_threshold ) {
            exit_reason = cg_exit_reason::CONVERGED;
            break;
        }

        if ( iter > parms.max_iter ) {
            exit_reason = cg_exit_reason::MAX_ITER_REACHED;
            break;
        }

        if ( nr/nr0 > parms.divergence_threshold ) {
            exit_reason = cg_exit_reason::DIVERGED;
            break;
        }

        iMr = parms.apply_preconditioner ? iA.cwiseProduct(r) : r;
        beta = r.dot(iMr)/rho;
        d = iMr + beta * d;
        iter++;
    }

    if ( parms.histfile.size() != 0 )
    {
        ofs << nr/nr0 << std::endl;
        ofs.close();
    }

    if ( parms.verbose )
        std::cout << erase_line << " -> Iteration " << iter << ", rr = " << nr/nr0 << std::endl;

    return exit_reason;
}
