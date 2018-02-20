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

#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <Eigen/SparseLU>
#include <unsupported/Eigen/SparseExtra>

using namespace Eigen;

#include "core/core"
#include "core/solvers"
#include "methods/hho"

int main(void)
{
	using RealType = double;

	size_t N = 10;
	size_t k = 1;

	hho_degree_info hdi(k, k);


    mesh_init_params<RealType> mip;
    mip.Nx = N;
    mip.Ny = N;

    quad_mesh<RealType> msh(mip);


    auto rhs_fun = [](const typename quad_mesh<RealType>::point_type& pt) -> RealType {
        return 2.0 * M_PI * M_PI * std::sin(M_PI*pt.x()) * std::sin(M_PI*pt.y());
        //return pt.x();
    };


    RealType error = 0.0;

    for (auto& cl : msh.cells)
    {
    	auto gr = make_hho_laplacian(msh, cl, hdi);

        Matrix<RealType, Dynamic, Dynamic> stab;   
        stab = make_hho_fancy_stabilization(msh, cl, gr.first, hdi);

        Matrix<RealType, Dynamic, 1> proj = project_function(msh, cl, hdi, rhs_fun);

        error += proj.dot(stab*proj);
    }

    std::cout << std::sqrt(error) << std::endl;

}