/*
 *       /\        Matteo Cicuttin (C) 2017
 *      /__\       matteo.cicuttin@enpc.fr
 *     /_\/_\      École Nationale des Ponts et Chaussées - CERMICS
 *    /\    /\
 *   /__\  /__\    This is a prototype implementation of the CutHHO method,
 *  /_\/_\/_\/_\   an unfitted Hybrid High-order method.
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
#include <fstream>
#include <vector>
#include <array>

#include "point.hpp"

/*****************************************************************************
 *   Mesh stuff
 *****************************************************************************/
template<typename UserData>
struct mesh_element
{
    UserData    user_data;
};

template<>
struct mesh_element<void>
{};

template<typename UserData, size_t N>
struct cell : public mesh_element<UserData> {
    std::array<size_t, N>   ptids;

    cell()
    {}

    bool operator<(const cell& other) const
    {
        return (this->ptids < other.ptids);
    }

    bool operator==(const cell& other) const
    {
        return (this->ptids == other.ptids);
    }
};

template<typename UserData = void>
using quad_cell = cell<UserData, 4>;

template<typename UserData = void>
using tri_cell = cell<UserData, 3>;

template<typename UserData, size_t N>
std::ostream&
operator<<(std::ostream& os, const cell<UserData, N>& cl)
{
    os << "Cell: ";
    for (size_t i = 0; i < N; i++)
        os << cl.ptids[i] << " ";

    return os;
}

enum class boundary
{
    NONE,
    DIRICHLET,
    NEUMANN,
    ROBIN
};

template<typename UserData>
struct face : public mesh_element<UserData> {
    std::array<size_t, 2>   ptids;
    bool                    is_boundary;
    boundary                bndtype;

    face() :
        is_boundary(false),
        bndtype(boundary::NONE)
    {}

    bool operator<(const face& other) const
    {
        assert(ptids[0] < ptids[1]);
        assert(other.ptids[0] < other.ptids[1]);
        return (this->ptids < other.ptids);
    }

    bool operator==(const face& other) const
    {
        assert(ptids[0] < ptids[1]);
        assert(other.ptids[0] < other.ptids[1]);
        return (this->ptids == other.ptids);
    }
};

template<typename UserData>
struct node : public mesh_element<UserData> {
    size_t      ptid;

    bool operator<(const node& other) const
    {
        return (this->ptid < other.ptid);
    }

    bool operator==(const node& other) const
    {
        return (this->ptid == other.ptid);
    }
};

template<typename UserData>
std::ostream&
operator<<(std::ostream& os, const face<UserData>& fc)
{
    os << "Face: " << fc.ptids[0] << " " << fc.ptids[1];
    if (fc.is_boundary && fc.bndtype == boundary::NONE)
        os << " [Boundary, unspecified kind]";
    if (fc.is_boundary && fc.bndtype == boundary::DIRICHLET)
        os << " [Dirichlet boundary]";
    if (fc.is_boundary && fc.bndtype == boundary::NEUMANN)
        os << " [Neumann boundary]";
    if (fc.is_boundary && fc.bndtype == boundary::ROBIN)
        os << " [Robin boundary]";
    return os;
}

template<typename T>
struct mesh_init_params {
    T           min_x, max_x;
    T           min_y, max_y;
    size_t      Nx, Ny;

    mesh_init_params()
        : min_x(0.0), max_x(1.0),
          min_y(0.0), max_y(1.0),
          Nx(4), Ny(4)
    {}

    T hx() const {
        return (max_x - min_x)/Nx;
    }

    T hy() const {
        return (max_y - min_y)/Ny;
    }
};




struct elem_simplex;
struct elem_quad;
struct elem_mixed;

template<typename T, typename KT, typename CellUD = void, typename FaceUD = void, typename NodeUD = void>
struct mesh_impl;


template<typename T, typename CellUD, typename FaceUD, typename NodeUD>
struct mesh_impl<T, elem_quad, CellUD, FaceUD, NodeUD> {

    typedef point<T,2>          point_type;
    typedef quad_cell<CellUD>   cell_type;
    typedef face<FaceUD>        face_type;
    typedef node<NodeUD>        node_type;
    typedef CellUD              cell_ud_type;
    typedef FaceUD              face_ud_type;
    typedef NodeUD              node_ud_type;
    typedef T                   coordinate_type;

    std::vector<point_type>     points;
    std::vector<node_type>      nodes;
    std::vector<face_type>      faces;
    std::vector<cell_type>      cells;

    mesh_impl() : mesh_impl( mesh_init_params<T>() )
    {}

    mesh_impl(const mesh_init_params<T>& parms)
    {
        auto hx = parms.hx();
        auto hy = parms.hy();

        size_t numpoints = (parms.Nx + 1) * (parms.Ny + 1);
        points.reserve(numpoints);

        size_t point_num = 0;
        for (size_t j = 0; j < parms.Ny+1; j++)
        {
            for(size_t i = 0; i < parms.Nx+1; i++)
            {
                auto px = parms.min_x + i*hx;
                auto py = parms.min_y + j*hy;
                point_type pt(px, py);
                points.push_back(pt);
                node_type n;
                n.ptid = point_num++;
                nodes.push_back(n);
            }
        }

        for (size_t j = 0; j < parms.Ny; j++)
        {
            for (size_t i = 0; i < parms.Nx; i++)
            {
                auto pt0_idx = j*(parms.Nx+1) + i;
                auto pt1_idx = pt0_idx + 1;
                auto pt2_idx = pt0_idx + parms.Nx + 2;
                auto pt3_idx = pt0_idx + parms.Nx + 1;

                cell_type cl;
                cl.ptids = {{pt0_idx, pt1_idx, pt2_idx, pt3_idx}};
                cells.push_back(cl);

                face_type f0;
                f0.ptids = {pt0_idx, pt1_idx};
                if (j == 0) f0.is_boundary = true;
                faces.push_back(f0);

                face_type f1;
                f1.ptids = {pt1_idx, pt2_idx};
                if (i == parms.Nx-1) f1.is_boundary = true;
                faces.push_back(f1);

                face_type f2;
                f2.ptids = {pt3_idx, pt2_idx};
                if (j == parms.Ny-1) f2.is_boundary = true;
                faces.push_back(f2);

                face_type f3;
                f3.ptids = {pt0_idx, pt3_idx};
                if (i == 0) f3.is_boundary = true;
                faces.push_back(f3);

            }
        }

        std::sort(cells.begin(), cells.end());
        std::sort(faces.begin(), faces.end());
        faces.erase( std::unique(faces.begin(), faces.end()), faces.end() );

        for (auto& fc : faces)
        {
            if (fc.is_boundary)
                fc.bndtype = boundary::DIRICHLET;
        }
    }
};


template<typename T, typename ET, typename CellUD = void, typename FaceUD = void, typename NodeUD = void>
using mesh = mesh_impl<T, ET, CellUD, FaceUD, NodeUD>;

template<typename T, typename CellUD = void, typename FaceUD = void, typename NodeUD = void>
using quad_mesh = mesh_impl<T, elem_quad, CellUD, FaceUD, NodeUD>;



#if 0






template<typename T>
bool load_netgen_2d(const std::string& filename, mesh<T>& msh)
{
    std::ifstream ifs(filename);
    if (!ifs.is_open())
    {
        std::cout << "Problem opening input mesh" << std::endl;
        return false;
    }

    size_t count;

    ifs >> count;
    msh.points.reserve(count);

    for (size_t i = 0; i < count; i++)
    {
        T x, y;
        ifs >> x >> y;
        msh.points.push_back( typename mesh<T>::point_type(x,y) );
    }

    std::cout << "Points: " << msh.points.size() << std::endl;

    ifs >> count;
    msh.cells.reserve(count);

    for (size_t i = 0; i < count; i++)
    {
        size_t dom, p0, p1, p2;
        ifs >> dom >> p0 >> p1 >> p2;

        if (dom == 0 || p0 == 0 || p1 == 0 || p2 == 0 )
        {
            std::cout << "Indices in netgen file should be 1-based, found a 0.";
            std::cout << std::endl;
            return false;
        }

        polygon p;
        p.domain_id = dom-1;
        p.point_ids.push_back(p0-1);
        p.point_ids.push_back(p1-1);
        p.point_ids.push_back(p2-1);
        msh.cells.push_back(p);
    }

    std::cout << "Cells: " << msh.cells.size() << std::endl;
    std::sort(msh.cells.begin(), msh.cells.end());

    msh.faces.reserve( 3 * msh.cells.size() );

    for (auto& cl : msh.cells)
    {
        assert(cl.point_ids.size() == 3);

        for (size_t i = 0; i < 3; i++)
        {
            auto p0 = cl.point_ids[i];
            auto p1 = cl.point_ids[(i+1)%3];

            if (p1 < p0)
                std::swap(p0, p1);

            typename mesh<T>::face_type face;
            face.point_ids[0] = p0;
            face.point_ids[1] = p1;

            msh.faces.push_back(face);
        }
    }

    std::sort(msh.faces.begin(), msh.faces.end());
    msh.faces.erase(std::unique(msh.faces.begin(), msh.faces.end()), msh.faces.end());

    std::cout << "Faces: " << msh.faces.size() << std::endl;

    for (size_t i = 0; i < msh.points.size(); i++)
    {
        typename mesh<T>::node_type node;
        node.point_id = i;
        msh.nodes.push_back(node);
    }

    std::cout << "Nodes: " << msh.nodes.size() << std::endl;

    ifs.close();
    return true;
}




#endif
