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
#include <iterator>
#include "cuthho_mesh.hpp"

template<typename T, size_t ET>
bool
is_cut(const cuthho_mesh<T, ET>& msh, const typename cuthho_mesh<T, ET>::cell_type& cl)
{
    assert(cl.user_data.location != element_location::UNDEF);
    return cl.user_data.location == element_location::ON_INTERFACE;
}

template<typename T, size_t ET>
bool
is_cut(const cuthho_mesh<T, ET>& msh, const typename cuthho_mesh<T, ET>::face_type& fc)
{
    assert(fc.user_data.location != element_location::UNDEF);
    return fc.user_data.location == element_location::ON_INTERFACE;
}

template<typename T, size_t ET>
element_location
location(const cuthho_mesh<T, ET>& msh, const typename cuthho_mesh<T, ET>::cell_type& cl)
{
    assert(cl.user_data.location != element_location::UNDEF);
    return cl.user_data.location;
}

template<typename T, size_t ET>
element_location
location(const cuthho_mesh<T, ET>& msh, const typename cuthho_mesh<T, ET>::face_type& fc)
{
    assert(fc.user_data.location != element_location::UNDEF);
    return fc.user_data.location;
}

template<typename T, size_t ET>
element_location
location(const cuthho_mesh<T, ET>& msh, const typename cuthho_mesh<T, ET>::node_type& n)
{
    assert(n.user_data.location != element_location::UNDEF);
    return n.user_data.location;
}


template<typename T, typename Function>
point<T, 2>
find_zero_crossing(const point<T,2>& p0, const point<T,2>& p1, const Function& level_set_function,
                   const T& threshold)
{
    /* !!! We assume that the level set function *has* a zero crossing
     * between p0 and p1 !!! */
    auto pa = p0;
    auto pb = p1;
    auto pm = (pa+pb)/2.0;
    auto pm_prev = pm;

    T x_diff_sq, y_diff_sq;

    /* A threshold of 1/10000 the diameter of the element is considered
     * acceptable. Since with 24 iterations we reduce the error by 16384
     * and the worst case is that the two points are at the opposite sides
     * of the element, we put 30 as limit. */
    size_t max_iter = 50;

    do {
        auto la = level_set_function(pa);
        auto lb = level_set_function(pb);
        auto lm = level_set_function(pm);

        if ( (lb >= 0 && lm >= 0) || (lb < 0 && lm < 0) )
        {   /* intersection is between pa and pm */
            pm_prev = pm;
            pb = pm;
            pm = (pa+pb)/2.0;
        }
        else
        {   /* intersection is between pm and pb */
            pm_prev = pm;
            pa = pm;
            pm = (pa+pb)/2.0;
        }

        x_diff_sq = (pm_prev.x() - pm.x()) * (pm_prev.x() - pm.x());
        y_diff_sq = (pm_prev.y() - pm.y()) * (pm_prev.y() - pm.y());

    } while ( (sqrt(x_diff_sq + y_diff_sq) > threshold) && max_iter-- );

    return pm;

    /* Affine zero crossing was like that: */
    //auto t = l0/(l0-l1);
    //auto ip = (pts[1] - pts[0]) * t + pts[0];
}

template<typename T, size_t ET, typename Function>
void
detect_node_position(cuthho_mesh<T, ET>& msh, const Function& level_set_function)
{
    for (auto& n : msh.nodes)
    {
        auto pt = points(msh, n);
        if ( level_set_function(pt) < 0 )
            n.user_data.location = element_location::IN_NEGATIVE_SIDE;
        else
            n.user_data.location = element_location::IN_POSITIVE_SIDE;
    }
}

template<typename T, size_t ET, typename Function>
void
detect_cut_faces(cuthho_mesh<T, ET>& msh, const Function& level_set_function)
{
    for (auto& fc : msh.faces)
    {
        auto pts = points(msh, fc);
        auto l0 = level_set_function(pts[0]);
        auto l1 = level_set_function(pts[1]);
        if (l0 >= 0 && l1 >= 0)
        {
            fc.user_data.location = element_location::IN_POSITIVE_SIDE;
            continue;
        }

        if (l0 < 0 && l1 < 0)
        {
            fc.user_data.location = element_location::IN_NEGATIVE_SIDE;
            continue;
        }

        auto threshold = diameter(msh, fc) / 1e20;
        auto pm = find_zero_crossing(pts[0], pts[1], level_set_function, threshold);

        /* If node 0 is in the negative region, mark it as node inside, otherwise mark node 1 */
        fc.user_data.node_inside = ( l0 < 0 ) ? 0 : 1;
        fc.user_data.location = element_location::ON_INTERFACE;
        fc.user_data.intersection_point = pm;
    }
}

template<typename T, size_t ET, typename Function>
void
detect_cell_agglo_set(cuthho_mesh<T, ET>& msh, const Function& level_set_function)
{
    typedef typename cuthho_mesh<T, ET>::face_type  face_type;
    typedef typename cuthho_mesh<T, ET>::point_type point_type;

    const T threshold = 0.3;
    const T threshold_cells = 0.2;

    for (auto& cl : msh.cells)
    {
        auto fcs = faces(msh, cl);
        auto pts = points(msh, cl);
        auto nds = nodes(msh, cl);

        if (fcs.size() != 4)
            throw std::invalid_argument("This works only on quads for now");

        if( !is_cut(msh, cl) )
        {
            cl.user_data.agglo_set = cell_agglo_set::T_OK;
            continue;
        }

        //// another criterion on the area of the cell
        if( measure(msh, cl, element_location::IN_NEGATIVE_SIDE)
            < threshold_cells * measure(msh, cl) )
        {
            cl.user_data.agglo_set = cell_agglo_set::T_KO_NEG;
            continue;
        }
        else if( measure(msh, cl, element_location::IN_POSITIVE_SIDE)
            < threshold_cells * measure(msh, cl) )
        {
            cl.user_data.agglo_set = cell_agglo_set::T_KO_POS;
            continue;
        }

        /* If it is a quadrilateral we have 6 possible configurations of the
         * element-cut intersection. */

        auto agglo_set_single_node = [&](size_t n) -> void
        {
            auto f1 = (n == 0) ? fcs.size()-1 : n-1;
            auto f2 = n;

            auto ma = measure(msh, fcs[f1]);
            auto pa = (pts[n] - fcs[f1].user_data.intersection_point);
            auto da = pa.to_vector().norm() / ma;

            auto mb = measure(msh, fcs[f2]);
            auto pb = (pts[n] - fcs[f2].user_data.intersection_point);
            auto db = pb.to_vector().norm() / mb;

            assert(da >= 0 && da <= 1);
            assert(db >= 0 && db <= 1);

            if ( std::min(da, db) > threshold )
            {
                cl.user_data.agglo_set = cell_agglo_set::T_OK;
                return;
            }

            if ( location(msh, nds[n]) == element_location::IN_NEGATIVE_SIDE )
                cl.user_data.agglo_set = cell_agglo_set::T_KO_NEG;
            else
                cl.user_data.agglo_set = cell_agglo_set::T_KO_POS;
        };
        
        auto agglo_set_double_node = [&](size_t f1, size_t f2) -> void
        {
            assert ( (f1 == 0 && f2 == 2) || ( f1 == 1 && f2 == 3 ) );

            auto n1 = f1;
            auto n2 = (f2+1) % fcs.size();

            auto ma = measure(msh, fcs[f1]);
            auto pa = (pts[n1] - fcs[f1].user_data.intersection_point);
            auto da = pa.to_vector().norm() / ma;

            auto mb = measure(msh, fcs[f2]);
            auto pb = (pts[n2] - fcs[f2].user_data.intersection_point);
            auto db = pb.to_vector().norm() / mb;

            auto m1 = std::max(da, db);
            auto m2 = std::max(1-da, 1-db);

            if ( std::min(m1, m2) > threshold )
            {
                cl.user_data.agglo_set = cell_agglo_set::T_OK;
                return;
            }

            if ( location(msh, nds[n1]) == element_location::IN_NEGATIVE_SIDE )
                cl.user_data.agglo_set = (m1 <= threshold) ? cell_agglo_set::T_KO_NEG : cell_agglo_set::T_KO_POS;
            else
                cl.user_data.agglo_set = (m2 <= threshold) ? cell_agglo_set::T_KO_NEG : cell_agglo_set::T_KO_POS;
        };

        for (size_t i = 0; i < fcs.size(); i++)
        {
            auto f1 = i;
            auto f2 = (i+1) % fcs.size();
            auto n = (i+1) % fcs.size();

            if ( is_cut(msh, fcs[f1]) && is_cut(msh, fcs[f2]) )
                agglo_set_single_node(n);

        }

        if ( is_cut(msh, fcs[0]) && is_cut(msh, fcs[2]) )
            agglo_set_double_node(0,2);

        if ( is_cut(msh, fcs[1]) && is_cut(msh, fcs[3]) )
            agglo_set_double_node(1,3);
/*
        if ( is_cut(msh, fcs[0]) && is_cut(msh, fcs[3]) )
            agglo_set_case_1();
        else if ( is_cut(msh, fcs[0]) && is_cut(msh, fcs[1]) )
            agglo_set_case_2();
        else if ( is_cut(msh, fcs[1]) && is_cut(msh, fcs[2]) )
            agglo_set_case_3();
        else if ( is_cut(msh, fcs[2]) && is_cut(msh, fcs[3]) )
            agglo_set_case_4();
        else if ( is_cut(msh, fcs[0]) && is_cut(msh, fcs[2]) )
            agglo_set_case_5();
        else if ( is_cut(msh, fcs[1]) && is_cut(msh, fcs[3]) )
            agglo_set_case_6();
            */
    }
}

template<typename T, size_t ET, typename Function>
void
detect_cut_cells(cuthho_mesh<T, ET>& msh, const Function& level_set_function)
{
    typedef typename cuthho_mesh<T, ET>::face_type  face_type;
    typedef typename cuthho_mesh<T, ET>::point_type point_type;

    size_t cell_i = 0;
    for (auto& cl : msh.cells)
    {
        auto fcs = faces(msh, cl);
        std::array< std::pair<size_t, point_type>, 2 >  cut_faces;

        size_t k = 0;
        for (size_t i = 0; i < fcs.size(); i++)
        {
            if ( is_cut(msh, fcs[i]) )
                cut_faces.at(k++) = std::make_pair(i, fcs[i].user_data.intersection_point);
        }

        /* If a face is cut, the cells that own the face are cut. Is this
         * unconditionally true? It should...fortunately this isn't avionics
         * software */

        if (k == 0)
        {
            auto is_positive = [&](const point_type& pt) -> bool {
                return level_set_function(pt) > 0;
            };

            auto pts = points(msh, cl);
            if ( std::all_of(pts.begin(), pts.end(), is_positive) )
                cl.user_data.location = element_location::IN_POSITIVE_SIDE;
            else
                cl.user_data.location = element_location::IN_NEGATIVE_SIDE;
        }

        if (k == 2)
        {
            cl.user_data.location = element_location::ON_INTERFACE;
            auto p0 = cut_faces[0].second;
            auto p1 = cut_faces[1].second;
            auto pt = p1 - p0;
            auto pn = p0 + point<T,2>(-pt.y(), pt.x());

            if ( level_set_function(pn) >= 0 )
            {
                cl.user_data.p0 = p1;
                cl.user_data.p1 = p0;
            }
            else
            {
                cl.user_data.p0 = p0;
                cl.user_data.p1 = p1;
            }

            cl.user_data.interface.push_back(cl.user_data.p0);
            cl.user_data.interface.push_back(cl.user_data.p1);
        }

        if ( k != 0 && k != 2 )
            throw std::logic_error("invalid number of cuts in cell");

        cell_i++;
    }
}

/* this creates Delta(T) */
// two neighbors have at least one common face
template<typename T, size_t ET>
void
make_neighbors_info(cuthho_mesh<T, ET>& msh)
{
    for (size_t i = 0; i < msh.cells.size(); i++)
    {
        auto fc_i = faces(msh,msh.cells[i]);
        for (size_t j = i+1; j < msh.cells.size(); j++)
        {
            auto &cl1 = msh.cells.at(i);
            auto &cl2 = msh.cells.at(j);

            // two f_neighbors have at least one common face
            bool are_f_neighbors = false;
            auto fc_j = faces(msh,msh.cells[j]);
            for (size_t i_face = 0; i_face < fc_i.size(); i_face++)
                for (size_t j_face = 0; j_face < fc_j.size(); j_face++)
                    if ( fc_i[i_face] == fc_j[j_face] )
                        are_f_neighbors = true;

            if (are_f_neighbors)
            {
                auto ofs_cl1 = offset(msh, cl1);
                auto ofs_cl2 = offset(msh, cl2);

                cl1.user_data.f_neighbors.insert(ofs_cl2);
                cl2.user_data.f_neighbors.insert(ofs_cl1);
            }
            else
            {
                // two d_neighbors have at least one common node
                bool are_d_neighbors = false;
                for (size_t ip = 0; ip < cl1.ptids.size(); ip++)
                    for (size_t jp = 0; jp < cl2.ptids.size(); jp++)
                        if ( cl1.ptids[ip] == cl2.ptids[jp] )
                            are_d_neighbors = true;

                if (are_d_neighbors)
                {
                    auto ofs_cl1 = offset(msh, cl1);
                    auto ofs_cl2 = offset(msh, cl2);

                    cl1.user_data.d_neighbors.insert(ofs_cl2);
                    cl2.user_data.d_neighbors.insert(ofs_cl1);
                }

            }
        }
    }

    /*
    for (auto& cl : msh.cells)
    {
        for (auto& n : cl.user_data.neighbors)
            std::cout << n << " ";
        std::cout << std::endl;
    }
    */
}


//// version for cartesian meshes -> very quick
/* this creates Delta(T) */
// there are at least two row and two columns of cells
template<typename T, size_t ET>
void
make_neighbors_info_cartesian(cuthho_mesh<T, ET>& msh)
{
    std::cout << "WARNING : make_neighbors_info_cartesian "
              << "works for cartesian meshes only !!"
              << std::endl;

    size_t N = sqrt(msh.cells.size());

    //////////////////  face neighbors  ///////////////////
    // first row of cells -> look left
    for (size_t i = 1; i < N; i++)
    {
        auto &cl1 = msh.cells.at(i);
        auto &cl2 = msh.cells.at(i-1);

        cl1.user_data.f_neighbors.insert(i-1);
        cl2.user_data.f_neighbors.insert(i);
    }

    // other rows of cells
    for (size_t j = 1; j < N; j++)
    {
        // first cell of the row -> look bottom
        auto &cl1 = msh.cells.at( j*N );
        auto &cl2 = msh.cells.at( (j-1)*N );
        cl1.user_data.f_neighbors.insert( (j-1)*N );
        cl2.user_data.f_neighbors.insert( j*N );

        // other cells -> look left and bottom
        for (size_t i = 1; i < N; i++)
        {
            auto &cl_c = msh.cells.at( j*N + i ); // current
            auto &cl_l = msh.cells.at( j*N + i - 1 ); // left
            auto &cl_b = msh.cells.at( (j-1)*N + i ); // bottom

            cl_c.user_data.f_neighbors.insert( j*N + i - 1 );
            cl_c.user_data.f_neighbors.insert( (j-1)*N + i );

            cl_l.user_data.f_neighbors.insert( j*N + i );
            cl_b.user_data.f_neighbors.insert( j*N + i );
        }
    }

    //////////////////////  diagonal neighbors  //////////////////////
    // first row of cells -> look left top
    for (size_t i = 1; i < N; i++)
    {
        auto &cl1 = msh.cells.at(i);
        auto &cl2 = msh.cells.at(N + i-1);

        cl1.user_data.d_neighbors.insert(N + i-1);
        cl2.user_data.d_neighbors.insert(i);
    }

    // other rows of cells
    for (size_t j = 1; j < N-1; j++)
    {
        // first cell of the row -> nothing to do
        // other cells -> look left top and left bottom
        for (size_t i = 1; i < N; i++)
        {
            auto &cl_c = msh.cells.at( j*N + i ); // current
            auto &cl_l = msh.cells.at( (j+1)*N + i - 1 ); // left top
            auto &cl_b = msh.cells.at( (j-1)*N + i - 1 ); // left bottom

            cl_c.user_data.d_neighbors.insert( (j+1)*N + i - 1 );
            cl_c.user_data.d_neighbors.insert( (j-1)*N + i - 1 );

            cl_l.user_data.d_neighbors.insert( j*N + i );
            cl_b.user_data.d_neighbors.insert( j*N + i );
        }
    }

    // last row -> look left bottom
    for (size_t i = 1; i < N; i++)
    {
        auto &cl1 = msh.cells.at( (N-1)*N + i);
        auto &cl2 = msh.cells.at( (N-2)*N + i-1);

        cl1.user_data.d_neighbors.insert((N-2)*N + i-1);
        cl2.user_data.d_neighbors.insert((N-1)*N + i);
    }
}

//#define USE_OLD_DISPLACEMENT

#ifdef USE_OLD_DISPLACEMENT
template<typename T, size_t ET, typename Function>
void
move_nodes(cuthho_mesh<T, ET>& msh, const Function& level_set_function)
{
    typedef typename cuthho_mesh<T, ET>::face_type  face_type;
    typedef typename cuthho_mesh<T, ET>::point_type point_type;

    for (auto& fc : msh.faces)
    {
        if ( location(msh, fc) == element_location::ON_INTERFACE )
        {
            auto nds = nodes(msh, fc);
            auto pts = points(msh, fc);
            auto lf = (pts[1] - pts[0]).to_vector().norm();
            auto dp = (fc.user_data.intersection_point - pts[0]).to_vector().norm();

            auto closeness = dp/lf;

            typename cuthho_mesh<T, ET>::node_type ntc;

            if (closeness < 0.45) //pts[0] is too close
                ntc = nds[0];
            else if (closeness > 0.55) // pts[1] is too close
                ntc = nds[1];
            else // nothing to do
                continue;

            auto ofs = offset(msh, ntc);

            msh.nodes.at( ofs ).user_data.displaced = true;

            T displacement;

            if ( location(msh, ntc) == element_location::IN_NEGATIVE_SIDE )
                displacement = -std::abs(0.5-closeness)*lf*0.7;
            else
                displacement = std::abs(0.5-closeness)*lf*0.7;

            Matrix<T, 2, Dynamic> normal = level_set_function.normal(fc.user_data.intersection_point) * displacement;

            typename cuthho_mesh<T, ET>::point_type p({normal(0), normal(1)});

            msh.points.at(ofs) = msh.points.at(ofs) + p;
        }
    }

    for (auto& cl : msh.cells)
    {
        auto nds = nodes(msh, cl);
        for (auto& n : nds)
            if (n.user_data.displaced)
                cl.user_data.distorted = true;
    }

    for (auto& cl : msh.cells) /* Check we didn't generate concave cells */
    {
        if (!cl.user_data.distorted)
            continue;

        auto pts = points(msh, cl);

        if (pts.size() < 4)
            continue;

        for (size_t i = 0; i < pts.size(); i++)
        {
            auto pa = pts[i];
            auto pb = pts[(i+1)%pts.size()];
            auto pc = pts[(i+2)%pts.size()];
            auto v1 = (pb - pa);
            auto v2 = (pc - pb);
            auto cross = v1.x() * v2.y() - v2.x() * v1.y();

            if ( cross < 0 )
                std::cout << red << "[ !WARNING! ] A concave polygon was generated (cell " << offset(msh, cl) << ")." << nocolor << std::endl;
        }
    }
}

#else

template<typename T, size_t ET, typename Function>
void
move_nodes(cuthho_mesh<T, ET>& msh, const Function& level_set_function)
{
    typedef typename cuthho_mesh<T, ET>::face_type  face_type;
    typedef typename cuthho_mesh<T, ET>::point_type point_type;

    T closeness_thresh = 0.4;

    for (auto& fc : msh.faces)
    {
        if ( location(msh, fc) == element_location::ON_INTERFACE )
        {
            auto nds = nodes(msh, fc);
            auto pts = points(msh, fc);
            auto bar = (pts[1] + pts[0])/2.0;

            auto lf = (pts[1] - pts[0]).to_vector().norm();
            auto dp = (fc.user_data.intersection_point - pts[0]).to_vector().norm();

            auto closeness = dp/lf;

            typename cuthho_mesh<T, ET>::node_type ntc;

            if (closeness < closeness_thresh) //pts[0] is too close
                ntc = nds[0];
            else if (closeness > 1.0-closeness_thresh) // pts[1] is too close
                ntc = nds[1];
            else // nothing to do
                continue;

            auto ofs = offset(msh, ntc);
            auto delta = (bar - fc.user_data.intersection_point)/2;
            msh.nodes.at( ofs ).user_data.displacement = msh.nodes.at( ofs ).user_data.displacement - delta;
            msh.nodes.at( ofs ).user_data.displaced = true;
        }
    }

    for (auto& nd : msh.nodes)
        if (nd.user_data.displaced)
            msh.points.at( nd.ptid ) = msh.points.at( nd.ptid ) + nd.user_data.displacement;


    for (auto& cl : msh.cells)
    {
        auto nds = nodes(msh, cl);
        for (auto& n : nds)
            if (n.user_data.displaced)
                cl.user_data.distorted = true;
    }

    for (auto& cl : msh.cells) /* Check we didn't generate concave cells */
    {
        if (!cl.user_data.distorted)
            continue;

        auto pts = points(msh, cl);

        if (pts.size() < 4)
            continue;

        for (size_t i = 0; i < pts.size(); i++)
        {
            auto pa = pts[i];
            auto pb = pts[(i+1)%pts.size()];
            auto pc = pts[(i+2)%pts.size()];
            auto v1 = (pb - pa);
            auto v2 = (pc - pb);
            auto cross = v1.x() * v2.y() - v2.x() * v1.y();

            if ( cross < 0 )
            {
                std::cout << red << "[ !WARNING! ] A concave polygon was generated (cell " << offset(msh, cl) << ")." << nocolor << std::endl;
                throw std::logic_error("concave poly");
            }
        }
    }
}

#endif

template<typename T, size_t ET>
std::array< typename cuthho_mesh<T, ET>::point_type, 2 >
points(const cuthho_mesh<T, ET>& msh,
       const typename cuthho_mesh<T, ET>::face_type& fc,
       const element_location& where)
{
    if ( location(msh, fc) != where && location(msh, fc) != element_location::ON_INTERFACE )
        throw std::invalid_argument("This face has no points where requested");

    auto pts = points(msh, fc);
    if ( !is_cut(msh, fc) )
        return pts;

    auto nds = nodes(msh, fc);
    if ( location(msh, nds[0]) == where && location(msh, nds[1]) != where )
        pts[1] = fc.user_data.intersection_point;
    else if ( location(msh, nds[0]) != where && location(msh, nds[1]) == where )
        pts[0] = fc.user_data.intersection_point;
    else
        throw std::logic_error("Invalid point configuration");

    return pts;
}

template<typename T>
auto
barycenter(const std::vector< point<T, 2> >& pts)
{
    /*
    point<T, 2> ret;

    T den = 0.0;

    for (size_t i = 2; i < pts.size(); i++)
    {
        auto pprev  = pts[i-1]-pts[0];
        auto pcur   = pts[i]-pts[0];
        auto d      = det(pprev, pcur) / 2.0;
        ret         = ret + (pprev + pcur) * d;
        den         += d;
    }

    return pts[0] + ret/(den*3);
    */
    return barycenter(pts.begin(), pts.end());
}

template<typename T, size_t ET>
auto
barycenter(const cuthho_mesh<T, ET>& msh,
           const typename cuthho_mesh<T, ET>::cell_type& cl,
           element_location where)
{
    if ( !is_cut(msh, cl) )
        return barycenter(msh, cl);

    auto tp = collect_triangulation_points(msh, cl, where);
    auto bar = barycenter(tp);

    return bar;
}

template<typename T, size_t ET, typename Function>
void
refine_interface(cuthho_mesh<T, ET>& msh, typename cuthho_mesh<T, ET>::cell_type& cl,
                 const Function& level_set_function, size_t min, size_t max)
{
    if ( (max-min) < 2 )
        return;

    typedef typename cuthho_mesh<T, ET>::point_type     point_type;

    size_t mid = (max+min)/2;
    auto p0 = cl.user_data.interface.at(min);
    auto p1 = cl.user_data.interface.at(max);
    auto pm = (p0+p1)/2.0;
    auto pt = p1 - p0;
    auto pn = point_type(-pt.y(), pt.x());
    auto ps1 = pm + pn;
    auto ps2 = pm - pn;

    auto lm = level_set_function(pm);
    auto ls1 = level_set_function(ps1);
    auto ls2 = level_set_function(ps2);

    point_type ip;

    if ( !((lm >= 0 && ls1 >= 0) || (lm < 0 && ls1 < 0)) )
    {
        auto threshold = diameter(msh, cl) / 1e20;
        ip = find_zero_crossing(pm, ps1, level_set_function, threshold);
    }
    else if ( !((lm >= 0 && ls2 >= 0) || (lm < 0 && ls2 < 0)) )
    {
        auto threshold = diameter(msh, cl) / 1e20;
        ip = find_zero_crossing(pm, ps2, level_set_function, threshold);
    }
    else
        throw std::logic_error("interface not found in search range");

    cl.user_data.interface.at(mid) = ip;

    refine_interface(msh, cl, level_set_function, min, mid);
    refine_interface(msh, cl, level_set_function, mid, max);
}

template<typename T, size_t ET, typename Function>
void
refine_interface(cuthho_mesh<T, ET>& msh, const Function& level_set_function, size_t levels)
{
    if (levels == 0)
        return;

    size_t interface_points = iexp_pow(2, levels);

    for (auto& cl : msh.cells)
    {
        if ( !is_cut(msh, cl) )
            continue;

        cl.user_data.interface.resize(interface_points+1);
        cl.user_data.interface.at(0)                = cl.user_data.p0;
        cl.user_data.interface.at(interface_points) = cl.user_data.p1;

        refine_interface(msh, cl, level_set_function, 0, interface_points);
    }
}

template<typename T, size_t ET>
std::vector< typename cuthho_mesh<T, ET>::point_type >
collect_triangulation_points(const cuthho_mesh<T, ET>& msh,
                             const typename cuthho_mesh<T, ET>::cell_type& cl,
                             const element_location& where)
{
    typedef typename cuthho_mesh<T, ET>::point_type     point_type;
    typedef typename cuthho_mesh<T, ET>::node_type      node_type;

    assert( is_cut(msh, cl) );
    auto ns = nodes(msh, cl);

    std::vector< point_type > ret;

    auto node2pt = [&](const node_type& n) -> auto {
        return msh.points.at(n.ptid);
    };

    auto insert_interface = [&](void) -> void {
        if (where == element_location::IN_NEGATIVE_SIDE)
            ret.insert(ret.end(), cl.user_data.interface.begin(), cl.user_data.interface.end());
        else if (where == element_location::IN_POSITIVE_SIDE)
            ret.insert(ret.end(), cl.user_data.interface.rbegin(), cl.user_data.interface.rend());
        else
            throw std::logic_error("If you've got here there is some issue...");
    };

    bool case1 = location(msh, ns.front()) == where && location(msh, ns.back()) != where;
    bool case2 = location(msh, ns.front()) != where && location(msh, ns.back()) == where;
    bool case3 = location(msh, ns.front()) != where && location(msh, ns.back()) != where;
    //bool case4 = location(msh, ns.front()) == where && location(msh, ns.back()) == where;

    if ( case1 || case2 || case3 )
    {
        for (size_t i = 0; i < ns.size(); i++)
            if ( location(msh, ns[i]) == where )
                ret.push_back( node2pt(ns[i]) );

        insert_interface();
    }
    else // case4, the only possible remaining
    {
        size_t i = 0;
        while ( i < ns.size() && location(msh, ns.at(i)) == where )
            ret.push_back( node2pt(ns.at(i++)) );
        insert_interface();
        while ( i < ns.size() && location(msh, ns.at(i)) != where )
            i++;
        while ( i < ns.size() && location(msh, ns.at(i)) == where )
            ret.push_back( node2pt(ns.at(i++)) );
    }

    return ret;
}


template<typename T>
struct temp_tri
{
    std::array< point<T,2>, 3 > pts;

    T area() const {
        auto v1 = pts[1] - pts[0];
        auto v2 = pts[2] - pts[0];

        return ( v1.x()*v2.y() - v2.x()*v1.y() ) / 2.0;
        // can be negative
    }
};

template<typename T>
std::ostream&
operator<<(std::ostream& os, const temp_tri<T>& t)
{
    os << "line([" << t.pts[0].x() << "," << t.pts[1].x() << "],[" << t.pts[0].y() << "," << t.pts[1].y() << "]);" << std::endl;
    os << "line([" << t.pts[1].x() << "," << t.pts[2].x() << "],[" << t.pts[1].y() << "," << t.pts[2].y() << "]);" << std::endl;
    os << "line([" << t.pts[2].x() << "," << t.pts[0].x() << "],[" << t.pts[2].y() << "," << t.pts[0].y() << "]);" << std::endl;
    return os;
}


template<typename T, size_t ET>
typename cuthho_mesh<T, ET>::point_type
tesselation_center(const cuthho_mesh<T, ET>& msh, const typename cuthho_mesh<T, ET>::cell_type& cl,
                    element_location where)
{
    auto fcs = faces(msh, cl);
    auto pts = points(msh, cl);
    auto nds = nodes(msh, cl);

    if (fcs.size() != 4)
        throw std::invalid_argument("This works only on quads for now");

    if( !is_cut(msh, cl) )
        throw std::invalid_argument("No tesselation centers for uncut cells");

    // if two consecutive faces are cut
    // return either the common node or the opposite node
    for (size_t i = 0; i < fcs.size(); i++)
    {
        auto f1 = i;
        auto f2 = (i+1) % fcs.size();
        auto n = (i+1) % fcs.size();

        if ( is_cut(msh, fcs[f1]) && is_cut(msh, fcs[f2]) )
        {
            if ( location(msh, nds[n]) == where )
                return pts[n];
            else
                return pts[(n+2)%4];
        }
    }

    // if two opposite faces are cut
    // return the center of one of the other faces
    for (size_t i = 0; i < 2; i++)
    {
        auto f1 = i;
        auto f2 = i+2;
        auto n = i+1;

        if ( is_cut(msh, fcs[f1]) && is_cut(msh, fcs[f2]) )
        {
            if( location(msh, nds[n]) == where )
                return 0.5*(pts[n] + pts[n+1]);
            else
                return 0.5*(pts[(n+2)%4] + pts[(n+3)%4]);
        }
    }

    // normally the tesselation center is already found
    throw std::logic_error("we shouldn't arrive here !!");
}

template<typename T, size_t ET>
std::vector<temp_tri<T>>
triangulate(const cuthho_mesh<T, ET>& msh, const typename cuthho_mesh<T, ET>::cell_type& cl,
            element_location where)
{
    assert( is_cut(msh, cl) );

    auto tp = collect_triangulation_points(msh, cl, where);
    // auto bar = barycenter(tp);
    auto bar = tesselation_center(msh, cl, where);

    std::vector<temp_tri<T>> tris;

    for (size_t i = 0; i < tp.size(); i++)
    {
        temp_tri<T> t;
        t.pts[0] = bar;
        t.pts[1] = tp[i];
        t.pts[2] = tp[(i+1)%tp.size()];

        tris.push_back(t);
    }

    return tris;
}

template<typename T, size_t ET>
T
measure(const cuthho_mesh<T, ET>& msh, const typename cuthho_mesh<T, ET>::cell_type& cl,
        const element_location& where)
{
    if ( !is_cut(msh, cl) ) /* Element is not cut, use std. integration */
        return measure(msh, cl);

    std::vector< std::pair<point<T,2>, T> > ret;
    auto tris = triangulate(msh, cl, where);

    T totmeas = 0.0;

    for (auto& tri : tris)
        totmeas += tri.area();

    return totmeas;
}

template<typename T, size_t ET>
std::vector< std::pair<point<T,2>, T> >
make_integrate(const cuthho_mesh<T, ET>& msh, const typename cuthho_mesh<T, ET>::cell_type& cl,
               size_t degree, const element_location& where)
{
    std::vector< std::pair<point<T,2>, T> > ret;

    if ( location(msh, cl) != where && location(msh, cl) != element_location::ON_INTERFACE )
        return ret;

    if ( !is_cut(msh, cl) ) /* Element is not cut, use std. integration */
        return integrate(msh, cl, degree);

    auto tris = triangulate(msh, cl, where);
    for (auto& tri : tris)
    {
        auto qpts = triangle_quadrature(tri.pts[0], tri.pts[1], tri.pts[2], degree);
        ret.insert(ret.end(), qpts.begin(), qpts.end());
    }

    return ret;
}


template<typename T, size_t ET>
std::vector< std::pair<point<T,2>, T> >
integrate(const cuthho_mesh<T, ET>& msh, const typename cuthho_mesh<T, ET>::cell_type& cl,
          size_t degree, const element_location& where)
{
    if( cl.user_data.integration_n.size() != 0 && where == element_location::IN_NEGATIVE_SIDE)
        return cl.user_data.integration_n;

    if( cl.user_data.integration_p.size() != 0 && where == element_location::IN_POSITIVE_SIDE)
        return cl.user_data.integration_p;

    return make_integrate(msh, cl, degree, where);
}




template<typename T, size_t ET>
std::vector< std::pair<point<T,2>, T> >
integrate(const cuthho_mesh<T, ET>& msh, const typename cuthho_mesh<T, ET>::face_type& fc,
          size_t degree, const element_location& where)
{
    std::vector< std::pair<point<T,2>, T> > ret;

    if ( location(msh, fc) != where && location(msh, fc) != element_location::ON_INTERFACE )
        return ret;

    if ( !is_cut(msh, fc) ) /* Element is not cut, use std. integration */
        return integrate(msh, fc, degree);

    auto pts = points(msh, fc, where);

    auto scale = pts[1] - pts[0];
    auto meas = scale.to_vector().norm();

    auto qps = edge_quadrature<T>(degree);

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

template<typename T, size_t ET>
std::vector< std::pair<point<T,2>, T> >
integrate_interface(const cuthho_mesh<T, ET>& msh, const typename cuthho_mesh<T, ET>::cell_type& cl,
                    size_t degree, element_location where)
{
    assert( is_cut(msh, cl) );

    typedef typename cuthho_mesh<T, ET>::point_type point_type;

    std::vector< std::pair<point<T,2>, T> > ret;

    auto pa = cl.user_data.interface.at(0);
    auto pb = cl.user_data.interface.at(1);
    auto bar = barycenter(msh, cl, where);

    auto va = (pa - bar).to_vector();
    auto vb_temp = pb - pa;
    auto vb = point_type({vb_temp.y(), -vb_temp.x()}).to_vector();

    auto int_sign = va.dot(vb) < 0 ? -1.0 : +1.0;

    auto qps = edge_quadrature<T>(degree);

    for (size_t i = 1; i < cl.user_data.interface.size(); i++)
    {
        auto p0 = cl.user_data.interface.at(i-1);
        auto p1 = cl.user_data.interface.at(i);

        auto scale = p1 - p0;
        auto meas = scale.to_vector().norm();

        for (auto itor = qps.begin(); itor != qps.end(); itor++)
        {
            auto qp = *itor;
            //auto p = qp.first.x() * scale + p0;
            auto t = qp.first.x();
            auto p = 0.5*(1-t)*p0 + 0.5*(1+t)*p1;
            auto w = int_sign * qp.second * meas * 0.5;

            ret.push_back( std::make_pair(p, w) );
        }
    }

    return ret;
}


template<typename T, typename Function>
std::vector< typename cuthho_quad_mesh<T>::point_type >
make_test_points(const cuthho_quad_mesh<T>& msh, const typename cuthho_quad_mesh<T>::cell_type& cl,
                 const Function& level_set_function, element_location where)
{
    const size_t N = 10;

    auto trans = make_reference_transform(msh, cl);

    std::vector< typename cuthho_quad_mesh<T>::point_type > ret;

    auto cell_pts = points(msh, cl);

    auto min = -1.0;
    auto h = 2.0/N;

    for (size_t j = 0; j < N+1; j++)
    {
        for(size_t i = 0; i < N+1; i++)
        {
            auto p_xi = min + i*h;
            auto p_eta = min + j*h;
            typename cuthho_quad_mesh<T>::point_type pt(p_xi, p_eta);

            auto phys_pt = trans.ref_to_phys(pt);

            if ( level_set_function(phys_pt) < 0 && where == element_location::IN_NEGATIVE_SIDE)
              ret.push_back( phys_pt );
            else if ( level_set_function(phys_pt) > 0 && where == element_location::IN_POSITIVE_SIDE)
              ret.push_back( phys_pt );
        }
    }

    return ret;
}




template<typename T, size_t ET>
void dump_mesh(const cuthho_mesh<T, ET>& msh)
{
    std::ofstream ofs("mesh_dump.m");
    size_t i = 0;
    ofs << "hold on;" << std::endl;
    for (auto& fc : msh.faces)
    {
        auto pts = points(msh, fc);
        if (fc.is_boundary)
            ofs << "line([" << pts[0].x() << ", " << pts[1].x() << "], [" << pts[0].y() << ", " << pts[1].y() << "], 'Color', 'r');" << std::endl;
        else if ( is_cut(msh, fc) )
            ofs << "line([" << pts[0].x() << ", " << pts[1].x() << "], [" << pts[0].y() << ", " << pts[1].y() << "], 'Color', 'g');" << std::endl;
        else
            ofs << "line([" << pts[0].x() << ", " << pts[1].x() << "], [" << pts[0].y() << ", " << pts[1].y() << "], 'Color', 'k');" << std::endl;

        auto bar = barycenter(msh, fc);
        ofs << "text(" << bar.x() << ", " << bar.y() << ", '" << i << "');" << std::endl;

        i++;
    }

    i = 0;
    for (auto& cl : msh.cells)
    {
        auto bar = barycenter(msh, cl);
        ofs << "text(" << bar.x() << ", " << bar.y() << ", '" << i << "');" << std::endl;

        if ( is_cut(msh, cl) )
        {
            auto p0 = cl.user_data.p0;
            auto p1 = cl.user_data.p1;
            auto q = p1 - p0;
            ofs << "quiver(" << p0.x() << ", " << p0.y() << ", " << q.x() << ", " << q.y() << ", 0)" << std::endl;

            for (size_t i = 1; i < cl.user_data.interface.size(); i++)
            {
                auto s = cl.user_data.interface.at(i-1);
                auto l = cl.user_data.interface.at(i) - s;
                ofs << "quiver(" << s.x() << ", " << s.y() << ", " << l.x() << ", " << l.y() << ", 0)" << std::endl;
            }

            for (auto& ip : cl.user_data.interface)
                ofs << "plot(" << ip.x() << ", " << ip.y() << ", '*k');" << std::endl;

            auto tpn = collect_triangulation_points(msh, cl, element_location::IN_NEGATIVE_SIDE);
            auto bn = barycenter(tpn);
            ofs << "plot(" << bn.x() << ", " << bn.y() << ", 'dr');" << std::endl;

            auto tpp = collect_triangulation_points(msh, cl, element_location::IN_POSITIVE_SIDE);
            auto bp = barycenter(tpp);
            ofs << "plot(" << bp.x() << ", " << bp.y() << ", 'db');" << std::endl;

        }
        i++;
    }

    ofs << "t = linspace(0,2*pi,1000);plot(sqrt(0.15)*cos(t)+0.5, sqrt(0.15)*sin(t)+0.5)" << std::endl;

    ofs.close();
}


#if 0

template<typename T, typename ET>
class assembler<cuthho_mesh<T, ET>>
{
    std::vector<size_t>                 cell_compress_table, face_compress_table;
    std::vector<size_t>                 cell_expand_table, face_expand_table;

    hho_degree_info                     di;

    std::vector< Triplet<T> >           triplets;

    size_t num_all_cells, num_asm_cells, num_notasm_cells;
    size_t num_all_faces, num_asm_faces, num_notasm_faces;

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

    bool cell_needs_assembly(const cuthho_mesh<T, ET>& msh,
                             const typename cuthho_mesh<T, ET>::cell_type& cl)
    {
        return location(msh, cl) == element_location::IN_NEGATIVE_SIDE ||
               location(msh, cl) == element_location::ON_INTERFACE;
    }

    bool face_needs_assembly(const cuthho_mesh<T, ET>& msh,
                             const typename cuthho_mesh<T, ET>::face_type& fc)
    {
        return location(msh, fc) == element_location::IN_NEGATIVE_SIDE ||
               location(msh, fc) == element_location::ON_INTERFACE;
    }

public:

    SparseMatrix<T>         LHS;
    Matrix<T, Dynamic, 1>   RHS;

    assembler(const cuthho_mesh<T, ET>& msh, hho_degree_info hdi)
        : di(hdi)
    {
        auto fna = [&](const typename cuthho_mesh<T, ET>::face_type& fc) -> bool {
            return face_needs_assembly(msh, fc);
        };

        num_all_faces = msh.faces.size();
        num_asm_faces = std::count_if(msh.faces.begin(), msh.faces.end(), fna);
        num_notasm_faces = num_all_faces - num_asm_faces;

        auto cna = [&](const typename cuthho_mesh<T, ET>::cell_type& cl) -> bool {
            return cell_needs_assembly(msh, cl);
        }
        ;
        num_all_cells = msh.cells.size();
        num_asm_cells = std::count_if(msh.cells.begin(), msh.cells.end(), cna);
        num_notasm_cells = num_all_cells - num_asm_cells;

        cell_compress_table.resize( num_all_cells );
        cell_expand_table.resize( num_asm_cells );
        face_compress_table.resize( num_all_faces );
        face_expand_table.resize( num_asm_faces );

        size_t compressed_offset = 0;
        for (size_t i = 0; i < num_all_cells; i++)
        {
            auto cl = msh.cells[i];
            if ( cell_needs_assembly(msh, cl) )
            {
                cell_compress_table.at(i) = compressed_offset;
                cell_expand_table.at(compressed_offset) = i;
                compressed_offset++;
            }
        }

        compressed_offset = 0;
        for (size_t i = 0; i < num_all_faces; i++)
        {
            auto fc = msh.faces[i];
            if ( face_needs_assembly(msh, fc) )
            {
                face_compress_table.at(i) = compressed_offset;
                face_expand_table.at(compressed_offset) = i;
                compressed_offset++;
            }
        }

        auto celdeg = di.cell_degree();
        auto facdeg = di.face_degree();

        auto cbs = cell_basis<cuthho_mesh<T, ET>,T>::size(celdeg);
        auto fbs = face_basis<cuthho_mesh<T, ET>,T>::size(facdeg);

        auto system_size = cbs * num_asm_cells + fbs * num_asm_faces;

        LHS = SparseMatrix<T>( system_size, system_size );
        RHS = Matrix<T, Dynamic, 1>::Zero( system_size );
    }

    void dump_tables() const
    {
        std::cout << "Compress table: " << std::endl;
        for (size_t i = 0; i < cell_compress_table.size(); i++)
            std::cout << i << " -> " << cell_compress_table.at(i) << std::endl;
    }

    template<typename Function>
    void
    assemble(const cuthho_mesh<T, ET>& msh, const typename cuthho_mesh<T, ET>::cell_type& cl,
             const Matrix<T, Dynamic, Dynamic>& lhs, const Matrix<T, Dynamic, 1>& rhs,
             const Function& dirichlet_bf)
    {
        if ( !cell_needs_assembly(msh, cl) )
            return;

        auto celdeg = di.cell_degree();
        auto facdeg = di.face_degree();

        auto cbs = cell_basis<cuthho_mesh<T, ET>,T>::size(celdeg);
        auto fbs = face_basis<cuthho_mesh<T, ET>,T>::size(facdeg);

        std::vector<assembly_index> asm_map;
        asm_map.reserve(cbs + 4*fbs);

        auto cell_offset        = offset(msh, cl);
        auto cell_LHS_offset    = cell_compress_table.at(cell_offset) * cbs;

        for (size_t i = 0; i < cbs; i++)
            asm_map.push_back( assembly_index(cell_LHS_offset+i, true) );

        auto fcs = faces(msh, cl);
        for (size_t face_i = 0; face_i < 4; face_i++)
        {
            auto fc = fcs[face_i];
            auto face_offset = offset(msh, fc);
            auto face_LHS_offset = cbs * num_asm_cells + face_compress_table.at(face_offset)*fbs;

            bool asm_face = face_needs_assembly(msh, fc);

            for (size_t i = 0; i < fbs; i++)
                asm_map.push_back( assembly_index(face_LHS_offset+i, asm_face) );
        }

        assert( asm_map.size() == lhs.rows() && asm_map.size() == lhs.cols() );

        for (size_t i = 0; i < lhs.rows(); i++)
        {
            if (!asm_map[i].assemble())
                continue;

            for (size_t j = 0; j < lhs.cols(); j++)
                if ( asm_map[j].assemble() )
                    triplets.push_back( Triplet<T>(asm_map[i], asm_map[j], lhs(i,j)) );
        }

        RHS.block(cell_LHS_offset, 0, cbs, 1) += rhs.block(0, 0, cbs, 1);
    } // assemble()

    template<typename Function>
    Matrix<T, Dynamic, 1>
    take_local_data(const cuthho_mesh<T, ET>& msh, const typename cuthho_mesh<T, ET>::cell_type& cl,
    const Matrix<T, Dynamic, 1>& solution, const Function& dirichlet_bf)
    {
        auto celdeg = di.cell_degree();
        auto facdeg = di.face_degree();

        auto cbs = cell_basis<cuthho_mesh<T, ET>,T>::size(celdeg);
        auto fbs = face_basis<cuthho_mesh<T, ET>,T>::size(facdeg);

        if ( !cell_needs_assembly(msh, cl) )
        {
            Matrix<T, Dynamic, 1> ret = Matrix<T, Dynamic, 1>::Zero(cbs + 4*fbs);
            return ret;
        }

        auto cell_offset        = offset(msh, cl);
        auto cell_SOL_offset    = cell_compress_table.at(cell_offset) * cbs;

        Matrix<T, Dynamic, 1> ret = Matrix<T, Dynamic, 1>::Zero(cbs + 4*fbs);
        ret.block(0, 0, cbs, 1) = solution.block(cell_SOL_offset, 0, cbs, 1);

        auto fcs = faces(msh, cl);
        for (size_t face_i = 0; face_i < 4; face_i++)
        {
            auto fc = fcs[face_i];

            if ( face_needs_assembly(msh, fc) )
            {
                auto face_offset = offset(msh, fc);
                auto face_SOL_offset = cbs * num_asm_cells + face_compress_table.at(face_offset)*fbs;
                ret.block(cbs+face_i*fbs, 0, fbs, 1) = solution.block(face_SOL_offset, 0, fbs, 1);
            }
        }

        return ret;
    }

    Matrix<T, Dynamic, 1>
    expand_solution(const cuthho_mesh<T, ET>& msh,
                    const Matrix<T, Dynamic, 1>& solution)
    {
        auto celdeg = di.cell_degree();
        auto facdeg = di.face_degree();

        auto cbs = cell_basis<cuthho_mesh<T, ET>,T>::size(celdeg);
        auto fbs = face_basis<cuthho_mesh<T, ET>,T>::size(facdeg);

        auto solsize = msh.cells.size() * cbs + msh.faces.size() * fbs;

        Matrix<T, Dynamic, 1> ret = Matrix<T, Dynamic, 1>::Zero(solsize);

        for (size_t i = 0; i < num_asm_cells; i++)
        {
            auto exp_index = cell_expand_table.at(i);
            ret.block(exp_index*cbs, 0, cbs, 1) = solution.block(i*cbs, 0, cbs, 1);
        }

        return ret;
    }

    void finalize(void)
    {
        LHS.setFromTriplets( triplets.begin(), triplets.end() );
        triplets.clear();
    }
};


template<typename T, typename ET>
auto make_assembler(const cuthho_mesh<T, ET>& msh, hho_degree_info hdi)
{
    return assembler<cuthho_mesh<T, ET>>(msh, hdi);
}

#endif


///////////////////   AGGLOMERATION   ///////////////////

//////////////  MERGE_CELLS_FACE
/// merge cl1 and cl2 through the common face fc
// output : new cell
template<typename cell_type, typename face_type>
cell_type
merge_cells_face(cell_type cl1, cell_type cl2, face_type com_f)
{
    cell_type ret;

    size_t f_pt1 = com_f.ptids[0];
    size_t f_pt2 = com_f.ptids[1];

    // list of points
    std::vector<size_t> pts1, pts2;
    // in order to be consistent with the cell representation, start with the smallest index
    if(cl1.ptids[0] < cl2.ptids[0])
    {
        pts1 = cl1.ptids;
        pts2 = cl2.ptids;
    }
    else
    {
        pts1 = cl2.ptids;
        pts2 = cl1.ptids;
    }

    // write points of pts1 until we reach the common face
    size_t ref_pt = pts1[0];
    size_t cp = 0;
    ret.ptids.push_back(ref_pt);

    bool on_face = false;
    if(ref_pt == f_pt1 || ref_pt == f_pt2) on_face = true;

    while(!on_face)
    {
        cp++;
        ref_pt = pts1[cp];
        ret.ptids.push_back(ref_pt);
        if(ref_pt == f_pt1 || ref_pt == f_pt2) on_face = true;
    }

    // look for the corresponding point in pts2
    for(size_t i = 0; i < pts2.size(); i++)
    {
        if(ref_pt == pts2[i])
        {
            cp = i;
            break;
        }
    }
    // write points of pts2 until we reach once more the common face
    on_face = false;
    while( !on_face )
    {
        cp = (cp + 1) % pts2.size();
        ref_pt = pts2[cp];
        ret.ptids.push_back(ref_pt);
        if(ref_pt == f_pt1 || ref_pt == f_pt2) on_face = true;
    }
    // look for the corresponding point in pts1
    for(size_t i=0; i < pts1.size(); i++)
    {
        if(ref_pt == pts1[i])
        {
            cp = i;
            break;
        }
    }

    // finish to write the points of pts1
    cp++;
    while(cp < pts1.size())
    {
        ret.ptids.push_back(pts1[cp]);
        cp++;
    }

    return ret;
}



//////////////  MERGE_CELLS_TWO_FACES
/// merge cl1 and cl2 through the list com_f of common faces
// output : new cell
template<typename cell_type, typename face_type>
cell_type
merge_cells_two_faces(cell_type cl1, cell_type cl2, typename std::vector<face_type> com_f)
{
    cell_type ret;

    assert( com_f.size() == 2 );

    size_t f1_pt1 = com_f[0].ptids[0];
    size_t f1_pt2 = com_f[0].ptids[1];
    size_t f2_pt1 = com_f[1].ptids[0];
    size_t f2_pt2 = com_f[1].ptids[1];

    std::set<size_t> com_pts;
    if(f1_pt1 == f2_pt1 || f1_pt2 == f2_pt1)
    {
        com_pts.insert(f1_pt1);
        com_pts.insert(f1_pt2);
        com_pts.insert(f2_pt2);
    }
    else if(f1_pt1 == f2_pt2 || f1_pt2 == f2_pt2)
    {
        com_pts.insert(f1_pt1);
        com_pts.insert(f1_pt2);
        com_pts.insert(f2_pt1);
    }
    else
        throw std::logic_error("com_f : no common points");

    // list of points
    std::vector<size_t> pts1, pts2;
    // in order to be consistent with the cell representation, start with the smallest index
    if(cl1.ptids[0] < cl2.ptids[0])
    {
        pts1 = cl1.ptids;
        pts2 = cl2.ptids;
    }
    else
    {
        pts1 = cl2.ptids;
        pts2 = cl1.ptids;
    }

    // write points of pts1 until we reach the common face
    size_t ref_pt = pts1[0];
    size_t next_pt= pts1[1];
    size_t cp = 0;
    ret.ptids.push_back(ref_pt);

    bool on_face = false;
    if( com_pts.find(ref_pt) != com_pts.end() && com_pts.find(next_pt) != com_pts.end() )
        on_face = true;

    while(!on_face)
    {
        cp++;
        ref_pt = next_pt;
        next_pt= pts1[(cp+1)%pts1.size()];
        ret.ptids.push_back(ref_pt);
        if( com_pts.find(ref_pt) != com_pts.end() && com_pts.find(next_pt) != com_pts.end() )
            on_face = true;
    }

    // look for the corresponding point in pts2
    for(size_t i = 0; i < pts2.size(); i++)
    {
        if(ref_pt == pts2[i])
        {
            cp = i;
            break;
        }
    }

    // write points of pts2 until we reach once more the common face
    on_face = false;
    while( !on_face )
    {
        cp = (cp + 1) % pts2.size();
        ref_pt = pts2[cp];
        next_pt= pts2[(cp+1) % pts2.size()];
        ret.ptids.push_back(ref_pt);
        if( com_pts.find(ref_pt) != com_pts.end() && com_pts.find(next_pt) != com_pts.end() )
            on_face = true;
    }
    // look for the corresponding point in pts1
    for(size_t i=0; i < pts1.size(); i++)
    {
        if(ref_pt == pts1[i])
        {
            cp = i;
            break;
        }
    }

    // finish to write the points of pts1
    cp++;
    while(cp < pts1.size())
    {
        ret.ptids.push_back(pts1[cp]);
        cp++;
    }

    return ret;
}


//////////////  MERGE_CELLS_DIAG
/// merge cl1 and cl2 through the common node com_n
// output : new cell
template<typename cell_type>
cell_type
merge_cells_diag(cell_type cl1, cell_type cl2, size_t com_n)
{
    // abort the process if both cells are on the interface
    // (this case is not yet supported for the list of points on the interface)
    if( cl1.user_data.location == element_location::ON_INTERFACE
        && cl2.user_data.location == element_location::ON_INTERFACE )
        throw std::invalid_argument("Cannot merge diagonally two cells on the interface.");

    cell_type ret;

    // list of points
    std::vector<size_t> pts1, pts2;
    // in order to be consistent with the cell representation, start with the smallest index
    if(cl1.ptids[0] < cl2.ptids[0])
    {
        pts1 = cl1.ptids;
        pts2 = cl2.ptids;
    }
    else
    {
        pts1 = cl2.ptids;
        pts2 = cl1.ptids;
    }

    // write points of pts1 until we reach the common face
    size_t ref_pt = pts1[0];
    size_t cp = 0;
    ret.ptids.push_back(ref_pt);

    bool on_interface = false;
    if(ref_pt == com_n) on_interface = true;

    while(!on_interface)
    {
        cp++;
        ref_pt = pts1[cp];
        ret.ptids.push_back(ref_pt);
        if(ref_pt == com_n) on_interface = true;
    }

    // look for the corresponding point in pts2
    for(size_t i = 0; i < pts2.size(); i++)
    {
        if(ref_pt == pts2[i])
        {
            cp = i;
            break;
        }
    }
    // write points of pts2 until we reach once more the common node
    on_interface = false;
    while( !on_interface )
    {
        cp = (cp + 1) % pts2.size();
        ref_pt = pts2[cp];
        ret.ptids.push_back(ref_pt);
        if(ref_pt == com_n) on_interface = true;
    }

    // look for the corresponding point in pts1
    for(size_t i=0; i < pts1.size(); i++)
    {
        if(ref_pt == pts1[i])
        {
            cp = i;
            break;
        }
    }

    // finish to write the points of pts1
    cp++;
    while(cp < pts1.size())
    {
        ret.ptids.push_back(pts1[cp]);
        cp++;
    }

    return ret;
}

//////////////  MERGE_CELLS
/// merge cl1 and cl2
// output : the agglomerated cell + list of faces to withdraw
/////  For the moment we can merge only cells that have at least a common face
/////  This procedure currently cannot be iterated
template<typename Mesh>
std::pair<   typename Mesh::cell_type, std::vector<typename Mesh::face_type> >
merge_cells(Mesh& msh, const typename Mesh::cell_type cl1,
            const typename Mesh::cell_type cl2)
{
    //////////////////  TESTS ON INPUTS  //////////////////
    // verify that the two cells are different
    if(cl1 == cl2)
        throw std::invalid_argument("Cannot merge a cell with itself.");

    // identify the common faces
    const auto fcs1 = faces(msh, cl1);
    const auto fcs2 = faces(msh, cl2);

    std::vector<typename Mesh::face_type> com_faces;
    for(size_t i = 0; i < fcs1.size(); i++)
    {
        const auto fc1 = fcs1[i];
        for(size_t j = 0; j < fcs2.size(); j++)
        {
            const auto fc2 = fcs2[j];
            if(fc1 == fc2) com_faces.push_back(fc1);
        }
    }

    // identify the common nodes
    std::set<size_t> com_nodes;
    size_t com_n;
    for(size_t i = 0; i < cl1.ptids.size(); i++)
    {
        for(size_t j = 0; j < cl2.ptids.size(); j++)
        {
            if(cl1.ptids[i] == cl2.ptids[j])
            {
                com_nodes.insert(cl1.ptids[i]);
                com_n = cl1.ptids[i];
            }
        }
    }

    // choose the agglomeration technique
    typename Mesh::cell_type cl;
    if(com_faces.size() == 0)
    {
        std::cout << "com nodes nb = " << com_nodes.size() << std::endl;
        if( com_nodes.size() == 1 )
        {
            cl = merge_cells_diag(cl1, cl2, com_n);
        }
        else
            throw std::invalid_argument("The cells have no common faces.");
    }
    else if( com_faces.size() == 1 )
    {
        assert(com_nodes.size() == 2); // only the nodes of the common face
        cl = merge_cells_face(cl1, cl2, com_faces[0]);
    }
    else if( com_faces.size() == 2 )
    {
        std::cout << "com nodes nb = " << com_nodes.size() << std::endl;
        assert(com_nodes.size() == 3); // only the case with two adjascent common faces
        cl = merge_cells_two_faces(cl1, cl2, com_faces);
    }
    if(com_faces.size() > 2)
        throw std::invalid_argument("The cells have too many common faces.");

    //////////////    COMPLETE THE MERGED CELL   //////////////

    ///////////// build the cell_cuthho_info (using the sub_cells info)
    // location
    if(cl1.user_data.location == element_location::ON_INTERFACE || cl2.user_data.location == element_location::ON_INTERFACE)
        cl.user_data.location = element_location::ON_INTERFACE;
    else if(cl1.user_data.location == element_location::IN_NEGATIVE_SIDE)
        cl.user_data.location = element_location::IN_NEGATIVE_SIDE;
    else if(cl1.user_data.location == element_location::IN_POSITIVE_SIDE)
        cl.user_data.location = element_location::IN_POSITIVE_SIDE;
    else
        throw std::logic_error("we shouldn't arrive here (cuthho_info) !!!");

    // agglo_set ---> NOT DONE : needed to iterate the agglomeration procedure
    cl.user_data.agglo_set = cell_agglo_set::T_OK;

    // p0, p1 and interface
    bool cut1 = cl1.user_data.location == element_location::ON_INTERFACE;
    bool cut2 = cl2.user_data.location == element_location::ON_INTERFACE;
    if(cut1 && !cut2)
    {
        cl.user_data.interface = cl1.user_data.interface;
        cl.user_data.p0 = cl1.user_data.p0;
        cl.user_data.p1 = cl1.user_data.p1;
    }
    else if(!cut1 && cut2)
    {
        cl.user_data.interface = cl2.user_data.interface;
        cl.user_data.p0 = cl2.user_data.p0;
        cl.user_data.p1 = cl2.user_data.p1;
    }
    else if(cut1 && cut2)
    {
        if(cl1.user_data.p0[0] == cl2.user_data.p1[0] &&
           cl1.user_data.p0[1] == cl2.user_data.p1[1])
        {
            cl.user_data.interface = cl2.user_data.interface;
            for(size_t i = 0; i < cl1.user_data.interface.size(); i++ )
            {
                cl.user_data.interface.push_back(cl1.user_data.interface[i]);
            }
            cl.user_data.p0 = cl2.user_data.p0;
            cl.user_data.p1 = cl1.user_data.p1;
        }
        else if(cl2.user_data.p0[0] == cl1.user_data.p1[0] &&
                cl2.user_data.p0[1] == cl1.user_data.p1[1])
        {
            cl.user_data.interface = cl1.user_data.interface;
            for(size_t i = 0; i < cl2.user_data.interface.size(); i++ )
            {
                cl.user_data.interface.push_back(cl2.user_data.interface[i]);
            }
            cl.user_data.p0 = cl1.user_data.p0;
            cl.user_data.p1 = cl2.user_data.p1;
        }
        else
            throw std::logic_error("we shouldn't arrive here (interface) !!!");
    }
    // distorted -> has to be updated for more general merges (if a node is withdrawn)
    if(cl1.user_data.distorted || cl2.user_data.distorted )
        cl.user_data.distorted = true;


    // for tests
    cl.user_data.highlight = true;


    // integration -> save composite quadrature
    size_t degree_max = 8; //////// VERY IMPORTANT !!!!!!! -> max deg for quadratures = 8
    auto integration1_n = integrate(msh, cl1, degree_max, element_location::IN_NEGATIVE_SIDE);
    auto integration1_p = integrate(msh, cl1, degree_max, element_location::IN_POSITIVE_SIDE);

    auto integration2_n = integrate(msh, cl2, degree_max, element_location::IN_NEGATIVE_SIDE);
    auto integration2_p = integrate(msh, cl2, degree_max, element_location::IN_POSITIVE_SIDE);

    cl.user_data.integration_n = integration1_n;
    cl.user_data.integration_p = integration1_p;
    for(size_t i = 0; i < integration2_n.size(); i++)
        cl.user_data.integration_n.push_back( integration2_n.at(i) );

    for(size_t i = 0; i < integration2_p.size(); i++)
        cl.user_data.integration_p.push_back( integration2_p.at(i) );

    // neighbors ---> NOT DONE : needed to iterate the agglomeration procedure

    return std::make_pair(cl, com_faces);
}


////// loc_agglo
// container for information about local agglomerations
// main_cell is defined when they are at least 3 cells
template<typename cell_type>
class loc_agglo
{
public:
    cell_type main_cell;
    std::vector<cell_type> cells;

    cell_type new_cell;

    loc_agglo(cell_type cl1, cell_type cl2, cell_type n_cell)
    {
        // check that the two cells are neighbors
        auto pts1 = cl1.ptids;
        auto pts2 = cl2.ptids;
        std::vector<size_t> com_nodes;
        for(size_t i = 0; i < pts1.size(); i++)
        {
            for(size_t j = 0; j < pts2.size(); j++)
            {
                if(pts1[i] == pts2[j]) com_nodes.push_back(pts1[i]);
            }
        }

        assert( com_nodes.size() > 0 );

        // init
        cells.push_back( cl1 );
        cells.push_back( cl2 );

        new_cell = n_cell;
    }

    bool is_agglo_possible(size_t offset)
    {
        // when there are 2 cells, the main cell is not defined -> check all cells
        if( cells.size() == 2 )
        {
            return (cells[0].user_data.f_neighbors.find(offset) != cells[0].user_data.f_neighbors.end())
                || (cells[1].user_data.f_neighbors.find(offset) != cells[1].user_data.f_neighbors.end())
                || (cells[0].user_data.d_neighbors.find(offset) != cells[0].user_data.d_neighbors.end())
                || (cells[1].user_data.d_neighbors.find(offset) != cells[1].user_data.d_neighbors.end());
        }

        // else the main cell is defined -> can agglomerate only if it is a neighbor of that cell
        return (main_cell.user_data.f_neighbors.find(offset) != main_cell.user_data.f_neighbors.end())
            || (main_cell.user_data.d_neighbors.find(offset) != main_cell.user_data.d_neighbors.end());
    }

    bool is_in(cell_type cl)
    {
        bool ret = false;
        for(size_t i = 0; i < cells.size(); i++)
        {
            if( cells[i] == cl)
            {
                ret = true;
                break;
            }
        }
        return ret;
    }

    void add_cell(cell_type cl_added, size_t offset, cell_type new_new_cell)
    {
        assert( is_agglo_possible(offset) );
        
        
        // check that the cell is not already in the agglomeration
        if( is_in(cl_added) )
            throw std::logic_error("Cell already agglomerated !!");
            
        // if there are only two cells, we also have to define the main cell
        if( cells.size() == 2 )
        {
            if ( cells[0].user_data.f_neighbors.find(offset) != cells[0].user_data.f_neighbors.end()
                 || cells[0].user_data.d_neighbors.find(offset) != cells[0].user_data.d_neighbors.end() )
                main_cell = cells[0];
            else
                main_cell = cells[1];
        }

        cells.push_back(cl_added);
        new_cell = new_new_cell;
    }
};

////// find_good_neighbor
template<typename Mesh>
typename Mesh::cell_type
find_good_neighbor(const Mesh& msh, const typename Mesh::cell_type cl,
                   const element_location where,
                   std::vector< loc_agglo<typename Mesh::cell_type> > loc_agglos)
{
    typename Mesh::cell_type best_neigh = cl;



    // look for a neighboring cell to merge with
    typename Mesh::coordinate_type area = 1000000;

    auto f_neigh = cl.user_data.f_neighbors;

    for (std::set<size_t>::iterator it = f_neigh.begin(); it != f_neigh.end(); ++it)
    {
        auto cl_n = msh.cells[*it];

        // if cl_n is on the wrong size -> do not consider it
        if (location(msh, cl_n) != where
            && location(msh, cl_n) != element_location::ON_INTERFACE)
            continue;


        // if cl_n is a small cut of the same size -> do not consider it
        if (where == element_location::IN_NEGATIVE_SIDE && cl_n.user_data.agglo_set == cell_agglo_set::T_KO_NEG)
            continue;
        if (where == element_location::IN_POSITIVE_SIDE && cl_n.user_data.agglo_set == cell_agglo_set::T_KO_POS)
            continue;

        // if cl_n is already agglomerated : check if further agglomerations are possible
        bool agglo_possible = true;
        for (size_t i = 0; i < loc_agglos.size(); i++)
        {
            if( loc_agglos.at(i).is_in(cl_n) )
            {
                agglo_possible = loc_agglos.at(i).is_agglo_possible( offset(msh, cl) );
                break;
            }
        }
        if( !agglo_possible )
            continue;

        // search for the "best" neighbor -> the one with the smallest volume
        if( area > measure(msh, cl_n, where) )
        {
            area = measure(msh, cl_n, where);
            best_neigh = cl_n;
        }
    }

    if(best_neigh == cl) // no possible face agglomerations
    {
        // look for a diagonal agglomeration
        auto d_neigh = cl.user_data.d_neighbors;

        for (std::set<size_t>::iterator it = d_neigh.begin(); it != d_neigh.end(); ++it)
        {
            auto cl_n = msh.cells[*it];

            // if cl_n is on the wrong size or cut -> do not consider it
            if (location(msh, cl_n) != where)
                continue;


            // if cl_n is already agglomerated : check if further agglomerations are possible
            bool agglo_possible = true;
            for (size_t i = 0; i < loc_agglos.size(); i++)
            {
                if( loc_agglos.at(i).is_in(cl_n) )
                {
                    agglo_possible = loc_agglos.at(i).is_agglo_possible( offset(msh, cl) );
                    break;
                }
            }
            if( !agglo_possible )
                continue;

            // search for the "best" neighbor -> the one with the smallest volume
            if( area > measure(msh, cl_n, where) )
            {
                area = measure(msh, cl_n, where);
                best_neigh = cl_n;
            }
        }
    }

    if(best_neigh == cl)
    {
        throw std::logic_error("No possible agglomerations !!!");
    }

    
    return best_neigh;
}

/////// check_corner
// do another agglomeration if the agglomerated cell is cut four times
// needed for the case of a square interface
// cl_agglo : a cell that is already agglomerated with other cells
// cl : a cell that is not yet agglomerated
// return: a boolean: whether another agglomeration is needed
//         a cell: the cell added to the agglomeration
template<typename Mesh>
std::pair< bool, typename Mesh::cell_type >
check_corner(Mesh& msh, typename Mesh::cell_type cl_agglo, typename Mesh::cell_type cl)
{
    typename Mesh::cell_type ret_cl;
    
    // check is another agglo is needed
    bool need_other_agglo = true;

    if( cl_agglo.user_data.p0.x() == cl.user_data.p1.x()
        && cl_agglo.user_data.p0.y() == cl.user_data.p1.y() )
        need_other_agglo = false;
    if( cl_agglo.user_data.p1.x() == cl.user_data.p0.x()
        && cl_agglo.user_data.p1.y() == cl.user_data.p0.y() )
        need_other_agglo = false;
    if( cl_agglo.user_data.p0.x() == cl.user_data.p0.x()
        && cl_agglo.user_data.p0.y() == cl.user_data.p0.y() )
        need_other_agglo = false;
    if( cl_agglo.user_data.p1.x() == cl.user_data.p1.x()
        && cl_agglo.user_data.p1.y() == cl.user_data.p1.y() )
        need_other_agglo = false;

    if(!need_other_agglo)
        return std::make_pair(false, ret_cl);

    std::cout << "agglomerate one more cell !!" << std::endl;

    // look for a cell that shares two intersection points with cl_agglo and cl
    // this cell is searched among the face neighbors
    auto f_neigh = cl.user_data.f_neighbors;
    typename Mesh::cell_type added_cell;
    bool found_added_cell = false;
    size_t offset_added_cell;
    for (std::set<size_t>::iterator it = f_neigh.begin(); it != f_neigh.end(); ++it)
    {
        auto cl_f = msh.cells[*it];
        if(location(msh, cl_f) != element_location::ON_INTERFACE)
            continue;

        if( (  cl_f.user_data.p0.x() == cl.user_data.p0.x()
               && cl_f.user_data.p0.y() == cl.user_data.p0.y() )
            || (cl_f.user_data.p1.x() == cl.user_data.p0.x()
                && cl_f.user_data.p1.y() == cl.user_data.p0.y() )
            || (cl_f.user_data.p0.x() == cl.user_data.p1.x()
                && cl_f.user_data.p0.y() == cl.user_data.p1.y() )
            || (cl_f.user_data.p1.x() == cl.user_data.p1.x()
                && cl_f.user_data.p1.y() == cl.user_data.p1.y() )
            )
        { // this cell is connected with cl
            if( (  cl_f.user_data.p0.x() == cl_agglo.user_data.p0.x()
                   && cl_f.user_data.p0.y() == cl_agglo.user_data.p0.y() )
                || (cl_f.user_data.p1.x() == cl_agglo.user_data.p0.x()
                    && cl_f.user_data.p1.y() == cl_agglo.user_data.p0.y() )
                || (cl_f.user_data.p0.x() == cl_agglo.user_data.p1.x()
                    && cl_f.user_data.p0.y() == cl_agglo.user_data.p1.y() )
                || (cl_f.user_data.p1.x() == cl_agglo.user_data.p1.x()
                    && cl_f.user_data.p1.y() == cl_agglo.user_data.p1.y() )
                )
            { // this cell is connected with cl_agglo
                added_cell = cl_f;
                offset_added_cell = *it;
                found_added_cell = true;
                break;
            }
        }
    }
    
    if(!found_added_cell)    
        throw std::logic_error("added_cell not found !!");

    return std::make_pair(true, added_cell);

}

////// make_agglomeration
// the main agglomeration routine
// currently, the mesh obtained must have convex cells
// obtained by merging sub_cells with one face in common
// for non-convex cells -> modify measure and integrate
// for merging more general cells -> modify merge_cells
// for diagonal cells -> modify make_neighbors_info
// no iterations after merging bad cells once
template<typename Mesh, typename Function>
void
make_agglomeration(Mesh& msh, const Function& level_set_function)
{
    std::vector<typename Mesh::face_type> removed_faces;
    std::vector<typename Mesh::cell_type> removed_cells;
    std::vector< loc_agglo<typename Mesh::cell_type> > loc_agglos;

    // initiate lists to store the agglomeration infos
    std::vector<int> agglo_table_neg, agglo_table_pos;
    size_t nb_cells = msh.cells.size();
    agglo_table_neg.resize(nb_cells);
    agglo_table_pos.resize(nb_cells);
    
    for(size_t i=0; i < nb_cells; i++)
    {
        agglo_table_neg.at(i) = -1;
        agglo_table_pos.at(i) = -1;
    }
    
    // start the process for domain 1, and then domain 2
    for(size_t domain=1; domain < 3; domain++)
    {
        // loop on the cells
        for (auto cl : msh.cells)
        {
            element_location where;

            if(domain == 1)
                where = element_location::IN_NEGATIVE_SIDE;
            else if(domain == 2)
                where = element_location::IN_POSITIVE_SIDE;
            else 
                throw std::logic_error("pb with domain");
                
            if(cl.user_data.agglo_set == cell_agglo_set::T_OK)
                continue;
            else if(cl.user_data.agglo_set == cell_agglo_set::UNDEF)
                throw std::logic_error("UNDEF agglo_set");
            else if(cl.user_data.agglo_set == cell_agglo_set::T_KO_NEG
                    && where != element_location::IN_NEGATIVE_SIDE)
                continue;
            else if(cl.user_data.agglo_set == cell_agglo_set::T_KO_POS
                    && where != element_location::IN_POSITIVE_SIDE)
                continue;


            // if cl is already agglomerated : no need for further agglomerations
            bool already_agglo = false;
            for (size_t i = 0; i < loc_agglos.size(); i++)
            {
                if( loc_agglos.at(i).is_in(cl) )
                {
                    already_agglo = true;
                    break;
                }
            }
            if( already_agglo )
                continue;


            typename Mesh::cell_type best_neigh = find_good_neighbor(msh, cl, where, loc_agglos);

            auto f_neigh = cl.user_data.f_neighbors;

            // prepare agglomeration of cells cl and best_neigh
            size_t offset1 = offset(msh,cl);
            size_t offset2 = offset(msh,best_neigh);

            if(where == element_location::IN_NEGATIVE_SIDE)
                agglo_table_neg.at(offset1) = offset2;
            else
                agglo_table_pos.at(offset1) = offset2;

            // check if best_neigh is already agglomerated
            int agglo_offset = -1;
            for (size_t i = 0; i < loc_agglos.size(); i++)
            {
                if( loc_agglos.at(i).is_in(best_neigh) )
                {
                    agglo_offset = i;
                    break;
                }
            }

            // if best_neigh is not agglomerated -> create a new agglomeration
            if( agglo_offset == -1 )
            {
                auto MC = merge_cells(msh, cl, best_neigh);

                loc_agglos.push_back( loc_agglo<typename Mesh::cell_type>(cl, best_neigh, MC.first) );

                for(size_t i=0; i<MC.second.size(); i++)
                {
                    removed_faces.push_back( MC.second[i] );
                }
                removed_cells.push_back(cl);
                removed_cells.push_back(best_neigh);
            }
            else // else : use the previous agglomeration
            {
                // if the merged cell is cut four times -> need to agglomerate one more cell
                bool need_other_agglo = true;
                auto NC = loc_agglos.at(agglo_offset).new_cell;
                if( NC.user_data.p0.x() == cl.user_data.p1.x()
                    && NC.user_data.p0.y() == cl.user_data.p1.y() )
                    need_other_agglo = false;
                if( NC.user_data.p1.x() == cl.user_data.p0.x()
                    && NC.user_data.p1.y() == cl.user_data.p0.y() )
                    need_other_agglo = false;
                if( NC.user_data.p0.x() == cl.user_data.p0.x()
                    && NC.user_data.p0.y() == cl.user_data.p0.y() )
                    need_other_agglo = false;
                if( NC.user_data.p1.x() == cl.user_data.p1.x()
                    && NC.user_data.p1.y() == cl.user_data.p1.y() )
                    need_other_agglo = false;

                if( need_other_agglo )
                {
                    std::cout << "agglomerate one more cell !!" << std::endl;

                    // look for a cell that share two intersection points with cl and NC
                    // this cell is searched among the face neighbors
                    typename Mesh::cell_type cell_agglo;
                    bool found_cell_agglo = false;
                    size_t offset_cell_bis;
                    for (std::set<size_t>::iterator it = f_neigh.begin(); it != f_neigh.end(); ++it)
                    {
                        auto cl_f = msh.cells[*it];
                        if(location(msh, cl_f) != element_location::ON_INTERFACE)
                            continue;

                        if( (  cl_f.user_data.p0.x() == cl.user_data.p0.x()
                               && cl_f.user_data.p0.y() == cl.user_data.p0.y() )
                            || (cl_f.user_data.p1.x() == cl.user_data.p0.x()
                                && cl_f.user_data.p1.y() == cl.user_data.p0.y() )
                            || (cl_f.user_data.p0.x() == cl.user_data.p1.x()
                                && cl_f.user_data.p0.y() == cl.user_data.p1.y() )
                            || (cl_f.user_data.p1.x() == cl.user_data.p1.x()
                                && cl_f.user_data.p1.y() == cl.user_data.p1.y() )
                            )
                        { // this cell is connected with cl
                            if( (  cl_f.user_data.p0.x() == NC.user_data.p0.x()
                                   && cl_f.user_data.p0.y() == NC.user_data.p0.y() )
                                || (cl_f.user_data.p1.x() == NC.user_data.p0.x()
                                    && cl_f.user_data.p1.y() == NC.user_data.p0.y() )
                                || (cl_f.user_data.p0.x() == NC.user_data.p1.x()
                                    && cl_f.user_data.p0.y() == NC.user_data.p1.y() )
                                || (cl_f.user_data.p1.x() == NC.user_data.p1.x()
                                    && cl_f.user_data.p1.y() == NC.user_data.p1.y() )
                                )
                            { // this cell is connected with NC
                                cell_agglo = cl_f;
                                offset_cell_bis = *it;
                                found_cell_agglo = true;
                                break;
                            }
                        }
                    }

                    if(found_cell_agglo)
                    {
                        std::cout << "cell_agglo_found" << std::endl;
                        auto MC_bis = merge_cells(msh, cell_agglo,
                                                  loc_agglos.at(agglo_offset).new_cell);
                        loc_agglos.at(agglo_offset).add_cell(cell_agglo,
                                                             offset_cell_bis, MC_bis.first);


                        for(size_t i=0; i<MC_bis.second.size(); i++)
                        {
                            removed_faces.push_back( MC_bis.second[i] );
                        }
                        removed_cells.push_back(cell_agglo);
                    }
                    else
                        throw std::logic_error("Cell agglo not found !!");

                }

                auto MC = merge_cells(msh, cl, loc_agglos.at(agglo_offset).new_cell);

                loc_agglos.at(agglo_offset).add_cell(cl, offset1, MC.first);

                for(size_t i=0; i<MC.second.size(); i++)
                {
                    removed_faces.push_back( MC.second[i] );
                }
                removed_cells.push_back(cl);
            }
        }
    }
    //////////////   CHANGE THE AGGLO FOR THE CELLS OF DOMAIN 1 THAT ARE TARGETTED ///////
    for (auto cl : msh.cells)
    {
        if(cl.user_data.agglo_set != cell_agglo_set::T_KO_POS)
            continue;

        size_t offset1 = offset(msh,cl);

        // are there cells that try to agglomerate with cl ?
        bool agglo = false;
        size_t cl2_offset;
        for(size_t i = 0; i < agglo_table_neg.size(); i++)
        {
            if(agglo_table_neg.at(i) == offset1)
            {
                agglo = true;
                cl2_offset = i;
                break;
            }
        }
        if(!agglo)
            continue;

        // at this point cl2_offset tries to agglomerate with cl
        size_t cl1_agglo = agglo_table_pos.at(offset1);
        
        // -> check that no one tries to agglomerate with cl1_agglo
        agglo = false;
        for(size_t i = 0; i < agglo_table_pos.size(); i++)
        {
            if(agglo_table_pos.at(i) == cl1_agglo)
            {
                agglo = true;
                break;
            }
        }
        if(!agglo)
            continue;

        // at this point we risk chain agglomerations
        // -> change the target of cl
        // agglo_table_pos.at(offset1) = cl2_offset;
        agglo_table_pos.at(offset1) = -1;
    }
    
    ///////////////////  BUILD LOCAL AGGLOMERATIONS  //////////////////
    std::vector< loc_agglo<typename Mesh::cell_type> > loc_agglos_bis;
    std::vector<typename Mesh::face_type> removed_faces_bis;
    std::vector<typename Mesh::cell_type> removed_cells_bis;
    for (auto cl : msh.cells)
    {
        auto offset_cl = offset(msh, cl);

        bool agglo = false;
        size_t offset_neigh;
        if(agglo_table_pos.at(offset_cl) != -1)
        {
            offset_neigh = agglo_table_pos.at(offset_cl);
            agglo = true;
        }
        if(agglo_table_neg.at(offset_cl) != -1)
        {
            offset_neigh = agglo_table_neg.at(offset_cl);
            agglo = true;
        }

        if(!agglo)
            continue;

        typename Mesh::cell_type neigh = msh.cells.at(offset_neigh);

        
        // test if one of the two cells is already agglomerated
        size_t agglo_offset;
        bool already_agglo_cl = false;
        for (size_t i = 0; i < loc_agglos_bis.size(); i++)
        {
            if( loc_agglos_bis.at(i).is_in(cl) )
            {
                already_agglo_cl = true;
                agglo_offset = i;
                break;
            }
        }
        bool already_agglo_neigh = false;
        for (size_t i = 0; i < loc_agglos_bis.size(); i++)
        {
            if( loc_agglos_bis.at(i).is_in(neigh) )
            {
                already_agglo_neigh = true;
                agglo_offset = i;
                break;
            }
        }

        // the two cells can not be both already agglomerated 
        if(already_agglo_cl && already_agglo_neigh)
            throw std::logic_error("Both cells already agglomerated !!");
        else if(!already_agglo_cl && !already_agglo_neigh)
        {
            // create a new local agglomeration
            auto MC = merge_cells(msh, cl, neigh);

            loc_agglos_bis.push_back( loc_agglo<typename Mesh::cell_type>(cl, neigh, MC.first) );

            for(size_t i=0; i<MC.second.size(); i++)
            {
                removed_faces_bis.push_back( MC.second[i] );
            }
            removed_cells_bis.push_back(cl);
            removed_cells_bis.push_back(neigh);
        }
        else // only one cell is already agglomerated
        {
            typename Mesh::cell_type cl1, cl2;
            size_t offset_cl2;
            if(already_agglo_cl)
            {
                removed_cells_bis.push_back(neigh);
                cl2 = neigh;
                offset_cl2 = offset_neigh;
            }
            else if(already_agglo_neigh)
            {
                removed_cells_bis.push_back(cl);
                cl2 = cl;
                offset_cl2 = offset_cl;
            }

            // get the agglomerated cell
            cl1 = loc_agglos_bis.at(agglo_offset).new_cell;
            
            // check if we need one more agglomeration here
            auto CC = check_corner(msh, cl1, cl2);
            if(CC.first)
            {
                auto offset_added_cell = offset(msh, CC.second);

                auto MC_bis = merge_cells(msh, CC.second, cl1);
                loc_agglos_bis.at(agglo_offset).add_cell(CC.second, offset_added_cell,
                                                         MC_bis.first);
                for(size_t i=0; i<MC_bis.second.size(); i++)
                {
                    removed_faces_bis.push_back( MC_bis.second[i] );
                }
                removed_cells_bis.push_back(CC.second);
            }

            // end the merge procedure
            auto MC = merge_cells(msh, cl2, loc_agglos_bis.at(agglo_offset).new_cell);

            loc_agglos_bis.at(agglo_offset).add_cell(cl2, offset_cl2, MC.first);

            for(size_t i=0; i<MC.second.size(); i++)
            {
                removed_faces_bis.push_back( MC.second[i] );
            }
        }
    }
    
    //////////////////////   UPDATE THE MESH   ////////////////////////

    // remove the agglomerated cells
    typename std::vector<typename Mesh::cell_type>::iterator it_RC;
    for(it_RC = removed_cells.begin(); it_RC != removed_cells.end(); it_RC++) {
        msh.cells.erase(std::remove(begin(msh.cells), end(msh.cells), *it_RC ), end(msh.cells));
    }
    
    // // remove the agglomerated cells
    // typename std::vector<typename Mesh::cell_type>::iterator it_RC;
    // for(it_RC = removed_cells_bis.begin(); it_RC != removed_cells_bis.end(); it_RC++) {
    //     msh.cells.erase(std::remove(begin(msh.cells), end(msh.cells), *it_RC ), end(msh.cells));
    // }

    // add new cells
    for (size_t i = 0; i < loc_agglos.size(); i++)
    {
        msh.cells.push_back(loc_agglos.at(i).new_cell);
    }

    
    // // add new cells
    // for (size_t i = 0; i < loc_agglos_bis.size(); i++)
    // {
    //     msh.cells.push_back(loc_agglos_bis.at(i).new_cell);
    // }

    // sort the new list of cells
    std::sort(msh.cells.begin(), msh.cells.end());

    // remove faces
    typename std::vector<typename Mesh::face_type>::iterator it_RF;
    for(it_RF = removed_faces.begin(); it_RF != removed_faces.end(); it_RF++) {
        msh.faces.erase(std::remove(begin(msh.faces), end(msh.faces), *it_RF ), end(msh.faces));
    }

    
    // // remove faces
    // typename std::vector<typename Mesh::face_type>::iterator it_RF;
    // for(it_RF = removed_faces_bis.begin(); it_RF != removed_faces_bis.end(); it_RF++) {
    //     msh.faces.erase(std::remove(begin(msh.faces), end(msh.faces), *it_RF ), end(msh.faces));
    // }

    // sort the new list of faces
    std::sort(msh.faces.begin(), msh.faces.end());
}


/////////////  OLD VERSION
// ////// make_agglomeration
// // the main agglomeration routine
// // currently, the mesh obtained must have convex cells
// // obtained by merging sub_cells with one face in common
// // for non-convex cells -> modify measure and integrate
// // for merging more general cells -> modify merge_cells
// // for diagonal cells -> modify make_neighbors_info
// // no iterations after merging bad cells once
// template<typename Mesh, typename Function>
// void
// make_agglomeration(Mesh& msh, const Function& level_set_function)
// {
//     std::vector<typename Mesh::face_type> removed_faces;
//     std::vector<int> old_cells;
//     std::vector<typename Mesh::cell_type> new_cells, removed_cells;

//     std::vector< loc_agglo<typename Mesh::cell_type> > loc_agglos;

//     old_cells.reserve(msh.cells.size());
//     for(size_t i=0; i<msh.cells.size(); i++)
//         old_cells.push_back(-1);

//     // loop on the cells
//     for (auto cl : msh.cells)
//     {
//         element_location where;

//         if(cl.user_data.agglo_set == cell_agglo_set::T_OK)
//             continue;
//         else if(cl.user_data.agglo_set == cell_agglo_set::UNDEF)
//             throw std::logic_error("UNDEF agglo_set");
//         else if(cl.user_data.agglo_set == cell_agglo_set::T_KO_NEG)
//             where = element_location::IN_NEGATIVE_SIDE;
//         else if(cl.user_data.agglo_set == cell_agglo_set::T_KO_POS)
//             where = element_location::IN_POSITIVE_SIDE;

//         // look for a neighboring cell to merge with
//         typename Mesh::coordinate_type area = 1000000;
//         typename Mesh::cell_type best_neigh = cl;

//         auto neigh = cl.user_data.neighbors;

//         for (std::set<size_t>::iterator it = neigh.begin(); it != neigh.end(); ++it)
//         {
//             auto cl_n = msh.cells[*it];

//             if (location(msh, cl_n) != where
//                 && location(msh, cl_n) != element_location::ON_INTERFACE)
//                 continue;

//             // search for the "best" neighbor -> the one with the smallest volume
//             if( area > measure(msh, cl_n, where) )
//             {
//                 area = measure(msh, cl_n, where);
//                 best_neigh = cl_n;
//             }
//         }

//         if(best_neigh == cl)
//         {
//             throw std::logic_error("No possible agglomerations !!!");
//         }

//         // prepare agglomeration of cells cl and best_neigh
//         size_t offset1 = offset(msh,cl);
//         size_t offset2 = offset(msh,best_neigh);
//         if(old_cells[offset1] == -1 && old_cells[offset2] == -1)
//         { // those cells are not yet agglomerated
//             old_cells[offset1] = new_cells.size();
//             old_cells[offset2] = new_cells.size();

//             removed_cells.push_back(cl);
//             removed_cells.push_back(best_neigh);

//             auto MC = merge_cells(msh, cl, best_neigh);

//             new_cells.push_back( MC.first );

//             for(size_t i=0; i<MC.second.size(); i++)
//             {
//                 removed_faces.push_back( MC.second[i] );
//             }
//         }
//         else
//         { // at least one cell has already been agglomerated

//             // if the desired agglomeration is already done
//             // -> nothing to do
//             if( old_cells[offset1] == old_cells[offset2] )
//                 continue;

//             std::vector<int> cells_agglo;
//             if (old_cells[offset1] != -1)
//             {
//                 cl = new_cells[ old_cells[offset1] ];
//                 cells_agglo.push_back( old_cells[offset1] );
//             }
//             if (old_cells[offset2] != -1)
//             {
//                 best_neigh = new_cells[ old_cells[offset2] ];
//                 cells_agglo.push_back( old_cells[offset2] );
//             }

//             if( cells_agglo.size() == 0 || cells_agglo.size() > 2 )
//                 throw std::logic_error("we shouldn't arrive here (prepare agglomeration) !!!");

//             if( cells_agglo.size() == 2 )
//             { // both cells are already agglomerated

//                 // for convenience, we make sure we delete the cell with the larger offset
//                 if( cells_agglo[0] > cells_agglo[1] )
//                 {
//                     size_t max = cells_agglo[0];
//                     cells_agglo[0] = cells_agglo[1];
//                     cells_agglo[1] = max;
//                 }

//                 // the cells pointing on the second new cell must now point on the first one
//                 for(size_t i=0; i<msh.cells.size(); i++)
//                 {
//                     size_t oc = old_cells[i];
//                     if( oc == cells_agglo[1] )
//                     {
//                         old_cells[i] == cells_agglo[0];
//                     }
//                 }

//                 // remove the second new cell
//                 new_cells.erase(std::remove(begin(new_cells), end(new_cells),
//                                             new_cells[ cells_agglo[1] ]), end(new_cells));

//                 // the indices in the list have to be updated
//                 for(size_t i=0; i<msh.cells.size(); i++)
//                 {
//                     if(old_cells[i] > cells_agglo[1])
//                         old_cells[i] = old_cells[i] - 1;
//                 }

//             }
//             else // only one cell is already agglomerated
//             {
//                 if( old_cells[offset1] == cells_agglo[0])
//                 {
//                     removed_cells.push_back( best_neigh );
//                     old_cells[offset2] = cells_agglo[0];
//                 }
//                 else if( old_cells[offset2] == cells_agglo[0])
//                 {
//                     removed_cells.push_back( cl );
//                     old_cells[offset1] = cells_agglo[0];
//                 }
//             }

//             auto MC = merge_cells(msh, cl, best_neigh);

//             new_cells[ cells_agglo[0] ] = MC.first;

//             for(size_t i=0; i<MC.second.size(); i++)
//             {
//                 removed_faces.push_back( MC.second[i] );
//             }
//         }
//     }

//     //////////////   UPDATE THE MESH   //////////

//     // remove the agglomerated cells
//     typename std::vector<typename Mesh::cell_type>::iterator it_RC;
//     for(it_RC = removed_cells.begin(); it_RC != removed_cells.end(); it_RC++) {
//         msh.cells.erase(std::remove(begin(msh.cells), end(msh.cells), *it_RC ), end(msh.cells));
//     }

//     // add new cells
//     typename std::vector<typename Mesh::cell_type>::iterator it_NC;
//     for(it_NC = new_cells.begin(); it_NC != new_cells.end(); it_NC++) {
//         msh.cells.push_back(*it_NC);
//     }

//     // sort the new list of cells
//     std::sort(msh.cells.begin(), msh.cells.end());

//     // remove faces
//     typename std::vector<typename Mesh::face_type>::iterator it_RF;
//     for(it_RF = removed_faces.begin(); it_RF != removed_faces.end(); it_RF++) {
//         msh.faces.erase(std::remove(begin(msh.faces), end(msh.faces), *it_RF ), end(msh.faces));
//     }

//     // sort the new list of faces
//     std::sort(msh.faces.begin(), msh.faces.end());
// }


//////////////////////////  CUT BASIS  ///////////////////////////


template<typename Mesh, typename VT>
class cut_face_basis
{
    typedef typename Mesh::coordinate_type  coordinate_type;
    typedef typename Mesh::point_type       point_type;

    point_type          face_bar;
    point_type          base;
    coordinate_type     face_h;
    size_t              basis_degree, basis_size;

public:
    cut_face_basis(const Mesh& msh, const typename Mesh::face_type& fc, size_t degree,
                   element_location where)
    {
        auto loc = location(msh,fc);
        if( loc != where && loc != element_location::ON_INTERFACE)
        {
            face_bar        = barycenter(msh, fc);
            face_h          = diameter(msh, fc);
            basis_degree    = degree;
            basis_size      = degree+1;

            auto pts = points(msh, fc);
            base = face_bar - pts[0];
        }
        else
        {
            face_bar        = barycenter(msh, fc, where);
            face_h          = diameter(msh, fc, where);
            basis_degree    = degree;
            basis_size      = degree+1;

            auto pts = points(msh, fc, where);
            base = face_bar - pts[0];
        }
    }
    Matrix<VT, Dynamic, 1>
    eval_basis(const point_type& pt)
    {
        Matrix<VT, Dynamic, 1> ret = Matrix<VT, Dynamic, 1>::Zero(basis_size);

        auto v = base.to_vector();
        auto t = (pt - face_bar).to_vector();
        auto dot = v.dot(t);
        auto ep = 4.0*dot/(face_h*face_h);

        coordinate_type coeff = sqrt(face_h / 2.0);

        ret(0) = sqrt(1.0 / 2.0) / coeff;
        if( basis_degree == 0)
            return ret;

        ret(1) = ep * sqrt(3.0 / 2.0) / coeff;
        if( basis_degree == 1)
            return ret;

        ret(2) = (3*ep*ep - 1) * sqrt(5.0/8.0) / coeff;
        if( basis_degree == 2)
            return ret;

        ret(3) = (5*ep*ep*ep - 3*ep) * sqrt(7.0 / 8.0) / coeff;
        if( basis_degree == 3)
            return ret;

        throw std::logic_error("bases : we shouldn't be here");
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



template<typename T, size_t ET>
auto
barycenter(const cuthho_mesh<T, ET>& msh,
           const typename cuthho_mesh<T, ET>::face_type& fc,
           element_location where)
{
    if ( !is_cut(msh, fc) )
        return barycenter(msh, fc);

    auto tp = points(msh, fc, where);

    point<T, 2> bar;
    bar[0] = 0.5 * (tp[0][0] + tp[1][0]);
    bar[1] = 0.5 * (tp[0][1] + tp[1][1]);

    return bar;
}


template<typename T, size_t ET>
auto
diameter(const cuthho_mesh<T, ET>& msh,
           const typename cuthho_mesh<T, ET>::face_type& fc,
           element_location where)
{
    if ( !is_cut(msh, fc) )
        return diameter(msh, fc);

    auto tp = points(msh, fc, where);
    auto vect = (tp[0] - tp[1]).to_vector();
    T ret = vect.norm();
    return ret;
}
