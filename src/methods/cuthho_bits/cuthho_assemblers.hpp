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

/////////////////////////////  ASSEMBLY  INDEX ///////////////////////
// used in all assemblers
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

//////////////////////////////   FICTITIOUS DOMAIN ASSEMBLERS   /////////////////////////////

template<typename Mesh, typename Function>
class virt_fict_assembler
{
    using T = typename Mesh::coordinate_type;

protected:
    std::vector< Triplet<T> >           triplets;
    std::vector<size_t>                 compress_face_table;
    std::vector<size_t>                 compress_cells_table;

    hho_degree_info                     di;
    Function                            dir_func;

    element_location loc_zone;
    size_t num_cells, num_other_faces, loc_cbs;

public:

    SparseMatrix<T>         LHS;
    Matrix<T, Dynamic, 1>   RHS;


    virt_fict_assembler(const Mesh& msh, const Function& dirichlet_bf,
                        hho_degree_info hdi, element_location where)
        : dir_func(dirichlet_bf), di(hdi), loc_zone(where)
    {
        auto is_removed = [&](const typename Mesh::face_type& fc) -> bool {
            bool is_dirichlet = fc.is_boundary && fc.bndtype == boundary::DIRICHLET;
            auto loc = location(msh,fc);
            bool is_where = (loc == where || loc == element_location::ON_INTERFACE);
            return is_dirichlet || (!is_where);
        };

        auto num_all_faces = msh.faces.size();
        auto num_removed_faces = std::count_if(msh.faces.begin(), msh.faces.end(), is_removed);
        num_other_faces = num_all_faces - num_removed_faces;

        compress_face_table.resize( num_all_faces );

        size_t compressed_offset = 0;
        for (size_t i = 0; i < num_all_faces; i++)
        {
            auto fc = msh.faces[i];
            if ( !is_removed(fc) )
            {
                compress_face_table.at(i) = compressed_offset;
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
        num_cells = compressed_offset;
    }

    size_t
    face_SOL_offset(const Mesh& msh, const typename Mesh::cell_type& cl,
                    const typename Mesh::face_type& fc)
    {
        auto facdeg = di.face_degree();
        auto fbs = face_basis<Mesh,T>::size(facdeg);

        auto fcs = faces(msh, cl);
        auto num_faces = fcs.size();
        auto f_dofs = num_faces * fbs;

        auto cbs = loc_cbs; // cbs = 0 if static condensation

        auto face_offset = offset(msh, fc);
        return num_cells * cbs + compress_face_table.at(face_offset) * fbs;
    }

    std::vector<assembly_index>
    init_asm_map(const Mesh& msh, const typename Mesh::cell_type& cl)
    {
        std::vector<assembly_index> asm_map;

        auto facdeg = di.face_degree();
        auto fbs = face_basis<Mesh,T>::size(facdeg);

        auto fcs = faces(msh, cl);
        auto num_faces = fcs.size();
        auto f_dofs = num_faces * fbs;
        auto cbs = loc_cbs;
        auto loc_size = cbs + f_dofs;
        asm_map.reserve( loc_size );

        size_t cell_offset = compress_cells_table.at( offset(msh, cl) );
        size_t cell_LHS_offset = cell_offset * cbs;

        for (size_t i = 0; i < cbs; i++)
            asm_map.push_back( assembly_index(cell_LHS_offset+i, true) );


        for (size_t face_i = 0; face_i < num_faces; face_i++)
        {
            auto fc = fcs[face_i];
            auto face_LHS_offset = face_SOL_offset(msh, cl, fc);

            element_location loc_fc = location(msh, fc);
            bool in_dom = (loc_fc == element_location::ON_INTERFACE ||
                           loc_fc == loc_zone);

            bool dirichlet = fc.is_boundary && fc.bndtype == boundary::DIRICHLET
                && in_dom;

            for (size_t i = 0; i < fbs; i++)
                asm_map.push_back( assembly_index(face_LHS_offset+i, !dirichlet) );
        }

        return asm_map;
    }

    Matrix<T, Dynamic, 1>
    get_dirichlet_data(const Mesh& msh, const typename Mesh::cell_type& cl)
    {
        auto facdeg = di.face_degree();
        auto fbs = face_basis<Mesh,T>::size(facdeg);
        auto fcs = faces(msh, cl);
        auto num_faces = fcs.size();
        auto f_dofs = num_faces * fbs;

        auto cbs = loc_cbs;

        Matrix<T, Dynamic, 1> dirichlet_data = Matrix<T, Dynamic, 1>::Zero(cbs + f_dofs);

        for (size_t face_i = 0; face_i < num_faces; face_i++)
        {
            auto fc = fcs[face_i];
            auto face_offset = offset(msh, fc);
            auto face_LHS_offset = face_SOL_offset(msh, cl, fc);

            element_location loc_fc = location(msh, fc);
            bool in_dom = (loc_fc == element_location::ON_INTERFACE ||
                           loc_fc == loc_zone);

            bool dirichlet = fc.is_boundary && fc.bndtype == boundary::DIRICHLET
                && in_dom;

            if (dirichlet)
            {
                Matrix<T, Dynamic, Dynamic> mass = make_mass_matrix(msh, fc, facdeg, loc_zone);
                Matrix<T, Dynamic, 1> loc_rhs = make_rhs(msh, fc, facdeg, loc_zone, dir_func);
                dirichlet_data.block(cbs + face_i*fbs, 0, fbs, 1) = mass.ldlt().solve(loc_rhs);
            }
        }

        return dirichlet_data;
    }

    virtual void
    assemble(const Mesh& msh, const typename Mesh::cell_type& cl,
             const Matrix<T, Dynamic, Dynamic>& lhs, const Matrix<T, Dynamic, 1>& rhs)
    {}

    void
    assemble_bis(const Mesh& msh, const typename Mesh::cell_type& cl,
                 const Matrix<T, Dynamic, Dynamic>& lhs, const Matrix<T, Dynamic, 1>& rhs)
    {
        if( !(location(msh, cl) == loc_zone ||
              location(msh, cl) == element_location::ON_INTERFACE) )
            return;

        auto asm_map = init_asm_map(msh, cl);
        auto dirichlet_data = get_dirichlet_data(msh, cl);

        assert( asm_map.size() == lhs.rows() && asm_map.size() == lhs.cols() );

        // LHS
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

        // RHS
        for (size_t i = 0; i < rhs.rows(); i++)
        {
            if (!asm_map[i].assemble())
                continue;

            RHS(asm_map[i]) += rhs(i);
        }
    } // assemble_bis()

    Matrix<T, Dynamic, 1>
    get_solF(const Mesh& msh, const typename Mesh::cell_type& cl,
             const Matrix<T, Dynamic, 1>& solution)
    {
        auto facdeg = di.face_degree();
        auto fbs = face_basis<Mesh,T>::size(facdeg);
        auto fcs = faces(msh, cl);
        auto num_faces = fcs.size();
        size_t f_dofs = num_faces*fbs;

        Matrix<T, Dynamic, 1> solF = Matrix<T, Dynamic, 1>::Zero(f_dofs);

        for (size_t face_i = 0; face_i < num_faces; face_i++)
        {
            auto fc = fcs[face_i];

            auto loc_fc = location(msh, fc);
            if( !(loc_fc == element_location::ON_INTERFACE || loc_fc == loc_zone) )
                continue;

            bool dirichlet = fc.is_boundary && fc.bndtype == boundary::DIRICHLET;
            if (dirichlet)
            {
                Matrix<T, Dynamic, Dynamic> mass = make_mass_matrix(msh, fc, facdeg);
                Matrix<T, Dynamic, 1> rhs = make_rhs(msh, fc, facdeg, dir_func);
                solF.block(face_i*fbs, 0, fbs, 1) = mass.llt().solve(rhs);
            }
            else
            {
                auto face_offset = offset(msh, fc);
                auto face_LHS_offset = face_SOL_offset(msh, cl, fc);
                solF.block(face_i*fbs, 0, fbs, 1) = solution.block(face_LHS_offset, 0, fbs, 1);
            }
        }

        return solF;
    }

    virtual Matrix<T, Dynamic, 1>
    take_local_data(const Mesh& msh, const typename Mesh::cell_type& cl,
                    const Matrix<T, Dynamic, 1>& solution)
    {}

    void finalize(void)
    {
        LHS.setFromTriplets( triplets.begin(), triplets.end() );
        triplets.clear();
    }
};

/////////////////////////////////////////

template<typename Mesh, typename Function>
class fict_assembler : public virt_fict_assembler<Mesh, Function>
{
    using T = typename Mesh::coordinate_type;

public:
    fict_assembler(const Mesh& msh, const Function& dirichlet_bf,
                    hho_degree_info hdi, element_location where)
        : virt_fict_assembler<Mesh, Function>(msh, dirichlet_bf, hdi, where)
    {
        this->loc_cbs = cell_basis<Mesh,T>::size( this->di.cell_degree() );

        auto fbs = face_basis<Mesh,T>::size( this->di.face_degree() );
        auto system_size = this->loc_cbs * this->num_cells + fbs * this->num_other_faces;

        this->LHS = SparseMatrix<T>( system_size, system_size );
        this->RHS = Matrix<T, Dynamic, 1>::Zero( system_size );
    }

    void
    assemble(const Mesh& msh, const typename Mesh::cell_type& cl,
             const Matrix<T, Dynamic, Dynamic>& lhs, const Matrix<T, Dynamic, 1>& rhs)
    {
        this->assemble_bis(msh, cl, lhs, rhs);
    }

    //// take_local_data
    Matrix<T, Dynamic, 1>
    take_local_data(const Mesh& msh, const typename Mesh::cell_type& cl,
                    const Matrix<T, Dynamic, 1>& solution)
    {
        auto loc_cl = location(msh, cl);
        if( !(loc_cl == element_location::ON_INTERFACE || loc_cl == this->loc_zone) )
            throw std::logic_error("Bad cell !!");

        auto solF = this->get_solF(msh, cl, solution);

        auto cbs = this->loc_cbs;

        Matrix<T, Dynamic, 1> ret = Matrix<T, Dynamic, 1>::Zero( cbs + solF.size() );

        auto cell_offset = this->compress_cells_table.at( offset(msh, cl) );
        auto cell_SOL_offset    = cell_offset * cbs;

        ret.head( cbs ) = solution.block(cell_SOL_offset, 0, cbs, 1);
        ret.tail( solF.size() ) = solF;

        return ret;
    }
};


template<typename Mesh, typename Function>
auto make_fict_assembler(const Mesh& msh, const Function dirichlet_bf,
                          hho_degree_info hdi, element_location where)
{
    return fict_assembler<Mesh, Function>(msh, dirichlet_bf, hdi, where);
}

/////////////////////////////////////////

template<typename Mesh, typename Function>
class fict_condensed_assembler : public virt_fict_assembler<Mesh, Function>
{
    using T = typename Mesh::coordinate_type;

    std::vector< Matrix<T, Dynamic, Dynamic> > loc_LHS;
    std::vector< Matrix<T, Dynamic, 1> >       loc_RHS;

public:
    fict_condensed_assembler(const Mesh& msh, const Function& dirichlet_bf,
                              hho_degree_info hdi, element_location where)
        : virt_fict_assembler<Mesh, Function>(msh, dirichlet_bf, hdi, where)
    {
        this->loc_cbs = 0;

        auto fbs = face_basis<Mesh,T>::size( this->di.face_degree() );
        auto system_size = fbs * this->num_other_faces;

        this->LHS = SparseMatrix<T>( system_size, system_size );
        this->RHS = Matrix<T, Dynamic, 1>::Zero( system_size );

        loc_LHS.resize( this->num_cells );
        loc_RHS.resize( this->num_cells );
    }

    void
    assemble(const Mesh& msh, const typename Mesh::cell_type& cl,
             const Matrix<T, Dynamic, Dynamic>& lhs, const Matrix<T, Dynamic, 1>& rhs)
    {
        if( !(location(msh, cl) == this->loc_zone ||
              location(msh, cl) == element_location::ON_INTERFACE) )
            return;

        // save local matrices
        size_t cell_offset = this->compress_cells_table.at( offset(msh, cl) );
        loc_LHS.at( cell_offset ) = lhs;
        loc_RHS.at( cell_offset ) = rhs;

        auto cbs = cell_basis<Mesh,T>::size( this->di.cell_degree() );
        auto fbs = face_basis<Mesh,T>::size( this->di.face_degree() );
        auto num_faces = faces(msh, cl).size();
        size_t f_dofs = num_faces * fbs;

        // static condensation
        auto mat_sc = static_condensation_compute(lhs, rhs, cbs, f_dofs);
        Matrix<T, Dynamic, Dynamic> lhs_sc = mat_sc.first;
        Matrix<T, Dynamic, 1> rhs_sc = mat_sc.second;

        this->assemble_bis(msh, cl, lhs_sc, rhs_sc);
    }

    //// take_local_data
    Matrix<T, Dynamic, 1>
    take_local_data(const Mesh& msh, const typename Mesh::cell_type& cl,
                    const Matrix<T, Dynamic, 1>& solution)
    {
        auto loc_cl = location(msh, cl);
        if( !(loc_cl == element_location::ON_INTERFACE || loc_cl == this->loc_zone) )
            throw std::logic_error("Bad cell !!");

        auto solF = this->get_solF(msh, cl, solution);

        auto cbs = cell_basis<Mesh,T>::size( this->di.cell_degree() );
        auto fbs = face_basis<Mesh,T>::size( this->di.face_degree() );
        auto num_faces = faces(msh, cl).size();
        size_t f_dofs = num_faces * fbs;

        size_t offset_cl = this->compress_cells_table.at( offset(msh, cl) );

        return static_condensation_recover(loc_LHS.at(offset_cl), loc_RHS.at(offset_cl), cbs, f_dofs, solF);
    }
};


template<typename Mesh, typename Function>
auto make_fict_condensed_assembler(const Mesh& msh, const Function dirichlet_bf,
                                    hho_degree_info hdi, element_location where)
{
    return fict_condensed_assembler<Mesh, Function>(msh, dirichlet_bf, hdi, where);
}



////////////////////////////////  INTERFACE ASSEMBLERS  /////////////////////////



template<typename Mesh>
class interface_assembler
{
    using T = typename Mesh::coordinate_type;
    std::vector<size_t>                 cell_table, face_table;
    size_t num_all_cells, num_all_faces;

    hho_degree_info                     di;

    std::vector< Triplet<T> >           triplets;


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

        if( rhs.rows() > 2*cbs )
        {
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


/////////////////////////////////////////////////////////



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




/******************************************************************************************/
/*******************                                               ************************/
/*******************               STOKES ASSEMBLERS               ************************/
/*******************                                               ************************/
/******************************************************************************************/


////////////////////////  STATIC CONDENSATION  //////////////////////////

template<typename T>
std::pair<   Matrix<T, Dynamic, Dynamic>, Matrix<T, Dynamic, 1>  >
stokes_static_condensation_compute
(const Matrix<T, Dynamic, Dynamic> lhs_A, const Matrix<T, Dynamic, Dynamic> lhs_B,
 const Matrix<T, Dynamic, 1> rhs_A, const Matrix<T, Dynamic, 1> rhs_B,
 const size_t cell_size, const size_t face_size)
{
    using matrix = Matrix<T, Dynamic, Dynamic>;
    using vector = Matrix<T, Dynamic, 1>;

    size_t size_tot = cell_size + face_size;
    size_t p_size = lhs_B.rows();
    assert(lhs_A.cols() == size_tot && lhs_A.rows() == size_tot);
    assert(rhs_A.rows() == size_tot || rhs_A.rows() == cell_size);

    matrix lhs_sc = matrix::Zero(face_size + p_size, face_size + p_size);
    vector rhs_sc = vector::Zero(face_size + p_size);


    // sub--lhs
    matrix K_TT = lhs_A.topLeftCorner(cell_size, cell_size);
    matrix K_TF = lhs_A.topRightCorner(cell_size, face_size);
    matrix K_FT = lhs_A.bottomLeftCorner(face_size, cell_size);
    matrix K_FF = lhs_A.bottomRightCorner(face_size, face_size);
    matrix K_PT = lhs_B.block(0, 0, p_size, cell_size);
    matrix K_PF = lhs_B.block(0, cell_size, p_size, face_size);

    // sub--rhs
    vector cell_rhs = vector::Zero(cell_size);
    vector face_rhs = vector::Zero(face_size);
    if(rhs_A.rows() == cell_size)
        cell_rhs = rhs_A;
    else
    {
        cell_rhs = rhs_A.head(cell_size);
        face_rhs = rhs_A.tail(face_size);
    }

    // static condensation
    auto K_TT_ldlt = K_TT.ldlt();
    matrix AL = K_TT_ldlt.solve(K_TF);
    matrix BL = K_TT_ldlt.solve(K_PT.transpose());
    vector rL = K_TT_ldlt.solve(cell_rhs);

    lhs_sc.block(0, 0, face_size, face_size) = K_FF - K_FT * AL;
    lhs_sc.block(face_size, 0, p_size, face_size) = K_PF - K_PT * AL;
    lhs_sc.block(0, face_size, face_size, p_size)
        = lhs_sc.block(face_size, 0, p_size, face_size).transpose();
    lhs_sc.block(face_size, face_size, p_size, p_size) = - K_PT * BL;

    rhs_sc.head(face_size) = face_rhs - K_FT * rL;
    rhs_sc.tail(p_size) = rhs_B - K_PT * rL;

    return std::make_pair(lhs_sc, rhs_sc);
}


// full static condensation (velocity + pressure)
// in this version we keep only the velocity face dofs
// and one pressure dof per cell (that represents the mean pressure in the cell)
template<typename T>
std::pair<   Matrix<T, Dynamic, Dynamic>, Matrix<T, Dynamic, 1>  >
stokes_full_static_condensation_compute
(const Matrix<T, Dynamic, Dynamic> lhs_A, const Matrix<T, Dynamic, Dynamic> lhs_B,
 const Matrix<T, Dynamic, Dynamic> lhs_C,
 const Matrix<T, Dynamic, 1> rhs_A, const Matrix<T, Dynamic, 1> rhs_B,
 const Matrix<T, Dynamic, 1> mult, const size_t cell_size, const size_t face_size)
{
    using matrix = Matrix<T, Dynamic, Dynamic>;
    using vector = Matrix<T, Dynamic, 1>;

    size_t size_tot = cell_size + face_size;
    size_t p_size = lhs_B.rows();
    assert(lhs_A.cols() == size_tot && lhs_A.rows() == size_tot);
    assert(rhs_A.rows() == size_tot || rhs_A.rows() == cell_size);
    assert(lhs_B.rows() == p_size && lhs_B.cols() == size_tot);
    assert(rhs_B.rows() == p_size);
    assert(lhs_C.rows() == p_size && lhs_C.cols() == p_size);
    assert(mult.rows() == p_size - 1);

    matrix lhs_sc = matrix::Zero(face_size + 1, face_size + 1);
    vector rhs_sc = vector::Zero(face_size + 1);

    // sub--lhs
    matrix K_TT = lhs_A.topLeftCorner(cell_size, cell_size);
    matrix K_TF = lhs_A.topRightCorner(cell_size, face_size);
    matrix K_FT = lhs_A.bottomLeftCorner(face_size, cell_size);
    matrix K_FF = lhs_A.bottomRightCorner(face_size, face_size);
    matrix K_PT = lhs_B.block(0, 0, p_size, cell_size);
    matrix K_PF = lhs_B.block(0, cell_size, p_size, face_size);


    // sub--rhs
    vector cell_rhs = vector::Zero(cell_size);
    vector face_rhs = vector::Zero(face_size);
    if(rhs_A.rows() == cell_size)
        cell_rhs = rhs_A;
    else
    {
        cell_rhs = rhs_A.head(cell_size);
        face_rhs = rhs_A.tail(face_size);
    }

    // compute the new matrices
    matrix tB_PT = K_PT.block(0, 0, 1, cell_size);
    matrix tB_PF = K_PF.block(0, 0, 1, face_size);
    matrix temp_B_PT = K_PT.block(1, 0, p_size-1, cell_size);
    matrix temp_B_PF = K_PF.block(1, 0, p_size-1, face_size);
    matrix hB_PT = mult * tB_PT + temp_B_PT;
    matrix hB_PF = mult * tB_PF + temp_B_PF;

    matrix ttC_pp = lhs_C.block(0, 0, 1, p_size).transpose();
    matrix hhC_pp = lhs_C.block(1, 0, p_size - 1, p_size).transpose() + ttC_pp * mult.transpose();
    matrix C_tptp = ttC_pp.block(0, 0, 1, 1);
    matrix C_tphp = hhC_pp.block(0, 0, 1, p_size - 1);
    matrix C_hptp = ttC_pp.block(1, 0, p_size - 1, 1) + mult * C_tptp;
    matrix C_hphp = hhC_pp.block(1, 0, p_size - 1, p_size - 1) + mult * C_tphp;

    vector rhs_tp = rhs_B.block(0,0,1,1);
    vector rhs_temp = rhs_B.block(1,0,p_size-1,1);
    vector rhs_hp = rhs_temp + mult * rhs_tp;


    ////////////// static condensation
    // invert matrices
    auto K_TT_ldlt = K_TT.ldlt();
    matrix iAhB = K_TT_ldlt.solve(hB_PT.transpose());
    matrix iAK_TF = K_TT_ldlt.solve(K_TF);
    matrix iAtB = K_TT_ldlt.solve(tB_PT.transpose());
    vector iA_rhs_T = K_TT_ldlt.solve(cell_rhs);

    auto iBAB = ( hB_PT * iAhB - C_hphp ).ldlt();
    matrix iBAB_B_PF = iBAB.solve(hB_PF);
    matrix iBAB_B_PT = iBAB.solve(hB_PT);
    matrix iBAB_C_hptp = iBAB.solve(C_hptp);
    vector iBAB_rhs_hp = iBAB.solve(rhs_hp);

    // compute final matrices and rhs
    matrix AFF_1 = K_FF;
    matrix AFF_2 = - K_FT * iAK_TF;
    matrix AFF_3 = ( K_FT * iAhB - hB_PF.transpose() ) * (iBAB_B_PT * iAK_TF - iBAB_B_PF);
    matrix AFF = AFF_1 + AFF_2 + AFF_3;

    matrix BFP_1 = tB_PF.transpose();
    matrix BFP_2 = - K_FT * iAtB;
    matrix BFP_3 = ( hB_PF.transpose() - K_FT * iAhB ) * ( iBAB_C_hptp - iBAB_B_PT * iAtB );
    matrix BFP = BFP_1 + BFP_2 + BFP_3;

    vector RHS_F_1 = face_rhs;
    vector RHS_F_2 = - K_FT * iA_rhs_T;
    vector RHS_F_3 = ( hB_PF.transpose() - K_FT * iAhB ) * ( iBAB_rhs_hp - iBAB_B_PT * iA_rhs_T );
    vector RHS_F = RHS_F_1 + RHS_F_2 + RHS_F_3;

    matrix BPF_1 = tB_PF;
    matrix BPF_2 = - tB_PT * iAK_TF;
    matrix BPF_3 = (C_tphp - tB_PT * iAhB) * ( iBAB_B_PF - iBAB_B_PT * iAK_TF );
    matrix BPF = BPF_1 + BPF_2 + BPF_3;

    matrix CPP_1 = C_tptp;
    matrix CPP_2 = - tB_PT * iAtB;
    matrix CPP_3 = ( C_tphp - tB_PT * iAhB ) * ( iBAB_C_hptp - iBAB_B_PT * iAtB );
    matrix CPP = CPP_1 + CPP_2 + CPP_3;

    vector RHS_P_1 = rhs_tp;
    vector RHS_P_2 = - tB_PT * iA_rhs_T;
    vector RHS_P_3 = ( C_tphp - tB_PT * iAhB) * ( iBAB_rhs_hp - iBAB_B_PT * iA_rhs_T );
    vector RHS_P = RHS_P_1 + RHS_P_2 + RHS_P_3;

    lhs_sc.block(0, 0, face_size, face_size) = AFF;
    lhs_sc.block(0, face_size, face_size, 1) = BFP;
    lhs_sc.block(face_size, 0, 1, face_size) = BPF;
    lhs_sc.block(face_size, face_size, 1, 1) = CPP;

    rhs_sc.head(face_size) = RHS_F;
    rhs_sc.tail(1) = RHS_P;

    return std::make_pair(lhs_sc, rhs_sc);
}


template<typename T>
std::pair<   Matrix<T, Dynamic, Dynamic>, Matrix<T, Dynamic, 1>  >
stokes_full_static_condensation_compute
(const Matrix<T, Dynamic, Dynamic> lhs_A, const Matrix<T, Dynamic, Dynamic> lhs_B,
 const Matrix<T, Dynamic, 1> rhs_A, const Matrix<T, Dynamic, 1> rhs_B,
 const Matrix<T, Dynamic, 1> mult, const size_t cell_size, const size_t face_size)
{
    size_t p_size = lhs_B.rows();
    Matrix<T, Dynamic, Dynamic> lhs_C = Matrix<T, Dynamic, Dynamic>::Zero(p_size, p_size);

    return stokes_full_static_condensation_compute
        (lhs_A, lhs_B, lhs_C, rhs_A, rhs_B, mult, cell_size, face_size);
}



template<typename T>
Matrix<T, Dynamic, 1>
stokes_full_static_condensation_recover_v
(const Matrix<T, Dynamic, Dynamic> lhs_A, const Matrix<T, Dynamic, Dynamic> lhs_B,
 const Matrix<T, Dynamic, Dynamic> lhs_C,
 const Matrix<T, Dynamic, 1> rhs_A, const Matrix<T, Dynamic, 1> rhs_B,
 const Matrix<T, Dynamic, 1> mult,
 const size_t cell_size, const size_t face_size, const Matrix<T, Dynamic, 1> sol_sc)
{
    using matrix = Matrix<T, Dynamic, Dynamic>;
    using vector = Matrix<T, Dynamic, 1>;

    size_t size_tot = cell_size + face_size;
    size_t p_size = lhs_B.rows();
    assert(lhs_A.cols() == size_tot && lhs_A.rows() == size_tot);
    assert(rhs_A.rows() == size_tot || rhs_A.rows() == cell_size);
    assert(lhs_B.rows() == p_size && lhs_B.cols() == size_tot);
    assert(rhs_B.rows() == p_size);
    assert(lhs_C.rows() == p_size && lhs_C.cols() == p_size);
    assert(mult.rows() == p_size - 1);
    assert(sol_sc.rows() == face_size + 1);


    // sub--lhs
    matrix K_TT = lhs_A.topLeftCorner(cell_size, cell_size);
    matrix K_TF = lhs_A.topRightCorner(cell_size, face_size);
    matrix K_TP = lhs_B.topLeftCorner(p_size, cell_size).transpose();
    matrix K_PT = lhs_B.block(0, 0, p_size, cell_size);
    matrix K_PF = lhs_B.block(0, cell_size, p_size, face_size);

    // sub--rhs
    vector cell_rhs = vector::Zero(cell_size);
    cell_rhs = rhs_A.head(cell_size);

    // compute the new matrices
    matrix tB_PT = K_PT.block(0, 0, 1, cell_size);
    matrix tB_PF = K_PF.block(0, 0, 1, face_size);
    matrix temp_B_PT = K_PT.block(1, 0, p_size-1, cell_size);
    matrix temp_B_PF = K_PF.block(1, 0, p_size-1, face_size);
    matrix hB_PT = mult * tB_PT + temp_B_PT;
    matrix hB_PF = mult * tB_PF + temp_B_PF;

    matrix ttC_pp = lhs_C.block(0, 0, 1, p_size).transpose();
    matrix hhC_pp = lhs_C.block(1, 0, p_size - 1, p_size).transpose() + ttC_pp * mult.transpose();
    matrix C_tptp = ttC_pp.block(0, 0, 1, 1);
    matrix C_tphp = hhC_pp.block(0, 0, 1, p_size - 1);
    matrix C_hptp = ttC_pp.block(1, 0, p_size - 1, 1) + mult * C_tptp;
    matrix C_hphp = hhC_pp.block(1, 0, p_size - 1, p_size - 1) + mult * C_tphp;

    vector rhs_tp = rhs_B.block(0,0,1,1);
    vector rhs_temp = rhs_B.block(1,0,p_size-1,1);
    vector rhs_hp = rhs_temp + mult * rhs_tp;

    // recover velocity cell solution
    auto K_TT_ldlt = K_TT.ldlt();
    matrix iAhB = K_TT_ldlt.solve(hB_PT.transpose());
    matrix iAK_TF = K_TT_ldlt.solve(K_TF);
    matrix iAtB = K_TT_ldlt.solve(tB_PT.transpose());
    vector iA_rhs_T = K_TT_ldlt.solve(cell_rhs);

    auto iBAB = ( hB_PT * iAhB - C_hphp ).ldlt();
    matrix iBAB_B_PF = iBAB.solve(hB_PF);
    matrix iBAB_B_PT = iBAB.solve(hB_PT);
    matrix iBAB_C_hptp = iBAB.solve(C_hptp);
    vector iBAB_rhs_hp = iBAB.solve(rhs_hp);

    vector ret = vector::Zero(size_tot);

    vector uF = sol_sc.head(face_size);
    vector solP = sol_sc.tail(1);

    vector uT_1 = iAhB * iBAB_B_PT * (iAK_TF * uF + iAtB * solP - iA_rhs_T);
    vector uT_2 = - iAhB * iBAB_B_PF * uF;
    vector uT_3 = iAhB * iBAB_rhs_hp;
    vector uT_4 = - iAhB * iBAB_C_hptp * solP;
    vector uT_5 = iA_rhs_T - iAK_TF * uF - iAtB * solP;
    vector uT = uT_1 + uT_2 + uT_3 + uT_4 + uT_5;

    ret.head(cell_size) = uT;
    ret.block(cell_size, 0, face_size, 1) = uF;
    return ret;
}


template<typename T>
Matrix<T, Dynamic, 1>
stokes_full_static_condensation_recover_v
(const Matrix<T, Dynamic, Dynamic> lhs_A, const Matrix<T, Dynamic, Dynamic> lhs_B,
 const Matrix<T, Dynamic, 1> rhs_A, const Matrix<T, Dynamic, 1> rhs_B,
 const Matrix<T, Dynamic, 1> mult,
 const size_t cell_size, const size_t face_size, const Matrix<T, Dynamic, 1> sol_sc)
{
    size_t p_size = lhs_B.rows();
    Matrix<T, Dynamic, Dynamic> lhs_C = Matrix<T, Dynamic, Dynamic>::Zero(p_size, p_size);

    return stokes_full_static_condensation_recover_v
        (lhs_A, lhs_B, lhs_C, rhs_A, rhs_B, mult, cell_size, face_size, sol_sc);
}


template<typename T>
Matrix<T, Dynamic, 1>
stokes_full_static_condensation_recover_p
(const Matrix<T, Dynamic, Dynamic> lhs_A, const Matrix<T, Dynamic, Dynamic> lhs_B,
 const Matrix<T, Dynamic, Dynamic> lhs_C,
 const Matrix<T, Dynamic, 1> rhs_A, const Matrix<T, Dynamic, 1> rhs_B,
 const Matrix<T, Dynamic, 1> mult,
 const size_t cell_size, const size_t face_size, const Matrix<T, Dynamic, 1> sol_sc)
{
    using matrix = Matrix<T, Dynamic, Dynamic>;
    using vector = Matrix<T, Dynamic, 1>;

    size_t size_tot = cell_size + face_size;
    size_t p_size = lhs_B.rows();
    assert(lhs_A.cols() == size_tot && lhs_A.rows() == size_tot);
    assert(rhs_A.rows() == size_tot || rhs_A.rows() == cell_size);
    assert(lhs_B.rows() == p_size && lhs_B.cols() == size_tot);
    assert(rhs_B.rows() == p_size);
    assert(lhs_C.rows() == p_size && lhs_C.cols() == p_size);
    assert(mult.rows() == p_size - 1);
    assert(sol_sc.rows() == face_size + 1);

    // sub--lhs
    matrix K_TT = lhs_A.topLeftCorner(cell_size, cell_size);
    matrix K_TF = lhs_A.topRightCorner(cell_size, face_size);
    matrix K_TP = lhs_B.topLeftCorner(p_size, cell_size).transpose();
    matrix K_PT = lhs_B.block(0, 0, p_size, cell_size);
    matrix K_PF = lhs_B.block(0, cell_size, p_size, face_size);

    // sub--rhs
    vector cell_rhs = vector::Zero(cell_size);
    cell_rhs = rhs_A.head(cell_size);

    // compute the new matrices
    matrix tB_PT = K_PT.block(0, 0, 1, cell_size);
    matrix tB_PF = K_PF.block(0, 0, 1, face_size);
    matrix temp_B_PT = K_PT.block(1, 0, p_size-1, cell_size);
    matrix temp_B_PF = K_PF.block(1, 0, p_size-1, face_size);
    matrix hB_PT = mult * tB_PT + temp_B_PT;
    matrix hB_PF = mult * tB_PF + temp_B_PF;

    matrix ttC_pp = lhs_C.block(0, 0, 1, p_size).transpose();
    matrix hhC_pp = lhs_C.block(1, 0, p_size - 1, p_size).transpose() + ttC_pp * mult.transpose();
    matrix C_tptp = ttC_pp.block(0, 0, 1, 1);
    matrix C_tphp = hhC_pp.block(0, 0, 1, p_size - 1);
    matrix C_hptp = ttC_pp.block(1, 0, p_size - 1, 1) + mult * C_tptp;
    matrix C_hphp = hhC_pp.block(1, 0, p_size - 1, p_size - 1) + mult * C_tphp;

    vector rhs_tp = rhs_B.block(0,0,1,1);
    vector rhs_temp = rhs_B.block(1,0,p_size-1,1);
    vector rhs_hp = rhs_temp + mult * rhs_tp;

    // pressure solution
    auto K_TT_ldlt = K_TT.ldlt();
    matrix iAhB = K_TT_ldlt.solve(hB_PT.transpose());
    matrix iAK_TF = K_TT_ldlt.solve(K_TF);
    matrix iAtB = K_TT_ldlt.solve(tB_PT.transpose());
    vector iA_rhs_T = K_TT_ldlt.solve(cell_rhs);

    auto iBAB = ( hB_PT * iAhB - C_hphp ).ldlt();
    matrix iBAB_B_PF = iBAB.solve(hB_PF);
    matrix iBAB_B_PT = iBAB.solve(hB_PT);
    matrix iBAB_C_hptp = iBAB.solve(C_hptp);
    vector iBAB_rhs_hp = iBAB.solve(rhs_hp);

    vector ret = vector::Zero(p_size);

    vector uF = sol_sc.head(face_size);
    vector solP = sol_sc.tail(1);

    vector p_1 = iBAB_B_PF * uF;
    vector p_2 = - iBAB_B_PT * iAK_TF * uF;
    vector p_3 = - iBAB_B_PT * iAtB * solP;
    vector p_4 = iBAB_C_hptp * solP;
    vector p_5 = - iBAB_rhs_hp;
    vector p_6 = iBAB_B_PT * iA_rhs_T;
    vector hP = p_1 + p_2 + p_3 + p_4 + p_5 + p_6;

    ret(0) = solP(0) + mult.dot(hP);
    ret.tail(p_size-1) = hP;
    return ret;
}


template<typename T>
Matrix<T, Dynamic, 1>
stokes_full_static_condensation_recover_p
(const Matrix<T, Dynamic, Dynamic> lhs_A, const Matrix<T, Dynamic, Dynamic> lhs_B,
 const Matrix<T, Dynamic, 1> rhs_A, const Matrix<T, Dynamic, 1> rhs_B,
 const Matrix<T, Dynamic, 1> mult,
 const size_t cell_size, const size_t face_size, const Matrix<T, Dynamic, 1> sol_sc)
{
    size_t p_size = lhs_B.rows();
    Matrix<T, Dynamic, Dynamic> lhs_C = Matrix<T, Dynamic, Dynamic>::Zero(p_size, p_size);

    return stokes_full_static_condensation_recover_p
        (lhs_A, lhs_B, lhs_C, rhs_A, rhs_B, mult, cell_size, face_size, sol_sc);
}


///////////////////////////   ASSEMBLERS  /////////////////////////

// template<typename Mesh>
// class virt_stokes_fict_assembler
// {
//     using T = typename Mesh::coordinate_type;
// protected:
//     std::vector< Triplet<T> >           triplets;
//     std::vector<size_t>                 compress_face_table;
//     // std::vector<size_t>                 expand_face_table;
//     std::vector<size_t>                 compress_cells_table;

//     hho_degree_info                     di;

//     element_location loc_zone;
//     size_t num_cells;
//     size_t num_other_faces;

// public:
//     SparseMatrix<T>         LHS;
//     Matrix<T, Dynamic, 1>   RHS;

//     virtual size_t
//     system_size()
//     {}

//     virtual size_t
//     v_size()
//     {}

//     virt_stokes_fict_assembler(const Mesh& msh, hho_degree_info hdi, element_location where)
//         : di(hdi), loc_zone(where)
//     {
//         auto is_removed = [&](const typename Mesh::face_type& fc) -> bool {
//             bool is_dirichlet = fc.is_boundary && fc.bndtype == boundary::DIRICHLET;
//             auto loc = location(msh,fc);
//             bool is_where = (loc == where || loc == element_location::ON_INTERFACE);
//             return is_dirichlet || (!is_where);
//         };

//         auto num_all_faces = msh.faces.size();
//         auto num_removed_faces = std::count_if(msh.faces.begin(), msh.faces.end(), is_removed);
//         num_other_faces = num_all_faces - num_removed_faces;

//         compress_face_table.resize( num_all_faces );
//         // expand_face_table.resize( num_other_faces );

//         size_t compressed_offset = 0;
//         for (size_t i = 0; i < num_all_faces; i++)
//         {
//             auto fc = msh.faces[i];
//             if ( !is_removed(fc) )
//             {
//                 compress_face_table.at(i) = compressed_offset;
//                 // expand_face_table.at(compressed_offset) = i;
//                 compressed_offset++;
//             }
//         }

//         compress_cells_table.resize( msh.cells.size() );
//         compressed_offset = 0;
//         for (size_t i = 0; i < msh.cells.size(); i++)
//         {
//             auto cl = msh.cells[i];
//             if (location(msh, cl) == where || location(msh, cl) == element_location::ON_INTERFACE)
//             {
//                 compress_cells_table.at(i) = compressed_offset;
//                 compressed_offset++;
//             }
//         }
//         num_cells = compressed_offset;

//         // auto celdeg = di.cell_degree();
//         // auto facdeg = di.face_degree();

//         // auto cbs_B = cell_basis<Mesh,T>::size(facdeg);
//         // auto cbs_A = vector_cell_basis<Mesh,T>::size(celdeg);
//         // auto fbs = vector_face_basis<Mesh,T>::size(facdeg);

//         // auto system_size = (cbs_A + cbs_B) * num_cells + fbs * num_other_faces + 1;

//         auto system_size = system_size();

//         LHS = SparseMatrix<T>( system_size, system_size );
//         RHS = Matrix<T, Dynamic, 1>::Zero( system_size );
//     }

//     template<typename Function>
//     void
//     assemble(const Mesh& msh, const typename Mesh::cell_type& cl,
//              const Matrix<T, Dynamic, Dynamic>& lhs_A, const Matrix<T, Dynamic, Dynamic>& lhs_B,
//              const Matrix<T, Dynamic, 1>& rhs_A, const Matrix<T, Dynamic, 1>& rhs_B,
//              const Function& dirichlet_bf)
//     {
//         if( !(location(msh, cl) == loc_zone ||
//               location(msh, cl) == element_location::ON_INTERFACE) )
//             return;

//         auto celdeg = di.cell_degree();
//         auto facdeg = di.face_degree();

//         auto cbs_B = cell_basis<Mesh,T>::size(facdeg);
//         auto cbs_A = vector_cell_basis<Mesh,T>::size(celdeg);
//         auto fbs_A = vector_face_basis<Mesh,T>::size(facdeg);

//         auto fcs = faces(msh, cl);
//         auto num_faces = fcs.size();
//         size_t f_dofs = num_faces * fbs_A;

//         // size_t v_size = v_size();

//         std::vector<assembly_index> asm_map;
//         asm_map.reserve( v_size() );

//         size_t cell_offset = compress_cells_table.at( offset(msh, cl) );
//         auto cell_LHS_offset    = cell_offset * cbs_A;
//         auto B_offset = cbs_A * num_cells + fbs_A * num_other_faces + cbs_B * cell_offset;

//         for (size_t i = 0; i < cbs_A; i++)
//             asm_map.push_back( assembly_index(cell_LHS_offset+i, true) );

//         Matrix<T, Dynamic, 1> dirichlet_data = Matrix<T, Dynamic, 1>::Zero(cbs_A + f_dofs);

//         for (size_t face_i = 0; face_i < num_faces; face_i++)
//         {
//             auto fc = fcs[face_i];
//             auto face_offset = offset(msh, fc);
//             auto face_LHS_offset = cbs_A * num_cells + compress_face_table.at(face_offset)*fbs_A;

//             element_location loc_fc = location(msh, fc);
//             bool in_dom = (loc_fc == element_location::ON_INTERFACE ||
//                            loc_fc == loc_zone);

//             bool dirichlet = fc.is_boundary && fc.bndtype == boundary::DIRICHLET
//                 && in_dom;

//             for (size_t i = 0; i < fbs_A; i++)
//                 asm_map.push_back( assembly_index(face_LHS_offset+i, !dirichlet) );

//             if (dirichlet)
//             {
//                 Matrix<T, Dynamic, Dynamic> mass = make_vector_mass_matrix(msh,fc,facdeg,loc_zone);
//                 Matrix<T, Dynamic, 1> loc_rhs=make_vector_rhs(msh,fc,facdeg,loc_zone,dirichlet_bf);
//                 dirichlet_data.block(cbs_A + face_i*fbs_A, 0, fbs_A, 1) = mass.ldlt().solve(loc_rhs);
//             }
//         }

//         assert( asm_map.size() == lhs_A.rows() && asm_map.size() == lhs_A.cols() );

//         // assemble A
//         for (size_t i = 0; i < lhs_A.rows(); i++)
//         {
//             if (!asm_map[i].assemble())
//                 continue;

//             for (size_t j = 0; j < lhs_A.cols(); j++)
//             {
//                 if ( asm_map[j].assemble() )
//                     triplets.push_back( Triplet<T>(asm_map[i], asm_map[j], lhs_A(i,j)) );
//                 else
//                     RHS(asm_map[i]) -= lhs_A(i,j) * dirichlet_data(j);
//             }
//         }

//         // assemble B
//         for (size_t i = 0; i < lhs_B.rows(); i++)
//         {
//             for (size_t j = 0; j < lhs_B.cols(); j++)
//             {
//                 auto global_i = B_offset + i;
//                 auto global_j = asm_map[j];
//                 if ( asm_map[j].assemble() )
//                 {
//                     triplets.push_back( Triplet<T>(global_i, global_j, lhs_B(i,j)) );
//                     triplets.push_back( Triplet<T>(global_j, global_i, lhs_B(i,j)) );
//                 }
//                 else
//                     RHS(global_i) -= lhs_B(i,j)*dirichlet_data(j);
//             }
//         }

//         // null pressure mean condition
//         cell_basis<cuthho_poly_mesh<T>, T> cb(msh, cl, di.face_degree());
//         auto qpsi = integrate(msh, cl, di.face_degree(), loc_zone);
//         Matrix<T, Dynamic, 1> mult = Matrix<T, Dynamic, 1>::Zero( cbs_B );
//         for (auto& qp : qpsi)
//         {
//             auto phi = cb.eval_basis(qp.first);
//             mult += qp.second * phi;
//         }
//         auto mult_offset = cbs_A * num_cells + fbs_A * num_other_faces + cbs_B * num_cells;

//         for (size_t i = 0; i < mult.rows(); i++)
//         {
//             triplets.push_back( Triplet<T>(B_offset+i, mult_offset, mult(i)) );
//             triplets.push_back( Triplet<T>(mult_offset, B_offset+i, mult(i)) );
//         }


//         // RHS A
//         for (size_t i = 0; i < rhs_A.rows(); i++)
//         {
//             if (!asm_map[i].assemble())
//                 continue;

//             RHS(asm_map[i]) += rhs_A(i);
//         }

//         // RHS B
//         for (size_t i = 0; i < rhs_B.rows(); i++)
//             RHS(B_offset + i) += rhs_B(i);

//     } // assemble()
// };

//////////////////////////////
template<typename Mesh>
class stokes_fict_assembler
{
    using T = typename Mesh::coordinate_type;
    std::vector< Triplet<T> >           triplets;

protected:
    std::vector<size_t>                 compress_face_table;
    std::vector<size_t>                 expand_face_table;
    std::vector<size_t>                 compress_cells_table;

    hho_degree_info                     di;

    element_location loc_zone;
    size_t num_cells;
    size_t num_other_faces;

public:

    SparseMatrix<T>         LHS;
    Matrix<T, Dynamic, 1>   RHS;

    stokes_fict_assembler(const Mesh& msh, hho_degree_info hdi, element_location where)
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
        num_other_faces = num_all_faces - num_removed_faces;

        compress_face_table.resize( num_all_faces );
        expand_face_table.resize( num_other_faces );

        size_t compressed_offset = 0;
        for (size_t i = 0; i < num_all_faces; i++)
        {
            auto fc = msh.faces[i];
            if ( !is_removed(fc) )
            {
                compress_face_table.at(i) = compressed_offset;
                expand_face_table.at(compressed_offset) = i;
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
        num_cells = compressed_offset;

        auto celdeg = di.cell_degree();
        auto facdeg = di.face_degree();

        auto cbs_B = cell_basis<Mesh,T>::size(facdeg);
        auto cbs_A = vector_cell_basis<Mesh,T>::size(celdeg);
        auto fbs = vector_face_basis<Mesh,T>::size(facdeg);

        auto system_size = (cbs_A + cbs_B) * num_cells + fbs * num_other_faces + 1;

        LHS = SparseMatrix<T>( system_size, system_size );
        RHS = Matrix<T, Dynamic, 1>::Zero( system_size );
    }

    template<typename Function>
    void
    assemble(const Mesh& msh, const typename Mesh::cell_type& cl,
             const Matrix<T, Dynamic, Dynamic>& lhs_A, const Matrix<T, Dynamic, Dynamic>& lhs_B,
             const Matrix<T, Dynamic, 1>& rhs_A, const Matrix<T, Dynamic, 1>& rhs_B,
             const Function& dirichlet_bf)
    {
        if( !(location(msh, cl) == loc_zone ||
              location(msh, cl) == element_location::ON_INTERFACE) )
            return;

        auto celdeg = di.cell_degree();
        auto facdeg = di.face_degree();

        auto cbs_B = cell_basis<Mesh,T>::size(facdeg);
        auto cbs_A = vector_cell_basis<Mesh,T>::size(celdeg);
        auto fbs_A = vector_face_basis<Mesh,T>::size(facdeg);

        auto fcs = faces(msh, cl);
        auto num_faces = fcs.size();
        size_t f_dofs = num_faces * fbs_A;

        std::vector<assembly_index> asm_map;
        asm_map.reserve(cbs_A + f_dofs);

        size_t cell_offset = compress_cells_table.at( offset(msh, cl) );
        auto cell_LHS_offset    = cell_offset * cbs_A;
        auto B_offset = cbs_A * num_cells + fbs_A * num_other_faces + cbs_B * cell_offset;

        for (size_t i = 0; i < cbs_A; i++)
            asm_map.push_back( assembly_index(cell_LHS_offset+i, true) );

        Matrix<T, Dynamic, 1> dirichlet_data = Matrix<T, Dynamic, 1>::Zero(cbs_A + f_dofs);

        for (size_t face_i = 0; face_i < num_faces; face_i++)
        {
            auto fc = fcs[face_i];
            auto face_offset = offset(msh, fc);
            auto face_LHS_offset = cbs_A * num_cells + compress_face_table.at(face_offset)*fbs_A;

            element_location loc_fc = location(msh, fc);
            bool in_dom = (loc_fc == element_location::ON_INTERFACE ||
                           loc_fc == loc_zone);

            bool dirichlet = fc.is_boundary && fc.bndtype == boundary::DIRICHLET
                && in_dom;

            for (size_t i = 0; i < fbs_A; i++)
                asm_map.push_back( assembly_index(face_LHS_offset+i, !dirichlet) );

            if (dirichlet)
            {
                Matrix<T, Dynamic, Dynamic> mass = make_vector_mass_matrix(msh,fc,facdeg,loc_zone);
                Matrix<T, Dynamic, 1> loc_rhs=make_vector_rhs(msh,fc,facdeg,loc_zone,dirichlet_bf);
                dirichlet_data.block(cbs_A + face_i*fbs_A, 0, fbs_A, 1) = mass.ldlt().solve(loc_rhs);
            }
        }

        assert( asm_map.size() == lhs_A.rows() && asm_map.size() == lhs_A.cols() );

        // assemble A
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

        // assemble B
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
        auto qpsi = integrate(msh, cl, di.face_degree(), loc_zone);
        Matrix<T, Dynamic, 1> mult = Matrix<T, Dynamic, 1>::Zero( cbs_B );
        for (auto& qp : qpsi)
        {
            auto phi = cb.eval_basis(qp.first);
            mult += qp.second * phi;
        }
        auto mult_offset = cbs_A * num_cells + fbs_A * num_other_faces + cbs_B * num_cells;

        for (size_t i = 0; i < mult.rows(); i++)
        {
            triplets.push_back( Triplet<T>(B_offset+i, mult_offset, mult(i)) );
            triplets.push_back( Triplet<T>(mult_offset, B_offset+i, mult(i)) );
        }


        // RHS A
        for (size_t i = 0; i < rhs_A.rows(); i++)
        {
            if (!asm_map[i].assemble())
                continue;

            RHS(asm_map[i]) += rhs_A(i);
        }

        // RHS B
        for (size_t i = 0; i < rhs_B.rows(); i++)
            RHS(B_offset + i) += rhs_B(i);

    } // assemble()

    template<typename Function>
    Matrix<T, Dynamic, 1>
    take_velocity(const Mesh& msh, const typename Mesh::cell_type& cl,
    const Matrix<T, Dynamic, 1>& solution, const Function& dirichlet_bf)
    {
        auto loc_cl = location(msh, cl);
        if( !(loc_cl == element_location::ON_INTERFACE || loc_cl == loc_zone) )
            throw std::logic_error("Bad cell !!");

        auto celdeg = di.cell_degree();
        auto facdeg = di.face_degree();

        auto cbs_A = vector_cell_basis<Mesh,T>::size(celdeg);
        auto fbs_A = vector_face_basis<Mesh,T>::size(facdeg);

        auto cell_offset = compress_cells_table.at( offset(msh, cl) );
        auto cell_SOL_offset    = cell_offset * cbs_A;

        auto fcs = faces(msh, cl);
        auto num_faces = fcs.size();

        Matrix<T, Dynamic, 1> ret = Matrix<T, Dynamic, 1>::Zero(cbs_A + num_faces*fbs_A);
        ret.block(0, 0, cbs_A, 1) = solution.block(cell_SOL_offset, 0, cbs_A, 1);

        for (size_t face_i = 0; face_i < num_faces; face_i++)
        {
            auto fc = fcs[face_i];

            auto loc_fc = location(msh, fc);
            if( !(loc_fc == element_location::ON_INTERFACE || loc_fc == loc_zone) )
                continue;

            bool dirichlet = fc.is_boundary && fc.bndtype == boundary::DIRICHLET;

            if (dirichlet)
            {
                Matrix<T, Dynamic, Dynamic> mass = make_vector_mass_matrix(msh, fc, facdeg);
                Matrix<T, Dynamic, 1> loc_rhs = make_vector_rhs(msh, fc, facdeg, dirichlet_bf);
                ret.block(cbs_A+face_i*fbs_A, 0, fbs_A, 1) = mass.ldlt().solve(loc_rhs);
            }
            else
            {
                auto face_offset = offset(msh, fc);
                auto face_SOL_offset = cbs_A * num_cells
                    + compress_face_table.at(face_offset)*fbs_A;
                ret.block(cbs_A+face_i*fbs_A, 0, fbs_A, 1)
                    = solution.block(face_SOL_offset, 0, fbs_A, 1);
            }
        }

        return ret;
    }


    Matrix<T, Dynamic, 1>
    take_pressure(const Mesh& msh, const typename Mesh::cell_type& cl,
                  const Matrix<T, Dynamic, 1>& sol) const
    {
        auto loc_cl = location(msh, cl);
        if( !(loc_cl == element_location::ON_INTERFACE || loc_cl == loc_zone) )
            throw std::logic_error("Bad cell !!");

        auto celdeg = di.cell_degree();
        auto facdeg = di.face_degree();

        auto cbs_A = vector_cell_basis<Mesh,T>::size(celdeg);
        auto fbs_A = vector_face_basis<Mesh,T>::size(facdeg);
        auto cbs_B = cell_basis<Mesh,T>::size(facdeg);

        auto cell_offset = compress_cells_table.at( offset(msh, cl) );
        auto pres_offset    = cbs_A * num_cells + fbs_A * num_other_faces
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
auto make_stokes_fict_assembler(const Mesh& msh, hho_degree_info hdi, element_location where)
{
    return stokes_fict_assembler<Mesh>(msh, hdi, where);
}




//////////////////////////////////////////////////////


template<typename Mesh>
class stokes_fict_condensed_assembler : public stokes_fict_assembler<Mesh>
{
    using T = typename Mesh::coordinate_type;

    std::vector< Triplet<T> >           triplets;

    std::vector< Matrix<T, Dynamic, Dynamic> > loc_LHS_A;
    std::vector< Matrix<T, Dynamic, 1> > loc_RHS_A;

public:

    SparseMatrix<T>         LHS;
    Matrix<T, Dynamic, 1>   RHS;

    stokes_fict_condensed_assembler(const Mesh& msh, hho_degree_info hdi, element_location where)
        : stokes_fict_assembler<Mesh>(msh, hdi, where)
    {
        loc_LHS_A.resize( this->num_cells );
        loc_RHS_A.resize( this->num_cells );

        auto facdeg = this->di.face_degree();
        auto fbs = vector_face_basis<Mesh,T>::size(facdeg);
        auto cbs_B = cell_basis<Mesh,T>::size(facdeg);

        auto system_size = fbs * this->num_other_faces + cbs_B * this->num_cells + 1;

        LHS = SparseMatrix<T>( system_size, system_size );
        RHS = Matrix<T, Dynamic, 1>::Zero( system_size );
    }

    template<typename Function>
    void
    assemble(const Mesh& msh, const typename Mesh::cell_type& cl,
             const Matrix<T, Dynamic, Dynamic>& lhs_A, const Matrix<T, Dynamic, Dynamic>& lhs_B,
             const Matrix<T, Dynamic, 1>& rhs_A, const Matrix<T, Dynamic, 1>& rhs_B,
             const Function& dirichlet_bf)
    {
        if( !(location(msh, cl) == this->loc_zone ||
              location(msh, cl) == element_location::ON_INTERFACE) )
            return;

        // save local matrices
        size_t cell_offset = this->compress_cells_table.at( offset(msh, cl) );
        loc_LHS_A.at( cell_offset ) = lhs_A;
        loc_RHS_A.at( cell_offset ) = rhs_A;

        auto celdeg = this->di.cell_degree();
        auto facdeg = this->di.face_degree();

        auto cbs_B = cell_basis<Mesh,T>::size(facdeg);
        auto cbs_A = vector_cell_basis<Mesh,T>::size(celdeg);
        auto fbs_A = vector_face_basis<Mesh,T>::size(facdeg);

        auto fcs = faces(msh, cl);
        auto num_faces = fcs.size();
        size_t f_dofs = num_faces * fbs_A;

        // static condensation
        auto mat_sc = stokes_static_condensation_compute(lhs_A, lhs_B, rhs_A, rhs_B,
                                                           cbs_A, f_dofs);
        Matrix<T, Dynamic, Dynamic> lhs_sc = mat_sc.first;
        Matrix<T, Dynamic, 1> rhs_sc = mat_sc.second;

        std::vector<assembly_index> asm_map;
        asm_map.reserve(f_dofs + cbs_B);

        Matrix<T, Dynamic, 1> dirichlet_data = Matrix<T, Dynamic, 1>::Zero(f_dofs);

        for (size_t face_i = 0; face_i < num_faces; face_i++)
        {
            auto fc = fcs[face_i];
            auto face_offset = offset(msh, fc);
            auto face_LHS_offset = this->compress_face_table.at(face_offset) * fbs_A;

            element_location loc_fc = location(msh, fc);
            bool in_dom = (loc_fc == element_location::ON_INTERFACE ||
                           loc_fc == this->loc_zone);

            bool dirichlet = fc.is_boundary && fc.bndtype == boundary::DIRICHLET
                && in_dom;

            for (size_t i = 0; i < fbs_A; i++)
                asm_map.push_back( assembly_index(face_LHS_offset+i, !dirichlet) );

            if (dirichlet)
            {
                Matrix<T, Dynamic, Dynamic> mass = make_vector_mass_matrix(msh, fc, facdeg, this->loc_zone);
                Matrix<T, Dynamic, 1> loc_rhs = make_vector_rhs(msh, fc, facdeg, this->loc_zone, dirichlet_bf);
                dirichlet_data.block(face_i*fbs_A, 0, fbs_A, 1) = mass.ldlt().solve(loc_rhs);
            }
        }

        auto B_offset = fbs_A * this->num_other_faces + cbs_B * cell_offset;
        for (size_t i = 0; i < cbs_B; i++)
            asm_map.push_back( assembly_index(B_offset+i, true) );

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

        // null pressure mean condition
        cell_basis<cuthho_poly_mesh<T>, T> cb(msh, cl, this->di.face_degree());
        auto qpsi = integrate(msh, cl, this->di.face_degree(), this->loc_zone);
        Matrix<T, Dynamic, 1> mult = Matrix<T, Dynamic, 1>::Zero( cbs_B );
        for (auto& qp : qpsi)
        {
            auto phi = cb.eval_basis(qp.first);
            mult += qp.second * phi;
        }
        auto mult_offset = fbs_A * this->num_other_faces + cbs_B * this->num_cells;

        for (size_t i = 0; i < mult.rows(); i++)
        {
            triplets.push_back( Triplet<T>(B_offset+i, mult_offset, mult(i)) );
            triplets.push_back( Triplet<T>(mult_offset, B_offset+i, mult(i)) );
        }

        for (size_t i = 0; i < rhs_sc.rows(); i++)
        {
            if (!asm_map[i].assemble())
                continue;

            RHS(asm_map[i]) += rhs_sc(i);
        }
    } // assemble()

    //// take_velocity
    // loc_mat is the full loc matrix (cell + faces dofs)
    template<typename Function>
    Matrix<T, Dynamic, 1>
    take_velocity(const Mesh& msh, const typename Mesh::cell_type& cl,
                  const Matrix<T, Dynamic, 1>& solution, const Function& dirichlet_bf,
                  const Matrix<T, Dynamic, Dynamic> loc_mat_A, const Matrix<T, Dynamic, 1> loc_rhs_A)
    {
        auto celdeg = this->di.cell_degree();
        auto facdeg = this->di.face_degree();

        auto cbs_A = vector_cell_basis<Mesh,T>::size(celdeg);
        auto fbs_A = vector_face_basis<Mesh,T>::size(facdeg);

        auto fcs = faces(msh, cl);
        auto num_faces = fcs.size();
        size_t f_dofs = num_faces*fbs_A;


        Matrix<T, Dynamic, 1> solF = Matrix<T, Dynamic, 1>::Zero(f_dofs);

        for (size_t face_i = 0; face_i < num_faces; face_i++)
        {
            auto fc = fcs[face_i];

            auto loc_fc = location(msh, fc);
            if( !(loc_fc == element_location::ON_INTERFACE || loc_fc == this->loc_zone) )
                continue;

            bool dirichlet = fc.is_boundary && fc.bndtype == boundary::DIRICHLET;

            if (dirichlet)
            {
                Matrix<T, Dynamic, Dynamic> mass = make_vector_mass_matrix(msh, fc, facdeg);
                Matrix<T, Dynamic, 1> rhs = make_vector_rhs(msh, fc, facdeg, dirichlet_bf);
                solF.block(face_i*fbs_A, 0, fbs_A, 1) = mass.ldlt().solve(rhs);
            }
            else
            {
                auto face_offset = offset(msh, fc);
                auto face_SOL_offset = this->compress_face_table.at(face_offset)*fbs_A;
                solF.block(face_i*fbs_A, 0, fbs_A, 1) = solution.block(face_SOL_offset, 0, fbs_A, 1);
            }
        }

        // Recover the full solution
        return static_condensation_recover(loc_mat_A, loc_rhs_A, cbs_A, f_dofs, solF);
    }


    //// take_velocity
    // uses the local matrices stored previously
    template<typename Function>
    Matrix<T, Dynamic, 1>
    take_velocity(const Mesh& msh, const typename Mesh::cell_type& cl,
                  const Matrix<T, Dynamic, 1>& solution,
                  const Function& dirichlet_bf)
    {
        size_t offset_cl = this->compress_cells_table.at( offset(msh, cl) );

        return take_velocity(msh, cl, solution, dirichlet_bf, loc_LHS_A.at(offset_cl),
                             loc_RHS_A.at(offset_cl));
    }

    Matrix<T, Dynamic, 1>
    take_pressure(const Mesh& msh, const typename Mesh::cell_type& cl,
                  const Matrix<T, Dynamic, 1>& sol) const
    {
        auto loc_cl = location(msh, cl);
        if( !(loc_cl == element_location::ON_INTERFACE || loc_cl == this->loc_zone) )
            throw std::logic_error("Bad cell !!");

        auto celdeg = this->di.cell_degree();
        auto facdeg = this->di.face_degree();

        auto fbs_A = vector_face_basis<Mesh,T>::size(facdeg);
        auto cbs_B = cell_basis<Mesh,T>::size(facdeg);

        auto cell_offset = this->compress_cells_table.at( offset(msh, cl) );
        auto pres_offset = fbs_A * this->num_other_faces + cbs_B * cell_offset;

        Matrix<T, Dynamic, 1> spres = sol.block(pres_offset, 0, cbs_B, 1);
        return spres;
    }

    template<typename Function>
    Matrix<T, Dynamic, 1>
    take_pressure(const Mesh& msh, const typename Mesh::cell_type& cl,
                  const Matrix<T, Dynamic, 1>& sol, const Function& dirichlet_bf)
    {
        return take_pressure(msh, cl, sol);
    }

    void finalize(void)
    {
        LHS.setFromTriplets( triplets.begin(), triplets.end() );
        triplets.clear();
    }
};


template<typename Mesh>
auto make_stokes_fict_condensed_assembler(const Mesh& msh, hho_degree_info hdi, element_location where)
{
    return stokes_fict_condensed_assembler<Mesh>(msh, hdi, where);
}


//////////////////////////////////////////////////////
// assembler with full static condensation
//  (velocity + pressure)

template<typename Mesh>
class stokes_fict_condensed_assembler2 : public stokes_fict_assembler<Mesh>
{
    using T = typename Mesh::coordinate_type;

    std::vector< Triplet<T> >           triplets;

    std::vector< Matrix<T, Dynamic, Dynamic> > loc_LHS_A, loc_LHS_B;
    std::vector< Matrix<T, Dynamic, 1> > loc_RHS_A, loc_RHS_B;

public:

    SparseMatrix<T>         LHS;
    Matrix<T, Dynamic, 1>   RHS;

    stokes_fict_condensed_assembler2(const Mesh& msh, hho_degree_info hdi, element_location where)
        : stokes_fict_assembler<Mesh>(msh, hdi, where)
    {
        loc_LHS_A.resize( this->num_cells );
        loc_LHS_B.resize( this->num_cells );
        loc_RHS_A.resize( this->num_cells );
        loc_RHS_B.resize( this->num_cells );

        auto facdeg = this->di.face_degree();
        auto fbs = vector_face_basis<Mesh,T>::size(facdeg);
        auto cbs_B = cell_basis<Mesh,T>::size(facdeg);

        auto system_size = fbs * this->num_other_faces + this->num_cells + 1;

        LHS = SparseMatrix<T>( system_size, system_size );
        RHS = Matrix<T, Dynamic, 1>::Zero( system_size );
    }

    template<typename Function>
    void
    assemble(const Mesh& msh, const typename Mesh::cell_type& cl,
             const Matrix<T, Dynamic, Dynamic>& lhs_A, const Matrix<T, Dynamic, Dynamic>& lhs_B,
             const Matrix<T, Dynamic, 1>& rhs_A, const Matrix<T, Dynamic, 1>& rhs_B,
             const Function& dirichlet_bf)
    {
        if( !(location(msh, cl) == this->loc_zone ||
              location(msh, cl) == element_location::ON_INTERFACE) )
            return;

        // save local matrices
        size_t cell_offset = this->compress_cells_table.at( offset(msh, cl) );
        loc_LHS_A.at( cell_offset ) = lhs_A;
        loc_LHS_B.at( cell_offset ) = lhs_B;
        loc_RHS_A.at( cell_offset ) = rhs_A;
        loc_RHS_B.at( cell_offset ) = rhs_B;

        auto celdeg = this->di.cell_degree();
        auto facdeg = this->di.face_degree();

        auto cbs_B = cell_basis<Mesh,T>::size(facdeg);
        auto cbs_A = vector_cell_basis<Mesh,T>::size(celdeg);
        auto fbs_A = vector_face_basis<Mesh,T>::size(facdeg);

        auto fcs = faces(msh, cl);
        auto num_faces = fcs.size();
        size_t f_dofs = num_faces * fbs_A;


        // static condensation
        Matrix<T, Dynamic, Dynamic> lhs_sc;
        Matrix<T, Dynamic, 1> rhs_sc;
        if(facdeg == 0) // condensate only cell velocity dofs
        {
            auto mat_sc = stokes_static_condensation_compute(lhs_A, lhs_B, rhs_A, rhs_B,
                                                             cbs_A, f_dofs);
            lhs_sc = mat_sc.first;
            rhs_sc = mat_sc.second;
        }
        else // full static condensation
        {
            // compute mult_C
            cell_basis<cuthho_poly_mesh<T>, T> s_cb(msh, cl, this->di.face_degree());
            auto qpsi = integrate(msh, cl, this->di.face_degree(), this->loc_zone);
            Matrix<T, Dynamic, 1> mult_C = Matrix<T, Dynamic, 1>::Zero( cbs_B - 1);
            T area = 0.0;
            for (auto& qp : qpsi)
            {
                auto s_phi = s_cb.eval_basis(qp.first);
                mult_C -= qp.second * s_phi.tail(cbs_B - 1);
                area += qp.second;
            }
            mult_C = mult_C / area;

            auto mat_sc = stokes_full_static_condensation_compute
                (lhs_A, lhs_B, rhs_A, rhs_B, mult_C, cbs_A, f_dofs);
            lhs_sc = mat_sc.first;
            rhs_sc = mat_sc.second;
        }

        std::vector<assembly_index> asm_map;
        asm_map.reserve(f_dofs + 1);

        Matrix<T, Dynamic, 1> dirichlet_data = Matrix<T, Dynamic, 1>::Zero(f_dofs);

        for (size_t face_i = 0; face_i < num_faces; face_i++)
        {
            auto fc = fcs[face_i];
            auto face_offset = offset(msh, fc);
            auto face_LHS_offset = this->compress_face_table.at(face_offset) * fbs_A;

            element_location loc_fc = location(msh, fc);
            bool in_dom = (loc_fc == element_location::ON_INTERFACE ||
                           loc_fc == this->loc_zone);

            bool dirichlet = fc.is_boundary && fc.bndtype == boundary::DIRICHLET
                && in_dom;

            for (size_t i = 0; i < fbs_A; i++)
                asm_map.push_back( assembly_index(face_LHS_offset+i, !dirichlet) );

            if (dirichlet)
            {
                Matrix<T, Dynamic, Dynamic> mass = make_vector_mass_matrix(msh, fc, facdeg, this->loc_zone);
                Matrix<T, Dynamic, 1> loc_rhs = make_vector_rhs(msh, fc, facdeg, this->loc_zone, dirichlet_bf);
                dirichlet_data.block(face_i*fbs_A, 0, fbs_A, 1) = mass.ldlt().solve(loc_rhs);
            }
        }

        auto B_offset = fbs_A * this->num_other_faces + cell_offset;
        asm_map.push_back( assembly_index(B_offset, true) );

        // LHS
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

        // null pressure mean condition
        auto mult_offset = fbs_A * this->num_other_faces + this->num_cells;
        auto area = measure(msh, cl, this->loc_zone);
        triplets.push_back( Triplet<T>(B_offset, mult_offset, area) );
        triplets.push_back( Triplet<T>(mult_offset, B_offset, area) );

        // RHS
        for (size_t i = 0; i < rhs_sc.rows(); i++)
        {
            if (!asm_map[i].assemble())
                continue;

            RHS(asm_map[i]) += rhs_sc(i);
        }
    } // assemble()

    //// take_velocity
    // loc_mat is the full loc matrix (cell + faces dofs)
    template<typename Function>
    Matrix<T, Dynamic, 1>
    take_velocity(const Mesh& msh, const typename Mesh::cell_type& cl,
                  const Matrix<T, Dynamic, 1>& solution, const Function& dirichlet_bf,
                  const Matrix<T, Dynamic, Dynamic> loc_mat_A,
                  const Matrix<T, Dynamic, Dynamic> loc_mat_B,
                  const Matrix<T, Dynamic, 1> loc_rhs_A, const Matrix<T, Dynamic, 1> loc_rhs_B)
    {
        auto celdeg = this->di.cell_degree();
        auto facdeg = this->di.face_degree();

        auto cbs_A = vector_cell_basis<Mesh,T>::size(celdeg);
        auto fbs_A = vector_face_basis<Mesh,T>::size(facdeg);

        auto fcs = faces(msh, cl);
        auto num_faces = fcs.size();
        size_t f_dofs = num_faces*fbs_A;


        Matrix<T, Dynamic, 1> solF = Matrix<T, Dynamic, 1>::Zero(f_dofs);

        for (size_t face_i = 0; face_i < num_faces; face_i++)
        {
            auto fc = fcs[face_i];

            auto loc_fc = location(msh, fc);
            if( !(loc_fc == element_location::ON_INTERFACE || loc_fc == this->loc_zone) )
                continue;

            bool dirichlet = fc.is_boundary && fc.bndtype == boundary::DIRICHLET;

            if (dirichlet)
            {
                Matrix<T, Dynamic, Dynamic> mass = make_vector_mass_matrix(msh, fc, facdeg);
                Matrix<T, Dynamic, 1> rhs = make_vector_rhs(msh, fc, facdeg, dirichlet_bf);
                solF.block(face_i*fbs_A, 0, fbs_A, 1) = mass.ldlt().solve(rhs);
            }
            else
            {
                auto face_offset = offset(msh, fc);
                auto face_SOL_offset = this->compress_face_table.at(face_offset)*fbs_A;
                solF.block(face_i*fbs_A, 0, fbs_A, 1) = solution.block(face_SOL_offset, 0, fbs_A, 1);
            }
        }

        // Recover the full solution
        if(facdeg == 0)
            return static_condensation_recover(loc_mat_A, loc_rhs_A, cbs_A, f_dofs, solF);

        // at this point facdeg > 0
        Matrix<T, Dynamic, 1> sol_sc = Matrix<T, Dynamic, 1>::Zero(f_dofs + 1);
        sol_sc.head(f_dofs) = solF;
        size_t cell_offset = this->compress_cells_table.at( offset(msh, cl) );
        size_t B_offset = fbs_A * this->num_other_faces + cell_offset;
        sol_sc(f_dofs) = solution(B_offset);

        auto cbs_B = cell_basis<Mesh,T>::size(facdeg);
        cell_basis<cuthho_poly_mesh<T>, T> s_cb(msh, cl, this->di.face_degree());
        auto qpsi = integrate(msh, cl, this->di.face_degree(), this->loc_zone);
        Matrix<T, Dynamic, 1> mult_C = Matrix<T, Dynamic, 1>::Zero( cbs_B - 1);
        T area = 0.0;
        for (auto& qp : qpsi)
        {
            auto s_phi = s_cb.eval_basis(qp.first);
            mult_C -= qp.second * s_phi.tail(cbs_B - 1);
            area += qp.second;
        }
        mult_C = mult_C / area;

        return stokes_full_static_condensation_recover_v
            (loc_mat_A, loc_mat_B, loc_rhs_A, loc_rhs_B, mult_C, cbs_A, f_dofs, sol_sc);
    }


    //// take_velocity
    // uses the local matrices stored previously
    template<typename Function>
    Matrix<T, Dynamic, 1>
    take_velocity(const Mesh& msh, const typename Mesh::cell_type& cl,
                  const Matrix<T, Dynamic, 1>& solution,
                  const Function& dirichlet_bf)
    {
        size_t offset_cl = this->compress_cells_table.at( offset(msh, cl) );

        return take_velocity(msh, cl, solution, dirichlet_bf, loc_LHS_A.at(offset_cl),
                             loc_LHS_B.at(offset_cl),
                             loc_RHS_A.at(offset_cl), loc_RHS_B.at(offset_cl) );
    }

    template<typename Function>
    Matrix<T, Dynamic, 1>
    take_pressure(const Mesh& msh, const typename Mesh::cell_type& cl,
                  const Matrix<T, Dynamic, 1>& sol, const Function& dirichlet_bf)
    {
        auto loc_cl = location(msh, cl);
        if( !(loc_cl == element_location::ON_INTERFACE || loc_cl == this->loc_zone) )
            throw std::logic_error("Bad cell !!");

        auto celdeg = this->di.cell_degree();
        auto facdeg = this->di.face_degree();

        auto cbs_A = vector_cell_basis<Mesh,T>::size(celdeg);
        auto fbs_A = vector_face_basis<Mesh,T>::size(facdeg);
        auto cbs_B = cell_basis<Mesh,T>::size(facdeg);

        auto cell_offset = this->compress_cells_table.at( offset(msh, cl) );
        auto pres_offset = fbs_A * this->num_other_faces + cell_offset;

        Matrix<T, Dynamic, 1> spres = sol.block(pres_offset, 0, 1, 1);
        if(facdeg == 0)
            return spres;

        // at this point facdeg > 0
        auto fcs = faces(msh, cl);
        auto num_faces = fcs.size();
        size_t f_dofs = num_faces*fbs_A;


        Matrix<T, Dynamic, 1> solF = Matrix<T, Dynamic, 1>::Zero(f_dofs);

        for (size_t face_i = 0; face_i < num_faces; face_i++)
        {
            auto fc = fcs[face_i];

            auto loc_fc = location(msh, fc);
            if( !(loc_fc == element_location::ON_INTERFACE || loc_fc == this->loc_zone) )
                continue;

            bool dirichlet = fc.is_boundary && fc.bndtype == boundary::DIRICHLET;

            if (dirichlet)
            {
                Matrix<T, Dynamic, Dynamic> mass = make_vector_mass_matrix(msh, fc, facdeg);
                Matrix<T, Dynamic, 1> rhs = make_vector_rhs(msh, fc, facdeg, dirichlet_bf);
                solF.block(face_i*fbs_A, 0, fbs_A, 1) = mass.ldlt().solve(rhs);
            }
            else
            {
                auto face_offset = offset(msh, fc);
                auto face_SOL_offset = this->compress_face_table.at(face_offset)*fbs_A;
                solF.block(face_i*fbs_A, 0, fbs_A, 1) = sol.block(face_SOL_offset, 0, fbs_A, 1);
            }
        }
        Matrix<T, Dynamic, 1> sol_sc = Matrix<T, Dynamic, 1>::Zero(f_dofs + 1);
        sol_sc.head(f_dofs) = solF;
        size_t B_offset = fbs_A * this->num_other_faces + cell_offset;
        sol_sc(f_dofs) = sol(B_offset);

        cell_basis<cuthho_poly_mesh<T>, T> s_cb(msh, cl, this->di.face_degree());
        auto qpsi = integrate(msh, cl, this->di.face_degree(), this->loc_zone);
        Matrix<T, Dynamic, 1> mult_C = Matrix<T, Dynamic, 1>::Zero( cbs_B - 1);
        T area = 0.0;
        for (auto& qp : qpsi)
        {
            auto s_phi = s_cb.eval_basis(qp.first);
            mult_C -= qp.second * s_phi.tail(cbs_B - 1);
            area += qp.second;
        }
        mult_C = mult_C / area;

        return stokes_full_static_condensation_recover_p
            (loc_LHS_A.at(cell_offset), loc_LHS_B.at(cell_offset),
             loc_RHS_A.at(cell_offset), loc_RHS_B.at(cell_offset), mult_C,
             cbs_A, f_dofs, sol_sc);
    }

    void finalize(void)
    {
        LHS.setFromTriplets( triplets.begin(), triplets.end() );
        triplets.clear();
    }
};


template<typename Mesh>
auto make_stokes_fict_condensed_assembler2(const Mesh& msh, hho_degree_info hdi, element_location where)
{
    return stokes_fict_condensed_assembler2<Mesh>(msh, hdi, where);
}

///////////////////////////////   STOKES INTERFACE


template<typename Mesh>
class stokes_interface_assembler
{
    using T = typename Mesh::coordinate_type;
    std::vector<size_t>                 cell_table, face_table;
    size_t num_all_cells, num_all_faces, num_other_faces;

    hho_degree_info                     di;

    std::vector< Triplet<T> >           triplets;

public:

    SparseMatrix<T>         LHS;
    Matrix<T, Dynamic, 1>   RHS;

    stokes_interface_assembler(const Mesh& msh, hho_degree_info hdi)
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
        num_other_faces = num_all_faces - num_dirichlet_faces;
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
        auto pdeg = facdeg;

        auto cbs = vector_cell_basis<Mesh,T>::size(celdeg);
        auto fbs = vector_face_basis<Mesh,T>::size(facdeg);
        auto pbs = cell_basis<Mesh,T>::size(pdeg);

        auto system_size = (cbs+pbs) * num_all_cells + fbs * num_other_faces + 1;

        LHS = SparseMatrix<T>( system_size, system_size );
        RHS = Matrix<T, Dynamic, 1>::Zero( system_size );
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
        auto pdeg = facdeg;

        auto cbs = vector_cell_basis<Mesh,T>::size(celdeg);
        auto fbs = vector_face_basis<Mesh,T>::size(facdeg);
        auto pbs = cell_basis<Mesh,T>::size(pdeg);

        auto fcs = faces(msh, cl);
        auto num_faces = fcs.size();

        std::vector<assembly_index> asm_map;
        asm_map.reserve(cbs + num_faces*fbs + pbs);

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
                Matrix<T, Dynamic, Dynamic> mass = make_vector_mass_matrix(msh, fc, facdeg);
                Matrix<T, Dynamic, 1> rhs = make_vector_rhs(msh, fc, facdeg, dirichlet_bf);
                dirichlet_data.block(cbs+face_i*fbs, 0, fbs, 1) = mass.ldlt().solve(rhs);
            }
        }

        auto B_LHS_offset = num_all_cells * cbs + num_other_faces * fbs
            + cell_table.at(cell_offset) * pbs;

        for (size_t i = 0; i < pbs; i++)
            asm_map.push_back( assembly_index(B_LHS_offset+i, true) );

        assert( asm_map.size() == lhs.rows() && asm_map.size() == lhs.cols() );

        // LHS
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

        // null mean pressure condition
        cell_basis<cuthho_poly_mesh<T>, T> pb(msh, cl, pdeg);
        auto qpsi = integrate(msh, cl, pdeg);
        Matrix<T, Dynamic, 1> mult = Matrix<T, Dynamic, 1>::Zero( pbs );
        for (auto& qp : qpsi)
        {
            auto phi = pb.eval_basis(qp.first);
            mult += qp.second * phi;
        }
        auto mult_offset = cbs * num_all_cells + fbs * num_other_faces + pbs * num_all_cells;

        for (size_t i = 0; i < mult.rows(); i++)
        {
            triplets.push_back( Triplet<T>(B_LHS_offset+i, mult_offset, mult(i)) );
            triplets.push_back( Triplet<T>(mult_offset, B_LHS_offset+i, mult(i)) );
        }

        // RHS
        for (size_t i = 0; i < rhs.rows(); i++)
        {
            if (!asm_map[i].assemble())
                continue;

            RHS(asm_map[i]) += rhs(i);
        }

    } // assemble()

    void
    assemble_cut(const Mesh& msh, const typename Mesh::cell_type& cl,
             const Matrix<T, Dynamic, Dynamic>& lhs, const Matrix<T, Dynamic, 1>& rhs)
    {
        if (location(msh, cl) != element_location::ON_INTERFACE)
            throw std::invalid_argument("Cut cell expected.");

        auto celdeg = di.cell_degree();
        auto facdeg = di.face_degree();
        auto pdeg = facdeg;

        auto cbs = vector_cell_basis<Mesh,T>::size(celdeg);
        auto fbs = vector_face_basis<Mesh,T>::size(facdeg);
        auto pbs = cell_basis<Mesh,T>::size(pdeg);

        auto fcs = faces(msh, cl);
        auto num_faces = fcs.size();

        std::vector<assembly_index> asm_map;
        asm_map.reserve( 2*(cbs + num_faces*fbs + pbs) );

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

            for (size_t i = 0; i < fbs; i++)
                asm_map.push_back( assembly_index(face_LHS_offset+i, true) );
        }

        auto B_LHS_offset = num_all_cells * cbs + num_other_faces * fbs
            + cell_table.at(cell_offset) * pbs;

        for (size_t i = 0; i < 2*pbs; i++)
            asm_map.push_back( assembly_index(B_LHS_offset+i, true) );

        assert( asm_map.size() == lhs.rows() && asm_map.size() == lhs.cols() );

        // LHS
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

        // null mean pressure condition
        cell_basis<cuthho_poly_mesh<T>, T> pb(msh, cl, pdeg);
        auto qpsi_n = integrate(msh, cl, pdeg, element_location::IN_NEGATIVE_SIDE);
        Matrix<T, Dynamic, 1> mult_n = Matrix<T, Dynamic, 1>::Zero( pbs );
        for (auto& qp : qpsi_n)
        {
            auto phi = pb.eval_basis(qp.first);
            mult_n += qp.second * phi;
        }
        auto qpsi_p = integrate(msh, cl, pdeg, element_location::IN_POSITIVE_SIDE);
        Matrix<T, Dynamic, 1> mult_p = Matrix<T, Dynamic, 1>::Zero( pbs );
        for (auto& qp : qpsi_p)
        {
            auto phi = pb.eval_basis(qp.first);
            mult_p += qp.second * phi;
        }

        auto mult_offset = cbs * num_all_cells + fbs * num_other_faces + pbs * num_all_cells;
        auto B_offset = cbs * num_all_cells + fbs * num_other_faces
            + pbs * cell_table.at(cell_offset);
        for (size_t i = 0; i < mult_n.rows(); i++)
        {
            triplets.push_back( Triplet<T>(B_offset+i, mult_offset, mult_n(i)) );
            triplets.push_back( Triplet<T>(mult_offset, B_offset+i, mult_n(i)) );
        }
        for (size_t i = 0; i < mult_p.rows(); i++)
        {
            triplets.push_back( Triplet<T>(B_offset+pbs+i, mult_offset, mult_p(i)) );
            triplets.push_back( Triplet<T>(mult_offset, B_offset+pbs+i, mult_p(i)) );
        }

        // RHS
        for (size_t i = 0; i < rhs.rows(); i++)
        {
            if (!asm_map[i].assemble())
                continue;

            RHS(asm_map[i]) += rhs(i);
        }

    } // assemble_cut()


    template<typename Function>
    Matrix<T, Dynamic, 1>
    take_velocity(const Mesh& msh, const typename Mesh::cell_type& cl,
                  const Matrix<T, Dynamic, 1>& solution,
                  const Function& dirichlet_bf, element_location where)
    {
        auto celdeg = di.cell_degree();
        auto facdeg = di.face_degree();

        auto cbs = vector_cell_basis<Mesh,T>::size(celdeg);
        auto fbs = vector_face_basis<Mesh,T>::size(facdeg);

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
                Matrix<T, Dynamic, Dynamic> mass = make_vector_mass_matrix(msh, fc, facdeg);
                Matrix<T, Dynamic, 1> rhs = make_vector_rhs(msh, fc, facdeg, dirichlet_bf);
                ret.block(cbs+face_i*fbs, 0, fbs, 1) = mass.ldlt().solve(rhs);
            }
            else
            {
                ret.block(cbs+face_i*fbs, 0, fbs, 1) = solution.block(face_SOL_offset, 0, fbs, 1);
            }
        }

        return ret;
    }

    Matrix<T, Dynamic, 1>
    take_pressure(const Mesh& msh, const typename Mesh::cell_type& cl,
                  const Matrix<T, Dynamic, 1>& sol, element_location where) const
    {
        auto celdeg = di.cell_degree();
        auto facdeg = di.face_degree();
        auto pdeg = facdeg;

        auto cbs = vector_cell_basis<Mesh,T>::size(celdeg);
        auto fbs = vector_face_basis<Mesh,T>::size(facdeg);
        auto pbs = cell_basis<Mesh,T>::size(pdeg);

        auto cell_offset = offset(msh, cl);
        size_t P_SOL_offset;
        if( location(msh, cl) == element_location::ON_INTERFACE)
        {
            if (where == element_location::IN_NEGATIVE_SIDE)
                P_SOL_offset = cell_table.at(cell_offset) * pbs;
            else if (where == element_location::IN_POSITIVE_SIDE)
                P_SOL_offset = cell_table.at(cell_offset) * pbs + pbs;
            else
                throw std::invalid_argument("Invalid location");
        }
        else
            P_SOL_offset = cell_table.at(cell_offset) * pbs;

        P_SOL_offset += num_all_cells * cbs + num_other_faces * fbs;

        Matrix<T, Dynamic, 1> spres = sol.block(P_SOL_offset, 0, pbs, 1);
        return spres;
    }

    void finalize(void)
    {
        LHS.setFromTriplets( triplets.begin(), triplets.end() );
        triplets.clear();
    }
};


template<typename Mesh>
auto make_stokes_interface_assembler(const Mesh& msh, hho_degree_info hdi)
{
    return stokes_interface_assembler<Mesh>(msh, hdi);
}

////////////////////////////////


template<typename Mesh>
class stokes_interface_condensed_assembler
{
    using T = typename Mesh::coordinate_type;
    std::vector<size_t>                 cell_table, face_table;
    size_t num_all_cells, num_all_faces, num_other_faces;

    hho_degree_info                     di;

    std::vector< Triplet<T> >           triplets;
    std::vector< Matrix<T, Dynamic, Dynamic> > loc_LHS;
    std::vector< Matrix<T, Dynamic, 1> > loc_RHS;

public:

    SparseMatrix<T>         LHS;
    Matrix<T, Dynamic, 1>   RHS;

    stokes_interface_condensed_assembler(const Mesh& msh, hho_degree_info hdi)
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
        num_other_faces = num_all_faces - num_dirichlet_faces;
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


        loc_LHS.resize( msh.cells.size() );
        loc_RHS.resize( msh.cells.size() );

        auto celdeg = di.cell_degree();
        auto facdeg = di.face_degree();
        auto pdeg = facdeg;

        auto cbs = vector_cell_basis<Mesh,T>::size(celdeg);
        auto fbs = vector_face_basis<Mesh,T>::size(facdeg);
        auto pbs = cell_basis<Mesh,T>::size(pdeg);

        auto system_size = fbs * num_other_faces + msh.cells.size() + 1;

        LHS = SparseMatrix<T>( system_size, system_size );
        RHS = Matrix<T, Dynamic, 1>::Zero( system_size );
    }

    Matrix<T, Dynamic, 1>
    compute_multC_interface(const Mesh& msh, const typename Mesh::cell_type& cl, const size_t pdeg)
    {
        // compute mult_C
        auto pbs = cell_basis<Mesh,T>::size(pdeg);
        size_t p_dofs = 2*pbs;
        cell_basis<cuthho_poly_mesh<T>, T> pb(msh, cl, pdeg);
        auto qpsi_n = integrate(msh, cl, pdeg, element_location::IN_NEGATIVE_SIDE);
        Matrix<T, Dynamic, 1> mult_C = Matrix<T, Dynamic, 1>::Zero( p_dofs - 1 );
        T area_n = 0.0;
        if( pdeg == 0 )
        {
            for (auto& qp : qpsi_n)
            {
                area_n += qp.second;
            }
        }
        else // if( facdeg > 0 )
        {
            for (auto& qp : qpsi_n)
            {
                auto phi = pb.eval_basis(qp.first);
                mult_C.head(pbs - 1) -= qp.second * phi.tail(pbs - 1);
                area_n += qp.second;
            }
        }

        auto qpsi_p = integrate(msh, cl, pdeg, element_location::IN_POSITIVE_SIDE);
        for (auto& qp : qpsi_p)
        {
            auto phi = pb.eval_basis(qp.first);
            mult_C.block(pbs - 1, 0, pbs, 1) -= qp.second * phi;
        }

        mult_C = mult_C / area_n;

        return mult_C;
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
        auto pdeg = facdeg;

        auto cbs = vector_cell_basis<Mesh,T>::size(celdeg);
        auto fbs = vector_face_basis<Mesh,T>::size(facdeg);
        auto pbs = cell_basis<Mesh,T>::size(pdeg);

        auto fcs = faces(msh, cl);
        auto num_faces = fcs.size();
        auto f_dofs = num_faces * fbs;
        auto v_dofs = cbs + f_dofs;


        // static condensation
        Matrix<T, Dynamic, Dynamic> lhs_A = lhs.block(0, 0, v_dofs, v_dofs);
        Matrix<T, Dynamic, Dynamic> lhs_B = lhs.block(v_dofs, 0, pbs, v_dofs);
        Matrix<T, Dynamic, Dynamic> lhs_C = lhs.block(v_dofs, v_dofs, pbs, pbs);
        Matrix<T, Dynamic, 1> rhs_A = rhs.head( v_dofs );
        Matrix<T, Dynamic, 1> rhs_B = rhs.tail( pbs );


        Matrix<T, Dynamic, Dynamic> lhs_sc;
        Matrix<T, Dynamic, 1> rhs_sc;
        if(facdeg == 0) // condensate only cell velocity dofs
        {
            auto mat_sc = stokes_static_condensation_compute(lhs_A, lhs_B, rhs_A, rhs_B,
                                                             cbs, f_dofs);
            lhs_sc = mat_sc.first;
            rhs_sc = mat_sc.second;
        }
        else // full static condensation
        {
            // compute mult_C
            cell_basis<cuthho_poly_mesh<T>, T> s_cb(msh, cl, pdeg);
            auto qpsi = integrate(msh, cl, pdeg);
            Matrix<T, Dynamic, 1> mult_C = Matrix<T, Dynamic, 1>::Zero( pbs - 1 );
            T area = 0.0;
            for (auto& qp : qpsi)
            {
                auto s_phi = s_cb.eval_basis(qp.first);
                mult_C -= qp.second * s_phi.tail(pbs - 1);
                area += qp.second;
            }
            mult_C = mult_C / area;

            auto mat_sc = stokes_full_static_condensation_compute
                (lhs_A, lhs_B, rhs_A, rhs_B, mult_C, cbs, f_dofs);
            lhs_sc = mat_sc.first;
            rhs_sc = mat_sc.second;
        }

        std::vector<assembly_index> asm_map;
        asm_map.reserve(f_dofs + 1);

        Matrix<T, Dynamic, 1> dirichlet_data = Matrix<T, Dynamic, 1>::Zero(f_dofs);

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
                Matrix<T, Dynamic, Dynamic> mass = make_vector_mass_matrix(msh, fc, facdeg);
                Matrix<T, Dynamic, 1> rhs = make_vector_rhs(msh, fc, facdeg, dirichlet_bf);
                dirichlet_data.block(face_i*fbs, 0, fbs, 1) = mass.ldlt().solve(rhs);
            }
        }

        auto B_LHS_offset = num_other_faces * fbs + cell_offset;
        asm_map.push_back( assembly_index(B_LHS_offset, true) );

        assert( asm_map.size() == lhs_sc.rows() && asm_map.size() == lhs_sc.cols() );

        // LHS
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

        // null mean pressure condition
        auto mult_offset = fbs * num_other_faces + msh.cells.size();
        auto area = measure(msh,cl);
        triplets.push_back( Triplet<T>(B_LHS_offset, mult_offset, area) );
        triplets.push_back( Triplet<T>(mult_offset, B_LHS_offset, area) );

        // RHS
        for (size_t i = 0; i < rhs_sc.rows(); i++)
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
        auto pdeg = facdeg;

        auto cbs = vector_cell_basis<Mesh,T>::size(celdeg);
        auto fbs = vector_face_basis<Mesh,T>::size(facdeg);
        auto pbs = cell_basis<Mesh,T>::size(pdeg);

        auto fcs = faces(msh, cl);
        auto num_faces = fcs.size();
        size_t f_dofs = 2*num_faces * fbs;
        size_t v_dofs = 2*cbs + f_dofs;
        size_t p_dofs = 2*pbs;


        // static condensation
        Matrix<T, Dynamic, Dynamic> lhs_A = lhs.block(0, 0, v_dofs, v_dofs);
        Matrix<T, Dynamic, Dynamic> lhs_B = lhs.block(v_dofs, 0, p_dofs, v_dofs);
        Matrix<T, Dynamic, Dynamic> lhs_C = lhs.block(v_dofs, v_dofs, p_dofs, p_dofs);
        Matrix<T, Dynamic, 1> rhs_A = rhs.head( v_dofs );
        Matrix<T, Dynamic, 1> rhs_B = rhs.tail( p_dofs );

        Matrix<T, Dynamic, Dynamic> lhs_sc;
        Matrix<T, Dynamic, 1> rhs_sc;

        auto mult_C = compute_multC_interface(msh, cl, pdeg);

        auto mat_sc = stokes_full_static_condensation_compute
            (lhs_A, lhs_B, lhs_C, rhs_A, rhs_B, mult_C, 2*cbs, f_dofs);
        lhs_sc = mat_sc.first;
        rhs_sc = mat_sc.second;


        std::vector<assembly_index> asm_map;
        asm_map.reserve( f_dofs + 1 );

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

        auto B_LHS_offset = num_other_faces * fbs + cell_offset;
        asm_map.push_back( assembly_index(B_LHS_offset, true) );

        assert( asm_map.size() == lhs_sc.rows() && asm_map.size() == lhs_sc.cols() );

        // LHS
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

        auto mult_offset = fbs * num_other_faces + msh.cells.size();
        T area = measure(msh, cl);
        triplets.push_back( Triplet<T>(B_LHS_offset, mult_offset, area) );
        triplets.push_back( Triplet<T>(mult_offset, B_LHS_offset, area) );

        // RHS
        for (size_t i = 0; i < rhs_sc.rows(); i++)
        {
            if (!asm_map[i].assemble())
                continue;

            RHS(asm_map[i]) += rhs_sc(i);
        }

    } // assemble_cut()

    template<typename Function>
    Matrix<T, Dynamic, 1>
    get_sol_F(const Mesh& msh, const typename Mesh::cell_type& cl,
              const Matrix<T, Dynamic, 1>& solution, const Function& dirichlet_bf)
    {
        auto facdeg = di.face_degree();
        auto fbs = vector_face_basis<Mesh,T>::size(facdeg);
        auto fcs = faces(msh, cl);
        auto num_faces = fcs.size();
        size_t f_dofs = num_faces*fbs;

        Matrix<T, Dynamic, 1> solF;

        if( location(msh, cl) == element_location::ON_INTERFACE )
            solF = Matrix<T, Dynamic, 1>::Zero(2*f_dofs);
        else
            solF = Matrix<T, Dynamic, 1>::Zero(f_dofs);

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
                assert( solF.rows() == f_dofs );
                Matrix<T, Dynamic, Dynamic> mass = make_vector_mass_matrix(msh, fc, facdeg);
                Matrix<T, Dynamic, 1> rhs = make_vector_rhs(msh, fc, facdeg, dirichlet_bf);
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
        return solF;
    }

    template<typename Function>
    Matrix<T, Dynamic, 1>
    take_velocity(const Mesh& msh, const typename Mesh::cell_type& cl,
                  const Matrix<T, Dynamic, 1>& solution,
                  const Function& dirichlet_bf, element_location where)
    {
        auto celdeg = di.cell_degree();
        auto facdeg = di.face_degree();
        auto pdeg = facdeg;

        auto cbs = vector_cell_basis<Mesh,T>::size(celdeg);
        auto fbs = vector_face_basis<Mesh,T>::size(facdeg);
        auto pbs = cell_basis<Mesh,T>::size(pdeg);

        auto cell_offset        = offset(msh, cl);
        auto lhs = loc_LHS.at(cell_offset);
        auto rhs = loc_RHS.at(cell_offset);

        auto fcs = faces(msh, cl);
        auto num_faces = fcs.size();
        size_t f_dofs = num_faces*fbs;

        auto v_dofs = cbs + f_dofs;
        auto p_dofs = pbs;
        if( location(msh, cl) == element_location::ON_INTERFACE )
        {
            v_dofs = 2 * v_dofs;
            p_dofs = 2 * p_dofs;
        }

        auto solF = get_sol_F(msh, cl, solution, dirichlet_bf);

        // Recover the full solution
        Matrix<T, Dynamic, Dynamic> lhs_A = lhs.block(0, 0, v_dofs, v_dofs);
        Matrix<T, Dynamic, Dynamic> lhs_B = lhs.block(v_dofs, 0, p_dofs, v_dofs);
        Matrix<T, Dynamic, Dynamic> lhs_C = lhs.block(v_dofs, v_dofs, p_dofs, p_dofs);
        Matrix<T, Dynamic, 1> rhs_A = rhs.head( v_dofs );
        Matrix<T, Dynamic, 1> rhs_B = rhs.tail( p_dofs );

        if( location(msh, cl) != element_location::ON_INTERFACE )
        {
            if(facdeg == 0)
                return static_condensation_recover(lhs_A, rhs_A, cbs, f_dofs, solF);

            // else -> at this point facdeg > 0
            // compute sol_sc
            Matrix<T, Dynamic, 1> sol_sc = Matrix<T, Dynamic, 1>::Zero(f_dofs + 1);
            sol_sc.head(f_dofs) = solF;
            size_t B_offset = fbs * num_other_faces + cell_offset;
            sol_sc(f_dofs) = solution(B_offset);

            // compute mult_C
            cell_basis<cuthho_poly_mesh<T>, T> pb(msh, cl, pdeg);
            auto qpsi = integrate(msh, cl, pdeg);
            Matrix<T, Dynamic, 1> mult_C = Matrix<T, Dynamic, 1>::Zero( pbs - 1);
            T area = 0.0;
            for (auto& qp : qpsi)
            {
                auto s_phi = pb.eval_basis(qp.first);
                mult_C -= qp.second * s_phi.tail(pbs - 1);
                area += qp.second;
            }
            mult_C = mult_C / area;

            return stokes_full_static_condensation_recover_v
                (lhs_A, lhs_B, lhs_C, rhs_A, rhs_B, mult_C, cbs, f_dofs, sol_sc);
        }
        // else -> location(msh, cl) == element_location::ON_INTERFACE

        Matrix<T, Dynamic, 1> ret = Matrix<T, Dynamic, 1>::Zero(cbs + f_dofs);

        // compute sol_sc
        Matrix<T, Dynamic, 1> sol_sc = Matrix<T, Dynamic, 1>::Zero(2*f_dofs + 1);
        sol_sc.head(2*f_dofs) = solF;
        size_t B_offset = fbs * num_other_faces + cell_offset;
        sol_sc(2*f_dofs) = solution(B_offset);

        auto mult_C = compute_multC_interface(msh, cl, pdeg);

        auto loc_sol = stokes_full_static_condensation_recover_v
            (lhs_A, lhs_B, lhs_C, rhs_A, rhs_B, mult_C, 2*cbs, 2*f_dofs, sol_sc);

        if( where == element_location::IN_NEGATIVE_SIDE )
        {
            ret.head(cbs) = loc_sol.head(cbs);
            ret.tail(f_dofs) = solF.head(f_dofs);
        }

        if( where == element_location::IN_POSITIVE_SIDE )
        {
            ret.head(cbs) = loc_sol.block(cbs, 0, cbs, 1);
            ret.tail(f_dofs) = solF.tail(f_dofs);
        }
        return ret;
    }

    template<typename Function>
    Matrix<T, Dynamic, 1>
    take_pressure(const Mesh& msh, const typename Mesh::cell_type& cl,
                  const Matrix<T, Dynamic, 1>& sol, const Function& dirichlet_bf,
                  element_location where)
    {
        auto celdeg = di.cell_degree();
        auto facdeg = di.face_degree();
        auto pdeg = facdeg;

        auto cbs = vector_cell_basis<Mesh,T>::size(celdeg);
        auto fbs = vector_face_basis<Mesh,T>::size(facdeg);
        auto pbs = cell_basis<Mesh,T>::size(pdeg);

        auto fcs = faces(msh, cl);
        auto num_faces = fcs.size();
        size_t f_dofs = num_faces*fbs;

        auto v_dofs = cbs + f_dofs;
        auto p_dofs = pbs;
        if( location(msh, cl) == element_location::ON_INTERFACE )
        {
            v_dofs = 2 * v_dofs;
            p_dofs = 2 * p_dofs;
        }

        auto solF = get_sol_F(msh, cl, sol, dirichlet_bf);

        auto cell_offset = offset(msh, cl);
        auto lhs = loc_LHS.at(cell_offset);
        auto rhs = loc_RHS.at(cell_offset);

        Matrix<T, Dynamic, Dynamic> lhs_A = lhs.block(0, 0, v_dofs, v_dofs);
        Matrix<T, Dynamic, Dynamic> lhs_B = lhs.block(v_dofs, 0, p_dofs, v_dofs);
        Matrix<T, Dynamic, Dynamic> lhs_C = lhs.block(v_dofs, v_dofs, p_dofs, p_dofs);
        Matrix<T, Dynamic, 1> rhs_A = rhs.head( v_dofs );
        Matrix<T, Dynamic, 1> rhs_B = rhs.tail( p_dofs );

        size_t P_SOL_offset = num_other_faces * fbs + cell_offset;
        Matrix<T, Dynamic, 1> ret = Matrix<T, Dynamic, 1>::Zero(pbs);
        if( facdeg == 0 && location(msh, cl) != element_location::ON_INTERFACE )
        {
            ret(0) = sol(P_SOL_offset);
            return ret;
        }
        if( location(msh, cl) != element_location::ON_INTERFACE )
        {
            // compute sol_sc
            Matrix<T, Dynamic, 1> sol_sc = Matrix<T, Dynamic, 1>::Zero(f_dofs + 1);
            sol_sc.head(f_dofs) = solF;
            size_t B_offset = fbs * num_other_faces + cell_offset;
            sol_sc(f_dofs) = sol(B_offset);

            // compute mult_C
            cell_basis<cuthho_poly_mesh<T>, T> pb(msh, cl, pdeg);
            auto qpsi = integrate(msh, cl, pdeg);
            Matrix<T, Dynamic, 1> mult_C = Matrix<T, Dynamic, 1>::Zero( pbs - 1 );
            T area = 0.0;
            for (auto& qp : qpsi)
            {
                auto s_phi = pb.eval_basis(qp.first);
                mult_C -= qp.second * s_phi.tail(pbs - 1);
                area += qp.second;
            }
            mult_C = mult_C / area;

            return stokes_full_static_condensation_recover_p
                (lhs_A, lhs_B, lhs_C, rhs_A, rhs_B, mult_C, cbs, f_dofs, sol_sc);
        }
        // else -> location(msh, cl) == element_location::ON_INTERFACE
        // compute sol_sc
        Matrix<T, Dynamic, 1> sol_sc = Matrix<T, Dynamic, 1>::Zero(2*f_dofs + 1);
        sol_sc.head(2*f_dofs) = solF;
        size_t B_offset = fbs * num_other_faces + cell_offset;
        sol_sc(2*f_dofs) = sol(B_offset);

        Matrix<T, Dynamic, 1> mult_C = compute_multC_interface(msh, cl, pdeg);

        auto loc_sol = stokes_full_static_condensation_recover_p
            (lhs_A, lhs_B, lhs_C, rhs_A, rhs_B, mult_C, 2*cbs, 2*f_dofs, sol_sc);

        if( where == element_location::IN_NEGATIVE_SIDE )
            ret = loc_sol.head(pbs);

        if( where == element_location::IN_POSITIVE_SIDE )
            ret = loc_sol.tail(pbs);

        return ret;
    }

    void finalize(void)
    {
        LHS.setFromTriplets( triplets.begin(), triplets.end() );
        triplets.clear();
    }
};


template<typename Mesh>
auto make_stokes_interface_condensed_assembler(const Mesh& msh, hho_degree_info hdi)
{
    return stokes_interface_condensed_assembler<Mesh>(msh, hdi);
}


