/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2010 - 2022 by the deal.II authors and Tao Jin.
 *
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE.md at
 * the top level directory of deal.II.
 *
 * ---------------------------------------------------------------------
*/

/* Source code for paper draft "Finite element simulations of the 
 * thermomechanically coupled responses of thermal barrier coating
 * systems using an unconditionally stable staggered approach"
 *
 * Author: Tao Jin, PhD (https://www.taojinlab.com/)
 *         Department of Mechanical Engineering
 *         University of Ottawa
 *         Ottawa, Ontario K1N 6N5, Canada
 * Starting date: April. 2023
 * Release date: August, 2023
 */


/* 
 * Main reference:
 * "A NEW UNCONDITIONALLY STABLE FRACTIONAL STEP METHOD FOR
 *  NON-LINEAR COUPLED THERMOMECHANICAL PROBLEMS"
 * by F. Armero and J.C. Simo
 */
 

/* Main features of the code:
 * Small deformation linear elasticity
 * Thermoelastic vibration problem (2D or 3D)
 * Monolithic time integration approaches:
 * Multi-thread for system assembly (TBB)
 */

/**********************************************************
 * WARNING!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
 * For the monolithic approach, the global stiffness matrix
 * is unsymmetric. On the other hand, the stiffness matrix
 * for the isothermal and adiabatic splits should be
 * symmetric.
 **********************************************************/

/**********************************************************
 * operator* overloaded for different type
 * of tensors in deal.ii:
 *
 * 1. for two symmetric 2nd order tensors A and B, A * B means
 *    double contraction between A and B, that is,
 *    A : B  = A_{ij}B_{ij}, which generates a scalar;
 *
 * 2. for a symmetric 2nd order tensor A and a regular 2nd
 *    order tensor B, A * B or B * A means tensor product,
 *    that is, A * B = A_{ik} B_{kj} or B * A = B_{ik} A_{kj}
 *    which generates another 2nd order tensor;
 *
 * 3. for two regular 2nd order tensors A and B, A * B or
 *    B * A means tensor product,
 *    that is, A * B = A_{ik} B_{kj} or B * A = B_{ik} A_{kj}
 *    which generates another 2nd order tensor;
 *
 * 4. for two regular 2nd order tensors A and B, in order to
 *    perform the double contraction operation, call
 *    c = scalar_product(A, B), which c is a scalar. That is,
 *    c = A_{ij} B_{ij};
 *
 * 5. for two regular 2nd order tensors A and B, another way
 *    to perform double contraction A_{ij} B_{ij} is
 *    c = double_contract<0,0,1,1>(A, B), where c is a scalar.
 *    For double contraction A_{ij} B_{ji}, we call
 *    c = double_contract<0,1,1,0>(A, B);
 *
 * 6. for a 2nd order symmetric tensor A and a 4th order
 *    symmetric tensor C, A * C = A_{ij} C_{ijkl} generates
 *    another 2nd order symmetric tensor.
 *    C * A = C_{ijkl} A_{kl} also generates a 2nd order
 *    symmetric tensor. If either A or C is a regular tensor,
 *    then "*" means tensor product, that is,
 *    A * C = A_{im} C_{mjkl},
 *    C * A = C_{ijkm} A_{ml} generates another 4th order
 *    tensor;
 *
 * 7. for a 2nd order general tensor A and a 4th order general
 *    tensor C, in order to perform the double contraction,
 *    A : C = A_{ij} C_{ijkl}, we call
 *    double_contract<0,0,1,1>(A, C) to generate a second order
 *    tensor. Similarly, for C : A = C_{ijkl} A_{kl}, we call
 *    double_contract<2,0,3,1>(C, A) to generate a second order
 *    tensor;
 *
 * 8. Whenever call double_contract<>(A,B), both A and B need to
 *    be regular tensor and CANNOT be symmetric tensor;
 *
 * 9. Whenever call scalar_product(A,B), both A and B can be either
 *    a regular or symmetric 2nd order tensor. Since the function
 *    interfaces exist for all cases;
 *
 **********************************************************/

#include <deal.II/base/function.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/base/quadrature_point_data.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/manifold_lib.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q_eulerian.h>

#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition_selector.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_selector.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/affine_constraints.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>

#include <deal.II/physics/elasticity/standard_tensors.h>
#include <deal.II/base/utilities.h>

#include <deal.II/numerics/error_estimator.h>
#include <deal.II/grid/grid_refinement.h>

#include <iostream>
#include <fstream>


namespace LinearThermoElasticCoupling
{
  using namespace dealii;

  template <int dim>
  class InitialValues : public Function<dim>
  {
  public:
    InitialValues()
    : Function<dim>(dim+dim+1)
    {}

    virtual void vector_value(const Point<dim> & p,
                              Vector<double> &value ) const override
    {
      if (dim == 2)
	{
	  value(0) = std::sin(numbers::PI/100.0 * p[0]); // velocity
	  value(1) = 0.0;

	  value(2) = 0.0;                                // displacement
	  value(3) = 0.0;

	  value(4) = 0.0;                                // temperature
	}

      if (dim == 3)
	{
	  value(0) = std::sin(numbers::PI/100.0 * p[0]); // velocity
	  value(1) = 0.0;
	  value(2) = 0.0;

	  value(3) = 0.0;                                // displacement
	  value(4) = 0.0;
	  value(5) = 0.0;

	  value(6) = 0.0;                                // temperature
	}
    }
  };

  template <int dim>
  class BoundaryValues : public Function<dim>
  {
  public:
    BoundaryValues()
    : Function<dim>(dim+dim+1)
    {}

    virtual void vector_value(const Point<dim> & /*p*/,
                              Vector<double> &value ) const override
    {
      if (dim == 2)
	{
	  value(0) = 0.0; // velocity
	  value(1) = 0.0;

	  value(2) = 0.0; // displacement
	  value(3) = 0.0;

	  value(4) = 0.0; // temperature
	}

      if (dim == 3)
	{
	  value(0) = 0.0; // velocity
	  value(1) = 0.0;
	  value(2) = 0.0;

	  value(3) = 0.0; // displacement
	  value(4) = 0.0;
	  value(5) = 0.0;

	  value(6) = 0.0; // temperature
	}
    }
  };

  template <int dim>
  class BoundaryValueTBCTopSurface : public Function<dim>
  {
  public:
    BoundaryValueTBCTopSurface(const unsigned int n_components = dim+dim+1,
			       const double time = 0.0)
    : Function<dim>(n_components, time)
    {}

    virtual void vector_value(const Point<dim> & /*p*/,
                              Vector<double> &value ) const override
    {
      const double t = this->get_time();
      const double beta = 10.0;

      if (dim == 2)
	{
	  value(0) = 0.0; // velocity
	  value(1) = 0.0;

	  value(2) = 0.0; // displacement
	  value(3) = 0.0;

	  value(4) = 500.0 * (1.0 - std::exp(-beta * t)); // temperature
	}

      if (dim == 3)
	{
	  value(0) = 0.0; // velocity
	  value(1) = 0.0;
	  value(2) = 0.0;

	  value(3) = 0.0; // displacement
	  value(4) = 0.0;
	  value(5) = 0.0;

	  value(6) = 500.0 * (1.0 - std::exp(-beta * t)); // temperature
	}
    }
  };

  namespace Parameters
  {
    struct Scenario
    {
      unsigned int m_scenario;
      unsigned int m_number_of_elements;
      double m_total_length;
      unsigned int m_total_material_regions;
      std::string m_material_file_name;
      unsigned int m_adaptive_refine;

      static void declare_parameters(ParameterHandler &prm);
      void parse_parameters(ParameterHandler &prm);
    };

    void Scenario::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Scenario");
      {
        prm.declare_entry("Scenario number",
                          "1",
                          Patterns::Integer(0),
                          "Geometry, loading and boundary conditions scenario");

        prm.declare_entry("Number of elements",
                          "1",
                          Patterns::Integer(0),
                          "Number of elements in 1D");

        prm.declare_entry("Total length",
                          "1.0",
                          Patterns::Double(0.0),
                          "Total length of the 1D bar");

        prm.declare_entry("Material regions",
                          "1",
                          Patterns::Integer(0),
                          "Number of material regions");

        prm.declare_entry("Material data file",
                          "1",
                          Patterns::FileName(Patterns::FileName::input),
                          "Material data file");

        prm.declare_entry("Adaptive refinement",
                          "0",
                          Patterns::Integer(0),
                          "Number of adaptive refinement");
      }
      prm.leave_subsection();
    }

    void Scenario::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Scenario");
      {
        m_scenario = prm.get_integer("Scenario number");
        std::cout << "scenario number = " << m_scenario << std::endl;
        m_number_of_elements = prm.get_integer("Number of elements");
        std::cout << "Number of elements (1D) = " << m_number_of_elements << std::endl;
        m_total_length = prm.get_double("Total length");
        std::cout << "Length of the bar (1D) = " << m_total_length << std::endl;
        m_total_material_regions = prm.get_integer("Material regions");
        std::cout << "total number of material types = " << m_total_material_regions << std::endl;
        m_material_file_name = prm.get("Material data file");
        std::cout << "material data file name = " << m_material_file_name << std::endl;
        m_adaptive_refine = prm.get_integer("Adaptive refinement");
        std::cout << "adaptive refinement number = " << m_adaptive_refine << std::endl;
      }
      prm.leave_subsection();
    }

    struct FESystem
    {
      unsigned int m_poly_degree;
      unsigned int m_quad_order;

      static void declare_parameters(ParameterHandler &prm);

      void parse_parameters(ParameterHandler &prm);
    };


    void FESystem::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Finite element system");
      {
        prm.declare_entry("Polynomial degree",
                          "2",
                          Patterns::Integer(0),
                          "Displacement system polynomial order");

        prm.declare_entry("Quadrature order",
                          "3",
                          Patterns::Integer(0),
                          "Gauss quadrature order");
      }
      prm.leave_subsection();
    }

    void FESystem::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Finite element system");
      {
        m_poly_degree = prm.get_integer("Polynomial degree");
        m_quad_order  = prm.get_integer("Quadrature order");
      }
      prm.leave_subsection();
    }

    struct TimeInfo
    {
      double m_end_time;
      std::string m_time_file_name;

      static void declare_parameters(ParameterHandler &prm);

      void parse_parameters(ParameterHandler &prm);
    };

    void TimeInfo::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Time");
      {
        prm.declare_entry("End time", "1", Patterns::Double(), "End time");

        prm.declare_entry("Time data file",
                          "1",
                          Patterns::FileName(Patterns::FileName::input),
                          "Time data file");
      }
      prm.leave_subsection();
    }

    void TimeInfo::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Time");
      {
        m_end_time = prm.get_double("End time");
        std::cout << "End time = " << m_end_time << std::endl;

        m_time_file_name = prm.get("Time data file");
        std::cout << "Time data file name = " << m_time_file_name << std::endl;
      }
      prm.leave_subsection();
    }

    struct AllParameters : public Scenario,
	                   public FESystem,
                           public TimeInfo
    {
      AllParameters(const std::string &input_file);

      static void declare_parameters(ParameterHandler &prm);

      void parse_parameters(ParameterHandler &prm);
    };

    AllParameters::AllParameters(const std::string &input_file)
    {
      ParameterHandler prm;
      declare_parameters(prm);
      prm.parse_input(input_file);
      parse_parameters(prm);
    }

    void AllParameters::declare_parameters(ParameterHandler &prm)
    {
      Scenario::declare_parameters(prm);
      FESystem::declare_parameters(prm);
      TimeInfo::declare_parameters(prm);
    }

    void AllParameters::parse_parameters(ParameterHandler &prm)
    {
      Scenario::parse_parameters(prm);
      FESystem::parse_parameters(prm);
      TimeInfo::parse_parameters(prm);
    }
  } // namespace Parameters


  class Time
  {
  public:
    Time(const double time_end)
      : m_timestep(0)
      , m_time_current(0.0)
      , m_time_end(time_end)
      , m_delta_t(0.0)
    {}

    virtual ~Time() = default;

    double current() const
    {
      return m_time_current;
    }
    double end() const
    {
      return m_time_end;
    }
    double get_delta_t() const
    {
      return m_delta_t;
    }
    unsigned int get_timestep() const
    {
      return m_timestep;
    }
    void increment(std::vector<std::array<double, 3>> time_table)
    {
      double t_1, t_delta;
      for (auto & time_group : time_table)
        {
	  t_1 = time_group[1];
	  t_delta = time_group[2];

	  if (m_time_current < t_1 - 1.0e-6*t_delta)
	    {
	      m_delta_t = t_delta;
	      break;
	    }
        }

      m_time_current += m_delta_t;
      ++m_timestep;
    }

  private:
    unsigned int m_timestep;
    double       m_time_current;
    const double m_time_end;
    double m_delta_t;
  };

  template <int dim>
  class LinearIsotropicThermalElasticity
  {
  public:
    LinearIsotropicThermalElasticity(const double lame_lambda,
			             const double lame_mu,
				     const double thermal_expansion_coeff,
				     const double density,
				     const double specific_capacity,
				     const double thermal_conductivity_coeff,
				     const double temperature_ref)
      : m_lame_lambda(lame_lambda)
      , m_lame_mu(lame_mu)
      , m_thermal_expansion_coeff(thermal_expansion_coeff)
      , m_density(density)
      , m_specific_capacity(specific_capacity)
      , m_thermal_conductivity_coeff(thermal_conductivity_coeff)
      , m_temperature_ref(temperature_ref)
      , m_strain(SymmetricTensor<2, dim>())
      , m_temperature(temperature_ref)
    {
      Assert(  ( lame_lambda / (2*(lame_lambda + lame_mu)) <= 0.5)
	     & ( lame_lambda / (2*(lame_lambda + lame_mu)) >=-1.0),
	     ExcInternalError() );
      m_mechanical_C = m_lame_lambda * Physics::Elasticity::StandardTensors<dim>::IxI
	             + 2.0 * m_lame_mu * Physics::Elasticity::StandardTensors<dim>::S;
      m_m_coupling = m_thermal_expansion_coeff * (3*m_lame_lambda + 2*m_lame_mu)
		                               * Physics::Elasticity::StandardTensors<dim>::I;
    }

    SymmetricTensor<2, dim> get_m_coupling()
    {
      return m_m_coupling;
    }

    SymmetricTensor<4, dim> get_mechanical_C()
    {
      return m_mechanical_C;
    }

    SymmetricTensor<2, dim> get_thermal_conductivity()
    {
      return m_thermal_conductivity_coeff * Physics::Elasticity::StandardTensors<dim>::I;
    }

    double get_heat_capacity()
    {
      return m_density * m_specific_capacity;
    }

    double get_density()
    {
      return m_density;
    }

    double get_temperature_ref()
    {
      return m_temperature_ref;
    }

    SymmetricTensor<2, dim> get_cauchy_stress() const
    {
      return m_mechanical_C * m_strain
	  -  m_temperature * m_m_coupling;
    }

    void update_material_data(const SymmetricTensor<2, dim> & strain,
			      const double temperature)
    {
      m_strain = strain;
      m_temperature = temperature;
    }

  private:
    const double m_lame_lambda;
    const double m_lame_mu;
    const double m_thermal_expansion_coeff;
    const double m_density;
    const double m_specific_capacity;
    const double m_thermal_conductivity_coeff;
    const double m_temperature_ref;
    SymmetricTensor<2, dim> m_strain;
    double m_temperature;
    SymmetricTensor<4, dim> m_mechanical_C;
    SymmetricTensor<2, dim> m_m_coupling;
  };

  template <int dim>
  class PointHistory
  {
  public:
    PointHistory()
      : m_m_coupling(SymmetricTensor<2, dim>())
      , m_mechanical_C(SymmetricTensor<4, dim>())
      , m_thermal_conductivity(SymmetricTensor<2, dim>())
      , m_heat_capacity(0.0)
      , m_density(0.0)
      , m_temperature_ref(0.0)
    {}

    virtual ~PointHistory() = default;

    void setup_lqp(const double lame_lambda,
		   const double lame_mu,
		   const double thermal_expansion_coeff,
		   const double density,
		   const double specific_capacity,
		   const double thermal_conductivity_coeff,
		   const double temperature_ref)
    {
      m_material =
              std::make_shared<LinearIsotropicThermalElasticity<dim>>(lame_lambda,
        	                                                      lame_mu,
		                                                      thermal_expansion_coeff,
		                                                      density,
		                                                      specific_capacity,
		                                                      thermal_conductivity_coeff,
								      temperature_ref);
      m_m_coupling = m_material->get_m_coupling();
      m_mechanical_C = m_material->get_mechanical_C();
      m_thermal_conductivity = m_material->get_thermal_conductivity();
      m_heat_capacity = m_material->get_heat_capacity();
      m_density = m_material->get_density();
      m_temperature_ref = m_material->get_temperature_ref();

      update_values(SymmetricTensor<2, dim>(), 0.0);
    }

    void update_values(const SymmetricTensor<2, dim> & strain,
		       const double temperature)
    {
      m_material->update_material_data(strain, temperature);
    }

    const SymmetricTensor<2, dim> &get_m_coupling() const
    {
      return m_m_coupling;
    }

    const SymmetricTensor<4, dim> &get_mechanical_C() const
    {
      return m_mechanical_C;
    }

    const SymmetricTensor<2, dim> &get_thermal_conductivity() const
    {
      return m_thermal_conductivity;
    }

    double get_heat_capacity() const
    {
      return m_heat_capacity;
    }

    double get_density() const
    {
      return m_density;
    }

    double get_temperature_ref() const
    {
      return m_temperature_ref;
    }

    const SymmetricTensor<2, dim> get_cauchy_stress() const
    {
      return m_material->get_cauchy_stress();
    }
  private:
    std::shared_ptr<LinearIsotropicThermalElasticity<dim>> m_material;
    SymmetricTensor<2, dim> m_m_coupling;
    SymmetricTensor<4, dim> m_mechanical_C;
    SymmetricTensor<2, dim> m_thermal_conductivity;
    double m_heat_capacity;
    double m_density;
    double m_temperature_ref;
  };


  template <int dim>
  class MonolithicSolve
  {
  public:
    MonolithicSolve(const std::string &input_file);

    void run();

  private:
    struct PerTaskData_ASM;
    struct ScratchData_ASM;

    struct PerTaskData_UQPH;
    struct ScratchData_UQPH;

    void make_grid();
    void refine_grid();

    void make_grid_case_1();
    void make_grid_case_2();
    void make_grid_case_3();
    void make_grid_case_4();

    void system_setup();

    void make_constraints();

    void assemble_system(const BlockVector<double> & solution_old);

    void assemble_system_one_cell(
      const typename DoFHandler<dim>::active_cell_iterator &cell,
      ScratchData_ASM &                                     scratch,
      PerTaskData_ASM &                                     data) const;

    void output_results() const;

    void direct_linear_solve(bool recalculateLU);

    void read_time_data(const std::string &data_file,
    		        std::vector<std::array<double, 3>> & time_table) const;

    void write_history_data() const;

    // Should not make this function const
    void read_material_data(const std::string &data_file,
			    const unsigned int total_material_regions);

    void setup_qph();

    void update_qph(const BlockVector<double> & solution);

    void update_qph_one_cell(
      const typename DoFHandler<dim>::active_cell_iterator &cell,
      ScratchData_UQPH &                                    scratch,
      PerTaskData_UQPH &                                    data);

    void copy_local_to_global_UQPH(const PerTaskData_UQPH & /*data*/)
    {}

    Parameters::AllParameters m_parameters;

    double m_vol_reference;

    Triangulation<dim> m_triangulation;

    Time                m_time;
    mutable TimerOutput m_timer;

    CellDataStorage<typename Triangulation<dim>::cell_iterator,
                    PointHistory<dim>>
      m_quadrature_point_history;

    const unsigned int               m_degree;
    const FESystem<dim>              m_fe;
    DoFHandler<dim>                  m_dof_handler;
    const unsigned int               m_dofs_per_cell;
    const FEValuesExtractors::Vector m_v_fe;
    const FEValuesExtractors::Vector m_u_fe;
    const FEValuesExtractors::Scalar m_T_fe;

    static const unsigned int m_n_blocks          = 3;
    static const unsigned int m_n_components      = dim + dim + 1;
    static const unsigned int m_first_v_component = 0;
    static const unsigned int m_first_u_component = dim;
    static const unsigned int m_T_component       = dim + dim;

    enum
    {
      m_v_dof = 0,
      m_u_dof = 1,
      m_T_dof = 2
    };

    std::vector<types::global_dof_index> m_dofs_per_block;

    const QGauss<dim>     m_qf_cell;
    const QGauss<dim - 1> m_qf_face;
    const unsigned int    m_n_q_points;

    AffineConstraints<double> m_constraints;
    BlockSparsityPattern      m_sparsity_pattern;
    BlockSparseMatrix<double> m_tangent_matrix;
    BlockVector<double>       m_system_rhs;
    BlockVector<double>       m_solution_n;
    BlockVector<double>       m_solution_old;
    SparseDirectUMFPACK       m_A_direct;

    std::map<unsigned int, std::vector<double>> m_material_data;

    std::vector<std::pair<double, double>> m_history_disp_x;
    std::vector<std::pair<double, double>> m_history_disp_y;
    std::vector<std::pair<double, double>> m_history_T;
  };

  template <int dim>
  MonolithicSolve<dim>::MonolithicSolve(const std::string &input_file)
    : m_parameters(input_file)
    , m_vol_reference(0.)
    , m_triangulation(Triangulation<dim>::maximum_smoothing)
    , m_time(m_parameters.m_end_time)
    , m_timer(std::cout, TimerOutput::summary, TimerOutput::wall_times)
    , m_degree(m_parameters.m_poly_degree)
    , m_fe(FE_Q<dim>(m_parameters.m_poly_degree),
           dim, // velocity
           FE_Q<dim>(m_parameters.m_poly_degree),
           dim, // displacement
           FE_Q<dim>(m_parameters.m_poly_degree),
           1) // temperature
    , m_dof_handler(m_triangulation)
    , m_dofs_per_cell(m_fe.n_dofs_per_cell())
    , m_v_fe(m_first_v_component)
    , m_u_fe(m_first_u_component)
    , m_T_fe(m_T_component)
    , m_dofs_per_block(m_n_blocks)
    , m_qf_cell(m_parameters.m_quad_order)
    , m_qf_face(m_parameters.m_quad_order)
    , m_n_q_points(m_qf_cell.size())
  {
  }


  template <int dim>
  void MonolithicSolve<dim>::direct_linear_solve(bool recalculateLU)
  {
    std::cout << "    direct solve" << std::endl;

    m_timer.enter_subsection("Direct linear solve");
    if (recalculateLU)
      m_A_direct.initialize(m_tangent_matrix);
    m_A_direct.vmult(m_solution_n, m_system_rhs);
    m_constraints.distribute(m_solution_n);

    m_timer.leave_subsection();
  }

  template <int dim>
  void MonolithicSolve<dim>::write_history_data() const
  {
    std::cout << std::endl;
    std::cout << "Write history data ... \n"<<std::endl;

    std::ofstream myfile_disp_x_history ("disp_x_history.hist");
    if (myfile_disp_x_history.is_open())
    {
      for (auto const time_disp : m_history_disp_x)
	{
	  myfile_disp_x_history << time_disp.first << "\t";
	  myfile_disp_x_history << time_disp.second << std::endl;
	}
      myfile_disp_x_history.close();
    }
    else
      std::cout << "Unable to open file";

    std::ofstream myfile_disp_y_history ("disp_y_history.hist");
    if (myfile_disp_y_history.is_open())
    {
      for (auto const time_disp : m_history_disp_y)
	{
	  myfile_disp_y_history << time_disp.first << "\t";
	  myfile_disp_y_history << time_disp.second << std::endl;
	}
      myfile_disp_y_history.close();
    }
    else
      std::cout << "Unable to open file";

    std::ofstream myfile_T_history ("temperature_history.hist");
    if (myfile_T_history.is_open())
    {
      for (auto const time_temperature : m_history_T)
	{
	  myfile_T_history << time_temperature.first << "\t";
	  myfile_T_history << time_temperature.second << std::endl;
	}
      myfile_T_history.close();
    }
    else
      std::cout << "Unable to open file";
  }



  template <int dim>
  void MonolithicSolve<dim>::read_material_data(const std::string &data_file,
				                const unsigned int total_material_regions)
  {
    std::ifstream myfile (data_file);

    double lame_lambda, lame_mu, thermal_expansion_coeff;
    double density, specific_capacity, thermal_conductivity_coeff;
    double temperature_ref;
    int material_region;
    double poisson_ratio;
    if (myfile.is_open())
      {
        std::cout << "Reading material data file ..." << std::endl;

        while ( myfile >> material_region
                       >> lame_lambda
		       >> lame_mu
		       >> thermal_expansion_coeff
		       >> density
		       >> specific_capacity
		       >> thermal_conductivity_coeff
		       >> temperature_ref)
          {
            m_material_data[material_region] = {lame_lambda,
        	                                lame_mu,
						thermal_expansion_coeff,
						density,
						specific_capacity,
						thermal_conductivity_coeff,
                                                temperature_ref};
            poisson_ratio = lame_lambda / (2*(lame_lambda + lame_mu));
            Assert( (poisson_ratio <= 0.5)&(poisson_ratio >=-1.0) , ExcInternalError());

            std::cout << "Region " << material_region << " : " << std::endl;
            std::cout << "    lame lambda = " << lame_lambda << std::endl;
            std::cout << "    lame mu = "  << lame_mu << std::endl;
            std::cout << "    poisson ratio = "  << poisson_ratio << std::endl;
            std::cout << "    thermal expansion coefficient (beta) = " << thermal_expansion_coeff << std::endl;
            std::cout << "    density = "  << density << std::endl;
            std::cout << "    specific heat capacity = "  << specific_capacity << std::endl;
            std::cout << "    thermal conductivity (isotropic) = "  << thermal_conductivity_coeff << std::endl;
            std::cout << "    reference temperature (K) = "  << temperature_ref << std::endl;
          }

        if (m_material_data.size() != total_material_regions)
          {
            std::cout << "Material data file has " << m_material_data.size() << " rows. However, "
        	      << "the mesh has " << total_material_regions << " material regions."
		      << std::endl;
            Assert(m_material_data.size() == total_material_regions,
                       ExcDimensionMismatch(m_material_data.size(), total_material_regions));
          }
        myfile.close();
      }
    else
      {
	std::cout << "Material data file : " << data_file << " not exist!" << std::endl;
	Assert(false, ExcMessage("Failed to read material data file"));
      }
  }

  template <int dim>
  void MonolithicSolve<dim>::setup_qph()
  {
    std::cout << "     Setting up quadrature point data ("
	      << m_n_q_points
	      << " points per cell)" << std::endl;

    m_quadrature_point_history.clear();
    for (auto const & cell : m_triangulation.active_cell_iterators())
      {
	m_quadrature_point_history.initialize(cell, m_n_q_points);
      }

    unsigned int material_id;
    double lame_lambda = 0.0;
    double lame_mu = 0.0;
    double thermal_expansion_coeff = 0.0;
    double density = 0.0;
    double specific_capacity = 0.0;
    double thermal_conductivity_coeff = 0.0;
    double temperature_ref = 0.0;

    for (const auto &cell : m_triangulation.active_cell_iterators())
      {
        material_id = cell->material_id();
        if (m_material_data.find(material_id) != m_material_data.end())
          {
            lame_lambda                = m_material_data[material_id][0];
            lame_mu                    = m_material_data[material_id][1];
            thermal_expansion_coeff    = m_material_data[material_id][2];
            density                    = m_material_data[material_id][3];
            specific_capacity          = m_material_data[material_id][4];
            thermal_conductivity_coeff = m_material_data[material_id][5];
            temperature_ref            = m_material_data[material_id][6];
	  }
        else
          {
            std::cout << "Could not find material data for material id: " << material_id << std::endl;
            Assert(false, ExcMessage("Could not find material data for material id."));
          }

        const std::vector<std::shared_ptr<PointHistory<dim>>> lqph =
          m_quadrature_point_history.get_data(cell);
        Assert(lqph.size() == m_n_q_points, ExcInternalError());

        for (unsigned int q_point = 0; q_point < m_n_q_points; ++q_point)
          lqph[q_point]->setup_lqp(lame_lambda, lame_mu, thermal_expansion_coeff,
				   density, specific_capacity, thermal_conductivity_coeff,
				   temperature_ref);
      }
  }

  template <int dim>
  void MonolithicSolve<dim>::update_qph(const BlockVector<double> & solution)
  {
    m_timer.enter_subsection("Update QPH data");
    std::cout << "    UQPH" << std::endl;

    const UpdateFlags uf_UQPH(update_values | update_gradients);
    PerTaskData_UQPH  per_task_data_UQPH;
    ScratchData_UQPH  scratch_data_UQPH(m_fe, m_qf_cell, uf_UQPH, solution);

    WorkStream::run(m_dof_handler.active_cell_iterators(),
                    *this,
                    &MonolithicSolve::update_qph_one_cell,
                    &MonolithicSolve::copy_local_to_global_UQPH,
                    scratch_data_UQPH,
                    per_task_data_UQPH);

    m_timer.leave_subsection();
  }

  template <int dim>
  void MonolithicSolve<dim>::update_qph_one_cell(
    const typename DoFHandler<dim>::active_cell_iterator &cell,
    ScratchData_UQPH &                                    scratch,
    PerTaskData_UQPH & /*data*/)
  {
    const std::vector<std::shared_ptr<PointHistory<dim>>> lqph =
      m_quadrature_point_history.get_data(cell);
    Assert(lqph.size() == m_n_q_points, ExcInternalError());

    Assert(scratch.m_solution_sym_grads_u_total.size() == m_n_q_points,
           ExcInternalError());
    Assert(scratch.m_solution_values_temperature_total.size() == m_n_q_points,
           ExcInternalError());

    scratch.reset();

    scratch.m_fe_values.reinit(cell);
    scratch.m_fe_values[m_u_fe].get_function_symmetric_gradients(
      scratch.m_solution_total, scratch.m_solution_sym_grads_u_total);
    scratch.m_fe_values[m_T_fe].get_function_values(
      scratch.m_solution_total, scratch.m_solution_values_temperature_total);

    for (const unsigned int q_point :
         scratch.m_fe_values.quadrature_point_indices())
      lqph[q_point]->update_values(scratch.m_solution_sym_grads_u_total[q_point],
                                   scratch.m_solution_values_temperature_total[q_point]);
  }


  template <int dim>
  void MonolithicSolve<dim>::run()
  {
    read_material_data(m_parameters.m_material_file_name,
		       m_parameters.m_total_material_regions);

    std::vector<std::array<double, 3>> time_table;

    read_time_data(m_parameters.m_time_file_name, time_table);

    make_grid();
    system_setup();

    if (   m_parameters.m_scenario == 1
	|| m_parameters.m_scenario == 2
	|| m_parameters.m_scenario == 3 )
      // L2 projection for initial conditions
      VectorTools::project(m_dof_handler,
			   m_constraints,
			   QGauss<dim>(m_fe.degree + 1),
			   InitialValues<dim>(),
			   m_solution_n);
    else if (m_parameters.m_scenario == 4)
      {
	VectorTools::project(m_dof_handler,
			     m_constraints,
			     QGauss<dim>(m_fe.degree + 1),
			     Functions::ZeroFunction<dim>(m_n_components),
			     m_solution_n);
/*
	const double hot_temperature = 500.0; // Kelvin
	const double H1 = 1.0, H2 = 0.15, H3 = 0.006, H4 = 0.1;

	std::map<types::global_dof_index, Point<dim> > support_points_T;
	ComponentMask temperature_mask = m_fe.component_mask(m_T_fe);
	DoFTools::map_dofs_to_support_points (MappingQ1<dim>(),
					      m_dof_handler,
					      support_points_T,
					      temperature_mask);

	for (auto const & item : support_points_T)
	  {
	    if (std::fabs(item.second[1] - (H1 + H2 + H3 + H4)) < 1.0e-9 )
	      {
		m_solution_n(item.first) = hot_temperature;
	      }
	  }
*/
      }
    else
      Assert(false, ExcMessage("The scenario has not been implemented!"));


/*
    // point interpolation for initial conditions
    VectorTools::interpolate(m_dof_handler,
			     InitialValues<dim>(),
			     m_solution_n);
*/
    // output initial conditions
    output_results();

    types::global_dof_index disp_x_target_dof = 0;
    types::global_dof_index disp_y_target_dof = 0;
    types::global_dof_index temperature_target_dof = 0;
    std::pair<double, double> time_disp_x;
    std::pair<double, double> time_disp_y;
    std::pair<double, double> time_temperature;
    if (m_parameters.m_scenario == 2)
      {
	const double disp_x_target_coordinate_x = m_parameters.m_total_length/2.0;
	const double disp_x_target_coordinate_y = m_parameters.m_total_length/2.0;

	const double disp_y_target_coordinate_x = m_parameters.m_total_length/4.0;
	const double disp_y_target_coordinate_y = m_parameters.m_total_length/4.0;

	const double temperature_target_coordinate_x = m_parameters.m_total_length/4.0;
	const double temperature_target_coordinate_y = m_parameters.m_total_length/2.0;

	std::map<types::global_dof_index, Point<dim> > support_points_disp_x;
	const FEValuesExtractors::Scalar x_displacement(dim);
	ComponentMask x_displacement_mask = m_fe.component_mask(x_displacement);
	DoFTools::map_dofs_to_support_points (MappingQ1<dim>(),
					      m_dof_handler,
					      support_points_disp_x,
					      x_displacement_mask);

	std::map<types::global_dof_index, Point<dim> > support_points_disp_y;
	const FEValuesExtractors::Scalar y_displacement(dim+1);
	ComponentMask y_displacement_mask = m_fe.component_mask(y_displacement);
	DoFTools::map_dofs_to_support_points (MappingQ1<dim>(),
					      m_dof_handler,
					      support_points_disp_y,
					      y_displacement_mask);

	std::map<types::global_dof_index, Point<dim> > support_points_T;
	ComponentMask temperature_mask = m_fe.component_mask(m_T_fe);
	DoFTools::map_dofs_to_support_points (MappingQ1<dim>(),
					      m_dof_handler,
					      support_points_T,
					      temperature_mask);

	for (auto const & item : support_points_disp_x)
	  {
	    if (    (std::fabs(item.second[0] - disp_x_target_coordinate_x) < 1.0e-9)
		 && (std::fabs(item.second[1] - disp_x_target_coordinate_y) < 1.0e-9) )
	      {
		if (dim == 3)
		  {
		    if (std::fabs(item.second[2] - 0.0) < 1.0e-9)
		      {
			disp_x_target_dof = item.first;
			break;
		      }
		  }
		else
		  {
		    disp_x_target_dof = item.first;
		    break;
		  }
	      }
	  }

	for (auto const & item : support_points_disp_y)
	  {
	    if (    (std::fabs(item.second[0] - disp_y_target_coordinate_x) < 1.0e-9)
		 && (std::fabs(item.second[1] - disp_y_target_coordinate_y) < 1.0e-9) )
	      {
		if (dim == 3)
		  {
		    if (std::fabs(item.second[2] - 0.0) < 1.0e-9)
		      {
			disp_y_target_dof = item.first;
			break;
		      }
		  }
		else
		  {
		    disp_y_target_dof = item.first;
		    break;
		  }
	      }
	  }

	for (auto const & item : support_points_T)
	  {
	    if (    (std::fabs(item.second[0] - temperature_target_coordinate_x) < 1.0e-9)
		 && (std::fabs(item.second[1] - temperature_target_coordinate_y) < 1.0e-9) )
	      {
		if (dim == 3)
		  {
		    if (std::fabs(item.second[2] - 0.0) < 1.0e-9)
		      {
			temperature_target_dof = item.first;
			break;
		      }
		  }
		else
		  {
		    temperature_target_dof = item.first;
		    break;
		  }
	      }
	  }

	// initial conditions
	time_disp_x.first = m_time.current();
	time_disp_x.second = m_solution_n(disp_x_target_dof);
	m_history_disp_x.push_back(time_disp_x);

	time_disp_y.first = m_time.current();
	time_disp_y.second = m_solution_n(disp_y_target_dof);
	m_history_disp_y.push_back(time_disp_y);

	time_temperature.first = m_time.current();
	time_temperature.second = m_solution_n(temperature_target_dof);
	m_history_T.push_back(time_temperature);
      }

    m_time.increment(time_table);

    // The first time step, we adaptively refine the mesh
    if (m_time.get_timestep() == 1)
      {
	const unsigned int n_cycle = m_parameters.m_adaptive_refine;
	for (unsigned int cycle = 0; cycle <= n_cycle; ++cycle)
	  {
	    std::cout << "Time step " << m_time.get_timestep()
	              << " at t=" << m_time.current()
	              << std::endl;
	    std::cout << "  refinement cycle " << cycle << std::endl;
	    if (cycle > 0)
	      {
		refine_grid();
		system_setup();

		if (   m_parameters.m_scenario == 1
		    || m_parameters.m_scenario == 2
		    || m_parameters.m_scenario == 3 )
		  // L2 projection for initial conditions
		  VectorTools::project(m_dof_handler,
				       m_constraints,
				       QGauss<dim>(m_fe.degree + 1),
				       InitialValues<dim>(),
				       m_solution_n);
		else if (m_parameters.m_scenario == 4)
		  {
		    VectorTools::project(m_dof_handler,
					 m_constraints,
					 QGauss<dim>(m_fe.degree + 1),
					 Functions::ZeroFunction<dim>(m_n_components),
					 m_solution_n);
		  }
		else
		  Assert(false, ExcMessage("The scenario has not been implemented!"));
	      }

	    m_solution_old = m_solution_n;

	    m_solution_n = 0.0;

	    make_constraints();
	    assemble_system(m_solution_old);

	    direct_linear_solve(true);
	    update_qph(m_solution_n);

	    output_results();
	  }

	if (m_parameters.m_scenario == 2)
	  {
	    time_disp_x.first = m_time.current();
	    time_disp_x.second = m_solution_n(disp_x_target_dof);
	    m_history_disp_x.push_back(time_disp_x);

	    time_disp_y.first = m_time.current();
	    time_disp_y.second = m_solution_n(disp_y_target_dof);
	    m_history_disp_y.push_back(time_disp_y);

	    time_temperature.first = m_time.current();
	    time_temperature.second = m_solution_n(temperature_target_dof);
	    m_history_T.push_back(time_temperature);
	  }
      } // first time step, adaptive mesh refinement

    while(m_time.current() < m_time.end() - m_time.get_delta_t()*1.0e-6)
      {
	m_time.increment(time_table);

        std::cout << "Time step " << m_time.get_timestep()
                  << " at t=" << m_time.current()
                  << std::endl;

	m_solution_old = m_solution_n;

	m_solution_n = 0.0;

	make_constraints();
	assemble_system(m_solution_old);

        direct_linear_solve(false);
        update_qph(m_solution_n);

	output_results();

	if (m_parameters.m_scenario == 2)
	  {
	    time_disp_x.first = m_time.current();
	    time_disp_x.second = m_solution_n(disp_x_target_dof);
	    m_history_disp_x.push_back(time_disp_x);

	    time_disp_y.first = m_time.current();
	    time_disp_y.second = m_solution_n(disp_y_target_dof);
	    m_history_disp_y.push_back(time_disp_y);

	    time_temperature.first = m_time.current();
	    time_temperature.second = m_solution_n(temperature_target_dof);
	    m_history_T.push_back(time_temperature);
	  }
      }

    if (m_parameters.m_scenario == 2)
      {
        write_history_data();
      }
  }

  template <int dim>
  struct MonolithicSolve<dim>::PerTaskData_ASM
  {
    FullMatrix<double>                   m_cell_matrix;
    Vector<double>                       m_cell_rhs;
    std::vector<types::global_dof_index> m_local_dof_indices;

    PerTaskData_ASM(const unsigned int dofs_per_cell)
      : m_cell_matrix(dofs_per_cell, dofs_per_cell)
      , m_cell_rhs(dofs_per_cell)
      , m_local_dof_indices(dofs_per_cell)
    {}

    void reset()
    {
      m_cell_matrix = 0.0;
      m_cell_rhs    = 0.0;
    }
  };


  template <int dim>
  struct MonolithicSolve<dim>::ScratchData_ASM
  {
    FEValues<dim>     m_fe_values;
    FEFaceValues<dim> m_fe_face_values;

    std::vector<std::vector<Tensor<1, dim>>>          m_Nx_velo; // shape function values for velocity field
    std::vector<std::vector<Tensor<1, dim>>>          m_Nx_disp; // shape function values for displacement field
    std::vector<std::vector<double>>                  m_Nx_T;  // shape function values for temperature

    std::vector<std::vector<Tensor<2, dim>>>          m_grad_Nx_velo;
    std::vector<std::vector<Tensor<2, dim>>>          m_grad_Nx_disp;
    std::vector<std::vector<Tensor<1, dim>>>          m_grad_Nx_T;

    std::vector<std::vector<SymmetricTensor<2, dim>>> m_symm_grad_Nx_velo;
    std::vector<std::vector<SymmetricTensor<2, dim>>> m_symm_grad_Nx_disp;

    const BlockVector<double>&  m_solution_previous;

    std::vector<Tensor<1, dim>> m_previous_velo;
    std::vector<Tensor<1, dim>> m_previous_disp;
    std::vector<double>         m_previous_T;

    std::vector<Tensor<2, dim>> m_previous_grad_velo;
    std::vector<Tensor<2, dim>> m_previous_grad_disp;
    std::vector<Tensor<1, dim>> m_previous_grad_T;


    ScratchData_ASM(const FiniteElement<dim> &fe_cell,
                    const QGauss<dim> &       qf_cell,
                    const UpdateFlags         uf_cell,
                    const QGauss<dim - 1> &   qf_face,
                    const UpdateFlags         uf_face,
		    const BlockVector<double> &solution_old)
      : m_fe_values(fe_cell, qf_cell, uf_cell)
      , m_fe_face_values(fe_cell, qf_face, uf_face)
      , m_Nx_velo(qf_cell.size(),
		  std::vector<Tensor<1, dim>>(fe_cell.n_dofs_per_cell()))
      , m_Nx_disp(qf_cell.size(),
		  std::vector<Tensor<1, dim>>(fe_cell.n_dofs_per_cell()))
      , m_Nx_T(qf_cell.size(),
	       std::vector<double>(fe_cell.n_dofs_per_cell()))
      , m_grad_Nx_velo(qf_cell.size(),
		       std::vector<Tensor<2, dim>>(fe_cell.n_dofs_per_cell()))
      , m_grad_Nx_disp(qf_cell.size(),
                       std::vector<Tensor<2, dim>>(fe_cell.n_dofs_per_cell()))
      , m_grad_Nx_T(qf_cell.size(),
		    std::vector<Tensor<1, dim>>(fe_cell.n_dofs_per_cell()))
      , m_symm_grad_Nx_velo(qf_cell.size(),
			    std::vector<SymmetricTensor<2, dim>>(fe_cell.n_dofs_per_cell()))
      , m_symm_grad_Nx_disp(qf_cell.size(),
                            std::vector<SymmetricTensor<2, dim>>(fe_cell.n_dofs_per_cell()))
      , m_solution_previous(solution_old)
      , m_previous_velo(qf_cell.size())
      , m_previous_disp(qf_cell.size())
      , m_previous_T(qf_cell.size())
      , m_previous_grad_velo(qf_cell.size())
      , m_previous_grad_disp(qf_cell.size())
      , m_previous_grad_T(qf_cell.size())
    {}

    ScratchData_ASM(const ScratchData_ASM &rhs)
      : m_fe_values(rhs.m_fe_values.get_fe(),
                    rhs.m_fe_values.get_quadrature(),
                    rhs.m_fe_values.get_update_flags())
      , m_fe_face_values(rhs.m_fe_face_values.get_fe(),
                         rhs.m_fe_face_values.get_quadrature(),
                         rhs.m_fe_face_values.get_update_flags())
      , m_Nx_velo(rhs.m_Nx_velo)
      , m_Nx_disp(rhs.m_Nx_disp)
      , m_Nx_T(rhs.m_Nx_T)
      , m_grad_Nx_velo(rhs.m_grad_Nx_velo)
      , m_grad_Nx_disp(rhs.m_grad_Nx_disp)
      , m_grad_Nx_T(rhs.m_grad_Nx_T)
      , m_symm_grad_Nx_velo(rhs.m_symm_grad_Nx_velo)
      , m_symm_grad_Nx_disp(rhs.m_symm_grad_Nx_disp)
      , m_solution_previous(rhs.m_solution_previous)
      , m_previous_velo(rhs.m_previous_velo)
      , m_previous_disp(rhs.m_previous_disp)
      , m_previous_T(rhs.m_previous_T)
      , m_previous_grad_velo(rhs.m_previous_grad_velo)
      , m_previous_grad_disp(rhs.m_previous_grad_disp)
      , m_previous_grad_T(rhs.m_previous_grad_T)
    {}

    void reset()
    {
      const unsigned int n_q_points      = m_Nx_T.size();
      const unsigned int n_dofs_per_cell = m_Nx_T[0].size();
      for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
        {
          Assert(m_Nx_velo[q_point].size() == n_dofs_per_cell, ExcInternalError());
          Assert(m_Nx_disp[q_point].size() == n_dofs_per_cell, ExcInternalError());
          Assert(m_Nx_T[q_point].size() == n_dofs_per_cell, ExcInternalError());

          Assert(m_grad_Nx_velo[q_point].size() == n_dofs_per_cell,
                 ExcInternalError());
          Assert(m_grad_Nx_disp[q_point].size() == n_dofs_per_cell,
                 ExcInternalError());
          Assert(m_grad_Nx_T[q_point].size() == n_dofs_per_cell,
                 ExcInternalError());

          Assert(m_symm_grad_Nx_velo[q_point].size() == n_dofs_per_cell,
                 ExcInternalError());
          Assert(m_symm_grad_Nx_disp[q_point].size() == n_dofs_per_cell,
                 ExcInternalError());

          m_previous_velo[q_point] = 0.0;
          m_previous_disp[q_point] = 0.0;
          m_previous_T[q_point] = 0.0;
          m_previous_grad_velo[q_point] = 0.0;
          m_previous_grad_disp[q_point] = 0.0;
          m_previous_grad_T[q_point] = 0.0;

          for (unsigned int k = 0; k < n_dofs_per_cell; ++k)
            {
	      m_Nx_velo[q_point][k]      = 0.0;
	      m_Nx_disp[q_point][k]      = 0.0;
              m_Nx_T[q_point][k]           = 0.0;

              m_grad_Nx_velo[q_point][k]      = 0.0;
              m_grad_Nx_disp[q_point][k]      = 0.0;
              m_grad_Nx_T[q_point][k]      = 0.0;

              m_symm_grad_Nx_velo[q_point][k] = 0.0;
              m_symm_grad_Nx_disp[q_point][k] = 0.0;
            }
        }
    }
  };

  template <int dim>
  struct MonolithicSolve<dim>::PerTaskData_UQPH
  {
    void reset()
    {}
  };

  template <int dim>
  struct MonolithicSolve<dim>::ScratchData_UQPH
  {
    const BlockVector<double> & m_solution_total;

    std::vector<SymmetricTensor<2, dim>> m_solution_sym_grads_u_total;
    std::vector<double>                  m_solution_values_temperature_total;

    FEValues<dim> m_fe_values;

    ScratchData_UQPH(const FiniteElement<dim> & fe_cell,
                     const QGauss<dim> &        qf_cell,
                     const UpdateFlags          uf_cell,
                     const BlockVector<double> &solution_total)
      : m_solution_total(solution_total)
      , m_solution_sym_grads_u_total(qf_cell.size())
      , m_solution_values_temperature_total(qf_cell.size())
      , m_fe_values(fe_cell, qf_cell, uf_cell)
    {}

    ScratchData_UQPH(const ScratchData_UQPH &rhs)
      : m_solution_total(rhs.m_solution_total)
      , m_solution_sym_grads_u_total(rhs.m_solution_sym_grads_u_total)
      , m_solution_values_temperature_total(rhs.m_solution_values_temperature_total)
      , m_fe_values(rhs.m_fe_values.get_fe(),
                    rhs.m_fe_values.get_quadrature(),
                    rhs.m_fe_values.get_update_flags())
    {}

    void reset()
    {
      const unsigned int n_q_points = m_solution_sym_grads_u_total.size();
      for (unsigned int q = 0; q < n_q_points; ++q)
        {
	  m_solution_sym_grads_u_total[q]  = 0.0;
	  m_solution_values_temperature_total[q] = 0.0;
        }
    }
  };


  template <int dim>
  void MonolithicSolve<dim>::make_grid_case_1()
  {
    for (unsigned int i = 0; i < 80; ++i)
      std::cout << "*";
    std::cout << std::endl;
    std::cout << "Emulate the 1D example in Simo's paper." << std::endl;
    for (unsigned int i = 0; i < 80; ++i)
      std::cout << "*";
    std::cout << std::endl;

    double const length = m_parameters.m_total_length;

    std::vector<unsigned int> repetitions(dim, 1);
    repetitions[0] = m_parameters.m_number_of_elements;

    GridGenerator::subdivided_hyper_rectangle(m_triangulation,
					      repetitions,
					      Point<dim>( 0.0,      0.0 ),
					      Point<dim>( length,   1.0 ) );

    for (const auto &cell : m_triangulation.active_cell_iterators())
      for (const auto &face : cell->face_iterators())
	{
	  if (face->at_boundary() == true)
	    {
	      if (    (std::fabs(face->center()[0] - 0.0   ) < 1.0e-9)
		   || (std::fabs(face->center()[0] - length) < 1.0e-9) )
		face->set_boundary_id(0);
	      else
	        face->set_boundary_id(1);
	    }
	}
  }

  template <int dim>
  void MonolithicSolve<dim>::make_grid_case_2()
  {
    for (unsigned int i = 0; i < 80; ++i)
      std::cout << "*";
    std::cout << std::endl;
    std::cout << "2D example." << std::endl;
    for (unsigned int i = 0; i < 80; ++i)
      std::cout << "*";
    std::cout << std::endl;

    double const length = m_parameters.m_total_length;

    std::vector<unsigned int> repetitions(dim, 1);
    repetitions[0] = m_parameters.m_number_of_elements;
    repetitions[1] = m_parameters.m_number_of_elements;

    GridGenerator::subdivided_hyper_rectangle(m_triangulation,
					      repetitions,
					      Point<dim>( 0.0,      0.0 ),
					      Point<dim>( length,   length ) );

    for (const auto &cell : m_triangulation.active_cell_iterators())
      for (const auto &face : cell->face_iterators())
	{
	  if (face->at_boundary() == true)
	    {
	      if (    (std::fabs(face->center()[0] - 0.0   ) < 1.0e-9)
		   || (std::fabs(face->center()[0] - length) < 1.0e-9) )
		face->set_boundary_id(0);
	      else
	        face->set_boundary_id(1);
	    }
	}
  }

  template <int dim>
  void MonolithicSolve<dim>::make_grid_case_3()
  {
    for (unsigned int i = 0; i < 80; ++i)
      std::cout << "*";
    std::cout << std::endl;
    std::cout << "3D example." << std::endl;
    for (unsigned int i = 0; i < 80; ++i)
      std::cout << "*";
    std::cout << std::endl;

    double const length = m_parameters.m_total_length;

    std::vector<unsigned int> repetitions(dim, 1);
    repetitions[0] = m_parameters.m_number_of_elements;

    GridGenerator::subdivided_hyper_rectangle(m_triangulation,
					      repetitions,
					      Point<dim>( 0.0,      0.0,    0.0  ),
					      Point<dim>( length,   1.0,    1.0  ) );

    for (const auto &cell : m_triangulation.active_cell_iterators())
      for (const auto &face : cell->face_iterators())
	{
	  if (face->at_boundary() == true)
	    {
	      if (    (std::fabs(face->center()[0] - 0.0   ) < 1.0e-9)
		   || (std::fabs(face->center()[0] - length) < 1.0e-9) )
		face->set_boundary_id(0);
	      else
	        face->set_boundary_id(1);
	    }
	}
  }

  template <int dim>
  void MonolithicSolve<dim>::make_grid_case_4()
  {
    for (unsigned int i = 0; i < 80; ++i)
      std::cout << "*";
    std::cout << std::endl;
    std::cout << "TBD-SineInterface" << std::endl;
    for (unsigned int i = 0; i < 80; ++i)
      std::cout << "*";
    std::cout << std::endl;

    GridIn<2> gridin;
    gridin.attach_triangulation(m_triangulation);
    std::ifstream f("TBC_rectangle.msh");
    gridin.read_msh(f);

    const double L = 0.03, A = 0.005;
    const double H1 = 1.0, H2 = 0.15, H3 = 0.006, H4 = 0.1;

    for (const auto &cell : m_triangulation.active_cell_iterators())
      for (const auto &face : cell->face_iterators())
	{
	  if (face->at_boundary() == true)
	    {
	      if (std::fabs(face->center()[1] - 0.0 ) < 1.0e-9 )
		face->set_boundary_id(0);
	      else if (std::fabs(face->center()[1] - (H1 + H2 + H3 + H4) ) < 1.0e-9)
	        face->set_boundary_id(1);
	      else if (std::fabs(face->center()[0] - 0.0 ) < 1.0e-9 )
	        face->set_boundary_id(2);
	      else if (std::fabs(face->center()[0] - L ) < 1.0e-9 )
	        face->set_boundary_id(3);
	    }
	}

    // if the number of DOFs are too high, the SparseDirectUMFPACK direct solver
    // will crush
    for (int i = 0; i < 2; i++)
      {
	for (const auto &cell : m_triangulation.active_cell_iterators())
	  {
	    // TGO layer
	    if (cell->material_id() > 1)
	      {
		cell->set_refine_flag();
	      }
	  }
	m_triangulation.execute_coarsening_and_refinement();
      }

    typename Triangulation<dim>::vertex_iterator vertex_ptr;
    vertex_ptr = m_triangulation.begin_active_vertex();
    while (vertex_ptr != m_triangulation.end_vertex())
      {
	Point<dim> & vertex_point = vertex_ptr->vertex();
	if (   (vertex_point(1) > H1 + H2      - 1.0e-9)
	    && (vertex_point(1) < H1 + H2 + H3 + 1.0e-9) )
	  vertex_point(1) += A * std::sin(vertex_point(0)/L*2.0*numbers::PI);
	else if (   (vertex_point(1) > H1     )
	         && (vertex_point(1) < H1 + H2) )
	  vertex_point(1) += (vertex_point(1) - H1)/H2 *
	                      A * std::sin(vertex_point(0)/L*2.0*numbers::PI);
	else if (   (vertex_point(1) > H1 + H2 + H3      )
	         && (vertex_point(1) < H1 + H2 + H3 + H4 ) )
	  vertex_point(1) += (H1 + H2 + H3 + H4 - vertex_point(1))/H4 *
	                      A * std::sin(vertex_point(0)/L*2.0*numbers::PI);
        ++vertex_ptr;
      }
  }

  template <int dim>
  void MonolithicSolve<dim>::refine_grid()
  {
    Vector<float> estimated_error_per_cell(m_triangulation.n_active_cells());

    FEValuesExtractors::Vector displacements(m_u_fe);
    ComponentMask displacements_mask = m_fe.component_mask(displacements);

    KellyErrorEstimator<dim>::estimate(m_dof_handler,
                                       QGauss<dim - 1>(m_fe.degree + 1),
                                       {},
                                       m_solution_n,
                                       estimated_error_per_cell,
				       displacements_mask);

    GridRefinement::refine_and_coarsen_fixed_number(m_triangulation,
                                                    estimated_error_per_cell,
                                                    0.05,
                                                    0.3);

    m_triangulation.execute_coarsening_and_refinement();
  }

  template <int dim>
  void MonolithicSolve<dim>::make_grid()
  {
    if (m_parameters.m_scenario == 1)
      make_grid_case_1();
    else if (m_parameters.m_scenario == 2)
      make_grid_case_2();
    else if (m_parameters.m_scenario == 3)
      make_grid_case_3();
    else if (m_parameters.m_scenario == 4)
      make_grid_case_4();
    else
      Assert(false, ExcMessage("The scenario has not been implemented!"));

    std::ofstream out("original_mesh.vtu");
    GridOut       grid_out;
    grid_out.write_vtu(m_triangulation, out);

    m_vol_reference = GridTools::volume(m_triangulation);
    std::cout << "Grid:\n\t Reference volume: " << m_vol_reference << std::endl;
  }

  template <int dim>
  void MonolithicSolve<dim>::system_setup()
  {
    m_timer.enter_subsection("Setup system");

    std::vector<unsigned int> block_component(m_n_components,
                                              m_v_dof);   // velocity
    for (unsigned int i = 0; i < dim; i++)
      block_component[m_first_u_component + i] = m_u_dof; // displacement

    block_component[m_T_component] = m_T_dof;             // temperature

    m_dof_handler.distribute_dofs(m_fe);
    DoFRenumbering::Cuthill_McKee(m_dof_handler);
    DoFRenumbering::component_wise(m_dof_handler, block_component);

    m_constraints.clear();
    DoFTools::make_hanging_node_constraints(m_dof_handler, m_constraints);
    // m_constraints need to include the periodic constraint so that the
    // sparsity pattern can be set up properly
    if (m_parameters.m_scenario == 4)
      {
	const int left_boundary_id = 2, right_boundary_id = 3;
	const unsigned int direction = 0;
	DoFTools::make_periodicity_constraints(m_dof_handler,
					       left_boundary_id,
					       right_boundary_id,
	                                       direction,
	                                       m_constraints);
      }
    m_constraints.close();

    m_dofs_per_block =
      DoFTools::count_dofs_per_fe_block(m_dof_handler, block_component);

    std::cout << "Triangulation:"
              << "\n\t Number of active cells: "
              << m_triangulation.n_active_cells()
              << "\n\t Number of used vertices: "
              << m_triangulation.n_used_vertices()
              << "\n\t Number of degrees of freedom (total): "
	      << m_dof_handler.n_dofs()
	      << "\n\t Number of degrees of freedom (velocity): "
	      << m_dofs_per_block[m_v_dof]
	      << "\n\t Number of degrees of freedom (displacement): "
	      << m_dofs_per_block[m_u_dof]
	      << "\n\t Number of degrees of freedom (temperature): "
	      << m_dofs_per_block[m_T_dof]
              << std::endl;

    m_tangent_matrix.clear();
    {

      const types::global_dof_index n_dofs_v = m_dofs_per_block[m_v_dof];
      const types::global_dof_index n_dofs_u = m_dofs_per_block[m_u_dof];
      const types::global_dof_index n_dofs_T = m_dofs_per_block[m_T_dof];

      BlockDynamicSparsityPattern dsp(m_n_blocks, m_n_blocks);

      dsp.block(m_v_dof, m_v_dof).reinit(n_dofs_v, n_dofs_v);
      dsp.block(m_v_dof, m_u_dof).reinit(n_dofs_v, n_dofs_u);
      dsp.block(m_v_dof, m_T_dof).reinit(n_dofs_v, n_dofs_T);

      dsp.block(m_u_dof, m_v_dof).reinit(n_dofs_u, n_dofs_v);
      dsp.block(m_u_dof, m_u_dof).reinit(n_dofs_u, n_dofs_u);
      dsp.block(m_u_dof, m_T_dof).reinit(n_dofs_u, n_dofs_T);

      dsp.block(m_T_dof, m_v_dof).reinit(n_dofs_T, n_dofs_v);
      dsp.block(m_T_dof, m_u_dof).reinit(n_dofs_T, n_dofs_u);
      dsp.block(m_T_dof, m_T_dof).reinit(n_dofs_T, n_dofs_T);

      dsp.collect_sizes();

      Table<2, DoFTools::Coupling> coupling(m_n_components, m_n_components);
      for (unsigned int ii = 0; ii < m_n_components; ++ii)
        for (unsigned int jj = 0; jj < m_n_components; ++jj)
          if (
               ((ii < m_first_u_component) && (jj == m_T_component))  ||
               ((ii == m_T_component) && (jj >= m_first_u_component) && (jj < m_T_component))
             )
            coupling[ii][jj] = DoFTools::none;
          else
            coupling[ii][jj] = DoFTools::always;
      DoFTools::make_sparsity_pattern(
        m_dof_handler, coupling, dsp, m_constraints, false);
      m_sparsity_pattern.copy_from(dsp);
    }

    m_tangent_matrix.reinit(m_sparsity_pattern);

    m_system_rhs.reinit(m_dofs_per_block);
    m_system_rhs.collect_sizes();

    m_solution_n.reinit(m_dofs_per_block);
    m_solution_n.collect_sizes();

    m_solution_old.reinit(m_dofs_per_block);
    m_solution_old.collect_sizes();

    setup_qph();

    m_timer.leave_subsection();
  }

  template <int dim>
  void MonolithicSolve<dim>::assemble_system(const BlockVector<double> & solution_old)
  {
    std::cout << "    assemble system" << std::endl;
    m_timer.enter_subsection("Assemble system");

    m_tangent_matrix = 0.0;
    m_system_rhs     = 0.0;

    const UpdateFlags uf_cell(update_values | update_gradients |
			      update_quadrature_points | update_JxW_values);
    const UpdateFlags uf_face(update_values | update_normal_vectors |
                              update_JxW_values);

    PerTaskData_ASM per_task_data(m_dofs_per_cell);
    ScratchData_ASM scratch_data(m_fe, m_qf_cell, uf_cell, m_qf_face, uf_face, solution_old);

    auto worker =
      [this](const typename DoFHandler<dim>::active_cell_iterator &cell,
	     ScratchData_ASM &                                    scratch,
	     PerTaskData_ASM &                                    data)
      {
        this->assemble_system_one_cell(cell, scratch, data);
      };

    auto copier = [this](const PerTaskData_ASM &data)
      {
        this->m_constraints.distribute_local_to_global(data.m_cell_matrix,
                                                       data.m_cell_rhs,
                                                       data.m_local_dof_indices,
                                                       m_tangent_matrix,
                                                       m_system_rhs);
      };

    WorkStream::run(
      m_dof_handler.active_cell_iterators(),
      worker,
      copier,
      scratch_data,
      per_task_data);

    m_timer.leave_subsection();
  }

  template <int dim>
  void MonolithicSolve<dim>::assemble_system_one_cell(
    const typename DoFHandler<dim>::active_cell_iterator &cell,
    ScratchData_ASM &                                     scratch,
    PerTaskData_ASM &                                     data) const
  {
    data.reset();
    scratch.reset();
    scratch.m_fe_values.reinit(cell);
    cell->get_dof_indices(data.m_local_dof_indices);

    const std::vector<std::shared_ptr<const PointHistory<dim>>> lqph =
      m_quadrature_point_history.get_data(cell);
    Assert(lqph.size() == m_n_q_points, ExcInternalError());

    Assert(scratch.m_previous_velo.size() == m_n_q_points,
           ExcInternalError());
    Assert(scratch.m_previous_disp.size() == m_n_q_points,
           ExcInternalError());
    Assert(scratch.m_previous_T.size() == m_n_q_points,
           ExcInternalError());
    Assert(scratch.m_previous_grad_velo.size() == m_n_q_points,
           ExcInternalError());
    Assert(scratch.m_previous_grad_disp.size() == m_n_q_points,
           ExcInternalError());
    Assert(scratch.m_previous_grad_T.size() == m_n_q_points,
           ExcInternalError());

    scratch.m_fe_values[m_v_fe].get_function_values(
      scratch.m_solution_previous, scratch.m_previous_velo);
    scratch.m_fe_values[m_u_fe].get_function_values(
      scratch.m_solution_previous, scratch.m_previous_disp);
    scratch.m_fe_values[m_T_fe].get_function_values(
      scratch.m_solution_previous, scratch.m_previous_T);

    scratch.m_fe_values[m_v_fe].get_function_gradients(
      scratch.m_solution_previous, scratch.m_previous_grad_velo);
    scratch.m_fe_values[m_u_fe].get_function_gradients(
      scratch.m_solution_previous, scratch.m_previous_grad_disp);
    scratch.m_fe_values[m_T_fe].get_function_gradients(
      scratch.m_solution_previous, scratch.m_previous_grad_T);

    const double delta_time = m_time.get_delta_t();

    for (const unsigned int q_point :
         scratch.m_fe_values.quadrature_point_indices())
      {
        for (const unsigned int k : scratch.m_fe_values.dof_indices())
          {
            const unsigned int k_group = m_fe.system_to_base_index(k).first.first;

            if (k_group == m_v_dof)
              {
                scratch.m_Nx_velo[q_point][k] =
                  scratch.m_fe_values[m_v_fe].value(k, q_point);
                scratch.m_grad_Nx_velo[q_point][k] =
                  scratch.m_fe_values[m_v_fe].gradient(k, q_point);
                scratch.m_symm_grad_Nx_velo[q_point][k] =
                  symmetrize(scratch.m_grad_Nx_velo[q_point][k]);
              }
            else if (k_group == m_u_dof)
              {
                scratch.m_Nx_disp[q_point][k] =
                  scratch.m_fe_values[m_u_fe].value(k, q_point);
                scratch.m_grad_Nx_disp[q_point][k] =
                  scratch.m_fe_values[m_u_fe].gradient(k, q_point);
                scratch.m_symm_grad_Nx_disp[q_point][k] =
                  symmetrize(scratch.m_grad_Nx_disp[q_point][k]);
              }
            else if (k_group == m_T_dof)
              {
                scratch.m_Nx_T[q_point][k] =
                  scratch.m_fe_values[m_T_fe].value(k, q_point);
                scratch.m_grad_Nx_T[q_point][k] =
                  scratch.m_fe_values[m_T_fe].gradient(k, q_point);
              }
            else
              Assert(k_group <= m_T_dof, ExcInternalError());
          }
      }

    for (const unsigned int q_point :
         scratch.m_fe_values.quadrature_point_indices())
      {

        const SymmetricTensor<2, dim> m_coupling = lqph[q_point]->get_m_coupling();
        const SymmetricTensor<4, dim> mechanical_C = lqph[q_point]->get_mechanical_C();
        const SymmetricTensor<2, dim> thermal_conductivity = lqph[q_point]->get_thermal_conductivity();
        const double heat_capacity = lqph[q_point]->get_heat_capacity();
        const double density = lqph[q_point]->get_density();
        const double temperature_ref = lqph[q_point]->get_temperature_ref();

        const std::vector<Tensor<1,dim>> &          N_velo = scratch.m_Nx_velo[q_point];
        const std::vector<Tensor<1,dim>> &          N_disp = scratch.m_Nx_disp[q_point];
        const std::vector<double> &                 N_T = scratch.m_Nx_T[q_point];

        const std::vector<SymmetricTensor<2, dim>> &symm_grad_Nx_disp =
          scratch.m_symm_grad_Nx_disp[q_point];

        const std::vector<SymmetricTensor<2, dim>> &symm_grad_Nx_velo =
          scratch.m_symm_grad_Nx_velo[q_point];

        const std::vector<Tensor<1, dim>> & grad_Nx_T =
          scratch.m_grad_Nx_T[q_point];

        const Tensor<1, dim> previous_velo = scratch.m_previous_velo[q_point];
        const Tensor<1, dim> previous_disp = scratch.m_previous_disp[q_point];
        const double         previous_T    = scratch.m_previous_T[q_point];

        const double         JxW = scratch.m_fe_values.JxW(q_point);

        const SymmetricTensor<2, dim>
          previous_sym_grad_velo =
            symmetrize(scratch.m_previous_grad_velo[q_point]);
        const SymmetricTensor<2, dim>
          previous_sym_grad_disp =
            symmetrize(scratch.m_previous_grad_disp[q_point]);
        const Tensor<1, dim> previous_grad_T = scratch.m_previous_grad_T[q_point];


        for (const unsigned int i : scratch.m_fe_values.dof_indices())
          {
            const unsigned int i_group = m_fe.system_to_base_index(i).first.first;

            if (i_group == m_v_dof)
              {
                data.m_cell_rhs(i) += (  N_velo[i] * previous_disp
                                       + delta_time/2.0 * N_velo[i] * previous_velo) * JxW;
              }
            else if (i_group == m_u_dof)
              {
                data.m_cell_rhs(i) += (  N_disp[i] * previous_velo
                     - delta_time/2.0/density * symm_grad_Nx_disp[i] * mechanical_C * previous_sym_grad_disp
		     + delta_time/2.0/density * symm_grad_Nx_disp[i] * m_coupling * previous_T) * JxW;
              }
            else if (i_group == m_T_dof)
              {
        	// backward Euler could get rid of the temperature oscillation due to sudden temperature change
                // data.m_cell_rhs(i) += (  N_T[i] * previous_T ) * JxW;
                // Crank-Nicholson
                data.m_cell_rhs(i) += (  N_T[i] * previous_T
                     - delta_time/2.0/heat_capacity * grad_Nx_T[i] * thermal_conductivity * previous_grad_T
		     - delta_time/2.0/heat_capacity * temperature_ref
		     * N_T[i] * m_coupling * previous_sym_grad_velo ) * JxW;
              }
            else
              Assert(i_group <= m_T_dof, ExcInternalError());

            for (const unsigned int j : scratch.m_fe_values.dof_indices())
              {
                const unsigned int j_group =
                  m_fe.system_to_base_index(j).first.first;

                if ((i_group == m_v_dof) && (j_group == m_v_dof))
                  {
		    data.m_cell_matrix(i, j) += (-delta_time/2.0) * N_velo[i] * N_velo[j] * JxW;
                  }
                else if ((i_group == m_v_dof) && (j_group == m_u_dof))
                  {
                    data.m_cell_matrix(i, j) += N_velo[i] * N_disp[j] * JxW;
                  }
                else if ((i_group == m_u_dof) && (j_group == m_v_dof))
                  {
                    data.m_cell_matrix(i, j) += N_disp[i] * N_velo[j] * JxW;
                  }
                else if ((i_group == m_u_dof) && (j_group == m_u_dof))
                  {
                    data.m_cell_matrix(i, j) += delta_time/2.0/density *
                	                        symm_grad_Nx_disp[i] * mechanical_C * symm_grad_Nx_disp[j] * JxW;
                  }
                else if ((i_group == m_u_dof) && (j_group == m_T_dof))
                  {
                    data.m_cell_matrix(i, j) += (-delta_time/2.0/density) *
                	                        (symm_grad_Nx_disp[i] * m_coupling * N_T[j]) * JxW;
                  }
                else if ((i_group == m_T_dof) && (j_group == m_v_dof))
                  {
             	    // backward Euler
                    // data.m_cell_matrix(i, j) += delta_time/heat_capacity*temperature_ref * N_T[i] *
                    //                          (m_coupling * symm_grad_Nx_velo[j]) * JxW;
                    // Crank-Nicholson
                    data.m_cell_matrix(i, j) += delta_time/2.0/heat_capacity*temperature_ref * N_T[i] *
                                  	                      (m_coupling * symm_grad_Nx_velo[j]) * JxW;
                  }
                else if ((i_group == m_T_dof) && (j_group == m_T_dof))
                  {
            	    // backward Euler
                    //data.m_cell_matrix(i, j) += (  N_T[i] * N_T[j]
	            //		               + delta_time/heat_capacity
		    //	    		       * grad_Nx_T[i] * thermal_conductivity * grad_Nx_T[j] )
		    //			       * JxW;
                    // Crank-Nicholson
                    data.m_cell_matrix(i, j) += (  N_T[i] * N_T[j]
		             	             + delta_time/2.0/heat_capacity
				    	     * grad_Nx_T[i] * thermal_conductivity * grad_Nx_T[j] )
					     * JxW;
                  }
                else
                  Assert((i_group <= m_T_dof) && (j_group <= m_T_dof),
                         ExcInternalError());
              }
          }
      }
  }

  template <int dim>
  void MonolithicSolve<dim>::make_constraints()
  {
    m_constraints.clear();
    DoFTools::make_hanging_node_constraints(m_dof_handler, m_constraints);
    if (m_parameters.m_scenario == 1)
      {
	int boundary_id = 0;
	VectorTools::interpolate_boundary_values(m_dof_handler,
						 boundary_id,
						 Functions::ZeroFunction<dim>(m_n_components),
						 m_constraints );

	boundary_id = 1;
	const FEValuesExtractors::Scalar y_velocity(1);
	VectorTools::interpolate_boundary_values(m_dof_handler,
						 boundary_id,
						 Functions::ZeroFunction<dim>(m_n_components),
						 m_constraints,
						 m_fe.component_mask(y_velocity));

	const FEValuesExtractors::Scalar y_displacement(dim+1);
	VectorTools::interpolate_boundary_values(m_dof_handler,
						 boundary_id,
						 Functions::ZeroFunction<dim>(m_n_components),
						 m_constraints,
						 m_fe.component_mask(y_displacement));
      }
    else if (m_parameters.m_scenario == 2)
      {
	int boundary_id = 0;
	VectorTools::interpolate_boundary_values(m_dof_handler,
						 boundary_id,
						 Functions::ZeroFunction<dim>(m_n_components),
						 m_constraints );
	boundary_id = 1;
	VectorTools::interpolate_boundary_values(m_dof_handler,
						 boundary_id,
						 Functions::ZeroFunction<dim>(m_n_components),
						 m_constraints );

      }
    else if (m_parameters.m_scenario == 3)
      {
	int boundary_id = 0;
	VectorTools::interpolate_boundary_values(m_dof_handler,
						 boundary_id,
						 Functions::ZeroFunction<dim>(m_n_components),
						 m_constraints);

	boundary_id = 1;
	const FEValuesExtractors::Scalar y_velocity(1);
	VectorTools::interpolate_boundary_values(m_dof_handler,
						 boundary_id,
						 Functions::ZeroFunction<dim>(m_n_components),
						 m_constraints,
						 m_fe.component_mask(y_velocity));

	const FEValuesExtractors::Scalar z_velocity(2);
	VectorTools::interpolate_boundary_values(m_dof_handler,
						 boundary_id,
						 Functions::ZeroFunction<dim>(m_n_components),
						 m_constraints,
						 m_fe.component_mask(z_velocity));

	const FEValuesExtractors::Scalar y_displacement(dim+1);
	VectorTools::interpolate_boundary_values(m_dof_handler,
						 boundary_id,
						 Functions::ZeroFunction<dim>(m_n_components),
						 m_constraints,
						 m_fe.component_mask(y_displacement));

	const FEValuesExtractors::Scalar z_displacement(dim+2);
	VectorTools::interpolate_boundary_values(m_dof_handler,
						 boundary_id,
						 Functions::ZeroFunction<dim>(m_n_components),
						 m_constraints,
						 m_fe.component_mask(z_displacement));
      }
    else if (m_parameters.m_scenario == 4)
      {
	const FEValuesExtractors::Vector velocities(0);
	const FEValuesExtractors::Vector displacements(dim);
	const FEValuesExtractors::Scalar temperature(dim+dim);
	const double cool_temperature = 0.0; // Kelvin (relative temperature)
	//const double hot_temperature = 500.0; // Kelvin (relative temperature)

        // bottom surface
	int boundary_id = 0;
	VectorTools::interpolate_boundary_values(m_dof_handler,
						 boundary_id,
						 Functions::ZeroFunction<dim>(m_n_components),
						 m_constraints,
						 m_fe.component_mask(velocities));
	VectorTools::interpolate_boundary_values(m_dof_handler,
						 boundary_id,
						 Functions::ZeroFunction<dim>(m_n_components),
						 m_constraints,
						 m_fe.component_mask(displacements));
	VectorTools::interpolate_boundary_values(m_dof_handler,
						 boundary_id,
						 Functions::ConstantFunction<dim>(cool_temperature,
										  m_n_components),
						 m_constraints,
						 m_fe.component_mask(temperature));

	// top surface
	boundary_id = 1;
	VectorTools::interpolate_boundary_values(m_dof_handler,
						 boundary_id,
						 BoundaryValueTBCTopSurface<dim>(m_n_components,
										 m_time.current()),
						 m_constraints,
						 m_fe.component_mask(temperature));

	// periodic boundary conditions for the left and right surfaces
	const int left_boundary_id = 2, right_boundary_id = 3;
	const unsigned int direction = 0;
	DoFTools::make_periodicity_constraints(m_dof_handler,
					       left_boundary_id,
					       right_boundary_id,
	                                       direction,
	                                       m_constraints);
      }
    else
      Assert(false, ExcMessage("The scenario has not been implemented!"));

    m_constraints.close();
  }

  template <int dim>
  void MonolithicSolve<dim>::output_results() const
  {
    std::cout << "    output results" << std::endl;

    m_timer.enter_subsection("Output results");

    DataOut<dim> data_out;
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      data_component_interpretation(
        dim+dim, DataComponentInterpretation::component_is_part_of_vector);
    data_component_interpretation.push_back(
       DataComponentInterpretation::component_is_scalar);

    std::vector<std::string> solution_name(dim, "velocity");
    solution_name.emplace_back("displacement");
    solution_name.emplace_back("displacement");
    if (dim == 3)
      solution_name.emplace_back("displacement");

    solution_name.emplace_back("temperature");

    DataOutBase::VtkFlags output_flags;
    output_flags.write_higher_order_cells       = false;
    output_flags.physical_units["velocity"] = "m/s";
    output_flags.physical_units["displacement"] = "m";
    output_flags.physical_units["temperature"] = "K";

    data_out.set_flags(output_flags);

    data_out.attach_dof_handler(m_dof_handler);
    data_out.add_data_vector(m_solution_n,
                             solution_name,
                             DataOut<dim>::type_dof_data,
                             data_component_interpretation);

    Vector<double> cell_material_id(m_triangulation.n_active_cells());
    // output material ID for each cell
    for (const auto &cell : m_dof_handler.active_cell_iterators())
      {
	cell_material_id(cell->active_cell_index()) = cell->material_id();
      }
    data_out.add_data_vector(cell_material_id, "materialID");

    // Stress L2 projection
    DoFHandler<dim> stresses_dof_handler_L2(m_triangulation);
    FE_Q<dim>     stresses_fe_L2(m_parameters.m_poly_degree); //FE_Q element is continuous
    stresses_dof_handler_L2.distribute_dofs(stresses_fe_L2);
    AffineConstraints<double> constraints;
    constraints.clear();
    DoFTools::make_hanging_node_constraints(stresses_dof_handler_L2, constraints);
    constraints.close();
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
	  data_component_interpretation_stress(1,
					       DataComponentInterpretation::component_is_scalar);

    for (unsigned int i = 0; i < dim; ++i)
      for (unsigned int j = i; j < dim; ++j)
	{
	  Vector<double> stress_field_L2;
	  stress_field_L2.reinit(stresses_dof_handler_L2.n_dofs());

	  MappingQ<dim> mapping(m_parameters.m_poly_degree + 1);
	  VectorTools::project(mapping,
			       stresses_dof_handler_L2,
			       constraints,
			       m_qf_cell,
			       [&] (const typename DoFHandler<dim>::active_cell_iterator & cell,
				    const unsigned int q) -> double
			       {
				 return m_quadrature_point_history.get_data(cell)[q]->get_cauchy_stress()[i][j];
			       },
			       stress_field_L2);

	  std::string stress_name = "Cauchy_stress_" + std::to_string(i+1) + std::to_string(j+1)
				  + "_L2";

	  data_out.add_data_vector(stresses_dof_handler_L2,
				   stress_field_L2,
				   stress_name,
				   data_component_interpretation_stress);
	}

    // Native stress average over Gauss points
    for (unsigned int i = 0; i < dim; ++i)
      for (unsigned int j = i; j < dim; ++j)
	{
	  Vector<double> cell_cauchy_stress(m_triangulation.n_active_cells());
	  for (const auto &cell : m_dof_handler.active_cell_iterators())
	    {
	      // if output_results() is defined as const, then
	      // we also need to put a const in std::shared_ptr,
	      // that is, std::shared_ptr<const PointHistory<dim>>
	      const std::vector<std::shared_ptr<const PointHistory<dim>>> lqph =
		m_quadrature_point_history.get_data(cell);
	      Assert(lqph.size() == m_n_q_points, ExcInternalError());

	      double cauchy_stress_gp = 0.0;
	      for (unsigned int q_point = 0; q_point < m_n_q_points; ++q_point)
		{
		  cauchy_stress_gp += lqph[q_point]->get_cauchy_stress()[i][j];
		}
	      cell_cauchy_stress(cell->active_cell_index()) = cauchy_stress_gp/m_n_q_points;
	    }
	  std::string stress_name = "Cauchy_stress_" + std::to_string(i+1) + std::to_string(j+1)
				  + "_Avg";
	  data_out.add_data_vector(cell_cauchy_stress, stress_name);
	}

    data_out.build_patches();
    std::ofstream output("solution-" + std::to_string(dim) + "d-" +
			 Utilities::int_to_string(m_time.get_timestep(),3) + ".vtu");
    data_out.write_vtu(output);

    m_timer.leave_subsection();
  }

  template <int dim>
  void MonolithicSolve<dim>::read_time_data(const std::string &data_file,
				  std::vector<std::array<double, 3>> & time_table) const
  {
    std::ifstream myfile (data_file);

    double t_0, t_1, delta_t;

    if (myfile.is_open())
      {
	std::cout << "Reading time data file ..." << std::endl;

	while ( myfile >> t_0
		       >> t_1
		       >> delta_t)
	  {
	    Assert( t_0 < t_1,
		    ExcMessage("For each time pair, "
			       "the start time should be smaller than the end time"));
	    time_table.push_back({{t_0, t_1, delta_t}});
	  }

	Assert(std::fabs(t_1 - m_parameters.m_end_time) < 1.0e-9,
	       ExcMessage("End time in time table is inconsistent with input data in parameters.prm"))

	Assert(time_table.size() > 0,
	       ExcMessage("Time data file is empty."));
	myfile.close();
      }
    else
      {
        std::cout << "Time data file : " << data_file << " not exist!" << std::endl;
        Assert(false, ExcMessage("Failed to read time data file"));
      }

    for (auto & time_group : time_table)
      {
	std::cout << time_group[0] << ",\t"
	          << time_group[1] << ",\t"
		  << time_group[2] << std::endl;
      }
  }
} // namespace LinearThermoElasticCoupling


int main()
{
  using namespace LinearThermoElasticCoupling;

  try
    {
      const unsigned int dim = 2;
      MonolithicSolve<dim>         solid2D("parameters.prm");
      solid2D.run();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}
