## SmallDeformationThermoMechanicalCoupling
Monolithic approach and staggered approach for transient responses of small deformation thermomechanically coupled system. The code is written using deal.II (9.4.0).

### Purpose 
This repository provides the source code and the input files for the numerical examples used in the paper draft titled “Finite element simulations of the thermomechanically coupled responses of thermal barrier coating systems using an unconditionally stable staggered approach”. The repository contains the following content:
1. the source code of the monolithic approach and the staggered approach for thermomechanically coupled system. For the staggered approach, it has the option of the (conditionally stable) isothermal split and the (unconditionally stable) adiabatic split.
2. the input files for the numerical examples used in the aforementioned manuscript.

### How to compile
The monolithic approach and the staggered approach for the thermomechanically coupled system are implemented in [deal.II](https://www.dealii.org/) (version 9.4.0), which is an open source finite element library. In order to compile the code (**main.cc**) provided here, deal.II should be configured with MPI and at least with the interfaces to BLAS, LAPACK, Threading Building Blocks (TBB), and UMFPACK. For optional interfaces to other software packages, see https://www.dealii.org/developer/readme.html.

Once the deal.II library is compiled, for instance, to "~/dealii-9.4.0/bin/", follow the steps listed below:
1. cd MonolithicCode (assume you want to use the monolithic approach)
2. cmake -DDEAL_II_DIR=~/dealii-9.4.0/bin/ . (assume you installed dealii-9.4.0 under the $Home directory)
3. make debug or make release
4. make

### How to run
1. copy the input files contained in one of the example folders into the folder containing the source code (for instance, MonolithicCode or StaggeredCode)
2. make run

### Reference
This work is based on the algorithms proposed in the paper:

Armero, F., Simo, J.C., 1992. A new unconditionally stable fractional step method for non-linear coupled thermomechanical problems. International Journal for Numerical Methods in Engineering 35, 737–766. doi:https://doi.org/10.1002/nme.1620350408.
