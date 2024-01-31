# Finite Difference Methods for Solving Diel Vertical Migration

This repository is a result of the course "TMA4500 Industrial Mathematics, Specialization Project" on NTNU. The purpose of this course is to get experience with doing a larger project and writing a scientific article before the master thesis.

My project revolved around implementing finite difference methods for solving a Hamilton-Jacobi equation of the form $u_t+u_x^2-bu_{xx} + \mu u +f=0$ with Neumann boundary conditions in space and an initial value, and a Fokker-Planck equation of the form $m_t + 2u_x m + bm_{xx}=0$ with no-flux boundary conditions in space and a terminal value.

This repository contains the python file "proj_fnc.py" containing functions for solving the HJ- and the FP-equation along with some practical functions. The file "numerical_results.ipynb" contains implementation of test problems and numerical tests of the numerical methods from "proj_fnc.py".
