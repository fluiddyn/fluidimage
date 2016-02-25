
UVmat uses thin plate spline for the interpolation and the calculation
of the derivative of the fields.

Files (and functions) in UVmat related to that
----------------------------------------------

base functions
~~~~~~~~~~~~~~

- tps_coeff.m
- tps_eval.m
- tps_eval_dxy.m

These functions are implemented exactly as in Matlab in the file
tps_base.py.
  
example
~~~~~~~

- test_tps.m

This script has been translated in bad Python in the file try_tps.py.

Others
~~~~~~

- filter_tps.m: "find the thin plate spline coefficients for
  interpolation-smoothing"

- tps_coeff_field.m: "calculate the thin plate spline (tps)
  coefficients within subdomains for a field structure"

- set_subdomains.m: "sort a set of points defined by scattered
  coordinates in subdomains, as needed for tps interpolation"

- calc_field_tps.m: "Defines fields (velocity, vort, div...) from civ
  data and calculate them with tps interpolation"


The full documentations of these Matlab functions is now given:
  
filter_tps
^^^^^^^^^^

Find the thin plate spline coefficients for interpolation-smoothing
    
OUTPUT:

- SubRange(NbCoord,NbSubdomain,2): range (min, max) of the
  coordiantes x and y respectively, for each subdomain
    
- NbCentre(NbSubdomain): number of source points for each subdomain
    
- FF: false flags
    
- U_smooth, V_smooth: filtered velocity components at the positions of
  the initial data
    
- Coord_tps(NbCentre,NbCoord,NbSubdomain): positions of the tps centres
    
- U_tps,V_tps: weight of the tps for each subdomain
    
to get the interpolated field values, use the function calc_field.m
    
INPUT:
    
- coord=[X Y]: matrix whose first column is the x coordinates of the
  initial data, the second column the y coordiantes
    
- U,V: set of velocity components of the initial data

- Rho: smoothing parameter
    
- Threshold: max diff accepted between smoothed and initial data 

- Subdomain: estimated number of data points in each subdomain


tps_coeff_field
^^^^^^^^^^^^^^^

Calculate the thin plate spline (tps) coefficients within subdomains
for a field structure

DataOut=tps_coeff_field(DataIn,checkall) 

OUTPUT:

DataOut: output field structure, reproducing the input field structure
        DataIn and adding the fields: .Coord_tps .[VarName '_tps'] for
        each eligible input variable VarName (scalar or vector
        components)

errormsg: error message, = '' by default

INPUT:

- DataIn: input field structure checkall:

  * =1 if tps is needed for all fields ( a projection mode interp_tps
    has been chosen),

  * =0 otherwise (tps only needed to get spatial derivatives of
    scattered data)


called functions: 'find_field_cells', 'set_subdomains', 'tps_coeff'


set_subdomains
^^^^^^^^^^^^^^

Sort a set of points defined by scattered coordinates in subdomains,
as needed for tps interpolation

[SubRange,NbCentre,IndSelSubDomain] = set_subdomains(Coord,SubDomainNbPoint)

OUTPUT:

- SubRange(NbCoord,NbSubdomain,2): range (min, max) of the coordinates
  x and y respectively, for each subdomain

- NbCentre(NbSubdomain): number of source points for each subdomain

- IndSelSubDomain(SubDomainNbPointMax,NbSubdomain): set of indices of
  the input point array selected in each subdomain, =0 beyond NbCentre
  points

INPUT:

- coord=[X Y]: matrix whose first column is the x coordinates of the
  input data points, the second column the y coordinates

- SubdomainNbPoint: estimated number of data points whished for each
  subdomain

calc_field_tps
^^^^^^^^^^^^^^

Defines fields (velocity, vort, div...) from civ data and calculate
them with tps interpolation

[DataOut,VarAttribute,errormsg]=calc_field_tps(Coord_tps,NbCentre,SubRange,FieldVar,FieldName,Coord_interp)

OUTPUT:

- DataOut: structure representing the output fields
- VarAttribute: cell array of structures coontaining the variable attributes 
- errormsg: error msg , = '' by default

INPUT:

- Coord_tps: coordinates of the centres, of dimensions
  [nb_point,nb_coord,nb_subdomain], where

  * nb_point is the max number of data point in a subdomain,
  * nb_coord the space dimension, 
  * nb_subdomain the nbre of subdomains used for tps

- NbCentre: nbre of tps centres for each subdomain, of dimension nb_subdomain

- SubRange: coordinate range for each subdomain, of dimensions
  [nb_coord,2,nb_subdomain]

- FieldVar: array representing the input fields as tps weights with
  dimension (nbvec_sub+3,NbSubDomain,nb_dim)

  * nbvec_sub= max nbre of vectors in a subdomain  
  * NbSubDomain =nbre of subdomains
  * nb_dim: nbre of dimensions for vector components (x-> 1, y->2)

- FieldName: cell array representing the list of operations (eg
  div(U,V), rot(U,V))

- Coord_interp: coordinates of sites on which the fields need to be
  calculated of dimensions

  * [nb_site,nb_coord] for an array of interpolation sites

  * [nb_site_y,nb_site_x,nb_coord] for interpolation on a plane grid
    of size [nb_site_y,nb_site_x]
