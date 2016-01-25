Analysis of UVmat civ_series algorithm and parameters
=====================================================

See the `UVmat documentation <http://servforge.legi.grenoble-inp.fr/projects/soft-uvmat/wiki/UvmatHelp#Civ>`_.

Description of a typical .xml
-----------------------------

Input and Output files
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: matlab

   Series.InputTable = '/RootPath &amp; /SubDir/ &amp; RootFile &amp; _Index &amp; .ext'
   % Input files address
   % example: /home/users/campagne8a/Work/MATLAB/UVMAT/UVMAT_DEMO_SOURCES/UVMAT_DEMO01_pair &amp; /images &amp; /frame &amp; _1 &amp; .png
		
   Series.IndexRange.first_i = 1
   Series.IndexRange.incr_i = 1
   Series.IndexRange.last_i = 2
   Series.IndexRange.TimeUnit = s
   Series.IndexRange.MaxIndex_i = 2
   Series.IndexRange.MinIndex_i = 1
   Series.IndexRange.TimeSource =

   % compute from first_i to last_i every incr_i
   % MinIndex_i and MaxIndex_i are available image indices


   Series.OutputSubDir = '/images'
   Series.OutputDirExt = '.civ'
   Series.OutputRootFile = '/frame'
   %  Write in /RootPath/images.civ/frame1-2

   Series.CkeckOverwrite = 1

Program to execute
~~~~~~~~~~~~~~~~~~

.. code-block:: matlab

   Series.Action.ActionPath = 'U:\project\watu\2015\15GRAVIT\UVMAT\series\'
   %  address of program
   Series.Action.ActionName = 'civ_series'
   %  ???? same as  Series.ActionInput.Program ?????
   Series.Action.RUN = 1
   %  0 or 1
   Series.Action.status = 0
   % ?????


Parameters for computation
~~~~~~~~~~~~~~~~~~~~~~~~~~
.. code-block:: matlab

   %%  Set what to do
   % civ1: the initial image correlation process which by itself already provides a velocity field
   % Fix1: detection of 'false' velocity vectors according to different criteria. 
   % Patch1: interpolation and filtering on a regular grid, providing access to spatial derivatives of the velocity (divergence, curl, strain).
   % advanced image correlation algorithm using the result of civ1 as a first approximation.
   % fix2 and patch2: similar as fix1 and patch1, but applied to the civ2 results.

   Series.ActionInput.CheckCiv1 = 1
   Series.ActionInput.CheckFix1 = 1
   Series.ActionInput.CheckPatch1 = 1
   Series.ActionInput.CheckCiv2 = 1
   Series.ActionInput.CheckFix2 = 1
   Series.ActionInput.CheckPatch2 = 1
   % run or not run Civ1 these available steps

.. code-block:: matlab

   %% Set which pairs have to be correlated
   Series.ActionInput.PairIndices.ListPairCiv1 = 'Di= 0|1 :dt= 1'
   Series.ActionInput.PairIndices.ListPairCiv2 = 'Di= 0|1 :dt= 1'
   % examples
   % Di= 0|1 :dt= 1 means correlation between image i+0 and i+1 => dt=1-0=1
   % Di= -1|1 :dt= 2 means correlation between image i-1 and i+1 => dt=1-(-1)=2
   % Dj ....... same as Di but with j index
   % j= a-b :dt= 1000 means correlation between images j=a and b, dt is in ms
   % j= 1-2 :dt= 1000 means correlation between images j=1 and 2, dt is in ms
 
   Series.ActionInput.PairIndices.dt_unit = 'dt in mframe'
   Series.ActionInput.PairIndices.TimeSource =
   % If timing from an XML file <ImaDoc> has been detected, this is indicated in the edit box [ImaDoc] and the corresponding time intervals are indicated (in ms). For some applications, this time interval may evolve in time, so that reference indices ref_i and ref_j are chosen for the display. 

   Series.ActionInput.PairIndices.ListPairMode = 'series(Di)'
   % series(Di) or series(Dj) or pair j1-j2 .......
 
   Series.ActionInput.PairIndices.MinIndex_i = 1
   Series.ActionInput.PairIndices.MaxIndex_i = 2
   % ???? is this the same information as Series.IndexRange.MaxIndex_i and Series.IndexRange.MinIndex_i ?????
   Series.ActionInput.PairIndices.ref_i = 1
   % ????
   Series.ActionInput.PairIndices.TimeUnit = 'frame'

.. code-block:: matlab

   %% Parameters for Civ1
   Series.ActionInput.Civ1.CorrBoxSize = [25 25]
   Series.ActionInput.Civ1.SearchBoxSize = [55 55]
   Series.ActionInput.Civ1.SearchBoxShift = [0 0]
   % CorrBoxSize set the size (in pixels) of the 'correlation box', the sliding window used to get image correlations.
   % SearchBoxSize set the size of the 'search box' in which image correlation is calculated.
   % This search box can be shifted with respect to the correlation box by parameters (SearchBoxShift). This is useful in the presence of a known mean flow. 

  % This gives correlations between sub-images SubA and SubB of images A and B defined as
  % SubA = A( iref - lx : iref + lx , jref - ly: jref + ly)
  % SubB = B( iref - sx + shiftx: iref + sx + shiftx, jref - sy + shifty : jref + sy + shifty)

  % with
  % iref and  jref:  middle position of given subimage (defined by the grid)
  % shiftx and shifty : shift for this given pair of subimages
  % lx = CorrBoxSize(1)/2, ly = CorrBoxSize(2)/2
  % sx = SearchBoxSize(1)/2, sy = SearchBoxSize(2)/2

  Series.ActionInput.Civ1.CorrSmoot = 1
  % choose of the subpixel determination of the interpolation max: 1 for SUBPIXGAUSS, 2 for SUBPIX2DGAUSS
  Series.ActionInput.Civ1.Dx = 20
  Series.ActionInput.Civ1.Dy = 20
  % Dx, Dy: mesh for PIV calculation
  % ???? How is it defined???
 
  Series.ActionInput.Civ1.CheckGrid = 0
  Series.ActionInput.Civ1.CheckMask = 0
  Series.ActionInput.Civ1.CheckThreshold = 0
  % if 1: look for files to apply grid, mask etc...

  Series.ActionInput.TestCiv1 = 0
  % ????

.. code-block:: matlab

  %% Parameters for Fix1
  Series.ActionInput.Fix1.CheckFmin2 = 1
  Series.ActionInput.Fix1.CheckF3 = 1
  Series.ActionInput.Fix1.MinCorr = 0.2
  % ??????????????????????

.. code-block:: matlab

  %% Parameters for Patch1
  Series.ActionInput.Patch1.FieldSmooth = 10
  Series.ActionInput.Patch1.MaxDiff = 1.5
  Series.ActionInput.Patch1.SubDomainSize = 1000
  Series.ActionInput.Patch1.TestPatch1 = 0
  % see function filter_tps.m

.. code-block:: matlab

  %% Parameters for Civ2
  Series.ActionInput.Civ2.CorrBoxSize = [21 21]
  Series.ActionInput.Civ2.CorrSmooth = 1
  Series.ActionInput.Civ2.SearchBoxSize = [27 27]
  Series.ActionInput.Civ2.CheckDeformation = 0
  % for subpixel interpolation and image deformation (linear transform)
  % => use of DUDX DUDY etc... before crop of sub-images
  Series.ActionInput.Civ2.Dx = 10
  Series.ActionInput.Civ2.Dy = 10
  Series.ActionInput.Civ2.CheckGrid = 0
  Series.ActionInput.Civ2.CheckMask = 0
  Series.ActionInput.Civ2.CheckThreshold = 0 
  Series.ActionInput.TestCiv2 = 0

.. code-block:: matlab

  %%Parameters for Fix2
  Series.ActionInput.Fix2.CheckFmin2 = 1
  Series.ActionInput.Fix2.CheckF4 = 0
  Series.ActionInput.Fix2.CheckF3 = 1
  Series.ActionInput.Fix2.MinCorr = 0.2

.. code-block:: matlab

  %% Parameters for Patch2
  Series.ActionInput.Patch2.FieldSmooth = 2
  Series.ActionInput.Patch2.MaxDiff = 1.5
  Series.ActionInput.Patch2.SubDomainSize = 1000
  Series.ActionInput.Patch2.TestPatch2 = 0

.. code-block:: matlab

  #### ????
  Series.ActionInput.ListCompareMode = PIV
  % PIV or displacment or PIV volume
  Series.ActionInput.ConfigSource = default
  Series.ActionInput.Program = civ_series

