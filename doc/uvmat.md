# Analysis of UVmat civ_series algorithm and parameters

See the
[UVmat documentation](http://servforge.legi.grenoble-inp.fr/projects/soft-uvmat/wiki/UvmatHelp#Civ).

## Description of main loop on couple of images in civ_series.m

### Step 0: Initialisation

of a structure "par_civ" containing the 2 images A and B + all parameters for the
computation

### Step 2: Civ1: Call the function civ

Initialisation of grid, correlation and search boxes, mask

Loop on every grid points:

2.1: crop of subimages A and B of images A and B

subA is centered in \[xA, yA\] and has a size \[Lx, Ly\]

subB is centered in \[xA + shitfx, xB + shifty\] and has a size \[Sx, Sy\] with Sx,Sy >
Lx, Ly

2.2: if checkmask == 1 -> application of the mask on subimages

2.3: if checkdeformation == 1 -> apply deformation + interpolation (with function
interp2)

2.4.1 convolution of subA and subB with function conv2 => "conv"

```matlab
conv = conv2(subA, subB, 'valid');
% be careful on the ordering of subB
```

2.4.2 search displacement \[dx, dy\] corresponding such that

```matlab
conv(dx, dy) == corrmax
% corrmax = max( max( conv));
```

2.5 subpixel determination of displacement: \[vec_x, vec_y\] = fct(conv, dx, dy)

it exists 3 different fct depending on parameters: SUBPIXGAUSS, SUBPIX2DGAUSS and
quadr_fit

2.6 if the position \[xA + vec_x, yA + vec_y\] is in the mask -> vec_x, vec_y is put to 0

2.7 the maximum correlation is normalised

```matlab
norm = sum( sum( subA.*subA )) .* sum( sum( subsubB.*subsubB ))
% subsubB is a crop of subB at the same size as subA centered on [xA + dx, yA + dy]
corrmax = corrmax / sqrt(norm)
```

### Step 3: Fix1: Call the function fix

detection of vectors that don't satisfy parameters given to fix

### Step 4: Patch1: Call the function filter_tps

find the thin plate spline coefficients for interpolation-smoothing (See UVmat
documentation)

### Step 5: Civ2: Call the function civ

same as Civ1 + option deformation

### Step 6: Fix2: Call the function fix

same as Fix1 + option CheckF4

### Step 7: Patch2: Call the function filter_tps

same as Patch1

### Step 8: Write results in a netcdf file

## Description of a typical .xml

### Input and Output files

```matlab
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
```

### Program to execute

```matlab
Series.Action.ActionPath = 'U:\project\watu\2015\15GRAVIT\UVMAT\series\'
%  address of program
Series.Action.ActionName = 'civ_series'
%  ???? same as  Series.ActionInput.Program ?????
Series.Action.RUN = 1
%  0 or 1
Series.Action.status = 0
% ?????
```

### Parameters for computation

```matlab
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
```

```matlab
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
```

```matlab
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

Series.ActionInput.Civ1.CorrSmooth = 1
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
% to open a figure via uvmat showing correlations for each windows
```

```matlab
%% Parameters for Fix1
Series.ActionInput.Fix1.CheckFmin2 = 1
% remove vectors with maximum correlation too close to the border of the searchbox (<2pix or less)
Series.ActionInput.Fix1.CheckF3 = 1
% ??????
Series.ActionInput.Fix1.MinCorr = 0.2
% remove vectors with correlation below MinCorr
Series.ActionInput.Fix1.MaxVel = 0.2
Series.ActionInput.Fix1.MinVel = 0.2
% remove vectors with modulus not in [MinVel, MaxVel]
```

```matlab
%% Parameters for Patch1
Series.ActionInput.Patch1.FieldSmooth = 10
Series.ActionInput.Patch1.MaxDiff = 1.5
Series.ActionInput.Patch1.SubDomainSize = 1000
% see function filter_tps.m
Series.ActionInput.Patch1.TestPatch1 = 0
% open a window via uvmat to test values of Fieldsmooth around 10 here
```

```matlab
%% Parameters for Civ2
Series.ActionInput.Civ2.CorrBoxSize = [21 21]
Series.ActionInput.Civ2.CorrSmooth = 1
Series.ActionInput.Civ2.SearchBoxSize = [27 27]
Series.ActionInput.Civ2.CheckDeformation = 0
% for subpixel interpolation and image deformation (linear transform)
% => use of DUDX DUDY etc... before crop of sub-images
% if == 1 then Series.ActionInput.Civ2.CorrSmooth is removed from xml
Series.ActionInput.Civ2.Dx = 10
Series.ActionInput.Civ2.Dy = 10
Series.ActionInput.Civ2.CheckGrid = 0
Series.ActionInput.Civ2.CheckMask = 0
Series.ActionInput.Civ2.CheckThreshold = 0
Series.ActionInput.TestCiv2 = 0
```

```matlab
%%Parameters for Fix2
Series.ActionInput.Fix2.CheckFmin2 = 1
Series.ActionInput.Fix2.CheckF4 = 0
% ???????
Series.ActionInput.Fix2.CheckF3 = 1
Series.ActionInput.Fix2.MinCorr = 0.2
Series.ActionInput.Fix2.MaxVel = 0.2
Series.ActionInput.Fix2.MinVel = 0.2
```

```matlab
%% Parameters for Patch2
Series.ActionInput.Patch2.FieldSmooth = 2
Series.ActionInput.Patch2.MaxDiff = 1.5
Series.ActionInput.Patch2.SubDomainSize = 1000
Series.ActionInput.Patch2.TestPatch2 = 0
```

```matlab
#### ????
Series.ActionInput.ListCompareMode = PIV
% PIV or displacment or PIV volume
Series.ActionInput.ConfigSource = default
Series.ActionInput.Program = civ_series
```
