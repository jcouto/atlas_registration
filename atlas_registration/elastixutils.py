import numpy as np
from pathlib import Path
import os
import subprocess as sub
from .atlasutils import get_brainglobe_atlas

# the parameter files are adapted from DeepTrace, thanks to Laura DeNardo (UCLA)
# for sharing the parameter files.

################################################################
######### Transform parameters passed to ELASTIX ###############
################################################################
  
elastixpar0 = '''//Affine Transformation - updated November 2025

// Description: affine, MI, ASGD

// the casting can't be done safely
(FixedInternalImagePixelType "float") 
(FixedImageDimension 3)
(MovingInternalImagePixelType "float")
(UseDirectionCosines "true")
(UseFastAndLowMemoryVersion "true")

// Registration
(Registration "MultiResolutionRegistration")
// Pyramid
(FixedImagePyramid "FixedSmoothingImagePyramid")
(MovingImagePyramid "MovingSmoothingImagePyramid")
(NumberOfResolutions {number_of_resolutions})
(ImagePyramidSchedule {image_pyramid_schedule}) 
(MaximumNumberOfIterations {maximum_number_of_interactions} ) 

(NumberOfHistogramBins {number_of_histogram_bins})
(NumberOfMovingHistogramBins {number_of_histogram_bins})
(NumberOfFixedHistogramBins {number_of_histogram_bins})
(FixedLimitRangeRatio 0.0)
(MovingLimitRangeRatio 0.0)
(FixedKernelBSplineOrder 3)
(MovingKernelBSplineOrder 3)

(ErodeMask "false" )

(Metric "AdvancedMattesMutualInformation")
(UseMultiThreadingForMetrics "true")

(Optimizer "AdaptiveStochasticGradientDescent")
(Interpolator "BSplineInterpolator")
(ResampleInterpolator "FinalBSplineInterpolator")
(HowToCombineTransforms "Compose")
(Resampler "DefaultResampler")
(Transform "AffineTransform")

(AutomaticTransformInitialization "true")
(AutomaticTransformInitializationMethod "GeometricalCenter")
(AutomaticScalesEstimation "true")

(WriteTransformParametersEachIteration "false")
(WriteResultImage "false")
(CompressResultImage "false")
(WriteResultImageAfterEachResolution "false") 
(ShowExactMetricValue "false")

//Number of spatial samples used to compute the mutual information in each resolution level:
(ImageSampler "RandomCoordinate")
(FixedImageBSplineInterpolationOrder 3)
(UseRandomSampleRegion "false")
(NumberOfSpatialSamples {number_of_spatial_samples} )
(NewSamplesEveryIteration "true")
(CheckNumberOfSamples "true")

(MaximumNumberOfSamplingAttempts 10)
//Order of B-Spline interpolation used in each resolution level:
(BSplineInterpolationOrder 3)

//Order of B-Spline interpolation used for applying the final deformation:
(FinalBSplineInterpolationOrder 3)

//Default pixel value for pixels that come from outside the picture:
(DefaultPixelValue 0)

//SP: Param_A in each resolution level. a_k = a/(A+k+1)^alpha
(SP_A 20.0 )

'''

elastixpar1 = '''//Bspline Transformation - updated May 2012

//ImageTypes
(FixedInternalImagePixelType "float")
(FixedImageDimension 3)
(MovingInternalImagePixelType "float")
(MovingImageDimension 3)
(UseDirectionCosines "true")

//Components
(Registration "MultiResolutionRegistration")
(FixedImagePyramid "FixedSmoothingImagePyramid")
(MovingImagePyramid "MovingSmoothingImagePyramid")
(Interpolator "BSplineInterpolator")
(Metric "AdvancedMattesMutualInformation")
(Optimizer "StandardGradientDescent")
(ResampleInterpolator "FinalBSplineInterpolator")
(Resampler "DefaultResampler")

(Transform "BSplineTransform")
(NumberOfResolutions {number_of_resolutions_second})
(ImagePyramidSchedule {image_pyramid_schedule}) 

(FinalGridSpacingInVoxels {final_grid_spacing} {final_grid_spacing} {final_grid_spacing})

(ErodeMask "false") 
(UseJacobianPreconditioning "false")
(FiniteDifferenceDerivative "false")

(HowToCombineTransforms "Compose")

(WriteTransformParametersEachIteration "false")
(WriteResultImageAfterEachResolution "false")
(ShowExactMetricValue "false")
(WriteDiffusionFiles "false")

// Option supported in elastix 4.1:
(UseFastAndLowMemoryVersion "true")

//Maximum number of iterations in each resolution level:
(MaximumNumberOfIterations 5000)

//Number of grey level bins in each resolution level:
(NumberOfHistogramBins 32)
(FixedLimitRangeRatio 0.0)
(MovingLimitRangeRatio 0.0)
(FixedKernelBSplineOrder 3)
(MovingKernelBSplineOrder 3)

//Number of spatial samples used to compute the mutual information in each resolution level:
(NumberOfSpatialSamples 15000 )
(ImageSampler "RandomCoordinate")
(FixedImageBSplineInterpolationOrder 1)
(UseRandomSampleRegion "true")
(SampleRegionSize 150.0 150.0 150.0)
(NewSamplesEveryIteration "true")
(CheckNumberOfSamples "true")
(MaximumNumberOfSamplingAttempts 10)

//Order of B-Spline interpolation used in each resolution level:
(BSplineInterpolationOrder 3)
//Order of B-Spline interpolation used for applying the final deformation:
(FinalBSplineInterpolationOrder 3)

//Default pixel value for pixels that come from outside the picture:
(DefaultPixelValue 0)

//SP: Param_a in each resolution level. a_k = a/(A+k+1)^alpha
(SP_a 10000.0 )

//SP: Param_A in each resolution level. a_k = a/(A+k+1)^alpha
(SP_A 100.0 )

//SP: Param_alpha in each resolution level. a_k = a/(A+k+1)^alpha
(SP_alpha 0.6 )

(WriteResultImage "true")
(ResultImageFormat "tiff")
(ResultImagePixelType "float") // short overflows in some cases
(CompressResultImage "false")

'''
def random_string(length = 10):
    return ''.join(np.random.choice([a for a in 'abcdefghijz0123456789'],length))



def elastix_register_brain(stack,
                           atlas = 'allen_mouse_10um',
                           brain_geometry = 'left',
                           par0 = elastixpar0,
                           par1 = elastixpar1,
                           number_of_resolutions = 3,
                           number_of_resolutions_second = 5,
                           final_grid_spacing = 25.0, # 25
                           number_of_histogram_bins = 32,
                           maximum_number_of_interactions = 5000,
                           number_of_spatial_samples = 4000,
                           stack_gaussian_smoothing = None, # skip the smoothing
                           working_path = None,
                           elastix_binary_path = "elastix",
                           pbar = None,
                           debug = False):

    from tifffile import imwrite, imread

    if not stack_gaussian_smoothing is None:
        from scipy.ndimage import gaussian_filter
        stack = gaussian_filter(stack,stack_gaussian_smoothing)
    if type(atlas) is str:
        reference = get_brainglobe_atlas(atlas)
    else:
        reference = atlas

    if brain_geometry == 'left':
        reference  = reference[:,:,:reference.shape[2]//2]
    elif brain_geometry == 'right':
        reference  = reference[:,:,reference.shape[2]//2:]

    if working_path is None:
        working_path = Path.home()/'.elastix_temporary'/random_string()
        
    working_path.mkdir(exist_ok = True, parents=True)
    if (working_path/'result.1.tiff').exists():
        import os
        os.unlink(working_path/'result.1.tiff')

    pyramids = [] # get the image_pyramid_schedule from the number of resolutions
    for i in range(number_of_resolutions-1,-1,-1):
        pyramids.extend(3*[2**(i)])

    p0 = 'elastix_p0.txt'
    with open(working_path/p0,'w') as fd:
        fd.write(par0.format(number_of_resolutions = number_of_resolutions,
                             image_pyramid_schedule = ' '.join([str(p) for p in pyramids]),
                             number_of_histogram_bins = number_of_histogram_bins,
                             number_of_spatial_samples = number_of_spatial_samples,
                             maximum_number_of_interactions = maximum_number_of_interactions))
    
    pyramids = [] # get the image_pyramid_schedule from the number of resolutions
    for i in range(number_of_resolutions_second-1,-1,-1):
        pyramids.extend(3*[2**(i)])
    p1 = 'elastix_p1.txt'
    with open(working_path/p1,'w') as fd:
        fd.write(par1.format(number_of_resolutions_second = number_of_resolutions_second,
                             image_pyramid_schedule = ' '.join([str(p) for p in pyramids]),
                             final_grid_spacing=final_grid_spacing))
    
    stack_path = working_path/'im.tif'
    imwrite(stack_path,stack)

    registration_path = working_path/'template.tif'
    imwrite(registration_path,reference)

    elastixcmd = r'{elastix_binary_path} -f "{registration_path}" -m "{stack_path}" -out "{working_path}" -p "{p0}" -p "{p1}"'.format(
        elastix_binary_path = elastix_binary_path,
        registration_path = registration_path,
        stack_path = stack_path,
        working_path = working_path,
        p0 = working_path/p0,
        p1 = working_path/p1)
    if debug:
        print(elastixcmd)
    proc = sub.Popen(elastixcmd,
                 shell=True,
                 stdout=sub.PIPE)
                 # preexec_fn=os.setsid) # does not work on windows?
    if not pbar is None:
        pbar.set_description('Running elastix')
        pbar.reset()
    while True:
        out = proc.stdout.readline()
        ret = proc.poll()
        if not pbar is None:
            pbar.update(1)
        if ret is not None:
            break

    transformix_parameters_paths = (working_path/'TransformParameters.0.txt',
                                    working_path/'TransformParameters.1.txt') 
    transforms = []
    for t in transformix_parameters_paths:
        with open(t,'r') as fd:
            transforms.append(fd.read().replace(
                str(working_path/'TransformParameters.0.txt'),
                'TransformParameters.0.txt'))
            
    res = imread(working_path/'result.1.tiff')
    res = np.clip(res,np.iinfo(stack.dtype).min,np.iinfo(stack.dtype).max).astype(stack.dtype)

    if debug:
        print(f"Kept folder {working_path} with the parameters and transforms.")
    else:
        from shutil import rmtree
        rmtree(working_path)

    return res, transforms


def elastix_apply_transform(stack,transforms,
                            elastix_path = "transformix",
                            working_path = None,
                            pbar = None,
                            debug = False):
    '''

    '''
    from tifffile import imwrite, imread
    # make that it works with registration template being an array!
    if elastix_path is None:
        # assume that it is in path
        elastix_path = "transformix"
    if working_path is None:
        working_path = Path.home()/'.elastix_temporary'/random_string()

    working_path.mkdir(exist_ok = True)
    if (working_path/'result.tiff').exists():
        import os
        os.unlink(working_path/'result.tiff')
    
    for i,transf in enumerate(transforms):
        transform_path = working_path/f'TransformParameters.{i}.txt'
        with open(transform_path,'w') as fd:
            if 'TransformParameters.0.txt' in transf:
                transf = transf.replace('TransformParameters.0.txt',
                                        str(working_path/'TransformParameters.0.txt'))
            fd.write(transf)

    stack_path = working_path/'im.tif'
    imwrite(stack_path,stack)

    elastixcmd = r'{elastix_path} -in "{stack_path}" -out "{outpath}" -tp "{t1}"'.format(
        elastix_path = elastix_path,
        stack_path = stack_path,
        outpath = working_path,
        t1 = transform_path)
    if debug:
        print(elastixcmd)
    proc = sub.Popen(elastixcmd,
                 shell=True,
                 stdout=sub.PIPE)
                 #preexec_fn=os.setsid)  # does not work on windows
    if not pbar is None:
        pbar.set_description('Running transformix')
        pbar.reset()
    while True:
        out = proc.stdout.readline()
        ret = proc.poll()
        if not pbar is None:
            pbar.update(1)
        if ret is not None:
            break
    res = imread(working_path/'result.tiff')
    res = np.clip(res,np.iinfo(stack.dtype).min,np.iinfo(stack.dtype).max).astype(stack.dtype)
    if debug:
        print(f"Kept folder {working_path} with the parameters and transforms.")
    else:
        from shutil import rmtree
        rmtree(working_path)
    return res
