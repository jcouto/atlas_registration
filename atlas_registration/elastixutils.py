import numpy as np

# the parameter files are adapted from DeepTrace, thanks to Laura DeNardo (UCLA)
# for sharing the parameter files.

################################################################
######### Transform parameters passed to ELASTIX ###############
################################################################
  
elastixpar0 = '''//Affine Transformation - updated May 2012

// Description: affine, MI, ASGD

//ImageTypes
(FixedInternalImagePixelType "float")
(FixedImageDimension 3)
(MovingInternalImagePixelType "float")

//Components
(Registration "MultiResolutionRegistration")
(FixedImagePyramid "FixedSmoothingImagePyramid")
(MovingImagePyramid "MovingSmoothingImagePyramid")
(Interpolator "BSplineInterpolator")
(Metric "AdvancedMattesMutualInformation")
(Optimizer "AdaptiveStochasticGradientDescent")
(ResampleInterpolator "FinalBSplineInterpolator")
(Resampler "DefaultResampler")
(Transform "AffineTransform")

(ErodeMask "true" )

(NumberOfResolutions {number_of_resolutions})

(HowToCombineTransforms "Compose")
(AutomaticTransformInitialization "true")
(AutomaticScalesEstimation "true")

(WriteTransformParametersEachIteration "false")
(WriteResultImage "false")
(CompressResultImage "false")
(WriteResultImageAfterEachResolution "false") 
(ShowExactMetricValue "false")

//Maximum number of iterations in each resolution level:
(MaximumNumberOfIterations {maximum_number_of_interactions} ) 

//Number of grey level bins in each resolution level:
(NumberOfHistogramBins {number_of_histogram_bins} )
(FixedLimitRangeRatio 0.0)
(MovingLimitRangeRatio 0.0)
(FixedKernelBSplineOrder 3)
(MovingKernelBSplineOrder 3)

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

(ErodeMask "false" )

(NumberOfResolutions {number_of_resolutions_second})
(FinalGridSpacingInVoxels {final_grid_spacing} {final_grid_spacing} {final_grid_spacing})

(HowToCombineTransforms "Compose")

(WriteTransformParametersEachIteration "false")
(WriteResultImage "true")
(ResultImageFormat "tiff")
//unsigned char gives issues when values are very close to the max range (i.e. for 255)
//(ResultImagePixelType "unsigned char")
(ResultImagePixelType "short")
(CompressResultImage "false")
(WriteResultImageAfterEachResolution "false")
(ShowExactMetricValue "false")
(WriteDiffusionFiles "true")

// Option supported in elastix 4.1:
(UseFastAndLowMemoryVersion "true")

//Maximum number of iterations in each resolution level:
(MaximumNumberOfIterations 5000)

//Number of grey level bins in each resolution level:
(NumberOfHistogramBins 32 )
(FixedLimitRangeRatio 0.0)
(MovingLimitRangeRatio 0.0)
(FixedKernelBSplineOrder 3)
(MovingKernelBSplineOrder 3)

//Number of spatial samples used to compute the mutual information in each resolution level:
(ImageSampler "RandomCoordinate")
(FixedImageBSplineInterpolationOrder 1 )
(UseRandomSampleRegion "true")
(SampleRegionSize 150.0 150.0 150.0)
(NumberOfSpatialSamples 15000 )
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
'''

def elastix_register_brain(stack,
                           atlas = 'kim_mouse_10um',
                           brain_geometry = 'left',
                           par0 = elastixpar0,
                           par1 = elastixpar1,
                           number_of_resolutions = 6,
                           number_of_resolutions_second = 8,
                           final_grid_spacing = 25.0,
                           number_of_histogram_bins = 32,
                           maximum_number_of_interactions = 2500,
                           number_of_spatial_samples = 5000,
                           stack_gaussian_smoothing = None, # skip the smoothing
                           working_path = None,
                           elastix_binary_path = "elastix",
                           pbar = None):

    from tifffile import imwrite, imread

    if not stack_gaussian_smoothing is None:
        from scipy.ndimage import gaussian_filter
        stack = gaussian_filter(stack,stack_gaussian_smoothing)
    if type(atlas) is str:
        atlas_folder = list((Path.home()/'.brainglobe').glob(atlas+'*'))
        if not len(atlas_folder):
            raise(ValueError(f'Could not find {atlas} in the .brainglobe folder'))
        reference = imread(atlas_folder[0]/'reference.tiff')
    else:
        reference = atlas

    if brain_geometry == 'left':
        reference  = reference[:,:,:reference.shape[2]//2]
    elif brain_geometry == 'right':
        reference  = reference[:,:,reference.shape[2]//2:]

    if working_path is None:
        working_path = Path.home()/'.elastix_temporary'
    working_path.mkdir(exist_ok = True)
    if (working_path/'result.1.tiff').exists():
        import os
        os.unlink(working_path/'result.1.tiff')

    p0 = 'elastix_p0.txt'
    with open(p0,'w') as fd:
        fd.write(par0.format(number_of_resolutions = number_of_resolutions,
                             number_of_histogram_bins = number_of_histogram_bins,
                             maximum_number_of_interactions = maximum_number_of_interactions))
    p1 = 'elastix_p1.txt'
    with open(p1,'w') as fd:
        fd.write(par1.format(number_of_resolutions = number_of_resolutions,
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
        p0 = p0,
        p1 = p1)
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
    # The output filename will depend on the transforms..
    nstack = imread(working_path/'result.1.tiff')
    with open(transformix_parameters[1],'r') as fd:
        transform = fd.read()
    return nstack, transform


def elastix_apply_transform(stack,transform,
                            elastix_path = "transformix",
                            working_path = None,
                            pbar = None):
    '''

    '''
    
    # make that it works with registration template being an array!
    if elastix_path is None:
        # assume that it is in path
        elastix_path = "transformix"
    if working_path is None:
        working_path = Path.home()/'.elastix_temporary'
    working_path.mkdir(exist_ok = True)
    if (working_path/'result.tiff').exists():
        import os
        os.unlink(working_path/'result.tiff')
    
    transform_path = working_path/'transform_param.txt'
    with open(transform_path,'w') as fd:
        fd.write(transform)

    stack_path = working_path/'im.tif'
    imwrite(stack_path,stack)

    elastixcmd = r'{elastix_path} -in "{stack}" -out "{outpath}" -tp "{t1}"'.format(
        elastix_path = elastix_path,
        stack = stack_path,
        outpath = working_path,
        t1 = transform_path)
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
    return imread(working_path/'result.tiff')


def elastix_apply_transform(stack,transform,
                            elastix_path = "transformix",
                            working_path = None,
                            pbar = None):
    '''
    transform path is a folder with 2 files
    '''
    
    # make that it works with registration template being an array!
    if elastix_path is None:
        # assume that it is in path
        elastix_path = "transformix"
        
    elastixcmd = r'{elastix_path} -in "{stack}" -out "{outpath}" -tp "{t1}"'.format(
        elastix_path = elastix_path,
        stack = stack_path,
        outpath = outpath,
        t1 = transform_path)
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
    return imread(pjoin(outpath,'result.tiff'))


