
#ifndef _MODEL_3D_CODE_H
#define _MODEL_3D_CODE_H 1

// Model data path: D:\_WIP_Data\Dev_Models\Pig\Base_Scale\test\initial_pig_oviduct.ovm
// Curve data path: D:\_WIP_Data\Dev_Models\Pig\Base_Scale\test\muestra_2128_complete_scaled_x100_resampled_rotated2.crv
// Transformations data path: D:\_WIP_Data\Dev_Models\Pig\Base_Scale\test\transformations.trx

#ifdef _WIN32 
#define INCLUDE_DATA_FILE "D:\\_WIP_Data\\Dev_Models\\Pig\\Base_Scale\\17th_attempt\\export_cuda\\"
#else
#define INCLUDE_DATA_FILE "./"
#endif

#define NO_OF_SLICES 1201
#define NO_OF_SLICES_MINUS_ONE 1200
#define NO_OF_SEGMENTS NO_OF_SLICES_MINUS_ONE
#define NO_OF_SEGMENTS_MINUS_ONE 1199
#define NO_OF_VECTICES 326000
#define NO_OF_TRIANGLES 648930

//mid point and radius of spheres surrounding each triangle
static float4 TRIANGLE_BOUNDING_SPHERE_H[NO_OF_TRIANGLES];

//Pre-Calculated collision detection information for each triangle - contains triangle vertices relative to plane
static float4 TRIANGLE_PLANE_H[NO_OF_TRIANGLES * 4];

//The midpoint of each segment (defined by two adjacent slices)
static float4 SEGMENT_MIDPOINT_VECTOR_H[NO_OF_SEGMENTS];

//The cumulative index of each triangle within a tri-strip relative to each slice
static int SLICE_TRISTRIP_INDEX_H[NO_OF_SLICES];

//The definition of a plane for each slice
static float4 SLICE_PLANE_H[NO_OF_SLICES];

#endif

