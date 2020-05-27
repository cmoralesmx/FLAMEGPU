/*Functions for storing and processing Oviduct Model and collision detection structures from Texture memory for efficiency*/

#ifndef _MODEL_TEXTURE_ACCESS_H
#define _MODEL_TEXTURE_ACCESS_H 1

#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>
#include "Common/helper_cuda.h"
#include "Model3DCodeCUDA.h"

__device__ float4* TRIANGLE_BOUNDING_SPHERE_D;
__device__ TrianglePlane* TRIANGLE_PLANE_D;
__device__ float4* SEGMENT_MIDPOINT_VECTOR_D;
__device__ int* SLICE_TRISTRIP_INDEX_D;
__device__ float4* SLICE_PLANE_D;

texture<float4, 1, cudaReadModeElementType> TRIANGLE_BOUNDING_SPHERE_TEX;
texture<float4, 1, cudaReadModeElementType> TRIANGLE_PLANE_TEX;
texture<float4, 1, cudaReadModeElementType> SEGMENT_MIDPOINT_VECTOR_TEX;
texture<int, 1, cudaReadModeElementType> SLICE_TRISTRIP_INDEX_TEX;
texture<float4, 1, cudaReadModeElementType> SLICE_PLANE_TEX;


__device__ float4 getTriangleBoundingSphere(int triangleIndex) {
	//int offset = triangleIndex * 4;
	return tex1Dfetch(TRIANGLE_BOUNDING_SPHERE_TEX, triangleIndex);
	//return make_float4(tex1Dfetch(TRIANGLE_BOUNDING_SPHERE_TEX, offset), tex1Dfetch(TRIANGLE_BOUNDING_SPHERE_TEX, offset + 1), tex1Dfetch(TRIANGLE_BOUNDING_SPHERE_TEX, offset + 2), tex1Dfetch(TRIANGLE_BOUNDING_SPHERE_TEX, offset + 3));
}

__device__ TrianglePlane getTrianglePlane(int triangleIndex) {
	int offset = triangleIndex * 4;
	TrianglePlane tp;
	

	tp.plane = tex1Dfetch(TRIANGLE_PLANE_TEX, offset);

	float4 UA = tex1Dfetch(TRIANGLE_PLANE_TEX, offset + 1);
	float4 VP0 = tex1Dfetch(TRIANGLE_PLANE_TEX, offset + 2);
	float4 P1P2 = tex1Dfetch(TRIANGLE_PLANE_TEX, offset + 3);

	tp.U = make_float3(UA.x, UA.y, UA.z);
	tp.V = make_float3(UA.w, VP0.x, VP0.y);
	tp.p0 = make_float2(VP0.z, VP0.w);
	tp.p1 = make_float2(P1P2.x, P1P2.y);
	tp.p2 = make_float2(P1P2.z, P1P2.w);
	return tp;
}

__device__ float3 getSegmentMidpointVector(int segmentIndex) {
	//int offset = segmentIndex * 3;
	float4 vec = (tex1Dfetch(SEGMENT_MIDPOINT_VECTOR_TEX, segmentIndex));
	return make_float3(vec.x, vec.y, vec.z);
}


__device__ int getSliceTristripIndex(int sliceIndex) {
	return tex1Dfetch(SLICE_TRISTRIP_INDEX_TEX, sliceIndex);
}


__device__ float4 getSlicePlane(int sliceIndex) {
	//int offset = sliceIndex * 4;
	return (tex1Dfetch(SLICE_PLANE_TEX, sliceIndex));
}


void loadData(char* filename) {
	FILE* fp;
	fp = fopen(filename, "rb");

	fread(TRIANGLE_BOUNDING_SPHERE_H, sizeof(float4), NO_OF_TRIANGLES, fp);
	fread(TRIANGLE_PLANE_H, sizeof(float4), NO_OF_TRIANGLES * 4, fp);
	fread(SEGMENT_MIDPOINT_VECTOR_H, sizeof(float4), NO_OF_SEGMENTS, fp);
	fread(SLICE_TRISTRIP_INDEX_H, sizeof(int), NO_OF_SLICES, fp);
	fread(SLICE_PLANE_H, sizeof(float4), NO_OF_SLICES, fp);

	fclose(fp);
}

template<class T , int dim, enum cudaTextureReadMode readMode>
void TransferAndBindToTexture(void* devicePtr, const void* hostPtr, const struct texture< T, dim, readMode > &texPtr, int size) {
	checkCudaErrors(cudaMalloc((void**)&devicePtr, size));
	checkCudaErrors(cudaMemcpy(devicePtr, hostPtr, size, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaBindTexture(0, texPtr, devicePtr, size));

}

void initialiseGPUDataFromFile(char* filename) {

	loadData(filename);

	TransferAndBindToTexture((void*)&TRIANGLE_BOUNDING_SPHERE_D, TRIANGLE_BOUNDING_SPHERE_H, TRIANGLE_BOUNDING_SPHERE_TEX, sizeof(float4) * NO_OF_TRIANGLES);
	TransferAndBindToTexture((void*)&TRIANGLE_PLANE_D, TRIANGLE_PLANE_H, TRIANGLE_PLANE_TEX, sizeof(float4) * NO_OF_TRIANGLES * 4);
	TransferAndBindToTexture((void*)&SEGMENT_MIDPOINT_VECTOR_D, SEGMENT_MIDPOINT_VECTOR_H, SEGMENT_MIDPOINT_VECTOR_TEX, sizeof(float4) * NO_OF_SEGMENTS );
	TransferAndBindToTexture((void*)&SLICE_TRISTRIP_INDEX_D, SLICE_TRISTRIP_INDEX_H, SLICE_TRISTRIP_INDEX_TEX, sizeof(int) * NO_OF_SLICES);
	TransferAndBindToTexture((void*)&SLICE_PLANE_D, SLICE_PLANE_H, SLICE_PLANE_TEX, sizeof(float4) * NO_OF_SLICES);


	printf("Finished Loading Environment Data\n");
}

/*
Computes the full path of the .BIN file to use as the spatial environment for the simulation
the constant INCLUDE_DATA_FILE is hard coded in the Model3DCodeCUDA.h file
*/
void initialiseGPUData() {	
	char data[500];

	sprintf(data, "%senvironment_definition.bin", INCLUDE_DATA_FILE);

	initialiseGPUDataFromFile(data);
}


#endif