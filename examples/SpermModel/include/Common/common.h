/*Maths constants and vector functions*/

#ifndef _COMMON_H
#define _COMMON_H 1


//#define PI_OVER_180 ((4.0f*atanf(1.0f)) / 180)
#define PI_OVER_180 0.01745329f

//#define PI_UNDER_180 (180 / (4.0f*atanf(1.0f)))
#define PI_UNDER_180 57.29578f

#define TO_RADIANS(angle) ((angle) * PI_OVER_180)
#define TO_DEGREES(angle) ((angle) * PI_UNDER_180)

#define Plane float4

#define square(x) ((x) * (x))

#define EPSILON 0.00001f

inline __device__ float3 GetPerpendicular(const float3 &vec) {
	if (abs(vec.x) > abs(vec.y))
	{
		float len = sqrt(vec.x * vec.x + vec.z * vec.z);		
		return make_float3(vec.z / len, 0.0f, -vec.x / len);
	}
	else
	{
		float len = sqrt(vec.y * vec.y + vec.z * vec.z);		
		return make_float3(0.0f, vec.z / len, -vec.y / len);
	}
}

inline __host__ __device__ bool areEqual(const float3 &v1, const float3 &v2) {
	return ((abs(v1.x - v2.x) < EPSILON) && (abs(v1.y - v2.y) < EPSILON) && (abs(v1.z - v2.z) < EPSILON));
}

inline __host__ __device__ bool areOpposite(const float3 &v1, const float3 &v2) {
	return ((abs(v1.x + v2.x) < EPSILON) && (abs(v1.y + v2.y) < EPSILON) && (abs(v1.z + v2.z) < EPSILON));
}


inline __host__ __device__ bool isZero(const float3 &src) {
	return (src.x == 0 && src.y == 0 && src.z == 0);
}

inline __host__ __device__ bool isNan(const float3 &src) {
	return (isnan(src.x) || isnan(src.y) || isnan(src.z));
}


inline __host__ __device__ float distance(const float3 &a, const float3 &b) {
	return length(b - a);
}

inline __host__ __device__ float lenSq(const float2 &a) {
	float tmp = length(a);
	return tmp * tmp;
}

inline __device__ float GetRandomNumber(const float minNumber, const float maxNumber, RNG_rand48* rand48) {
	return minNumber + (rnd(rand48) * (maxNumber - minNumber));
}

inline __host__ __device__ void	swap(float &ioV1, float &ioV2)
{
	float tmp = ioV1;
	ioV1 = ioV2;
	ioV2 = tmp;
}

inline __host__ __device__ float3 Reflect(const float3 &I, const float3 &N) {
	float3 n = normalize(N);
	float3 i = normalize(I);
	//return  (2 * N * (dot(N, I))) - I;
	return i - (2 * n * (dot(n, i)));
}


inline __device__ float calculateDistanceFromPointToPlane(Plane plane, float3 inPoint) {
	return dot(inPoint, make_float3(plane)) + plane.w;
}


inline __device__ Plane makePlaneFromPointAndNormal(const float3 &pt, const float3 &n) {

	float d = -dot(pt, n);

	return make_float4(n.x, n.y, n.z, d);
}


inline __device__ Plane makePlaneFromPoints(const float3 &pt0, const float3 &pt1, const float3 &pt2) {
	float3 v0 = pt0 - pt1;
	float3 v1 = pt2 - pt1;

	float3 n = normalize(cross(v1, v0));

	return makePlaneFromPointAndNormal(pt0, n);
}



#endif