/*Implementation of Sphere Sweeping algorithm in CUDA - original source reference below*/

#ifndef _SPHERE_COLLISION_CUDA_H
#define _SPHERE_COLLISION_CUDA_H 1

/*
Sphere Sweeping Collision Detection Algorithm based on 
J. Rouwe. (2003, May) Collision detection with swept spheres and ellipsoids. 
[Online]. Available: http://www.jrouwe.nl/sweptellipsoid/SweptEllipsoid.pdf
*/

#include "float.h"
#include "Common/common.h"
#include "OviductCollisionDetectionV1/collision_detection_header.h"
#include "Model3DCodeCUDA.h"


#ifdef _MSC_VER
#pragma region Intersection Definitions
#endif

#ifdef _MSC_VER
#pragma endregion
#endif

__device__ IntersectionResult testIntersectionPlaneSphere(const Plane &plane, const float3 &oldPosition, const float3 &direction, float distance_to_move, const float radius);
__device__ IntersectionResult testIntersectionTriSphere(const TrianglePlane &tp, const float3 &sphere_centre, const float sphere_radius, const float3 &sphereVel);

#ifdef _MSC_VER
#pragma region Plane Functions
#endif


/*__device__ void	GetBasisVectors(const Plane &plane, float3 &outU, float3 &outV) { 
	float3 planeNormal = make_float3(plane);
	outU = GetPerpendicular(planeNormal);
	outV = cross(planeNormal, outU); 
} */

// Convert a point from world space to plane space (3D -> 2D)
__device__ float2 ConvertWorldToPlane(const float3 &inU, const float3 &inV, const float3 &inPoint) {
	return make_float2(dot(inU,inPoint), dot(inV,inPoint));
}

// Convert a point from world space to plane space (3D -> 2D)
/*__device__ float2	ConvertWorldToPlane(const Plane &plane, const float3 &inPoint) {
	float3 u, v;
	GetBasisVectors(plane, u, v);
	return sConvertWorldToPlane(u, v, inPoint);
}*/

__device__ float3 ConvertPlaneToWorld(const Plane &plane, const float3 &inU, const float3 &inV, const float2 &inPoint) {
	return (inU * inPoint.x + inV * inPoint.y - make_float3(plane) * plane.w);
}

#ifdef _MSC_VER
#pragma endregion
#endif


#ifdef _MSC_VER
#pragma region inner calculation functions
#endif


/*
 * Modified from PolygonContains for efficiency on GPU
*/
__device__ bool TriangleContains(const TrianglePlane &tp,  const float2 &inPoint) {
	//float2 v1;
	//float2 v2;
	float2 v1_v2;
	float2 v1_point;

	//v1 = tp.p0;
	//v2 = tp.p2;
	v1_v2 = tp.p2 - tp.p0;
	v1_point = inPoint - tp.p0;
	if (v1_v2.x * v1_point.y - v1_point.x * v1_v2.y > 0.0f) {
		return false;
	}

	//v1 = tp.p1;
	//v2 = tp.p0;
	v1_v2 = tp.p0 - tp.p1;
	v1_point = inPoint - tp.p1;
	if (v1_v2.x * v1_point.y - v1_point.x * v1_v2.y > 0.0f) {
		return false;
	}

	//v1 = tp.p2;
	//v2 = tp.p1;
	v1_v2 = tp.p1 - tp.p2;
	v1_point = inPoint - tp.p2;
	if (v1_v2.x * v1_point.y - v1_point.x * v1_v2.y > 0.0f) {
		return false;
	}

	return true;
}



/*
 * Modified from PolygonCircleIntersect for efficiency on GPU
*/
__device__ bool TriangleCircleIntersect(const TrianglePlane &tp, const float2 &inCenter, float inRadiusSq, float2 &outPoint) {
	// Check if the center is inside the polygon
	if (TriangleContains(tp, inCenter))
	{
		outPoint = inCenter;
		return true;
	}

	// Loop through edges
	bool collision = false;

	float2 v1;
	float2 v2;
	float2 v1_v2;
	float2 v1_center;
	float fraction;
	float dist_sq;
	float v1_v2_len_sq;
	float2 point;

	{
		v1 = tp.p0;
		v2 = tp.p2;
		v1_v2 = v2 - v1;
		v1_center = inCenter - v1;
		fraction = dot(v1_center, v1_v2);
		if (fraction < 0.0f) {
			// Closest point is v1
			dist_sq = lenSq(v1_center);
			if (dist_sq <= inRadiusSq) {
				collision = true;
				outPoint = v1;
				inRadiusSq = dist_sq;
			}
		}
		else 
		{
			v1_v2_len_sq = lenSq(v1_v2);
			if (fraction <= v1_v2_len_sq)
			{
				// Closest point is on line segment
				point = v1 + v1_v2 * (fraction / v1_v2_len_sq);
				dist_sq = lenSq(point - inCenter);
				if (dist_sq <= inRadiusSq) {
					collision = true;
					outPoint = point;
					inRadiusSq = dist_sq;
				}
			}
		}
	}
	{
		v1 = tp.p1;
		v2 = tp.p0;
		v1_v2 = v2 - v1;
		v1_center = inCenter - v1;
		fraction = dot(v1_center, v1_v2);
		if (fraction < 0.0f) {
			// Closest point is v1
			dist_sq = lenSq(v1_center);
			if (dist_sq <= inRadiusSq) {
				collision = true;
				outPoint = v1;
				inRadiusSq = dist_sq;
			}
		}
		else 
		{
			v1_v2_len_sq = lenSq(v1_v2);
			if (fraction <= v1_v2_len_sq)
			{
				// Closest point is on line segment
				point = v1 + v1_v2 * (fraction / v1_v2_len_sq);
				dist_sq = lenSq(point - inCenter);
				if (dist_sq <= inRadiusSq) {
					collision = true;
					outPoint = point;
					inRadiusSq = dist_sq;
				}
			}
		}
	}
	{
		v1 = tp.p2;
		v2 = tp.p1;
		v1_v2 = v2 - v1;
		v1_center = inCenter - v1;
		fraction = dot(v1_center, v1_v2);
		if (fraction < 0.0f) {
			// Closest point is v1
			dist_sq = lenSq(v1_center);
			if (dist_sq <= inRadiusSq) {
				collision = true;
				outPoint = v1;
				inRadiusSq = dist_sq;
			}
		}
		else 
		{
			v1_v2_len_sq = lenSq(v1_v2);
			if (fraction <= v1_v2_len_sq)
			{
				// Closest point is on line segment
				point = v1 + v1_v2 * (fraction / v1_v2_len_sq);
				dist_sq = lenSq(point - inCenter);
				if (dist_sq <= inRadiusSq) {
					collision = true;
					outPoint = point;
					inRadiusSq = dist_sq;
				}
			}
		}
	}

	return collision;
}

__device__ bool FindLowestRootInInterval(const float inA, const float inB, const float inC, const float inUpperBound, float &outX)
{
	// Check if a solution exists
	float determinant = inB * inB - 4.0f * inA * inC;
	if (determinant < 0.0f)
		return false;

	// The standard way of doing this is by computing: x = (-b +/- Sqrt(b^2 - 4 a c)) / 2 a 
	// is not numerically stable when a is close to zero. 
	// Solve the equation according to "Numerical Recipies in C" paragraph 5.6
	float q = -0.5f * (inB + (inB < 0.0f? -1.0f : 1.0f) * sqrt(determinant));

	// Both of these can return +INF, -INF or NAN that's why we test both solutions to be in the specified range below
	float x1 = q / inA;
	float x2 = inC / q;

	// Order the results
	if (x2 < x1)
		swap(x1, x2);

	// Check if x1 is a solution
	if (x1 >= 0.0f && x1 <= inUpperBound)
	{
		outX = x1;
		return true;
	}

	// Check if x2 is a solution
	if (x2 >= 0.0f && x2 <= inUpperBound)
	{
		outX = x2;
		return true;
	}

	return false;
}

__device__ bool SweptCircleTriangleEdgeVertexIntersect(const TrianglePlane &tp, const float2 &inBegin, const float2 &inDelta, const float inA, const float inB, const float inC, float2 &outPoint, float &outFraction) {
	// Loop through edges
	float upper_bound = 1.0f;
	bool collision = false;

	float a1 = inA - lenSq(inDelta);
	float t;
	float2 v1;
	float2 v2;
	float2 bv1;
	float b1,c1,a2;

	float2 v1v2;
	float v1v2_dot_delta;
	float v1v2_dot_bv1;
	float v1v2_len_sq;
	float f;

	{
		v1 = tp.p0;
		v2 = tp.p2;

		bv1 = v1 - inBegin;
		b1 = inB + 2.0f * dot(inDelta, bv1);
		c1 = inC - lenSq(bv1);
		if (FindLowestRootInInterval(a1, b1, c1, upper_bound, t)) {
			// We have a collision
			collision = true;
			upper_bound = t;
			outPoint = v1;
		}

		// Check if circle hits the edge
		v1v2 = v2 - v1;
		v1v2_dot_delta = dot(v1v2, inDelta);
		v1v2_dot_bv1 = dot(v1v2, bv1);
		v1v2_len_sq = lenSq(v1v2);
		a2 = v1v2_len_sq * a1 + v1v2_dot_delta * v1v2_dot_delta;
		b1 = v1v2_len_sq * b1 - 2.0f * v1v2_dot_bv1 * v1v2_dot_delta;
		c1 = v1v2_len_sq * c1 + v1v2_dot_bv1 * v1v2_dot_bv1;
		if (FindLowestRootInInterval(a2, b1, c1, upper_bound, t))
		{
			// Check if the intersection point is on the edge
			f = t * v1v2_dot_delta - v1v2_dot_bv1;
			if (f >= 0.0f && f <= v1v2_len_sq) {
				// We have a collision
				collision = true;
				upper_bound = t;
				outPoint = v1 + v1v2 * (f / v1v2_len_sq);
			}
		}

	}
	{
		v1 = tp.p1;
		v2 = tp.p0;

		bv1 = v1 - inBegin;
		b1 = inB + 2.0f * dot(inDelta, bv1);
		c1 = inC - lenSq(bv1);
		if (FindLowestRootInInterval(a1, b1, c1, upper_bound, t)) {
			// We have a collision
			collision = true;
			upper_bound = t;
			outPoint = v1;
		}

		// Check if circle hits the edge
		v1v2 = v2 - v1;
		v1v2_dot_delta = dot(v1v2, inDelta);
		v1v2_dot_bv1 = dot(v1v2, bv1);
		v1v2_len_sq = lenSq(v1v2);
		a2 = v1v2_len_sq * a1 + v1v2_dot_delta * v1v2_dot_delta;
		b1 = v1v2_len_sq * b1 - 2.0f * v1v2_dot_bv1 * v1v2_dot_delta;
		c1 = v1v2_len_sq * c1 + v1v2_dot_bv1 * v1v2_dot_bv1;
		if (FindLowestRootInInterval(a2, b1, c1, upper_bound, t))
		{
			// Check if the intersection point is on the edge
			f = t * v1v2_dot_delta - v1v2_dot_bv1;
			if (f >= 0.0f && f <= v1v2_len_sq) {
				// We have a collision
				collision = true;
				upper_bound = t;
				outPoint = v1 + v1v2 * (f / v1v2_len_sq);
			}
		}

	}
	{	
		v1 = tp.p2;
		v2 = tp.p1;

		bv1 = v1 - inBegin;
		b1 = inB + 2.0f * dot(inDelta, bv1);
		c1 = inC - lenSq(bv1);
		if (FindLowestRootInInterval(a1, b1, c1, upper_bound, t)) {
			// We have a collision
			collision = true;
			upper_bound = t;
			outPoint = v1;
		}

		// Check if circle hits the edge
		v1v2 = v2 - v1;
		v1v2_dot_delta = dot(v1v2, inDelta);
		v1v2_dot_bv1 = dot(v1v2, bv1);
		v1v2_len_sq = lenSq(v1v2);
		a2 = v1v2_len_sq * a1 + v1v2_dot_delta * v1v2_dot_delta;
		b1 = v1v2_len_sq * b1 - 2.0f * v1v2_dot_bv1 * v1v2_dot_delta;
		c1 = v1v2_len_sq * c1 + v1v2_dot_bv1 * v1v2_dot_bv1;
		if (FindLowestRootInInterval(a2, b1, c1, upper_bound, t))
		{
			// Check if the intersection point is on the edge
			f = t * v1v2_dot_delta - v1v2_dot_bv1;
			if (f >= 0.0f && f <= v1v2_len_sq) {
				// We have a collision
				collision = true;
				upper_bound = t;
				outPoint = v1 + v1v2 * (f / v1v2_len_sq);
			}
		}

	}



	// Check if we had a collision
	if (!collision)
		return false;
	outFraction = upper_bound;
	return true;
}

#ifdef _MSC_VER
#pragma endregion
#endif


__device__ bool PlaneSweptSphereIntersect(const Plane &inPlane, const float3 &inBegin, const float3 &inDelta, const float inRadius, float &outT1, float &outT2)
{
	// If the center of the sphere moves like: center = inBegin + t * inDelta for t e [0, 1]
	// then the sphere intersects the plane if: -R <= distance plane to center <= R

	//float n_dot_d = inPlane.mNormal.Dot(inDelta);
	float n_dot_d = dot(make_float3(inPlane), inDelta);

	//float dist_to_b = inPlane.GetSignedDistance(inBegin);
	float dist_to_b = calculateDistanceFromPointToPlane(inPlane, inBegin);
	if (n_dot_d == 0.0f)
	{
		// The sphere is moving nearly parallel to the plane, check if the distance
		// is smaller than the radius
		if (abs(dist_to_b) > inRadius)
			return false;

		// Intersection on the entire range
		outT1 = 0.0f;
		outT2 = 1.0f;
	}
	else
	{
		// Determine interval of intersection
		outT1 = (inRadius - dist_to_b) / n_dot_d;
		outT2 = (-inRadius - dist_to_b) / n_dot_d;

		// Order the results
		if (outT1 > outT2) 
			swap(outT1, outT2);

		// Early out if no hit possible
		if (outT1 > 1.0f || outT2 < 0.0f)
			return false;

		// Clamp it to the range [0, 1], the range of the swept sphere
		if (outT1 < 0.0f) outT1 = 0.0f;
		if (outT2 > 1.0f) outT2 = 1.0f;
	}

	return true;
}

__device__ IntersectionResult testIntersectionPlaneSphere(const Plane &plane, const float3 &oldPosition, const float3 &direction, const float distance_to_move, const float radius){
	IntersectionResult rslt;
	//rslt.
	float3 delta = direction * distance_to_move;
	float outT1, outT2;
	rslt.intersectionOccurred = PlaneSweptSphereIntersect(plane, oldPosition, delta, radius,  outT1, outT2);
	rslt.distanceToMove = outT1 * distance_to_move;
	return rslt;
}

__device__ bool PolygonSweptSphereIntersect(const TrianglePlane &tp, const float3 &inBegin, const float3 &inDelta, const float inRadius, float3 &outPoint, float &outFraction) {
	// Determine the range over which the sphere intersects the plane
	float t1, t2;
	float4 inPlane = tp.plane;
	if (!PlaneSweptSphereIntersect(inPlane, inBegin, inDelta, inRadius, t1, t2))
		return false;

	float3 planeNormal = make_float3(inPlane);
	// The radius of the circle is defined as: radius^2 = (sphere radius)^2 - (distance plane to center)^2
	// this can be written as: radius^2 = a * t^2 + b * t + c
	float n_dot_d = dot(planeNormal, inDelta);
	float dist_to_b =  calculateDistanceFromPointToPlane(inPlane, inBegin);
	float a = -n_dot_d * n_dot_d;
	float b = -2.0f * n_dot_d * dist_to_b;
	float c = inRadius * inRadius - dist_to_b * dist_to_b;

	// Get basis
	float3 u, v;
	//GetBasisVectors(inPlane, u, v);
	u = tp.U;
	v = tp.V;

	// Get begin and delta in plane space
	float2 begin = ConvertWorldToPlane(u, v, inBegin);
	float2 delta = ConvertWorldToPlane(u, v, inDelta);

	// Test if sphere intersects at t1
	float2 p;
	if (TriangleCircleIntersect(tp, begin + delta * t1, a * t1 * t1 + b * t1 + c, p)) {
		outFraction = t1;
		outPoint = ConvertPlaneToWorld(inPlane, u, v, p);
		return true;
	}

	// Test if sphere intersects with one of the edges or vertices
	if (SweptCircleTriangleEdgeVertexIntersect(tp, begin, delta, a, b, c, p, outFraction)) {
		outPoint = ConvertPlaneToWorld(inPlane, u, v, p);
		return true;
	}

	return false;
}

__device__ IntersectionResult testIntersectionTriSphere(const TrianglePlane &tp, const float3 &sphere_centre, const float sphere_radius, const float3 &delta) {

	IntersectionResult rslt;

	float3 outPoint;
	float outFraction;

	rslt.intersectionOccurred = PolygonSweptSphereIntersect(tp, sphere_centre, delta, sphere_radius, outPoint, outFraction);
	float intersectDistance = distance(outPoint, sphere_centre) - sphere_radius;
	if (intersectDistance < 0) intersectDistance = 0;
	float lenDelta = length(delta);
	rslt.distanceToMove = fmin(lenDelta, intersectDistance);


	return rslt;
}


#endif
