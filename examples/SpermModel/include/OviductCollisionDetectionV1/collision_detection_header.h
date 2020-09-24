/*Header file for collision detection structures and functions*/

#ifndef _COLLISION_DETECTION_HEADER_H
#define _COLLISION_DETECTION_HEADER_H 1



#define INTERSECTION_OCCURRED true
#define NO_INTERSECTION false

#define COLLISION_OCCURRED true
#define NO_COLLISION false


#define INTERSECTION_BUFFER 0.01
#define RADIUS_BUFFER 0.01


struct TrianglePlane {
	float4 plane;
	float3 U;
	float3 V;
	float2 p0;
	float2 p1;
	float2 p2;
};

struct CollisionResult
{
  float distanceToMove;
  bool collisionOccurred;
  int newSegmentIndex;
  int2 sliceRange;
  float3 collisionPlaneNormal;
};

struct IntersectionResult {
	float distanceToMove;
	bool intersectionOccurred;
};


__device__ IntersectionResult make_intersection_result(float distanceToMove, 
	bool intersectionOccurred) {
	IntersectionResult rslt;
	rslt.distanceToMove = distanceToMove;
	rslt.intersectionOccurred = intersectionOccurred;
	return rslt;
}




struct SphereLineIntersectionResult {
	int nbinter;
	float inter1;
	float inter2;
	bool intersectionOccurred;
};




#endif