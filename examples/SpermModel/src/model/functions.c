
/*
 * Copyright 2011 University of Sheffield.
 * Author: Dr Paul Richmond 
 * Contact: p.richmond@sheffield.ac.uk (http://www.paulrichmond.staff.shef.ac.uk)
 *
 * University of Sheffield retain all intellectual property and 
 * proprietary rights in and to this software and related documentation. 
 * Any use, reproduction, disclosure, or distribution of this software 
 * and related documentation without an express license agreement from
 * University of Sheffield is strictly prohibited.
 *
 * For terms of licence agreement please attached licence or view licence 
 * on www.flamegpu.com website.
 * 
 */

#ifndef _FLAMEGPU_FUNCTIONS
#define _FLAMEGPU_FUNCTIONS

#include <header.h>
#include "Common/helper_math.h"
#include "OviductCollisionDetectionV1/collision_detectionCUDA.h"
#include "Common/common.h"
#include "Common/cuda_matrix.h"

#define TOTAL_NO_OF_OOCYTES xmachine_memory_Oocyte_MAX

/*Function Implementations*/

/*Local collision detection structure*/
struct CollisionDetails {
	float3 collisionPlaneNormal;
	bool collisionOccurred;
};

/*Copies model / collision detection data stored in environment_definition.bin into GPU memory*/
__FLAME_GPU_INIT_FUNC__ void copyModelData() {

	char data[500];
	/*	The data path is loaded from the 0.XML file */
	int pathLength = getDataPath(data);

	if (pathLength == 0) {
		initialiseGPUData();
	}
	else {
		char dataFile[500];

		sprintf(dataFile, "%senvironment_definition.bin", data);

		initialiseGPUDataFromFile(dataFile);
	}
	const char* simDesc = "Simulation using final corrected model with mouse_oviduct a";
	setSimulationDescription(simDesc);

	//Perform the pre-initialisation step - distribute the sperm on the walls
	singleIteration();
}


__FLAME_GPU_STEP_FUNC__ void updateIterationNo() {
	unsigned int iter = getIterationNumber();
	set_currentIterationNo(&iter);
}

#ifdef _MSC_VER
#pragma region Helper Functions
#endif

/*Calculates movement speed for a single step*/
__device__ float GetSingleStepProgressiveVelocity(xmachine_memory_Sperm* sperm) {
	return Const_ProgressiveVelocity / ((float)Const_ProgressiveMovementSteps);
}

/*Performs a random number true/false test using the specified threshold [0..1]*/
__device__ bool TestCondition(float threshold, RNG_rand48* rand48) {
	return (rnd(rand48) < threshold);
}

/*Returns true if the specified Sperm agent has the specified bitwise state*/
__device__ bool HasState(xmachine_memory_Sperm* sperm, const int state) {
	return (sperm->activationState & state) == state;
}


/*Alters the specified Sperm agent to have the specified bitwise collision state*/
__device__ void SetCollisionState(xmachine_memory_Sperm* sperm, const int collisionState) {
	sperm->activationState = ((sperm->activationState & (~((int)COLLISION_STATE_MASK))) | collisionState);
}

/*Alters the specified Sperm agent to have the specified bitwise movement state*/
__device__ void SetMovementState(xmachine_memory_Sperm* sperm, const int movementState) {
	sperm->activationState = ((sperm->activationState & (~((int)MOVEMENT_STATE_MASK))) | movementState);
}

/*Alters the specified Sperm agent to have the specified bitwise activation state*/
__device__ void SetActivationState(xmachine_memory_Sperm* sperm, const int activationState) {
	sperm->activationState = ((sperm->activationState & (~((int)ACTIVATION_STATE_MASK))) | activationState);
}

/*Returns the current collision state*/
__device__ int GetCollisionState(xmachine_memory_Sperm* sperm) {
	return sperm->activationState & (COLLISION_STATE_MASK);
}

/*Returns the current movement state*/
__device__ int GetMovementState(xmachine_memory_Sperm* sperm) {
	return sperm->activationState & (MOVEMENT_STATE_MASK);
}

/*Returns the current activation state*/
__device__ int GetActivationState(xmachine_memory_Sperm* sperm) {
	return sperm->activationState & (ACTIVATION_STATE_MASK);
}

/*Returns true if the current index is out of bounds*/
__device__ bool SpermOutOfBounds() {
	int index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x; 
	return index >= d_xmachine_memory_Sperm_count;
}

/*Returns true if the current index is out of bounds*/
__device__ bool OocyteOutOfBounds() {
	int index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x; 
	return index >= d_xmachine_memory_Oocyte_count;
}

/*Converts the individual terms from the transformation matrix into a 4x4 matrix definition*/
__device__ Matrix getTransformationMatrix(xmachine_memory_Sperm* sperm) {
	return make_matrix(
		sperm->_mat0, sperm->_mat1, sperm->_mat2, 0, 
		sperm->_mat4, sperm->_mat5, sperm->_mat6, 0, 
		sperm->_mat8, sperm->_mat9, sperm->_mat10, 0, 
		sperm->_mat12, sperm->_mat13, sperm->_mat14, 1);
}

/*Sets the sperm matrix definition to the specified matrix*/
__device__ void setTransformationMatrix(xmachine_memory_Sperm* sperm, Matrix &mat) {
	sperm->_mat0 = mat.m[0];
	sperm->_mat1 = mat.m[1];
	sperm->_mat2 = mat.m[2];
	//sperm->_mat3 = mat.m[3];
	//sperm->_mat3 = 0;
	sperm->_mat4 = mat.m[4];
	sperm->_mat5 = mat.m[5];
	sperm->_mat6 = mat.m[6];
	//sperm->_mat7 = mat.m[7];
	//sperm->_mat7 = 0;
	sperm->_mat8 = mat.m[8];
	sperm->_mat9 = mat.m[9];
	sperm->_mat10 = mat.m[10];
	//sperm->_mat11 = mat.m[11];
	//sperm->_mat11 = 0;
	sperm->_mat12 = mat.m[12];
	sperm->_mat13 = mat.m[13];
	sperm->_mat14 = mat.m[14];
	//sperm->_mat15 = mat.m[15];
	//sperm->_mat15 = 1;
}

/*Returns a float3 definition of the oocyte position*/
__device__ float3 getOocytePosition(xmachine_message_oocytePosition* oocyte) {
	return make_float3(oocyte->positionX, oocyte->positionY, oocyte->positionZ);
}

#ifdef _MSC_VER
#pragma endregion
#endif

#ifdef _MSC_VER
#pragma region Sperm Angular Rotation Functions
#endif

// maxAngleDeg is angle between normal and cone radius
__device__ void ConicRotation(Matrix &spermMatrix, float maxAngleDeg, RNG_rand48* rand48) {

	//float3 I = MatrixGetDirection(spermMatrix);

	float factRad = TO_RADIANS(GetRandomNumber(0, 360, rand48));

	float sinFactRnd = sinf(factRad) * rnd(rand48);
	float cosFactRnd = cosf(factRad) * rnd(rand48);

	float pitchDeg = sinFactRnd * maxAngleDeg;
	float yawDeg =  cosFactRnd * maxAngleDeg;
		
	MatrixRotate(spermMatrix, TO_RADIANS(pitchDeg), TO_RADIANS(yawDeg), 0);
}

// maxAngleDeg is angle between normal and cone radius
__device__ void HalfConicReflection(Matrix &spermMatrix, float3 collisionPlaneNormal, float maxAngleDeg, RNG_rand48* rand48) {
	if (!isZero(collisionPlaneNormal)) {

		float3 N = collisionPlaneNormal;
		float3 I = MatrixGetDirection(spermMatrix);

		float3 outAxis;
		
		/* Calculate angle between incident and Normal and identify axis of rotation */
		float angRefDeg = VectorGetAngleBetween(I, N, outAxis);
		outAxis = normalize(outAxis);

		float pitchDeg = 0;
		float yawDeg = 0;

		float factRad = TO_RADIANS(GetRandomNumber(0, 360, rand48));

		float sinFactRnd = sinf(factRad) * rnd(rand48);
		float cosFactRnd = cosf(factRad) * rnd(rand48);

		pitchDeg = fabs(sinFactRnd * maxAngleDeg) + (angRefDeg - 90);
		yawDeg = cosFactRnd * maxAngleDeg;

		/* Apply pitch and yaw rotations */
		
		MatrixRotateAbsoluteAnyAxis(spermMatrix, TO_RADIANS(pitchDeg), outAxis);
		MatrixRotateAbsoluteAnyAxis(spermMatrix, TO_RADIANS(yawDeg), N);
	}
}

#ifdef _MSC_VER
#pragma endregion
#endif

#ifdef _MSC_VER
#pragma region Common Functions
#endif

/* Attach a sperm to oocyte - bind to the closest point between the sperm and the oocyte */
__device__ void AttachSpermToOocyte(xmachine_memory_Sperm* sperm, Matrix& spermMatrix, float3 oocytePosition, int oocyteCollisionID) {

	float3 position = MatrixGetPosition(spermMatrix);
	float3 direction = normalize(oocytePosition - position);
	float distanceToMove = distance(position, oocytePosition) - (Const_OocyteRadius + Const_SpermRadius);

	MatrixSetPosition(spermMatrix, position + (direction * distanceToMove));

	/* Move state to attached to oocyte */

	SetActivationState(sperm, ACTIVATION_STATE_POST_CAPACITATED);
	SetCollisionState(sperm, COLLISION_STATE_ATTACHED_TO_OOCYTE);
	sperm->attachedToOocyteTime = currentIterationNo;
	sperm->attachedToOocyteID = oocyteCollisionID;
}



/*Oocte position shared memory*/
__shared__ float3 SharedOocytePosition[TOTAL_NO_OF_OOCYTES];
__shared__ short SharedOocyteID[TOTAL_NO_OF_OOCYTES];
__shared__ short SharedOocyteUniqueEnvironment[TOTAL_NO_OF_OOCYTES];
__shared__ short NoOfOocytes;

/*Reads in all oocyte positions and puts them into shared memory*/
__device__ void GenerateOocytePositionCache(xmachine_message_oocytePosition_list* oocytePositionList) {
	if (threadIdx.x == 0) { NoOfOocytes = 0;}
	xmachine_message_oocytePosition* oocytePosition_message = get_first_oocytePosition_message(oocytePositionList);
	while(oocytePosition_message) {
		if(threadIdx.x == 0) { 
			SharedOocytePosition[NoOfOocytes] = getOocytePosition(oocytePosition_message); 
			SharedOocyteID[NoOfOocytes] = oocytePosition_message->id;
			SharedOocyteUniqueEnvironment[NoOfOocytes] = oocytePosition_message->uniqueEnvironmentNo;
			NoOfOocytes++;

		}
		oocytePosition_message = get_next_oocytePosition_message(oocytePosition_message, oocytePositionList);
	}

	__syncthreads();
}

/*The sperm agent state is updated to being attached to the epithelium*/
__device__ void AttachToEpithelium(xmachine_memory_Sperm* sperm) {
	SetCollisionState(sperm, COLLISION_STATE_ATTACHED_TO_EPITHELIUM);
}

/*Resolves agent to oviduct collisions*/
__device__ bool ResolveCollisions(xmachine_memory_Sperm* sperm, Matrix& spermMatrix, float movementDistance, float3 direction, CollisionDetails &collisionDetails, RNG_rand48* rand48) {

	float3 oldPosition;
	float3 newPosition;

	collisionDetails.collisionOccurred = false;

	bool outOfEnvironment = false;

	int oocyteCollisionID = -1;

	CollisionResult result;

	oldPosition = MatrixGetPosition(spermMatrix);

	int currentSegment = sperm->oviductSegment;

	result = resolve_environment_collisions(sperm->id, currentSegment, oldPosition, direction, movementDistance, Const_SpermRadius);

	sperm->oviductSegment = result.newSegmentIndex;

	if (sperm->oviductSegment >= (NO_OF_SEGMENTS_MINUS_ONE - 1)) {
		outOfEnvironment = true;
	}

	collisionDetails.collisionOccurred = result.collisionOccurred;
	newPosition = oldPosition + (direction * result.distanceToMove);

	float3 oocyteCollisionPosition;
	
	bool checkOocyteCollision = (d_current_iteration_no >= Const_OocyteFertilityStartTime);

	if (checkOocyteCollision) {
		for(int i=0;i<NoOfOocytes;i++) {
			float3 pointOnLine;
			if (SharedOocyteUniqueEnvironment[i] == sperm->uniqueEnvironmentNo) {
				float distToLine = CalculateDistanceFromPointToLine(SharedOocytePosition[i], oldPosition, newPosition, pointOnLine);
				if (distToLine >= 0 && distToLine < (Const_OocyteRadius + Const_SpermRadius)) {
					oocyteCollisionID = SharedOocyteID[i];
					oocyteCollisionPosition = SharedOocytePosition[i];
				}
			}
		}
	}

	

	/* Collide with oocyte? */

	if (oocyteCollisionID != -1) {
		AttachSpermToOocyte(sperm, spermMatrix, oocyteCollisionPosition, oocyteCollisionID);
		return true;
	}
	else {
		MatrixSetPosition(spermMatrix, newPosition);

		if (collisionDetails.collisionOccurred) {
			collisionDetails.collisionPlaneNormal = result.collisionPlaneNormal;

			SetCollisionState(sperm, COLLISION_STATE_TOUCHING_EPITHELIUM);
			if (outOfEnvironment) {
				SetActivationState(sperm, ACTIVATION_STATE_POST_CAPACITATED);
				return true;
			}
			else {
				return false;
			}
		}
		else {
			SetCollisionState(sperm, COLLISION_STATE_FREE);

			if (outOfEnvironment) {
				SetActivationState(sperm, ACTIVATION_STATE_POST_CAPACITATED);
				return true;
			}
			else {
				return false;
			}
		}

	}

}

#ifdef _MSC_VER
#pragma endregion
#endif

#define INITIAL_DISTRIBUTION_ANGLE 90

#ifdef _MSC_VER
#pragma region Sperm Functions
#endif

//Distribute Sperm on walls of current section - called at start of simulation only
//Removed random component for consistent deployment - direction calculated based on direction from 
//line between prev and next segment midpoints.
__FLAME_GPU_FUNC__ int Sperm_Init(xmachine_memory_Sperm* sperm, RNG_rand48* rand48) {

	if (SpermOutOfBounds()) { return 0; }

	Matrix spermMatrix = getTransformationMatrix(sperm);
	int currentSegment = sperm->oviductSegment;
	float3 currentSegmentNormal = normalize(make_float3(getSlicePlane(currentSegment)));
	Matrix m;
	MatrixToIdentity(m);

	MatrixSetDirection(m, currentSegmentNormal);

	float spermNoRnd = (float)sperm->spermNo / (float)MAX_NO_OF_SPERM;

	MatrixRotatePitch(m, TO_RADIANS(90 - (INITIAL_DISTRIBUTION_ANGLE*0.5f) + (spermNoRnd * INITIAL_DISTRIBUTION_ANGLE)));
	MatrixRotateAbsoluteAnyAxis(m, TO_RADIANS(spermNoRnd * 360), currentSegmentNormal);

	float3 direction = MatrixGetDirection(m);

	CollisionDetails collisionDetails;

	/* Identify Collisions */
	ResolveCollisions(sperm, spermMatrix, 300, direction, collisionDetails, rand48);

	
	if (HasState(sperm, ACTIVATION_STATE_CAPACITATED)) {
		SetCollisionState(sperm, COLLISION_STATE_TOUCHING_EPITHELIUM);
	}
	else {
		SetCollisionState(sperm, COLLISION_STATE_ATTACHED_TO_EPITHELIUM);
	}
	setTransformationMatrix(sperm, spermMatrix);

	return 0;
}


/*Tests if sperm should become capacitated*/
__FLAME_GPU_FUNC__ int Sperm_Capacitate(xmachine_memory_Sperm* sperm, RNG_rand48* rand48) {

	if (SpermOutOfBounds()) { return 0; }
	
		
	bool activate = TestCondition(Const_CapacitationThreshold, rand48);
	//Activate if random number is less than ACTIVATION_THRESHOLD

	if (activate) {
		SetActivationState(sperm, ACTIVATION_STATE_CAPACITATED);
		sperm->remainingLifeTime = Const_CapacitatedSpermLife;
	}
	else {
		//No State Change
	}

	return 0;
}

/*
	* If Attach to wall, reflect randomly 180 degrees (new random direction for after detachment) 
	* Otherwise reflect based on turn angle, 
*/
__device__ bool HandleSurfaceInteraction(xmachine_memory_Sperm* sperm, Matrix& spermMatrix, CollisionDetails collisionDetails, RNG_rand48* rand48, float attachmentThreshold) {
	bool resolved = false;
	if (TestCondition(attachmentThreshold, rand48)) {

		/* Reflection (180) */
		//Reflection180(spermMatrix, collisionDetails.collisionPlaneNormal, rand48);
		HalfConicReflection(spermMatrix, collisionDetails.collisionPlaneNormal, Const_DetachmentMaxRotationAngle, rand48);
	//	Reflection90(spermMatrix, collisionDetails.collisionPlaneNormal, rand48);
		/* Move to attached to epithelium state */
		AttachToEpithelium(sperm);
		resolved = true;
	}
	else {
		/* Reflection */
		HalfConicReflection(spermMatrix, collisionDetails.collisionPlaneNormal, Const_ReflectionMaxRotationAngle, rand48);
		//Reflection(spermMatrix, collisionDetails.collisionPlaneNormal, rand48);
	}
	
	return resolved;
}

#ifdef _MSC_VER
#pragma region Progressive Movement
#endif

/*Moves forward a single iteration*/
__device__ bool SingleProgressiveMovement(xmachine_memory_Sperm* sperm, Matrix& spermMatrix, RNG_rand48* rand48, float distanceToMove) {

	CollisionDetails collisionDetails;

	bool resolved = false;

	float3 direction = MatrixGetDirection(spermMatrix);

	/* Identify Collisions */
	resolved = ResolveCollisions(sperm, spermMatrix, distanceToMove, direction, collisionDetails, rand48);

	/* Collide with surface? */
	if (HasState(sperm, COLLISION_STATE_TOUCHING_EPITHELIUM)) {

		if (HandleSurfaceInteraction(sperm, spermMatrix, collisionDetails, rand48, Const_AttachmentThresholdProgressive)) {
			resolved = true;
		}
		else {
			//Reflection(spermMatrix, collisionDetails.collisionPlaneNormal, rand48);
		}

	}

	return resolved;

}


/*Moves forward at small steps, progressively performing collision detection*/
__FLAME_GPU_FUNC__ int Sperm_ProgressiveMovement(xmachine_memory_Sperm* sperm, xmachine_message_oocytePosition_list* oocytePositionList, RNG_rand48* rand48) {
	//Pre cache all oocyte positions in shared memory to allow for multiple iterative loops and early exit for out of bounds agents (limitations of flame GPU).
	GenerateOocytePositionCache(oocytePositionList);

	if (SpermOutOfBounds()) { return 0; }

	Matrix spermMatrix = getTransformationMatrix(sperm);

	bool resolved;
	for(int i=0;i<Const_ProgressiveMovementSteps;i++) {
		resolved = SingleProgressiveMovement(sperm, spermMatrix, rand48, GetSingleStepProgressiveVelocity(sperm));

		if (resolved) {
			break;
		}
	}
	setTransformationMatrix(sperm, spermMatrix);

	return 0;
}

#ifdef _MSC_VER
#pragma endregion
#endif

#ifdef _MSC_VER
#pragma region Non Progressive Movement
#endif
/*MOves non-progressively */
__device__ bool SingleNonProgressiveMovement(xmachine_memory_Sperm* sperm, Matrix& spermMatrix, RNG_rand48* rand48, float distanceToMove) {

	CollisionDetails collisionDetails;

	bool resolved = false;

	ConicRotation(spermMatrix, Const_NonProgressiveMaxRotationAngle, rand48);
	//ConstrainedRotation(spermMatrix, rand48);

	float3 direction = MatrixGetDirection(spermMatrix);

	resolved = ResolveCollisions(sperm, spermMatrix, distanceToMove, direction, collisionDetails, rand48);

	/* Collide with surface? */
	if (HasState(sperm, COLLISION_STATE_TOUCHING_EPITHELIUM)) {

		if (HandleSurfaceInteraction(sperm, spermMatrix, collisionDetails, rand48, Const_AttachmentThresholdNonProgressive)) {
			resolved = true;
		}
		else {
			//RandomDirection360(spermMatrix, rand48);
			//ConstrainedRotation(spermMatrix, rand48);
		}
	}

	return resolved;
}
/*Moves non-progressively*/
__FLAME_GPU_FUNC__ int Sperm_NonProgressiveMovement(xmachine_memory_Sperm* sperm, xmachine_message_oocytePosition_list* oocytePositionList, RNG_rand48* rand48) {
	//Pre cache all oocyte positions in shared memory to allow for multiple iterative loops and early exit for out of bounds agents (limitations of flame GPU).
	GenerateOocytePositionCache(oocytePositionList);

	if (SpermOutOfBounds()) { return 0; }

	Matrix spermMatrix = getTransformationMatrix(sperm);

	SingleNonProgressiveMovement(sperm, spermMatrix, rand48, Const_NonProgressiveVelocity);

	setTransformationMatrix(sperm, spermMatrix);

	return 0;

}

#ifdef _MSC_VER
#pragma endregion
#endif

/*Determins if an agent should detach from the oviduct*/
__FLAME_GPU_FUNC__ int Sperm_DetachFromEpithelium(xmachine_memory_Sperm* sperm, RNG_rand48* rand48) {
	if (SpermOutOfBounds()) { return 0; }

	float switchThreshold = HasState(sperm, MOVEMENT_STATE_NON_PROGRESSIVE) ? Const_DetachmentThresholdNonProgressive : Const_DetachmentThresholdProgressive;


	if (TestCondition(switchThreshold, rand48)) {
		SetCollisionState(sperm, COLLISION_STATE_TOUCHING_EPITHELIUM);
	}

	return 0;
}


/*Switches between progressive and non-progressive movement*/
__FLAME_GPU_FUNC__ int Sperm_SwitchMovementState(xmachine_memory_Sperm* sperm, RNG_rand48* rand48) {
	if (SpermOutOfBounds()) { return 0; }

	int alternateMovementState = MOVEMENT_STATE_NON_PROGRESSIVE;
	float mn = Const_NonProgressiveMin;
	float mx = Const_NonProgressiveMax;

	if (HasState(sperm, MOVEMENT_STATE_NON_PROGRESSIVE)) {
		alternateMovementState = MOVEMENT_STATE_PROGRESSIVE;
		 mn = Const_ProgressiveMin;
		 mx = Const_ProgressiveMax;
	}
	
	if (--sperm->movementStateTimer <= 0) {
		SetMovementState(sperm, alternateMovementState);
		sperm->movementStateTimer = (int)round( GetRandomNumber(mn, mx, rand48) /*SampleFromNormalDistribution(mean, sd, rand48)*/);
	}


	return 0;
}

/*Regulate sperm live*/
__FLAME_GPU_FUNC__ int Sperm_RegulateState(xmachine_memory_Sperm* sperm) {
	if (SpermOutOfBounds()) { return 0; }

	if (HasState(sperm, ACTIVATION_STATE_CAPACITATED)) {
		if (--sperm->remainingLifeTime <= 0) {
			SetActivationState(sperm, ACTIVATION_STATE_DEAD);
		}
	}
	//if (sperm->remainingLifeTime > 0)
	return 0;
}

#ifdef _MSC_VER
#pragma endregion
#endif

#ifdef _MSC_VER
#pragma region Oocyte Functions
#endif
/*Reports the position of the oocyte*/
__FLAME_GPU_FUNC__ int Oocyte_ReportPosition(xmachine_memory_Oocyte* oocyte, xmachine_message_oocytePosition_list* oocytePosition_messages){
	if (OocyteOutOfBounds()) { return 0; }

	add_oocytePosition_message(oocytePosition_messages, oocyte->id, oocyte->positionX, oocyte->positionY, oocyte->positionZ, oocyte->uniqueEnvironmentNo);
	return 0;
}

#ifdef _MSC_VER
#pragma endregion
#endif
  


#endif //_FLAMEGPU_FUNCTIONS
