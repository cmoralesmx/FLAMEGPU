/*Implementation of a 16 element (4x4) matrix and associated functions*/
/*Modified for CUDA from implementation of Matrix4x4 by Kevin Harris (2005)
http://artificialstudios.googlecode.com/svn-history/r57/trunk/artificialstudios/v1.1/Common/Matrix4x4.cpp
*/
#ifndef _CUDA_MATRIX_H
#define _CUDA_MATRIX_H 1

#include "common.h"

struct Matrix {
	float m[16];
};

__constant__ float ReferenceDirection[] = {0,0,1};

__device__ void MatrixToIdentity(Matrix &mat);

__device__ void MatrixTranslate(Matrix &mat, const float3 &v);
__device__ void MatrixRotate(Matrix &mat, float yawRad, float pitchRad, float rollRad);
__device__ void MatrixRotateAnyAxis(Matrix &mat, float angleRad, const float3 &normAxis);

__device__ void MatrixRotatePitch(Matrix &mat, float pitchRad);
__device__ void MatrixRotateYaw(Matrix &mat, float yawRad);
__device__ void MatrixRotateRoll(Matrix &mat, float rollRad);

__device__ void MatrixMultiply(Matrix &mat, const float* other);
__device__ void MatrixOverwrite(Matrix &mat, const float* val);
__device__ void MatrixOverwriteMatrix(Matrix &mat, const Matrix &newMat);



__device__ void MatrixRotateVector(const Matrix &mat, float3 &v);
__device__ void MatrixTransformVector(const Matrix &mat, float3 &v);

__device__ float3 MatrixGetDirection(const Matrix &mat);
__device__ float3 MatrixGetPosition(const Matrix &mat);
__device__ void MatrixSetPosition(Matrix &mat, const float3 position);
__device__ void MatrixSetDirection(Matrix &mat, const float3 dir);

__device__ float VectorGetAngleBetween(float3 v1, float3 v2, float3 &outAxis);


/// <summary>
/// Return the direction vector of the matrix
/// </summary>
__device__ float3 MatrixGetDirection(const Matrix &mat) {
	float3 dir = make_float3(ReferenceDirection[0],ReferenceDirection[1],ReferenceDirection[2]);
	MatrixRotateVector(mat, dir);
	if (isZero(dir)) {
		return make_float3(ReferenceDirection[0],ReferenceDirection[1],ReferenceDirection[2]);
	}
	else {
		return normalize(dir);
	}
}


__device__ float3 MatrixGetPosition(const Matrix &mat) {
	float3 vec = make_float3(0);
	MatrixTransformVector(mat, vec);
	return vec;
}

__device__ void MatrixSetPosition(Matrix &mat, const float3 position) {
	mat.m[12] = position.x;
	mat.m[13] = position.y;
	mat.m[14] = position.z;
}

__device__ void MatrixSetDirection(Matrix &mat, const float3 dir) {
	float3 refDirection = make_float3(ReferenceDirection[0],ReferenceDirection[1],ReferenceDirection[2]);
	float3 outAxis;
	float angleDeg = VectorGetAngleBetween(refDirection, dir, outAxis);
	if (angleDeg != 0) {
		float3 position = MatrixGetPosition(mat);

		Matrix newMatrix;
		MatrixToIdentity(newMatrix);
		outAxis = normalize(outAxis);
		MatrixRotateAnyAxis(newMatrix, TO_RADIANS(angleDeg), outAxis);
	
		MatrixSetPosition(newMatrix, position);

		MatrixOverwriteMatrix(mat, newMatrix);
	}
}

//Returns the angle between the two vectors in degrees
__device__ float VectorGetAngleBetween(float3 v1, float3 v2, float3 &outAxis) {
	v1 = normalize(v1);
	v2 = normalize(v2);
	outAxis = cross(v1, v2);

	if (length(outAxis) == 0) {
		outAxis = normalize(GetPerpendicular(v2));

		if (areEqual(v1, v2)) {
			return 0;
		}
		else {
			return 180;
		}
	}
	else {
		outAxis = normalize(outAxis);
		return TO_DEGREES(acosf(clamp(dot(v1, v2), -1.0f, 1.0f)));
	}
}
__device__ Matrix make_matrix(float m0, float m1, float m2, float m3, 
	float m4, float m5, float m6, float m7, 
	float m8, float m9, float m10, float m11,
	float m12, float m13, float m14, float m15) {
		Matrix mat;
		mat.m[0] = m0;
		mat.m[1] = m1;
		mat.m[2] = m2;
		mat.m[3] = m3;
		mat.m[4] = m4;
		mat.m[5] = m5;
		mat.m[6] = m6;
		mat.m[7] = m7;
		mat.m[8] = m8;
		mat.m[9] = m9;
		mat.m[10] = m10;
		mat.m[11] = m11;
		mat.m[12] = m12;
		mat.m[13] = m13;
		mat.m[14] = m14;
		mat.m[15] = m15;
		return mat;
}

__device__ void CreateIdentityArray(float* a) {
	a[0] = 1.0f; a[4] = 0.0f; a[8] = 0.0f; a[12] = 0.0f;
	a[1] = 0.0f; a[5] = 1.0f; a[9] = 0.0f; a[13] = 0.0f;
	a[2] = 0.0f; a[6] = 0.0f; a[10] = 1.0f; a[14] = 0.0f;
	a[3] = 0.0f; a[7] = 0.0f; a[11] = 0.0f; a[15] = 1.0f;
}

__device__ void MatrixToIdentity(Matrix &mat) {
	float* m = mat.m;
	m[0] = 1.0f; m[4] = 0.0f; m[8] = 0.0f; m[12] = 0.0f;
	m[1] = 0.0f; m[5] = 1.0f; m[9] = 0.0f; m[13] = 0.0f;
	m[2] = 0.0f; m[6] = 0.0f; m[10] = 1.0f; m[14] = 0.0f;
	m[3] = 0.0f; m[7] = 0.0f; m[11] = 0.0f; m[15] = 1.0f;
}

__device__ void MatrixTranslate(Matrix &mat, const float3 &v) {
	float t[16];
	CreateIdentityArray(t);
	t[12] = v.x;
	t[13] = v.y;
	t[14] = v.z;

	MatrixMultiply(mat, t);
}


__device__ void MatrixRotatePitch(Matrix &mat, float pitchRad) {
	MatrixRotateAnyAxis(mat, pitchRad, make_float3(1, 0, 0));
}

__device__ void MatrixRotateYaw(Matrix &mat, float yawRad) {
	MatrixRotateAnyAxis(mat, yawRad, make_float3(0, 1, 0));
}

__device__ void MatrixRotateRoll(Matrix &mat, float rollRad) {
	MatrixRotateAnyAxis(mat, rollRad, make_float3(0, 0, 1));
}

__device__ void MatrixRotate(Matrix &mat, float yawRad, float pitchRad, float rollRad) {
	if (yawRad != 0) {
		MatrixRotateYaw(mat, yawRad);
	}
	if (pitchRad != 0) {
		MatrixRotatePitch(mat, pitchRad);
	}
	if (rollRad != 0) {
		MatrixRotateRoll(mat, rollRad);
	}
}

/*
/// Rotate around arbitrary axis - formulas taken from:
/// http://www.euclideanspace.com/maths/geometry/rotations/conversions/angleToMatrix/index.htm
/// 1 + (1-cos(angle))*(x*x-1)	        -z*sin(angle)+(1-cos(angle))*x*y	    y*sin(angle)+(1-cos(angle))*x*z
/// z*sin(angle)+(1-cos(angle))*x*y	    1 + (1-cos(angle))*(y*y-1)	            -x*sin(angle)+(1-cos(angle))*y*z
/// -y*sin(angle)+(1-cos(angle))*x*z	x*sin(angle)+(1-cos(angle))*y*z	        1 + (1-cos(angle))*(z*z-1)
*/
__device__ void MatrixRotateAnyAxis(Matrix &mat, float angleRad, const float3 &normAxis) {

	float sinAngle = sinf(angleRad);
	float cosAngle = cosf(angleRad);

	float x = normAxis.x;
	float y = normAxis.y;
	float z = normAxis.z;

	float r[16];
	CreateIdentityArray(r);
	r[0] = 1 + (1 - cosAngle) * (x * x - 1);
	r[1] = z * sinAngle + (1 - cosAngle) * x * y;
	r[2] = -y * sinAngle + (1 - cosAngle) * x * z;

	r[4] = -z * sinAngle + (1 - cosAngle) * x * y;
	r[5] = 1 + (1 - cosAngle) * (y * y - 1);
	r[6] = x * sinAngle + (1 - cosAngle) * y * z;

	r[8] = y * sinAngle + (1 - cosAngle) * x * z;
	r[9] = -x * sinAngle + (1 - cosAngle) * y * z;
	r[10] = 1 + (1 - cosAngle) * (z * z - 1);

	MatrixMultiply(mat, r);

}

 __device__ void MatrixRotateAbsoluteAnyAxis(Matrix &mat, float angleRad, const float3 &normAxis) {
            //Vector3 currentDirection = GetDirection();
            Matrix m;
			MatrixToIdentity(m);
            MatrixRotateAnyAxis(m, angleRad, normAxis);

			float3 pos = MatrixGetPosition(mat);
			MatrixSetPosition(mat, make_float3(0,0,0));
			//MatrixTranslate(mat, -pos);
            MatrixMultiply(m, mat.m);
            MatrixOverwrite(mat, m.m);
			MatrixSetPosition(mat, pos);
            
        }

/*__device__ void MatrixLocalRotateAnyAxis(Matrix &mat, const float angleRad, const float3 &normAxis) {
	float3 pos = make_float3(mat.m[12], mat.m[13], mat.m[14]);
	MatrixTranslate(mat, -pos);
	MatrixRotateAnyAxis(mat, angleRad, normAxis);
	MatrixTranslate(mat, pos);
}*/



__device__ void MatrixTransformVector(const Matrix &mat, float3 &v) {
	const float* m = mat.m;
	float x = v.x;
	float y = v.y;
	float z = v.z;
	// float w = 1;

	v.x = x * m[0] +
		y * m[4] +
		z * m[8] +
		/*w * */m[12];

	v.y = x * m[1] +
		y * m[5] +
		z * m[9] +
		/*w * */m[13];

	v.z = x * m[2] +
		y * m[6] +
		z * m[10] +
		/*w * */m[14];
}

__device__ void MatrixRotateVector(const Matrix &mat, float3 &v) {
	const float* m = mat.m;
	float x = v.x;
	float y = v.y;
	float z = v.z;
	// float w = 0;

	v.x = x * m[0] +
		y * m[4] +
		z * m[8]/* +
				w * m[12]*/;

	v.y = x * m[1] +
		y * m[5] +
		z * m[9]/* +
				w * m[13]*/;

	v.z = x * m[2] +
		y * m[6] +
		z * m[10]/* +
				 w * m[14]*/;
}

__device__ void MatrixTranspose(Matrix &mat) {
	const float* m = mat.m;
	float result[16];

	result[0] = m[0];
	result[1] = m[4];
	result[2] = m[8];
	result[3] = m[12];

	result[4] = m[1];
	result[5] = m[5];
	result[6] = m[9];
	result[7] = m[13];

	result[8] = m[2];
	result[9] = m[6];
	result[10] = m[10];
	result[11] = m[14];

	result[12] = m[3];
	result[13] = m[7];
	result[14] = m[11];
	result[15] = m[15];

	MatrixOverwrite(mat, result);
}

__device__ void MatrixMultiply(Matrix &mat, const float* other) {
	float result[16];
	const float* m = mat.m;

	result[0] = (m[0] * other[0]) + (m[4] * other[1]) + (m[8] * other[2]) + (m[12] * other[3]);
	result[1] = (m[1] * other[0]) + (m[5] * other[1]) + (m[9] * other[2]) + (m[13] * other[3]);
	result[2] = (m[2] * other[0]) + (m[6] * other[1]) + (m[10] * other[2]) + (m[14] * other[3]);
	result[3] = (m[3] * other[0]) + (m[7] * other[1]) + (m[11] * other[2]) + (m[15] * other[3]);

	result[4] = (m[0] * other[4]) + (m[4] * other[5]) + (m[8] * other[6]) + (m[12] * other[7]);
	result[5] = (m[1] * other[4]) + (m[5] * other[5]) + (m[9] * other[6]) + (m[13] * other[7]);
	result[6] = (m[2] * other[4]) + (m[6] * other[5]) + (m[10] * other[6]) + (m[14] * other[7]);
	result[7] = (m[3] * other[4]) + (m[7] * other[5]) + (m[11] * other[6]) + (m[15] * other[7]);

	result[8] = (m[0] * other[8]) + (m[4] * other[9]) + (m[8] * other[10]) + (m[12] * other[11]);
	result[9] = (m[1] * other[8]) + (m[5] * other[9]) + (m[9] * other[10]) + (m[13] * other[11]);
	result[10] = (m[2] * other[8]) + (m[6] * other[9]) + (m[10] * other[10]) + (m[14] * other[11]);
	result[11] = (m[3] * other[8]) + (m[7] * other[9]) + (m[11] * other[10]) + (m[15] * other[11]);

	result[12] = (m[0] * other[12]) + (m[4] * other[13]) + (m[8] * other[14]) + (m[12] * other[15]);
	result[13] = (m[1] * other[12]) + (m[5] * other[13]) + (m[9] * other[14]) + (m[13] * other[15]);
	result[14] = (m[2] * other[12]) + (m[6] * other[13]) + (m[10] * other[14]) + (m[14] * other[15]);
	result[15] = (m[3] * other[12]) + (m[7] * other[13]) + (m[11] * other[14]) + (m[15] * other[15]);

	MatrixOverwrite(mat, result);
}

///
__device__ void MatrixOverwrite(Matrix &mat, const float* val) {
	mat.m[0] = val[0];
	mat.m[1] = val[1];
	mat.m[2] = val[2];
	mat.m[3] = val[3];
	mat.m[4] = val[4];
	mat.m[5] = val[5];
	mat.m[6] = val[6];
	mat.m[7] = val[7];
	mat.m[8] = val[8];
	mat.m[9] = val[9];
	mat.m[10] = val[10];
	mat.m[11] = val[11];
	mat.m[12] = val[12];
	mat.m[13] = val[13];
	mat.m[14] = val[14];
	mat.m[15] = val[15];
}

__device__ void MatrixOverwriteMatrix(Matrix &mat, const Matrix &newMat) {
	MatrixOverwrite(mat, newMat.m);
	/*mat.m[0] = newMat.m[0];
	mat.m[1] = newMat.m[1];
	mat.m[2] = newMat.m[2];
	mat.m[3] = newMat.m[3];
	mat.m[4] = newMat.m[4];
	mat.m[5] = newMat.m[5];
	mat.m[6] = newMat.m[6];
	mat.m[7] = newMat.m[7];
	mat.m[8] = newMat.m[8];
	mat.m[9] = newMat.m[9];
	mat.m[10] = newMat.m[10];
	mat.m[11] = newMat.m[11];
	mat.m[12] = newMat.m[12];
	mat.m[13] = newMat.m[13];
	mat.m[14] = newMat.m[14];
	mat.m[15] = newMat.m[15];*/
}


__device__ void MatrixInverse(Matrix &mat) {
	float* m = mat.m;

	float m11 = m[(0 * 4) + 0];
	float m12 = m[(1 * 4) + 0];
	float m13 = m[(2 * 4) + 0];
	float m14 = m[(3 * 4) + 0];

	float m21 = m[(0 * 4) + 1];
	float m22 = m[(1 * 4) + 1];
	float m23 = m[(2 * 4) + 1];
	float m24 = m[(3 * 4) + 1];

	float m31 = m[(0 * 4) + 2];
	float m32 = m[(1 * 4) + 2];
	float m33 = m[(2 * 4) + 2];
	float m34 = m[(3 * 4) + 2];

	float m41 = m[(0 * 4) + 3];
	float m42 = m[(1 * 4) + 3];
	float m43 = m[(2 * 4) + 3];
	float m44 = m[(3 * 4) + 3];


	float d12 = (m31 * m42 - m41 * m32);
	float d13 = (m31 * m43 - m41 * m33);
	float d23 = (m32 * m43 - m42 * m33);
	float d24 = (m32 * m44 - m42 * m34);
	float d34 = (m33 * m44 - m43 * m34);
	float d41 = (m34 * m41 - m44 * m31);


	float tmp[16];

	tmp[0] = (m22 * d34 - m23 * d24 + m24 * d23);
	tmp[1] = -(m21 * d34 + m23 * d41 + m24 * d13);
	tmp[2] = (m21 * d24 + m22 * d41 + m24 * d12);
	tmp[3] = -(m21 * d23 - m22 * d13 + m23 * d12);

	float det = m11 * tmp[0] + m12 * tmp[1] + m13 * tmp[2] + m14 * tmp[3];

	if (det == 0) {
		float a[16];
		CreateIdentityArray(a);
		//  System.Diagnostics.Debug.WriteLine("Warning: Call to invertMatrix produced a Singular matrix");
		MatrixOverwrite(mat, a);
	}
	else {
		float invDet = 1.0f / det;

		// Compute rest of inverse.
		tmp[0] *= invDet;
		tmp[1] *= invDet;
		tmp[2] *= invDet;
		tmp[3] *= invDet;

		tmp[4] = -(m12 * d34 - m13 * d24 + m14 * d23) * invDet;
		tmp[5] = (m11 * d34 + m13 * d41 + m14 * d13) * invDet;
		tmp[6] = -(m11 * d24 + m12 * d41 + m14 * d12) * invDet;
		tmp[7] = (m11 * d23 - m12 * d13 + m13 * d12) * invDet;

		// Pre-compute 2x2 dets for first two rows when computing cofactors 
		// of last two rows.
		d12 = m11 * m22 - m21 * m12;
		d13 = m11 * m23 - m21 * m13;
		d23 = m12 * m23 - m22 * m13;
		d24 = m12 * m24 - m22 * m14;
		d34 = m13 * m24 - m23 * m14;
		d41 = m14 * m21 - m24 * m11;

		tmp[8] = (m42 * d34 - m43 * d24 + m44 * d23) * invDet;
		tmp[9] = -(m41 * d34 + m43 * d41 + m44 * d13) * invDet;
		tmp[10] = (m41 * d24 + m42 * d41 + m44 * d12) * invDet;
		tmp[11] = -(m41 * d23 - m42 * d13 + m43 * d12) * invDet;
		tmp[12] = -(m32 * d34 - m33 * d24 + m34 * d23) * invDet;
		tmp[13] = (m31 * d34 + m33 * d41 + m34 * d13) * invDet;
		tmp[14] = -(m31 * d24 + m32 * d41 + m34 * d12) * invDet;
		tmp[15] = (m31 * d23 - m32 * d13 + m33 * d12) * invDet;

		MatrixOverwrite(mat, tmp);
	}

}

#endif
