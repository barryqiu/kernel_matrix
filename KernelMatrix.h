// KernelMatrix.cuh
// �����ˣ���Т��
//
// �˾���ļ��� ��Kernel Matrix��
// ����˵������������������������ø�˹�˺������к˾���ļ��㣬�����������
//          Ϊ m*n������ m Ϊ������n Ϊ����������������Ϊ m��ÿ��������ά��Ϊ n��
//          ��ô����ĺ˾���Ϊ m * m
//
// �޸���ʷ��
// 2015��04��09�գ���Т����
//    ��ʼ�汾

#ifndef __KERNELMATRIX_CUH__
#define __KERNELMATRIX_CUH__

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "ErrorCode.h"
#include "Matrix.h"

// �ࣺKernelMatrix������˾���
// �̳��ԣ���
// ��������������������ø�˹�˺������к˾���ļ��㣬�����������Ϊm*n��
// ����mΪ������nΪ����������������Ϊ�㣬ÿ��������ά��Ϊm����ô����ĺ�
// ����Ϊn*n
class KernelMatrix
{
protected:
    // ��Ա������sigma
    // sigma�Ǹ�˹�˱任���ֳ�Ϊ����������ˣ���һ�����ɲ���
    float sigma;

public:
    // ���캯����KernekMatrix
    // �޲����汾�Ĺ��캯������Ա������ʼ��ΪĬ��ֵ
    __host__ __device__ KernelMatrix()
    {
        sigma = 1;
    }

    // ���캯����KernekMatrix
    // �в����汾�Ĺ��캯�������ݸ����Ĳ������ø�����Ա������ֵ
    __host__ __device__ KernelMatrix(float sigma)
    {
        this->sigma = sigma;
    }

    // ��Ա������getSigma
    // ��ȡ��Ա���� sigma ��ֵ
    __host__ __device__ float getSigma() const
    {
        return this->sigma;
    }

    // ��Ա������setSigma
    // ���ó�Ա���� sigma ��ֵ
    __host__ __device__ int setSigma(float sigma)
    {
        this->sigma = sigma; 
        return NO_ERROR;

    }

    // Host ��Ա������calcKernelMatrix������˾���
    // ����ָ���� sigma ���������������������к˾���ļ��㣬���������������������
    __host__ int calcKernelMatrix(
            Matrix *inSample,        // �������������
            Matrix *outKernelMatrix  // ����ĺ˾���
    );

    // Host ��Ա����: calcKernelMatrixSeriel�����м���˾���
    // ����ָ���� sigma ��������������������Դ��еķ�ʽ���к˾���ļ��㣬
    // ���������������������
    __host__ int calcKernelMatrixSeriel(
            Matrix *inSample,        // �������������
            Matrix *outKernelMatrix  // ����ĺ˾���
    );
};

#endif