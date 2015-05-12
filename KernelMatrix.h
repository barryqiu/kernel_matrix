// KernelMatrix.cuh
// 创建人：邱孝兵
//
// 核矩阵的计算 （Kernel Matrix）
// 功能说明：对于输入的样例矩阵，利用高斯核函数进行核矩阵的计算，假设输入矩阵
//          为 m*n，其中 m 为行数，n 为列数，即样例个数为 m，每个样例的维度为 n，
//          那么输出的核矩阵为 m * m
//
// 修改历史：
// 2015年04月09日（邱孝兵）
//    初始版本

#ifndef __KERNELMATRIX_CUH__
#define __KERNELMATRIX_CUH__

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "ErrorCode.h"
#include "Matrix.h"

// 类：KernelMatrix（计算核矩阵）
// 继承自：无
// 对于输入的样例矩阵，利用高斯核函数进行核矩阵的计算，假设输入矩阵为m*n，
// 其中m为行数，n为列数，即样例个数为你，每个样例的维度为m，那么输出的核
// 矩阵为n*n
class KernelMatrix
{
protected:
    // 成员变量：sigma
    // sigma是高斯核变换（又称为径向基函数核）的一个自由参数
    float sigma;

public:
    // 构造函数：KernekMatrix
    // 无参数版本的构造函数，成员变量初始化为默认值
    __host__ __device__ KernelMatrix()
    {
        sigma = 1;
    }

    // 构造函数：KernekMatrix
    // 有参数版本的构造函数，根据给定的参数设置各个成员变量的值
    __host__ __device__ KernelMatrix(float sigma)
    {
        this->sigma = sigma;
    }

    // 成员方法：getSigma
    // 获取成员变量 sigma 的值
    __host__ __device__ float getSigma() const
    {
        return this->sigma;
    }

    // 成员方法：setSigma
    // 设置成员变量 sigma 的值
    __host__ __device__ int setSigma(float sigma)
    {
        this->sigma = sigma; 
        return NO_ERROR;

    }

    // Host 成员函数：calcKernelMatrix（计算核矩阵）
    // 根据指定的 sigma 参数对输入的样例矩阵进行核矩阵的计算，并将结果存放在输出矩阵中
    __host__ int calcKernelMatrix(
            Matrix *inSample,        // 输入的样例矩阵
            Matrix *outKernelMatrix  // 输出的核矩阵
    );

    // Host 成员函数: calcKernelMatrixSeriel（串行计算核矩阵）
    // 根据指定的 sigma 参数对输入的样例矩阵以串行的方式进行核矩阵的计算，
    // 并将结果存放在输出矩阵中
    __host__ int calcKernelMatrixSeriel(
            Matrix *inSample,        // 输入的样例矩阵
            Matrix *outKernelMatrix  // 输出的核矩阵
    );
};

#endif