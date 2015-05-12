#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include "Matrix.h"
#include "KernelMatrix.h"
#include <windows.h>
#include <ctime>

using namespace std;

int main()
{
    cudaEvent_t start, stop;
    float rumtime;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
	Matrix* matrix;			 // 输入样本矩阵
    Matrix* outMatrix;		 // 输出核矩阵
    int errcode;			 // 错误码
    int cylceTimes = 1;      // 循环次数

    time_t start_1,end_1,time_1; // 使用 ctime 的计时方式

     // 创建操作类
    KernelMatrix km = KernelMatrix();

    // 创建矩阵
    MatrixBasicOp::newMatrix(&matrix);
    MatrixBasicOp::newMatrix(&outMatrix);

    // 从文件中读取矩阵
  //  MatrixBasicOp::readFromFile("a1_raw.csv",18,100,matrix);
    MatrixBasicOp::readFromFile("a2_raw.csv",18,100,matrix);

    // 在当前设备创建矩阵
    MatrixBasicOp::makeAtCurrentDevice(outMatrix, 100,100);

    // 在主机端创建矩阵
    // MatrixBasicOp::makeAtHost(outMatrix, 1000,1000);
    
    // 将输入矩阵拷贝到当前设备上
     MatrixBasicOp::copyToCurrentDevice(matrix);

    // GPU开始计时
    cudaEventRecord(start, 0);

	// CPU 开始计时
    // DWORD start_time = GetTickCount();  
    //start_1 = clock();

    // 调用函数计算核矩阵
    for (int i = 0; i < cylceTimes; i++) 
    {
         errcode = km.calcKernelMatrix(matrix, outMatrix);
        //errcode = km.calcKernelMatrixSeriel(matrix, outMatrix);
    }

	// CPU 结束统计时间
    //end_1 = clock();
    //DWORD end_time = GetTickCount();
    //DWORD  used_time = end_time - start_time;
    // cout << used_time << endl;
    //cout << end_1 - start_1 << endl;
    
    // GPU 结束计时    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&rumtime, start, stop);

    if (errcode != NO_ERROR) {
        cout << "kernel matrix error:" << errcode << endl;
    }
    else {
        cout << "kernel matrix success" << endl;        
        cout << "the time is " << (rumtime) / 100 << " ms" << endl;
    }
    


    // 将结果拷贝到主机端内存
    MatrixBasicOp::copyToHost(outMatrix);
    
    // 将矩阵写入到文件中
    //MatrixBasicOp::writeToFile("output1.csv",outMatrix);
    MatrixBasicOp::writeToFile("output2.csv",outMatrix);
}

