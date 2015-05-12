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
	Matrix* matrix;			 // ������������
    Matrix* outMatrix;		 // ����˾���
    int errcode;			 // ������
    int cylceTimes = 1;      // ѭ������

    time_t start_1,end_1,time_1; // ʹ�� ctime �ļ�ʱ��ʽ

     // ����������
    KernelMatrix km = KernelMatrix();

    // ��������
    MatrixBasicOp::newMatrix(&matrix);
    MatrixBasicOp::newMatrix(&outMatrix);

    // ���ļ��ж�ȡ����
  //  MatrixBasicOp::readFromFile("a1_raw.csv",18,100,matrix);
    MatrixBasicOp::readFromFile("a2_raw.csv",18,100,matrix);

    // �ڵ�ǰ�豸��������
    MatrixBasicOp::makeAtCurrentDevice(outMatrix, 100,100);

    // �������˴�������
    // MatrixBasicOp::makeAtHost(outMatrix, 1000,1000);
    
    // ��������󿽱�����ǰ�豸��
     MatrixBasicOp::copyToCurrentDevice(matrix);

    // GPU��ʼ��ʱ
    cudaEventRecord(start, 0);

	// CPU ��ʼ��ʱ
    // DWORD start_time = GetTickCount();  
    //start_1 = clock();

    // ���ú�������˾���
    for (int i = 0; i < cylceTimes; i++) 
    {
         errcode = km.calcKernelMatrix(matrix, outMatrix);
        //errcode = km.calcKernelMatrixSeriel(matrix, outMatrix);
    }

	// CPU ����ͳ��ʱ��
    //end_1 = clock();
    //DWORD end_time = GetTickCount();
    //DWORD  used_time = end_time - start_time;
    // cout << used_time << endl;
    //cout << end_1 - start_1 << endl;
    
    // GPU ������ʱ    
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
    


    // ������������������ڴ�
    MatrixBasicOp::copyToHost(outMatrix);
    
    // ������д�뵽�ļ���
    //MatrixBasicOp::writeToFile("output1.csv",outMatrix);
    MatrixBasicOp::writeToFile("output2.csv",outMatrix);
}

