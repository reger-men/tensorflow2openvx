#pragma once
#include <iostream>
#include <assert.h>

#define PAD_VALID    0
#define PAD_SAME     1
#define MAXPOOL      0
#define AVGPOOL      1
#define NCHW         0
#define NHWC         1
namespace cnn
{
    struct DataInfo
    {
        ~DataInfo();

        void deallocate(std::string key);
        void allocate(std::string key, size_t data_size);
        bool isValid(std::string key) const;
        void setVal(std::string key, float val = 1.f);
        void print(std::string key, int n = 25) const;
        void NHWC2NCHW(size_t data_size, const float* inp_data);

        //Layer data
        std::string mLayerName{""};
        std::string mLayerType{""};


        //Kernel & Bias data
        size_t    mKernelDataSize;
        size_t    mBiasDataSize;
        char*     mpKernelData{ NULL };
        char*     mpBiasData{ NULL };
        bool      isKernelValid = false;
        bool      isBiasValid = false;

        //Input parameters (Kernel input shape
        int       mInput_h{ -1 };
        int       mInput_w{ -1 };

        //Kernel parameters
        int         mPadType{PAD_VALID};
        int         mPoolType{MAXPOOL};
        int         mN{ -1 };
        int         mC{ -1 };
        int         mkernel_h{ -1 };
        int         mkernel_w{ -1 };
        int         mpad_h{ -1 };
        int         mpad_w{ -1 };
        int         mstride_h{ -1 };
        int         mstride_w{ -1 };
        int         mbias_term{ -1 };
        int         mdilation_h{ -1 };
        int         mdilation_w{ -1 };
        int         mgroup{ -1 }; //not supported yet in TF
        int         mData_Format {NHWC};
    };
}
