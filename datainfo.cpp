#include <algorithm>
#include "datainfo.h"

cnn::DataInfo::~DataInfo()
{
    deallocate("Kernel");
    deallocate("Bias");
}

void cnn::DataInfo::deallocate(std::string key)
{
    if(key == "Kernel"){
        if (mpKernelData)
        {
            delete mpKernelData;
            mKernelDataSize = 0;
        }
    }else if(key == "Bias"){
        if (mpBiasData)
        {
            delete mpBiasData;
            mBiasDataSize = 0;
        }
    }else{
        std::cerr << "Key not defined!" << std::endl;
        return;
    }
}

bool cnn::DataInfo::isValid(std::string key) const
{
    if(key == "Kernel"){
        return isKernelValid;
    }else if(key == "Bias"){
        return isBiasValid;
    }else{
        std::cerr << "Key not defined!" << std::endl;
        return 0;
    }
}

void cnn::DataInfo::setVal(std::string key, float val /*= 1.f*/)
{
    if(key == "Kernel"){
        float * local_data = (float *)mpKernelData;
        for (int i = 0; i < mKernelDataSize / 4; i++)
        {
            local_data[i] = val;
        }
    }else if(key == "Bias"){
        float * local_data = (float *)mpBiasData;
        for (int i = 0; i < mBiasDataSize / 4; i++)
        {
            local_data[i] = val;
        }
    }else{
        std::cerr << "Key not defined!" << std::endl;
        return;
    }
}

void cnn::DataInfo::print(std::string key, int n /*= 25*/) const
{
    if (!isValid(key))
    {
        std::cerr << "Not initialized or requested output size is bigger then data size" << std::endl;
        return;
    }

    std::cout << "Coefficients begin:" << std::endl;
    std::cout.precision(17);
    if(key == "Kernel"){
        float * local_data = (float *)mpKernelData;
        int toPrint = std::min<int>(n, mKernelDataSize / 4);
        for (int i = 0; i < toPrint; i++)
        {
            std::cout << " " << local_data[i];
        }
        std::cout << std::endl;
    }else if(key == "Bias"){
        float * local_data = (float *)mpBiasData;
        int toPrint = std::min<int>(n, mBiasDataSize / 4);
        for (int i = 0; i < toPrint; i++)
        {
            std::cout << " " << local_data[i];
        }
        std::cout << std::endl;
    }else{
        std::cerr << "Key not defined!" << std::endl;
        return;
    }

}

//In TensorFlow's NHWC ordering (unlike Caffe's and MIopen NCHW).
//NHWC: [batch, in_height, in_width, in_channels]
//NCHW: [batch, in_channels, in_height, in_width]
void cnn::DataInfo::NHWC2NCHW(size_t data_size, const float *inp_data)
{
    int nSize = this->mN, cSize = this->mC;
    int hSize = (this->mkernel_h != -1) ? this->mkernel_h : this->mInput_h;
    int wSize = (this->mkernel_w != -1) ? this->mkernel_w : this->mInput_w;

    float * out_data = (float *)this->mpKernelData;

    bool dimsMatch = (static_cast<int>(data_size / 4) == (nSize * cSize * hSize * wSize));
    assert(dimsMatch && "data_size and dimensions ( n, c, h, w) do not fit together!");

    if (!dimsMatch)
    {
        std::cerr << "data_size and dimensions ( n, c, h, w) do not fit together! Aborting operation" << std::endl;
        return;
    }

    for (int n = 0; n < nSize; n++)
    {
        for (int c = 0; c < cSize; c++)
        {
            for (int h = 0; h < hSize; h++)
            {
                for (int w = 0; w < wSize; w++)
                {
                    //HWCN
                    size_t inpIdx = h * wSize * cSize * nSize + w * cSize * nSize + c * nSize + n;
                    //NCHW
                    size_t outIdx = n * cSize * hSize * wSize + c * hSize * wSize + h * wSize + w;
                    //std::cout << inpIdx << " v " << inp_data[inpIdx] << " to " << outIdx << " v " << out_data[outIdx] << std::endl;
                    out_data[outIdx] = inp_data[inpIdx];
                }
            }
        }
    }
}

void cnn::DataInfo::allocate(std::string key, size_t data_size)
{
    if(key == "Kernel"){
        if (mpKernelData != NULL)
            deallocate(key);
        this->mKernelDataSize = data_size;
        mpKernelData = new char[data_size];
        isKernelValid = true;
    }else if(key =="Bias"){
        if (mpBiasData != NULL)
            deallocate(key);
        this->mBiasDataSize = data_size;
        mpBiasData = new char[data_size];
        isBiasValid = true;
    }else{
        std::cerr << "Key not defined!" << std::endl;
        return;
    }
}

