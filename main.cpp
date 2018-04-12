#include "tfutil.h"

int loadTFModelFile(const char* fileName, std::vector<std::vector<std::string>>& net, int inputDim[4], std::string outputFolder)
{
    //verify the version of protobuf library.
    GOOGLE_PROTOBUF_VERIFY_VERSION;

    //read the TF model and initializing graph
    std:: cout<<"Reading the binary file from : "<< fileName << std::endl;
    std::ifstream input(fileName, std::ios::in | std::ios::binary);

    tensorflow::GraphDef graph;
    if (!input.is_open())
    {
        std::cout << "cant open :" << fileName << std::endl;
        return -1;
    }

    bool isSuccess = graph.ParseFromIstream(&input);
    if(isSuccess) {
        std::cout << "TF Model Read Successful" << std::endl;
        int node_param_size = graph.node_size();
        if(node_param_size > 0) {
            TFUtil tf_util = TFUtil(&graph);
            tf_util.PrintModel();

            //Write GDF
            const char * tensorType = "VX_TYPE_FLOAT32";
            const char * convertPolicy = "VX_CONVERT_POLICY_SATURATE";
            const char * roundPolicy = "VX_ROUND_POLICY_TO_NEAREST_EVEN";
            int fixedPointPosition = 0;
            bool isVirtualEnabled = true;
            bool bFuseScaleWithBatchNorm = true;
            tf_util.GenerateGDF(inputDim, outputFolder, tensorType, convertPolicy, roundPolicy, fixedPointPosition, isVirtualEnabled, bFuseScaleWithBatchNorm);
        }
        else {
            std::cerr << "ERROR: [Unsupported caffemodel] please upgrade this caffemodel, currently uses deprecated V1LayerParameters." << std::endl;
            return -1;
        }
    }
    else {
        std::cerr << "Tensorflow Model Read Failed" << std::endl;
    }


    input.close();
    return 0;
}

int main(int argc, char* argv[])
{
    const char *fileName = "frozen_inference_graph.pb"; //"output_graph.pb"; //"frozen_cityscapes.pb";

    std::vector<std::vector<std::string>> net;
    int inputDim[4] = { 1, 3, 299, 299 }; //[Batch Size, Channel, Height, Width]
    std::string outputFolder = ".";


    // make sure that weights and bias folder are created
    std::string dir = outputFolder + "/weights";
    mkdir(dir.c_str(), 0777);
    dir = outputFolder + "/bias";
    mkdir(dir.c_str(), 0777);
    // load TF model
    if(loadTFModelFile(fileName, net, inputDim, outputFolder) < 0) {
        return -1;
    }

    return 0;
}
