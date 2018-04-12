#ifndef TFUTIL_H
#define TFUTIL_H

#include <fstream>
#include <utility>
#include <vector>
#include <iomanip>

#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"



#include "datainfo.h"
#include <sys/stat.h>

#define BIAS    "Bias"
#define KERNEL  "Kernel"
#define error(...) printf("ERROR: " __VA_ARGS__), exit(1)
#define info(...)  printf("OK: " __VA_ARGS__)

using namespace tensorflow;
class TFUtil
{
public:
    explicit TFUtil(GraphDef* graph);
    NodeDef* GetNode(const std::string& name) const;
    const std::vector<NodeDef *> &GetInputs(const std::string& node_name) const;
    const std::vector<NodeDef*> &GetOutputs(const std::string &node_name) const;
    void ParseTFModel(int inputDim[], std::string outputFolder);
    int CalculateTensorDim(int inputDim[4], std::map<std::string, std::vector<int>>& tensorMap);
    int CalculateTensorDim(int inputDim[4], std::map<std::string, std::vector<int>>& tensorMap, std::vector<std::string> node);
    void PrintModel();

    int GenerateGDF(int inputDim[4], std::string outputFolder, const char *tensorType, const char *convertPolicy, const char *roundPolicy, int fixedPointPosition = 0,
                    bool isVirtualEnabled = true,  bool bFuseScaleWithBatchNorm = true );
    void WriteGDF(std::ostream& ofsGDF, std::vector<std::vector<std::string> > &net, std::map<std::string,std::vector<int>>& tensorMap,
        std::string tensorType, int fixedPointPosition, std::string convertPolicy,
        std::string roundPolicy, bool isVirtualEnabled, std::string outputFolder,
        bool bFuseScaleLayer);

    void GenerateCode(std::ostream& ofsCodeH, std::ostream& ofsCodeC,
        std::ofstream& ofsCodeM, std::ofstream& ofsCodeA,
        std::ofstream& ofsCodeD, std::vector<std::vector<std::string>>& net,
        std::map<std::string,std::vector<int>>& tensorMap,
        std::string tensorType, int fixedPointPosition,
        std::string convertPolicy, std::string roundPolicy,
        bool isVirtualEnabled, std::string outputFolder,
        bool bFuseScaleLayer);

private:
    GraphDef* mGraph;
    std::set<NodeDef*> mEmpty_set;
    std::vector<NodeDef*> mEmpty_vector;
    std::map<string, NodeDef*> mNodes;
    std::map<string, std::vector<NodeDef*>> mInputs;
    std::map<string, std::vector<NodeDef*>> mOutputs;
    std::vector<NodeDef*> GetNextNode(NodeDef& node, const std::string& type);
    std::vector<std::vector<std::string>> mNet;
    std::map<std::string,std::vector<int>> mTensorMap;

    //Model generator parameters
    const char* mTensorType;
    const char* mConvertPolicy;
    const char* mRoundPolicy;
    int mFixedPointPosition;
    bool mIsVirtualEnabled;
    bool mBFuseScaleWithBatchNorm;

    void UpdateIO(const std::string& node_name,const std::string& old_output_name, const std::string& new_output_name);
    void UpdateInput(const std::string& node_name, const std::string& old_input_name, const std::string& new_input_name);
    void UpdateOutput(const std::string& node_name,const std::string& old_output_name, const std::string& new_output_name);
    void UpdateUnsupportedNodes(const std::string& node_name, std::string root_name = "");
    void PrintAttr(const AttrValue& value);
    bool HasTensor(NodeDef &node);
    void GenerateVXC(std::string outputFolder = ".");

    void GetLayerParams(cnn::DataInfo& layer, std::string& params);
    void MapConvLayer(NodeDef& node, std::string outputFolder, cnn::DataInfo& params, NodeDef &hasPad);
    bool GetKernelParams(const std::string &fieldName, cnn::DataInfo *dInfo, std::vector<std::string> keys);
    void FormatFileName(std::string& str, const std::string& from, const std::string& to);
    bool GetKernelDataAndShape(const std::string& fieldName, cnn::DataInfo* dInfo, std::string dataType, std::string key="value");
    void CreateLayerFiles(std:: string layer_name, std::string outputFolder, cnn::DataInfo& params, std::string key);
    void MapActLayer(NodeDef &node);
    void MapPoolLayer(NodeDef &node, cnn::DataInfo &params);
    void MapMulLayer(NodeDef &node, std::string outputFolder, cnn::DataInfo &params);
    void MapUpNNLayer(NodeDef &node, cnn::DataInfo &params);

    bool isSupportedLayer(std::string node_name);
    std::string getIdentifierName(const std::string name);

    void WriteVXCode(std::ostream& ofsCodeC, std::vector<std::vector<std::string>>& net,
        std::map<std::string,std::vector<int>>& tensorMap, std::string tensorType,
        int fixedPosition, std::string convertPolicy, std::string roundPolicy,
        bool isVirtualEnabled, bool bFuseScaleLayer, std::string outputFolder, std::string codeType);

    void GenerateCopyTensorCode(std::ostream& ofsCodeC);


    std::map<std::string,std::string> LAYER_DESCRIPTORS = {
        //OVX Node Types & TF ops
        {"Placeholder"             ,"Input"},
        {"ArgMax"                  ,"ArgMax"},
        {"concat"                  ,"Concat"},
        {"ConcatV2"                ,"Concat"},
        {"Conv2D"                  ,"Convolution"},
        {"#"                       ,"Deconvolution"},
        {"#"                       ,"Data"},
        {"#"                       ,"Dropout"},
        {"#"                       ,"Flatten"},
        {"MatMul"                  ,"InnerProduct"},
        {"MaxPool"                 ,"Pooling"},
        {"AvgPool"                 ,"Pooling"},
        {"Relu"                    ,"ReLU"},
        {"#"                       ,"Sigmoid"},
        {"Softmax"                 ,"Softmax"},
        {"ResizeNearestNeighbor"   ,"Upsamle"}
    };
};



#endif // TFUTIL_H
