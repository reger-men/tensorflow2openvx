#include "tfutil.h"
#include "tensorflow/core/framework/tensor_util.h"

//TFUtil::~TFUtil()
//{
//}

TFUtil::TFUtil(GraphDef *graph): mGraph(graph)
{
    for (int i = 0; i < mGraph->node_size(); i++) {
        auto node = mGraph->mutable_node(i);
        auto rslt = mNodes.insert(std::make_pair(node->name(), node));
        // Check that the graph doesn't contain multiple nodes with the same name.
        //CHECK(rslt.second);
        for (const auto& input : node->input()) {
            //mOutputs[input].insert(mNodes[node->name()]);
            mOutputs[input].push_back(mNodes[node->name()]);
            mInputs[node->name()].push_back(mNodes[input]);
        }
    }
}

NodeDef *TFUtil::GetNode(const std::string &node_name) const
{
    auto it = mNodes.find(node_name);
    if (it == mNodes.end()) {
        return nullptr;
    }
    return it->second;
}

const std::vector<NodeDef *> &TFUtil::GetOutputs(const std::string &node_name) const
{
    auto it = mOutputs.find(node_name);
    if (it == mOutputs.end()) {
        return mEmpty_vector;
    }
    return it->second;
}

const std::vector<NodeDef *> &TFUtil::GetInputs(const std::string &node_name) const
{
    auto it = mInputs.find(node_name);
    if (it == mInputs.end()) {
        return mEmpty_vector;
    }
    return it->second;
}

std::vector<NodeDef*> TFUtil::GetNextNode(NodeDef& node, const std::string& type) {
    std::vector<NodeDef*> outputs;
    for (NodeDef* out: GetOutputs(node.name())){
        if(out->op() == type){
            outputs.push_back(out);
        }
    }
    return outputs;
}

void TFUtil::UpdateInput(const std::string& node_name, const std::string& old_input_name,
                         const std::string& new_input_name) {
    std::vector<NodeDef*>& inputs = mInputs[node_name];
    int pos = distance(inputs.begin(), find(inputs.begin(), inputs.end(), mNodes[old_input_name]));
    if(pos < inputs.size()) {
        inputs[pos] = mNodes[new_input_name];
    }
}

void TFUtil::UpdateOutput(const std::string& node_name,const std::string& old_output_name,
                          const std::string& new_output_name) {
    std::vector<NodeDef*>& outputs = mOutputs[node_name];
    int pos = distance(outputs.begin(), find(outputs.begin(), outputs.end(), mNodes[old_output_name]));
    if(pos < outputs.size()) {
        outputs[pos] = mNodes[new_output_name];
    }
}

void TFUtil::UpdateIO(const std::string& node_name,const std::string& old_output_name,
                      const std::string& new_output_name) {
    UpdateOutput(node_name, old_output_name, new_output_name);
    UpdateInput(new_output_name, old_output_name, node_name);
}

void TFUtil::UpdateUnsupportedNodes(const std::string& node_name, std::string root_name){
    root_name = (root_name.empty()) ? node_name : root_name;
    for (NodeDef* out: GetOutputs(node_name)){
        if(!isSupportedLayer(out->name())){
            UpdateUnsupportedNodes(out->name(), root_name);
        }else{
            UpdateIO(root_name, node_name, out->name());
            break;
        }
    }
}

void TFUtil::PrintModel() {
    for (int i = 0; i < mGraph->node_size(); i++) {
        std::cout << "node name: " << mGraph->node(i).name() << "\n";
        std::cout << "node op: " << mGraph->node(i).op() << "\n";
        std::cout << "node inputs: ";

        for (int j = 0; j < mGraph->node(i).input_size(); j++) {
            std::cout << mGraph->node(i).input(j) << " ";
        }

        std::cout << "\nnode attr:\n";
        for (auto& name : mGraph->node(i).attr()) {
            std::cout << " name.first " << name.first << ": ";
            PrintAttr(name.second);
            std::cout << "\n";
        }
        std::cout << "\n\n";
    }
}

bool TFUtil::HasTensor(NodeDef &node){
    bool hasTensor = false;
    for (auto& name : node.attr()) {
        hasTensor = (name.second.value_case() == tensorflow::AttrValue::kTensor);
        if (hasTensor)
            return (name.second.value_case() == tensorflow::AttrValue::kTensor);
    }
    return false;
}

void TFUtil::PrintAttr(const AttrValue& value) {
    switch (value.value_case()) {
    case tensorflow::AttrValue::kS:
        std::cout << "<s>"<< value.s();
        break;

    case tensorflow::AttrValue::kI:
        std::cout << "<i>"<< value.i();
        break;

    case tensorflow::AttrValue::kF:
        std::cout << "<f>"<< value.f();
        break;

    case tensorflow::AttrValue::kB:
        std::cout << "<b> "<< value.b();
        break;

    case tensorflow::AttrValue::kType:
        std::cout << "<type> "<< DataType_Name(value.type());
        break;

    case tensorflow::AttrValue::kShape: {
        std::cout << "<shape> {";
        std::string str = "";
        for (int i = 0; i < value.shape().dim_size(); i++) {
            str += std::to_string(value.shape().dim(i).size()) + ",";
        }
        str = str.substr(0, str.length() - 1);
        std::cout << str << "}";
    } break;

    case tensorflow::AttrValue::kTensor:
        std::cout << "<tensor> ";
        break;

    case tensorflow::AttrValue::kList:
        std::cout << "<list> ";
        break;

    case tensorflow::AttrValue::kFunc:
        std::cout << "<func> ";
        break;

    case tensorflow::AttrValue::kPlaceholder:
        std::cout << "<placeholder> ";
        break;

    case tensorflow::AttrValue::VALUE_NOT_SET:
        std::cout << "<not_set> ";
        break;
    }
}

void TFUtil::ParseTFModel(int inputDim[], std::string outputFolder)
{
    if(mGraph->versions().producer())
        std::cout << "Fetching the weights for graph version (producer): " << mGraph->versions().producer() << std::endl;
    std::cout << "#Nodes                                           : " << mGraph->node_size() << "\n" << std::endl;

    std::string input_node_name;
    std::map<std::string,std::vector<int>> _tensorMap;
    std::map<std::string,std::vector<int>> outputDims;

    bool isLayer = false;
    NodeDef hasPad;



    //extract layer information.
    for(NodeDef TFnode: mGraph->node()) {
        /*if(GetOutputs(TFnode.name()).size() == 0)
            continue;*/

        isLayer = false;
        std::cout << TFnode.name() << std::endl;

        cnn::DataInfo layer; // Kernel data structure
        for (NodeDef* in: GetInputs(TFnode.name())){
            if (in->name().find("/read") == std::string::npos && isSupportedLayer(in->name())){
                input_node_name = in->name();
                if(outputDims.find(input_node_name) != outputDims.end()) {
                    layer.mN = outputDims[input_node_name][0];
                    layer.mC = outputDims[input_node_name][1];
                    layer.mInput_h = outputDims[input_node_name][2];
                    layer.mInput_w = outputDims[input_node_name][3];
                }
                break;
            }
        }




        //Check layer name.
        if(TFnode.op() == "Placeholder" ) {
            if (TFnode.name().find("input") != std::string::npos || TFnode.name().find("Placeholder") != std::string::npos){
                input_node_name = TFnode.name();
                //set the input dim.
                //TODO: get the input shape from the TF Model.
                outputDims[input_node_name] = {inputDim[0], inputDim[1], inputDim[2], inputDim[3]};
            }
        }else if(TFnode.op() == "Conv2D" ) {
            //save layer data (weight and bias)
            layer.mLayerName = TFnode.name();
            layer.mLayerType = LAYER_DESCRIPTORS[TFnode.op()];
            MapConvLayer(TFnode, outputFolder, layer, hasPad);
            isLayer = true;
            hasPad.Clear();
        }else if(TFnode.op() == "Relu" || TFnode.op() == "Softmax") {
            //map activation layer
            layer.mLayerName = TFnode.name();
            layer.mLayerType = LAYER_DESCRIPTORS[TFnode.op()];
            MapActLayer(TFnode);
            isLayer = true;
        }else if(TFnode.op() == "MaxPool" || TFnode.op() == "AvgPool") {
            //save layer data (weight and bias)
            layer.mLayerName = TFnode.name();
            layer.mLayerType = LAYER_DESCRIPTORS[TFnode.op()];
            layer.mPoolType = (TFnode.op() == "MaxPool") ? MAXPOOL : AVGPOOL;
            MapPoolLayer(TFnode, layer);
            isLayer = true;
        }else if(TFnode.op() == "Pad" ) {
            //fusion Pad with the next convolution.
            for (NodeDef* out: GetOutputs(TFnode.name())){
                if(out->op() == "Conv2D"){
                    hasPad = TFnode;
                    break;
                }
            }
            continue;
        }else if(TFnode.op() == "ConcatV2" ) {
            layer.mLayerName = TFnode.name();
            layer.mLayerType = LAYER_DESCRIPTORS[TFnode.op()];
            isLayer = true;
        }else if(TFnode.op() == "ResizeNearestNeighbor" ) {
            layer.mLayerName = TFnode.name();
            layer.mLayerType = LAYER_DESCRIPTORS[TFnode.op()];
            MapUpNNLayer(TFnode, layer);
            isLayer = true;
        }else if(TFnode.op() == "MatMul" ) {
            //save layer data (weight and bias)
            assert(TFnode.input_size() == 2);
            layer.mLayerName = TFnode.name();
            layer.mLayerType = LAYER_DESCRIPTORS[TFnode.op()];
            MapMulLayer(TFnode, outputFolder, layer);
            isLayer = true;
        }


        //2.get layer information and add to net
        if(isLayer){
            if(!isSupportedLayer(TFnode.input(0))){
                UpdateInput(TFnode.name(), TFnode.input(0), input_node_name);
            }

            std::vector<std::string> node;
            std::string params;
            GetLayerParams(layer, params);
            node.push_back(layer.mLayerType); //ADD TYPE
            node.push_back(params);           //ADD PARAMS

            //ADD OUTPUT
            if(GetOutputs(TFnode.name()) != this->mEmpty_vector){
                for (NodeDef* out: GetOutputs(TFnode.name())){
                    if(isSupportedLayer(out->name())){
                        node.push_back(out->name());
                        break;
                    }
                }
            }
            //make sure to insert an output node.
            if(node.size() < 3)
                node.push_back(TFnode.name());

            node.push_back(layer.mLayerName); //ADD NAME

            //ADD INPUT's
            for (NodeDef* in: GetInputs(TFnode.name())){
                if (in->name().find("/read") == std::string::npos && isSupportedLayer(in->name())){
                    node.push_back(in->name());
                    if(layer.mLayerType != "Concat" && in->name().find("/axis") == std::string::npos)
                        break;
                }
            }
            //If last layer is softmax, change it to SoftmaxWithLoss and add new input argument
            //TODO: fix this
            if(TFnode.op() == "Softmax" && GetOutputs(TFnode.name()).size() == 0){
                node[0] = "SoftmaxWithLoss";
                node.push_back(TFnode.name());
            }

            mNet.push_back(node);
            // update output name with layer name
            input_node_name = layer.mLayerName;

            //update output dimension.
            int _inputDim[4] = {layer.mN , layer.mC, layer.mInput_h, layer.mInput_w};
            if(CalculateTensorDim(_inputDim, _tensorMap, node) == 0){
                auto&& it = _tensorMap.find(input_node_name);
                if(it != _tensorMap.end()) {
                    auto&& idim = it->second;
                    outputDims[input_node_name] = {idim[0], idim[1], idim[2], idim[3]}; //[Batch Size, Channel, Height, Width]
                }
            }
        }
    }
}

int TFUtil::CalculateTensorDim(int inputDim[], std::map<std::string, std::vector<int> > &tensorMap)
{
    tensorMap[mNet[0][4]] = std::vector<int>{inputDim[0], inputDim[1], inputDim[2], inputDim[3]};

    for(auto& node : mNet) {
        auto&& type = node[0];
        auto&& params = node[1];
        auto&& output = node[3];
        auto&& input = node[4];
        auto&& it = tensorMap.find(input);
        if(it == tensorMap.end()) {
            error("CalculateTensorDim: no dims found for %s\n", input.c_str());
        }

        auto&& idim = it->second;
        int n = idim[0], c = idim[1], H = idim[2], W = idim[3];
        int k = c, h = H, w = W;

        if (n < 1 || c < 1 || H < 1 || W < 1)
            error("calculateTensorDim: got invalid dim %dx%dx%dx%d for %s\n", n, c, H, W, input.c_str());

        if(type == "Convolution") {
            std::stringstream ss(params);
            int kernel_w, kernel_h, stride_w, stride_h, pad_w, pad_h, dilation_w, dilation_h, bias_term, group, pad_type;
            ss >> k >> kernel_w >> kernel_h >> stride_w >> stride_h >> pad_w >> pad_h >> dilation_w >> dilation_h >> bias_term >> group >> pad_type;
            if(pad_type == PAD_SAME){
                w = (W / stride_w);
                h = (H / stride_h);
            }else{
                w = ((W + 2 * pad_w - kernel_w - (kernel_w - 1) * (dilation_w - 1)) / stride_w) + 1;
                h = ((H + 2 * pad_h - kernel_h - (kernel_h - 1) * (dilation_h - 1)) / stride_h) + 1;
            }
            tensorMap[output + "_W"] = std::vector<int>{k, c, kernel_h, kernel_w};
            if(bias_term) {
                tensorMap[output + "_B"] = std::vector<int>{k};
            }
        }
        else if(type == "Deconvolution") {
            std::stringstream ss(params);
            int kernel_w, kernel_h, stride_w, stride_h, pad_w, pad_h, dilation_w, dilation_h, bias_term, group, pad_type;
            ss >> k >> kernel_w >> kernel_h >> stride_w >> stride_h >> pad_w >> pad_h >> dilation_w >> dilation_h >> bias_term >> group >> pad_type;
            if(pad_type == PAD_SAME){
                w = (W / stride_w);
                h = (H / stride_h);
            }else{
                w = stride_w * (W - 1) + dilation_w * (kernel_w - 1) + 1 - ( 2* pad_w );
                h = stride_h * (H - 1) + dilation_h * (kernel_h - 1) + 1 - ( 2* pad_h );
            }
            tensorMap[output + "_W"] = std::vector<int>{k, c, kernel_h, kernel_w};
            if(bias_term) {
                tensorMap[output + "_B"] = std::vector<int>{k};
            }
        }
        else if(type == "Pooling") {
            std::stringstream ss(params);
            int kernel_w, kernel_h, stride_w, stride_h, pad_w, pad_h, pool, global_pooling, pad_type;
            ss >> kernel_w >> kernel_h >> stride_w >> stride_h >> pad_w >> pad_h >> pool >> global_pooling >> pad_type;
            if(global_pooling) {
                // Compute kernel_w and kernel_h and write back the params for the GDF and C-code gen
                kernel_h = H;
                kernel_w = W;
                pad_h = pad_w = 0;
                stride_h = stride_w = 1;
                params =        std::to_string(kernel_w)
                        + " " + std::to_string(kernel_h)
                        + " " + std::to_string(stride_w)
                        + " " + std::to_string(stride_h)
                        + " " + std::to_string(pad_w)
                        + " " + std::to_string(pad_h)
                        + " " + std::to_string(pool)
                        + " " + std::to_string(global_pooling);
            }
            if(pad_type == PAD_SAME){
                h = ceil(float(H) / float(stride_h));
                w = ceil(float(W) / float(stride_w));
            }else{
                h = ceil(float(H - kernel_h + 1) / float(stride_h));
                w = ceil(float(W - kernel_w + 1) / float(stride_w));
            }
            //w = static_cast<int>(ceil( static_cast<float> (W + 2 * pad_w + stride_w - kernel_w)/ stride_w));
            //h = static_cast<int>(ceil( static_cast<float> (H + 2 * pad_h + stride_h - kernel_h)/ stride_h));
            if(pad_h > 0) if((h-1)*stride_h >= (H+pad_h)) h=h-1;
            if(pad_w > 0) if((w-1)*stride_w >= (W+pad_w)) w=w-1;
        }
        else if(type == "InnerProduct") {
            std::stringstream ss(params);
            ss >> k;
            w = 1;
            h = 1;
            tensorMap[output + "_W"] = std::vector<int>{k, c, H, W};
        }
        else if(type == "Concat") {
            for(int i = 5; i < node.size(); i++) {
                auto&& dim = tensorMap[node[i]];
                k += dim[1];
                if(dim[2] != H || dim[3] != W)
                    error("calculateTensorDim: Concat: got invalid dim %dx%dx%dx%d for %s (should be %dx*x%dx%d)\n", dim[0], dim[1], dim[2], dim[3], node[i].c_str(), n, H, W);
            }
        }
        else if(type == "SoftmaxWithLoss") {
            output = node[5];
        }
        else if (type == "BatchNorm") {
            std::stringstream ss(params);
            int use_global_stats;
            float eps;
            ss >> eps >> use_global_stats;
            tensorMap[output + "_W"] = std::vector<int>{k};
            tensorMap[output + "_B"] = std::vector<int>{k};
        }
        else if(type == "Scale") {
            std::stringstream ss(params);
            int bias_term;
            ss >> bias_term;
            tensorMap[output + "_W"] = std::vector<int>{k};
            if(bias_term) {
                tensorMap[output + "_B"] = std::vector<int>{k};
            }
        }
        else if(type == "Upsamle") {
            std::stringstream ss(params);
            int kernel_w, kernel_h;
            ss >> kernel_w >> kernel_h;
            w = (W * kernel_w);
            h = (H * kernel_h);
        }

        tensorMap[output] = std::vector<int>{n, k, h, w};
        if(n < 1 || k < 1 || h < 1 || w < 1)
            error("calculateTensorDim: got invalid dim %dx%dx%dx%d for %s\n", n, k, h, w, output.c_str());
    }
    return 0;
}

/* calculate dim for only one node*/
int TFUtil::CalculateTensorDim(int inputDim[], std::map<std::string, std::vector<int> > &tensorMap, std::vector<std::string> node)
{
    auto&& type = node[0];
    auto&& params = node[1];
    auto&& output = node[3];
    auto&& input = node[4];


    int n = inputDim[0], c = inputDim[1], H = inputDim[2], W = inputDim[3];
    int k = c, h = H, w = W;

    if (n < 1 || c < 1 || H < 1 || W < 1)
        error("calculateTensorDim: got invalid dim %dx%dx%dx%d for %s\n", n, c, H, W, input.c_str());

    if(type == "Convolution") {
        std::stringstream ss(params);
        int kernel_w, kernel_h, stride_w, stride_h, pad_w, pad_h, dilation_w, dilation_h, bias_term, group, pad_type;
        ss >> k >> kernel_w >> kernel_h >> stride_w >> stride_h >> pad_w >> pad_h >> dilation_w >> dilation_h >> bias_term >> group >> pad_type;
        if(pad_type == PAD_SAME){
            w = (W / stride_w);
            h = (H / stride_h);
        }else{
            w = ((W + 2 * pad_w - kernel_w - (kernel_w - 1) * (dilation_w - 1)) / stride_w) + 1;
            h = ((H + 2 * pad_h - kernel_h - (kernel_h - 1) * (dilation_h - 1)) / stride_h) + 1;
        }
        tensorMap[output + "_W"] = std::vector<int>{k, c, kernel_h, kernel_w};
        if(bias_term) {
            tensorMap[output + "_B"] = std::vector<int>{k};
        }
    }
    else if(type == "Deconvolution") {
        std::stringstream ss(params);
        int kernel_w, kernel_h, stride_w, stride_h, pad_w, pad_h, dilation_w, dilation_h, bias_term, group, pad_type;
        ss >> k >> kernel_w >> kernel_h >> stride_w >> stride_h >> pad_w >> pad_h >> dilation_w >> dilation_h >> bias_term >> group >> pad_type;
        if(pad_type == PAD_SAME){
            w = (W / stride_w);
            h = (H / stride_h);
        }else{
            w = stride_w * (W - 1) + dilation_w * (kernel_w - 1) + 1 - ( 2* pad_w );
            h = stride_h * (H - 1) + dilation_h * (kernel_h - 1) + 1 - ( 2* pad_h );
        }
        tensorMap[output + "_W"] = std::vector<int>{k, c, kernel_h, kernel_w};
        if(bias_term) {
            tensorMap[output + "_B"] = std::vector<int>{k};
        }
    }
    else if(type == "Pooling") {
        std::stringstream ss(params);
        int kernel_w, kernel_h, stride_w, stride_h, pad_w, pad_h, pool, global_pooling, pad_type;
        ss >> kernel_w >> kernel_h >> stride_w >> stride_h >> pad_w >> pad_h >> pool >> global_pooling >> pad_type;
        if(global_pooling) {
            // Compute kernel_w and kernel_h and write back the params for the GDF and C-code gen
            kernel_h = H;
            kernel_w = W;
            pad_h = pad_w = 0;
            stride_h = stride_w = 1;
            params =        std::to_string(kernel_w)
                    + " " + std::to_string(kernel_h)
                    + " " + std::to_string(stride_w)
                    + " " + std::to_string(stride_h)
                    + " " + std::to_string(pad_w)
                    + " " + std::to_string(pad_h)
                    + " " + std::to_string(pool)
                    + " " + std::to_string(global_pooling);
        }
        if(pad_type == PAD_SAME){
            h = ceil(float(H) / float(stride_h));
            w = ceil(float(W) / float(stride_w));
        }else{
            h = ceil(float(H - kernel_h + 1) / float(stride_h));
            w = ceil(float(W - kernel_w + 1) / float(stride_w));
        }
        //w = static_cast<int>(ceil( static_cast<float> (W + 2 * pad_w + stride_w - kernel_w)/ stride_w));
        //h = static_cast<int>(ceil( static_cast<float> (H + 2 * pad_h + stride_h - kernel_h)/ stride_h));
        if(pad_h > 0) if((h-1)*stride_h >= (H+pad_h)) h=h-1;
        if(pad_w > 0) if((w-1)*stride_w >= (W+pad_w)) w=w-1;
    }
    else if(type == "InnerProduct") {
        std::stringstream ss(params);
        ss >> k;
        w = 1;
        h = 1;
        tensorMap[output + "_W"] = std::vector<int>{k, c, H, W};
    }
    else if(type == "Concat") {
        for(int i = 5; i < node.size(); i++) {
            auto&& dim = tensorMap[node[i]];
            k += dim[1];
            if(dim[2] != H || dim[3] != W)
                error("calculateTensorDim: Concat: got invalid dim %dx%dx%dx%d for %s (should be %dx*x%dx%d)\n", dim[0], dim[1], dim[2], dim[3], node[i].c_str(), n, H, W);
        }
    }
    else if(type == "SoftmaxWithLoss") {
        output = node[5];
    }
    else if (type == "BatchNorm") {
        std::stringstream ss(params);
        int use_global_stats;
        float eps;
        ss >> eps >> use_global_stats;
        tensorMap[output + "_W"] = std::vector<int>{k};
        tensorMap[output + "_B"] = std::vector<int>{k};
    }
    else if(type == "Scale") {
        std::stringstream ss(params);
        int bias_term;
        ss >> bias_term;
        tensorMap[output + "_W"] = std::vector<int>{k};
        if(bias_term) {
            tensorMap[output + "_B"] = std::vector<int>{k};
        }
    }
    else if(type == "Upsamle") {
        std::stringstream ss(params);
        int kernel_w, kernel_h;
        ss >> kernel_w >> kernel_h;
        w = (W * kernel_w);
        h = (H * kernel_h);
    }

    tensorMap[output] = std::vector<int>{n, k, h, w};
    if(n < 1 || k < 1 || h < 1 || w < 1)
        error("calculateTensorDim: got invalid dim %dx%dx%dx%d for %s\n", n, k, h, w, output.c_str());

    return 0;
}

void TFUtil::GetLayerParams(cnn::DataInfo &layer, std::string &params)
{
    if(layer.mLayerType == "Convolution") {
        int pad_h = layer.mpad_h;
        int pad_w = layer.mpad_w;
        int stride_h = layer.mstride_h;
        int stride_w = layer.mstride_w;
        int kernel_h = layer.mkernel_h;
        int kernel_w = layer.mkernel_h;
        int k = layer.mN;
        int dilation_h = 1; //to do: get value from model
        int dilation_w = 1; //to do: get value from model
        int bias_term = layer.mbias_term;
        int group = 0;
        int pad_type = layer.mPadType;
        params =       std::to_string(k)
                + " " + std::to_string(kernel_w)
                + " " + std::to_string(kernel_h)
                + " " + std::to_string(stride_w)
                + " " + std::to_string(stride_h)
                + " " + std::to_string(pad_w)
                + " " + std::to_string(pad_h)
                + " " + std::to_string(dilation_w)
                + " " + std::to_string(dilation_h)
                + " " + std::to_string(bias_term)
                + " " + std::to_string(group)
                + " " + std::to_string(pad_type);
    }
    else if(layer.mLayerType == "Pooling") {
        int pad_h = layer.mpad_h;
        int pad_w = layer.mpad_w;
        int stride_h = layer.mstride_h;
        int stride_w = layer.mstride_w;
        int kernel_h = layer.mkernel_h;
        int kernel_w = layer.mkernel_w;
        int pool = (layer.mPoolType == MAXPOOL) ? MAXPOOL : AVGPOOL;
        int global_pooling = 0; //Default value
        int pad_type = layer.mPadType;
        params =       std::to_string(kernel_w)
                + " " + std::to_string(kernel_h)
                + " " + std::to_string(stride_w)
                + " " + std::to_string(stride_h)
                + " " + std::to_string(pad_w)
                + " " + std::to_string(pad_h)
                + " " + std::to_string(pool)
                + " " + std::to_string(global_pooling)
                + " " + std::to_string(pad_type);
    }
    else if(layer.mLayerType == "InnerProduct") {
        int k = layer.mN;
        int bias_term = layer.mbias_term;
        params = std::to_string(k) + " " + std::to_string(bias_term);
    }
    else if(layer.mLayerType == "LRN") {
        ///TODO
    }
    else if(layer.mLayerType == "BatchNorm") {
        ///TODO
    }
    else if(layer.mLayerType == "Scale") {
        ///TODO
    }
    else if(layer.mLayerType == "Dropout") {
        ///TODO
    }
    else if(layer.mLayerType == "Eltwise") {
        ///TODO
    }
    else if(layer.mLayerType == "Deconvolution") {
        ///TODO
    }
    else if(layer.mLayerType == "Upsamle") {
        int kernel_h = layer.mkernel_h;
        int kernel_w = layer.mkernel_h;
        params =       std::to_string(kernel_w)
                + " " + std::to_string(kernel_h);
    }
}

void TFUtil::MapConvLayer(NodeDef &node, std::string outputFolder, cnn::DataInfo &params, NodeDef &hasPad)
{
    bool status = false;
    std::string node_name = node.name();
    std::string new_output_name;

    //1.update padding value if hasPad.
    if(!hasPad.name().empty()){
        GetKernelDataAndShape(hasPad.input(1), &params, "");

        //Update Conv2D input & Pad parent output.
        node.set_input(0, hasPad.input(0));
        UpdateInput(node_name, node.input(0), hasPad.input(0));
    }


    //2.1.get weights data
    if(HasTensor(*GetNode(node.input(1)))){
        GetKernelDataAndShape(node.input(1), &params, KERNEL);
    }else{
        //try with the next node.
        GetKernelDataAndShape(GetNode(node.input(1))->input(0), &params, KERNEL);
    }

    //2.2.get conv params.
    std::vector<std::string> param_args = {"strides", "padding", "data_format"};
    GetKernelParams(node_name, &params, param_args);


    //2.3.Convert data format NHWC to NWCH
    if(params.mData_Format == NHWC){
        float * pbdata = new float[params.mKernelDataSize];
        memcpy(pbdata,params.mpKernelData,params.mKernelDataSize);
        params.NHWC2NCHW(params.mKernelDataSize, pbdata);
    }


    //3.pass layer name
    std:: string layer_name;
    if(!node.name().empty()) {
        layer_name = node.name();
        FormatFileName(layer_name,"/","_");
    }

    //4.create output files
    CreateLayerFiles(layer_name, outputFolder, params, KERNEL);

    //5.get bias data if exists.
    std::vector<NodeDef*> bias = GetNextNode(node,"BiasAdd");
    if(bias.size() == 0)
        bias = GetNextNode(node,"Add");

    if(bias.size() == 1){
        if(HasTensor(*GetNode((bias.at(0))->input(1)))){
            status = GetKernelDataAndShape((bias.at(0))->input(1), &params, BIAS);
        }else{
            //try with the next node.
            status = GetKernelDataAndShape((GetNode(bias.at(0)->input(1))->input(0)), &params, BIAS);
        }
        params.mbias_term = (status) ? 1: 0;

        //update conv output.
        for (NodeDef* out: GetOutputs(bias.at(0)->name())){
            new_output_name = out->name();
            break;
        }
    }

    //6.create output files
    CreateLayerFiles(layer_name, outputFolder, params, BIAS);

    //7.Update inputs outputs nodes.
    if(!new_output_name.empty())
        UpdateIO(node_name, bias.at(0)->name(), new_output_name);
}

void TFUtil::MapActLayer(NodeDef& node)
{
    std::string node_name = node.name();

    //1.Update outputs nodes.
    for (NodeDef* out: GetOutputs(node_name)){
        if(!isSupportedLayer(out->name())){
            UpdateUnsupportedNodes(node_name);
            break;
        }
    }
}

void TFUtil::MapPoolLayer(NodeDef &node, cnn::DataInfo &params)
{
    std::string node_name = node.name();

    //1.get pool params.
    std::vector<std::string> param_args = {"strides", "padding", "ksize"};
    GetKernelParams(node_name, &params, param_args);

    //2.pass layer name
    std:: string layer_name;
    if(!node.name().empty()) {
        layer_name = node.name();
        FormatFileName(layer_name,"/","_");
    }

    //3.Update outputs nodes.
    for (NodeDef* out: GetOutputs(node_name)){
        if(!isSupportedLayer(out->name())){
            UpdateUnsupportedNodes(node_name);
            break;
        }
    }

}

void TFUtil::MapMulLayer(NodeDef &node, std::string outputFolder, cnn::DataInfo &params)
{
    bool status = false;
    std::string node_name = node.name();
    std::string new_output_name;

    //1.1.get weights data
    if(HasTensor(*GetNode(node.input(1)))){
        GetKernelDataAndShape(node.input(1), &params, KERNEL);
    }else{
        //try with the next node.
        GetKernelDataAndShape(GetNode(node.input(1))->input(0), &params, KERNEL);
    }

    //1.2.Convert data format NHWC to NWCH
    if(params.mData_Format == NHWC){
        float * pbdata = new float[params.mKernelDataSize];
        memcpy(pbdata,params.mpKernelData,params.mKernelDataSize);
        params.NHWC2NCHW(params.mKernelDataSize, pbdata);
    }

    //2.pass layer name
    std:: string layer_name;
    if(!node.name().empty()) {
        layer_name = node.name();
        FormatFileName(layer_name,"/","_");
    }

    //3.create output files
    CreateLayerFiles(layer_name, outputFolder, params, KERNEL);

    //4.get bias data if exists.
    std::vector<NodeDef*> bias = GetNextNode(node,"BiasAdd");
    if(bias.size() == 0)
        bias = GetNextNode(node,"Add");

    if(bias.size() == 1){
        if(HasTensor(*GetNode((bias.at(0))->input(1)))){
            status = GetKernelDataAndShape((bias.at(0))->input(1), &params, BIAS);
        }else{
            //try with the next node.
            status = GetKernelDataAndShape((GetNode(bias.at(0)->input(1))->input(0)), &params, BIAS);
        }
        params.mbias_term = (status) ? 1: 0;

        //update MatMul output.
        for (NodeDef* out: GetOutputs(bias.at(0)->name())){
            new_output_name = out->name();
            break;
        }
    }

    //5.create output files
    CreateLayerFiles(layer_name, outputFolder, params, BIAS);

    //6.Update inputs outputs nodes.
    if(!new_output_name.empty())
        UpdateIO(node_name, bias.at(0)->name(), new_output_name);
}

void TFUtil::MapUpNNLayer(NodeDef &node, cnn::DataInfo &params)
{
    std::string node_name = node.name();

    //1.Get layer parameters
    if(HasTensor(*GetNode(node.input(1)))){
        GetKernelDataAndShape(node.input(1), &params, KERNEL);
    }else{
        //try with the next node.
        GetKernelDataAndShape(GetNode(node.input(1))->input(1), &params, KERNEL);
    }

    //2.Update outputs nodes.
    for (NodeDef* out: GetOutputs(node_name)){
        if(!isSupportedLayer(out->name())){
            UpdateUnsupportedNodes(node_name);
            break;
        }
    }
}

void TFUtil::FormatFileName(std::string &str, const std::string &from, const std::string &to)
{
    //Written to avoid conflicts with file creation with filenames that contain "/"
    size_t start_pos = 0;
    while ((start_pos = str.find(from, start_pos)) != std::string::npos) {
        str.replace(start_pos, from.length(), to);
        start_pos += to.length(); // Handles case where 'to' is a substring of 'from'
    }
}

bool TFUtil::GetKernelParams(const std::string &fieldName, cnn::DataInfo *dInfo, std::vector<std::string> keys)
{
    //const NodeDef * n = find(graph, "eval/batch/n");
    const NodeDef * n = GetNode(fieldName);

    assert((n != NULL) && "Node pointer is NULL!");
    if (n == NULL)
    {
        return false;
    }

    //getting dimensions
    for(std::string key: keys){
        const AttrValue &attr = n->attr().at(key);
        if(key == "strides"){
            const AttrValue_ListValue & strides = attr.list();
            dInfo->mstride_h = strides.i(1); //height
            dInfo->mstride_w = strides.i(2); //width

        }else if(key == "padding"){
            const std::string pad = attr.s();
            ///Formula to calcule the output size of any given conv:
            /// out_size = (W - K + 2P)/S +1
            /// W: the input width or height
            /// K: the Kernel width or height
            /// P: the padding
            /// S: the stride value
            if(pad == "SAME"){
                if(dInfo->mstride_h != -1 && dInfo->mstride_w != -1 && dInfo->mInput_h != -1 && dInfo->mInput_w != -1){
                    dInfo->mPadType = PAD_SAME;
                    int out_pad_h = std::max((ceil(dInfo->mInput_h/dInfo->mstride_h) - 1)*dInfo->mstride_h+dInfo->mkernel_h-dInfo->mInput_h, .0)/2;
                    int out_pad_w = std::max((ceil(dInfo->mInput_w/dInfo->mstride_w) - 1)*dInfo->mstride_w+dInfo->mkernel_w-dInfo->mInput_w, .0)/2;
                    dInfo->mpad_h = (dInfo->mpad_h == -1) ? out_pad_h : dInfo->mpad_h + out_pad_h;
                    dInfo->mpad_w = (dInfo->mpad_w == -1) ? out_pad_w : dInfo->mpad_w + out_pad_w;
                }else{
                    error("Padding SAME need input size values");
                }
            }else if(pad == "VALID"){
                dInfo->mPadType = PAD_VALID;
                dInfo->mpad_h = (dInfo->mpad_h == -1) ? 0 : dInfo->mpad_h;
                dInfo->mpad_w = (dInfo->mpad_w == -1) ? 0 : dInfo->mpad_w;
            }else{
                std::cerr << "Key not defined!" << std::endl;
                return false;
            }
        }else if(key == "ksize"){
            const AttrValue_ListValue & ksize = attr.list();
            dInfo->mkernel_h = ksize.i(1); //height
            dInfo->mkernel_w = ksize.i(2); //width
        }else if(key == "data_format"){
            const std::string & dFormat = attr.s();
            dInfo->mData_Format = (dFormat == "NHWC") ? NHWC : NCHW;
        }else{
            std::cerr << "Key not defined!" << std::endl;
            return false;
        }
    }

    return true;
}

bool TFUtil::GetKernelDataAndShape(const std::string &fieldName, cnn::DataInfo *dInfo, std::string dataType, std::string key)
{
    size_t data_size;
    const NodeDef * n = GetNode(fieldName);

    assert((n != NULL) && "Node pointer is NULL!");
    if (n == NULL)
    {
        return false;
    }

    //getting dimensions
    const AttrValue & attr = n->attr().at(key);

    assert(attr.has_tensor() && (std::stringstream() << "Tensor for " << fieldName << " is not set!").rdbuf());
    if (!attr.has_tensor())
    {
        return false;
    }

    const TensorShapeProto & shape = attr.tensor().tensor_shape();
    if (shape.dim_size() == 4) {
        //normal filter data 4d
        dInfo->mkernel_h = shape.dim(0).size();     //filter height
        dInfo->mkernel_w = shape.dim(1).size();     //filter width
        dInfo->mC = shape.dim(2).size();            //input channels
        dInfo->mN = shape.dim(3).size();            //output channels
    }
    else if (shape.dim_size() == 1)
    {
        if(shape.dim(0).size() == 2){ /*setting just kernel dimension, usually upsampling NN*/
            data_size = attr.tensor().tensor_content().size();
            if(data_size == 8 /*(2*4)*/){
                const int * nn_data = (const int *)attr.tensor().tensor_content().c_str();
                dInfo->mkernel_h = nn_data[0];
                dInfo->mkernel_w = nn_data[1];
                return true;
            }else{
                assert(false && (std::stringstream() << "unexpected datasize for filedName " << fieldName).rdbuf());
                return false;
            }
        }else{ /*setting just depth dimension, usually bias data*/
            dInfo->mkernel_h = (dInfo->mkernel_h == -1) ? 1: dInfo->mkernel_h;
            dInfo->mkernel_w = (dInfo->mkernel_w == -1) ? 1: dInfo->mkernel_w;
            dInfo->mC = (dInfo->mC == -1) ? shape.dim(0).size(): dInfo->mC; //input channels
            dInfo->mN = (dInfo->mN == -1) ? 1: dInfo->mN;
        }

    }else if (shape.dim_size() == 2)
    {
        //setting padding dimension, usually padding data
        if(shape.dim(1).size() == 2){
            data_size = attr.tensor().tensor_content().size();
            if(data_size == 32 /*(8*4)*/){
                const int * pad_data = (const int *)attr.tensor().tensor_content().c_str();
                /// Data stored as 12 34 56 78
                /// ............   N  H  W  C
                dInfo->mpad_h = pad_data[2];
                dInfo->mpad_w = pad_data[5];
            }
            return true;
        }else{
            //setting the number of filters, usually InnerProduct data
            dInfo->mN = shape.dim(1).size();
        }

    }
    else
    {
        assert(false && (std::stringstream() << "unexpected number of dimensions (not 4 and not 1) for filedName " << fieldName).rdbuf());
        return false;
    }

    data_size = attr.tensor().tensor_content().size();
    const float * pbdata = (const float *)attr.tensor().tensor_content().c_str(); //pointer to raw data in protobuf container

    dInfo->allocate(dataType, data_size);
    if(dataType == KERNEL){/*usually InnerProduct (MatMul)*/
        memcpy(dInfo->mpKernelData, pbdata, dInfo->mKernelDataSize);
    }else if(dataType == BIAS){
        memcpy(dInfo->mpBiasData, pbdata, dInfo->mBiasDataSize);
    }

    dInfo->print(KERNEL);
    std::cout << "Got size " << dInfo->mKernelDataSize << " " << dInfo->mC << std::endl;
    return true;
}

void TFUtil::CreateLayerFiles(std::string layer_name, std::string outputFolder, cnn::DataInfo &params, std::string key)
{
    //1.create output files
    std::string folder = (key == KERNEL) ? "/weights/" : "/bias/";
    std::string fileName = outputFolder + folder + layer_name + ".f32";
    FILE * fs;
    fs = fopen(fileName.c_str(), "wb");
    if(!fs) {
        printf("ERROR: unable to create dump files: make sure weights and bias folders are writable.\n");
        exit(1);
    }


    //2.save the data.
    if(params.isValid(key)){
        float * _data = (key == KERNEL) ? (float *)params.mpKernelData : (float *)params.mpBiasData;
        size_t  dataSize = (key == KERNEL) ? params.mKernelDataSize/4 : params.mBiasDataSize/4;
        for(int i=0; i < dataSize; i++) {
            float data = _data[i];
            fwrite(&data,sizeof(float),1,fs);
        }
    }
}

bool TFUtil::isSupportedLayer(std::string node_name){
    std::string ret = LAYER_DESCRIPTORS[GetNode(node_name)->op()];
    return !ret.empty();
}

std::string TFUtil::getIdentifierName(const std::string name)
{
    size_t N = name.size();
    const char * s = name.c_str();
    std::string cname = (N > 0 && std::isdigit(s[0])) ? "_" : "";
    for(size_t i = 0; i < N; i++) {
        cname += std::isalnum(s[i]) ? s[i] : '_';
    }
    return cname;
}

void TFUtil::WriteVXCode(
    std::ostream& ofsCodeC,
    std::vector<std::vector<std::string>>& net,
    std::map<std::string,std::vector<int>>& tensorMap,
    std::string tensorType,
    int fixedPosition,
    std::string convertPolicy,
    std::string roundPolicy,
    bool isVirtualEnabled,
    bool bFuseScaleLayer,
    std::string outputFolder,
    std::string codeType)
{
    auto&& inputTensorName = net[0][4];
    auto&& outputTensorName = net[net.size()-1][3];

    bool bfuse_scale_layer = bFuseScaleLayer;
    std::map<std::string,bool> declare_tensor_check;
    for(auto& node : net) {
        //declare input tensors.
        bool isFirstLayer = (&node == &net.front());
        bool isLastLayer = (&node == &net.back());

        std::string layerName = getIdentifierName(node[3]);
        std::string inputName = getIdentifierName(node[4]);
        if(codeType == "initialize") {
            ofsCodeC << "    // " << layerName <<" Layer" << std::endl;
        }
        for(size_t i=4; i < node.size(); i++) {
            if(node[i] != "" && declare_tensor_check.find(node[i]) == declare_tensor_check.end()) {
                auto&& dim = tensorMap[node[i]];
                if(codeType == "initialize") {
                    if(node[i] != inputTensorName && node[i] != outputTensorName) {
                        ofsCodeC << "    vx_size " << node[i] << "_dims[4] = { " << dim[3] << ", " << dim[2] << ", " << dim[1] << ", " << dim[0] << " };" << std::endl;
                        ofsCodeC << "    vx_tensor " << node[i] << ";" << std::endl;
                        ofsCodeC << "    " << node[i] << " = vxCreateTensor(context, 4, " << node[i] + "_dims,"<< tensorType <<", " << fixedPosition << ");" << std::endl;
                        ofsCodeC << "    " << "ERROR_CHECK_OBJECT("  << node[i] << ");" << std::endl;
                    }
                }
                else if(codeType == "release") {
                    if(node[i] != inputTensorName && node[i] != outputTensorName) {
                        ofsCodeC << "    " << "ERROR_CHECK_STATUS(vxReleaseTensor(&" << node[i] << "));" << std::endl;
                    }
                }
                declare_tensor_check[node[i]]= true;
            }
        }

        if (node[0] == "BatchNorm" && !isLastLayer && bfuse_scale_layer) {
            auto&& output = node[3];
            auto& next_node = *std::next(&node);
            if (next_node[0] == "Scale") {
                auto&& next_output = next_node[3];
                std::string nextOutput = getIdentifierName(next_node[3]);
                auto&& odim = tensorMap[next_output];
                if(!declare_tensor_check[next_output]) {
                    if(codeType == "initialize") {
                        ofsCodeC << "    vx_size " << nextOutput << "_dims[4] = { " << odim[3] << ", " << odim[2] << ", " << odim[1] << ", " << odim[0] << " };" << std::endl;
                        ofsCodeC << "    vx_tensor " << nextOutput << ";" << std::endl;
                        if(isVirtualEnabled){
                            ofsCodeC << "    " << nextOutput << " = vxCreateVirtualTensor(graph,4, " << nextOutput + "_dims, VX_TYPE_FLOAT32," << fixedPosition << ");" << std::endl;
                        }
                        else{
                            ofsCodeC << "    " << nextOutput << " = vxCreateTensor(context,4, " << nextOutput + "_dims, VX_TYPE_FLOAT32," << fixedPosition << ");" << std::endl;
                        }
                        ofsCodeC << "    " << "ERROR_CHECK_OBJECT("  << nextOutput << ");" << std::endl;
                    }
                    else if(codeType == "release") {
                        ofsCodeC << "    " << "ERROR_CHECK_STATUS(vxReleaseTensor(&" << nextOutput << "));" << std::endl;
                    }
                    declare_tensor_check[output] = true;
                }
                declare_tensor_check[next_output] = true;
                bfuse_scale_layer = true;
            }
        }
        if (node[0] == "Scale" && !isFirstLayer && bfuse_scale_layer) {
            auto& prev_node = *std::prev(&node);
            if (prev_node[0]=="BatchNorm"){
                if(codeType == "initialize")  {
                    ofsCodeC << "    // [NOTE -- Scale Layer Fused With Batch Norm Layer]" << std::endl<< std::endl;
                }
                continue;
            }
        }

        // declare output tensor.
        auto&& output = node[3];
        auto&& odim = tensorMap[output];
        if(!declare_tensor_check[output]) {
            if(codeType == "initialize") {
                if(layerName != outputTensorName) {
                    ofsCodeC << "    vx_size " << layerName << "_dims[4] = { " << odim[3] << ", " << odim[2] << ", " << odim[1] << ", " << odim[0] << " };" << std::endl;
                    ofsCodeC << "    vx_tensor " << layerName << ";" << std::endl;
                    if(isVirtualEnabled){
                        ofsCodeC << "    " << layerName << " = vxCreateVirtualTensor(graph,4, " << layerName + "_dims, VX_TYPE_FLOAT32," << fixedPosition << ");" << std::endl;
                    }
                    else{
                        ofsCodeC << "    " << layerName << " = vxCreateTensor(context,4, " << layerName + "_dims, VX_TYPE_FLOAT32," << fixedPosition << ");" << std::endl;
                    }
                    ofsCodeC << "    " << "ERROR_CHECK_OBJECT("  << layerName << ");" << std::endl;
                }
            }
            else if(codeType == "release") {
                if(layerName != outputTensorName) {
                    ofsCodeC << "    " << "ERROR_CHECK_STATUS(vxReleaseTensor(&" << layerName << "));" << std::endl;
                }
            }
            declare_tensor_check[output] = true;
        }

        auto&& type = node[0];
        auto&& params = node[1];
        if(type == "Convolution") {
            std::stringstream ss(params);
            int k, kernel_w, kernel_h, stride_w, stride_h, pad_w, pad_h, dilation_w, dilation_h, bias_term, group;
            ss >> k >> kernel_w >> kernel_h >> stride_w >> stride_h >> pad_w >> pad_h >> dilation_w >> dilation_h >> bias_term >> group;
            if(group > 1) {
                auto&& idim = tensorMap[inputName];

                if(codeType == "initialize") {
                    ofsCodeC << "    vx_size " << inputName << "_grp_dims[4] = { " << idim[3] << ", " << idim[2] << ", " << idim[1]/group << ", " << idim[0] << " };" << std::endl;
                    ofsCodeC << "    vx_size " << layerName << "_grp_dims[4] = { " << odim[3] << ", " << odim[2] << ", " << odim[1]/group << ", " << odim[0] << " };" << std::endl;
                    for(int g = 0; g < group; g++) {
                        // Input tensor for the group-g conv
                        ofsCodeC << "    vx_tensor " << inputName << "_grp" << g << ";" << std::endl;
                        if(isVirtualEnabled){
                            ofsCodeC << "    " << inputName << "_grp" << g << " = vxCreateVirtualTensor(graph,4, " << inputName << "_grp_dims, VX_TYPE_FLOAT32," << fixedPosition << ");" << std::endl;
                        }
                        else{
                            ofsCodeC << "    " << inputName << "_grp" << g << " = vxCreateTensor(context,4, " << inputName << "_grp_dims, VX_TYPE_FLOAT32," << fixedPosition << ");" << std::endl;
                        }
                        ofsCodeC << "    " << "ERROR_CHECK_OBJECT("  << inputName << "_grp" << g << ");" << std::endl;

                        // Output tensor for the group-g conv
                        ofsCodeC << "    vx_tensor " << layerName << "_grp" << g << ";" << std::endl;
                        if(isVirtualEnabled){
                            ofsCodeC << "    " << layerName << "_grp" << g << " = vxCreateVirtualTensor(graph,4, " << layerName << "_grp_dims, VX_TYPE_FLOAT32," << fixedPosition << ");" << std::endl;
                        }
                        else{
                            ofsCodeC << "    " << layerName << "_grp" << g << " = vxCreateTensor(context,4, " << layerName << "_grp_dims, VX_TYPE_FLOAT32," << fixedPosition << ");" << std::endl;
                        }
                        ofsCodeC << "    " << "ERROR_CHECK_OBJECT("  << layerName << "_grp" << g << ");" << std::endl;

                    }

                    // Slice conv input
                    ofsCodeC << "    vx_node " << inputName << "_grp_slice_node;" << std::endl;
                    ofsCodeC << "    " <<  inputName << "_grp_slice_node = " << "vxSliceLayer(graph, ";
                    ofsCodeC << inputName;
                    for(int g = 0; g < 8; g++) {
                        if(g < group)
                            ofsCodeC << ", " << inputName << "_grp" << g;
                        else
                            ofsCodeC << ", NULL";
                    }
                    ofsCodeC << ");" << std::endl;
                    ofsCodeC << "    " << "ERROR_CHECK_OBJECT(" << inputName << "_grp_slice_node);" << std::endl;
                    ofsCodeC << "    " << "ERROR_CHECK_STATUS(vxReleaseNode(&" << inputName << "_grp_slice_node));" << std::endl;

                    // Concat conv output
                    ofsCodeC << "    vx_node " << layerName << "_grp_concat_node;" << std::endl;
                    ofsCodeC << "    " <<  layerName << "_grp_concat_node = " << "vxConcatLayer(graph, ";
                    ofsCodeC << layerName;
                    for(int g = 0; g < 8; g++) {
                        if(g < group)
                            ofsCodeC << ", " << layerName << "_grp" << g;
                        else
                            ofsCodeC << ", NULL";
                    }
                    ofsCodeC << ");" << std::endl;
                    ofsCodeC << "    " << "ERROR_CHECK_OBJECT(" << layerName << "_grp_concat_node);" << std::endl;
                    ofsCodeC << "    " << "ERROR_CHECK_STATUS(vxReleaseNode(&" << layerName << "_grp_concat_node));" << std::endl;
                }
                else if(codeType == "release") {
                    for(int g = 0; g < group; g++) {
                        ofsCodeC << "    " << "ERROR_CHECK_STATUS(vxReleaseTensor(&" << inputName << "_grp" << g << "));" << std::endl;
                        ofsCodeC << "    " << "ERROR_CHECK_STATUS(vxReleaseTensor(&" << layerName << "_grp" << g << "));" << std::endl;
                    }
                }

                auto&& dim = tensorMap[output + "_W"];
                if(codeType == "initialize") {
                    ofsCodeC << "    vx_size " << layerName << "_W" << "_dims[4] = { " << dim[3] << ", " << dim[2] << ", " << dim[1]/group << ", " << dim[0]/group << " };" << std::endl;
                    for(int g = 0; g < group; g++) {
                        ofsCodeC << "    vx_tensor " << layerName << "_grp" << g << "_W" << ";" << std::endl;
                        ofsCodeC << "    " << layerName << "_grp" << g << "_W" << " = vxCreateTensor(context,4, " << layerName << "_W" << "_dims, " << tensorType << ", " << fixedPosition << ");" << std::endl;
                        ofsCodeC << "    " << "ERROR_CHECK_OBJECT(" << layerName << "_grp" << g << "_W" << "); " << std::endl;
                        ofsCodeC << "    " << "ERROR_CHECK_STATUS(copyTensor(" << layerName << "_grp" << g << "_W" << ", dataFolder + \"/weights/" << layerName << "_grp" << g << ".f32\"));" << std::endl;
                    }
                }
                else if(codeType == "release") {
                    for(int g = 0; g < group; g++) {
                        ofsCodeC << "    " << "ERROR_CHECK_STATUS(vxReleaseTensor(&" << layerName << "_grp" << g << "_W" << "));" << std::endl;
                    }
                }
                declare_tensor_check[output + "_W"] = true;
                if(bias_term) {
                    if(codeType == "initialize") {
                        ofsCodeC << "    vx_size " << layerName << "_B" << "_dims[1] = { " << k/group << " };" << std::endl;
                        for(int g = 0; g < group; g++) {
                            ofsCodeC << "    vx_tensor " << layerName << "_grp" << g << "_B" << ";" << std::endl;
                            ofsCodeC << "    " << layerName << "_grp" << g << "_B" << " = vxCreateTensor(context,1, " << layerName << "_B"  "_dims, " << tensorType << ", " << fixedPosition << ");" << std::endl;
                            ofsCodeC << "    " << "ERROR_CHECK_OBJECT(" << layerName << "_grp" << g << "_B" << "); " << std::endl;
                            ofsCodeC << "    " << "ERROR_CHECK_STATUS(copyTensor(" << layerName << "_grp" << g << "_B" << ", dataFolder + \"/bias/" << layerName << "_grp" << g << ".f32\"));" << std::endl;
                        }
                    }
                    else if(codeType == "release") {
                        for(int g = 0; g < group; g++) {
                            ofsCodeC << "    " << "ERROR_CHECK_STATUS(vxReleaseTensor(&" << layerName << "_grp" << g << "_B" << "));" << std::endl;
                        }
                    }
                    declare_tensor_check[layerName + "_B"] = true;
                }

                if(codeType == "initialize") {
                    ofsCodeC << "    vx_nn_convolution_params_t " << layerName << "_params;" << std::endl;
                    ofsCodeC << "    " << layerName + "_params.padding_x = " << pad_w << ";" << std::endl;
                    ofsCodeC << "    " << layerName + "_params.padding_y = " << pad_h << ";" << std::endl;
                    ofsCodeC << "    " << layerName + "_params.overflow_policy = " << convertPolicy << ";" << std::endl;
                    ofsCodeC << "    " << layerName + "_params.rounding_policy = " << roundPolicy << ";" << std::endl;
                    ofsCodeC << "    " << layerName + "_params.down_scale_size_rounding = " << "VX_NN_DS_SIZE_ROUNDING_FLOOR ;" << std::endl;
                    ofsCodeC << "    " << layerName + "_params.dilation_x = " << dilation_w - 1 << ";" << std::endl;
                    ofsCodeC << "    " << layerName + "_params.dilation_y = " << dilation_h - 1 << ";" << std::endl;

                    for(int g = 0; g < group; g++) {
                        ofsCodeC << "    vx_node " << layerName << "_grp" << g << "_node;" << std::endl;
                        ofsCodeC << "    " << layerName << "_grp" << g << "_node = " << "vxConvolutionLayer(graph, ";
                        ofsCodeC << inputName << "_grp" << g << ", ";
                        ofsCodeC << layerName << "_grp" << g << "_W, ";
                        if(bias_term)
                            ofsCodeC << layerName << "_grp" << g << "_B, ";
                        else
                            ofsCodeC << "NULL, ";
                        ofsCodeC << "&" << layerName + "_params, " << "sizeof(" << layerName + "_params ), ";
                        ofsCodeC << layerName << "_grp" << g << ");" << std::endl;
                        ofsCodeC << "    " << "ERROR_CHECK_OBJECT(" << layerName << "_grp" << g << "_node);" << std::endl;
                        ofsCodeC << "    " << "ERROR_CHECK_STATUS(vxReleaseNode(&" << layerName << "_grp" << g << "_node));" << std::endl;
                    }
                }
            }
            else {
                std::string weights = layerName + "_W";
                std::string dim_weights = output + "_W";
                auto&& dim = tensorMap[dim_weights];
                if(codeType == "initialize") {
                    ofsCodeC << "    vx_size " << weights << "_dims[4] = { " << dim[3] << ", " << dim[2] << ", " << dim[1] << ", " << dim[0] << " };" << std::endl;
                    ofsCodeC << "    vx_tensor " << weights << ";" << std::endl;
                    ofsCodeC << "    " << weights << " = vxCreateTensor(context,4, " << weights + "_dims, " << tensorType << ", " << fixedPosition << ");" << std::endl;
                    ofsCodeC << "    " << "ERROR_CHECK_OBJECT(" << weights << "); " << std::endl;
                    ofsCodeC << "    " << "ERROR_CHECK_STATUS(copyTensor(" << weights << ", dataFolder + \"/weights/" + layerName + ".f32\"));" << std::endl;
                }
                else if(codeType == "release") {
                    ofsCodeC << "    " << "ERROR_CHECK_STATUS(vxReleaseTensor(&" << weights << "));" << std::endl;
                }
                declare_tensor_check[weights] = true;
                std::string bias = "NULL";
                if(bias_term) {
                    bias = layerName + "_B";
                    if(codeType == "initialize") {
                        ofsCodeC << "    vx_size " << bias << "_dims[1] = { " << k << " };" << std::endl;
                        ofsCodeC << "    vx_tensor " << bias << ";" << std::endl;
                        ofsCodeC << "    " << bias << " = vxCreateTensor(context,1, " << bias + "_dims, " << tensorType << ", " << fixedPosition << ");" << std::endl;
                        ofsCodeC << "    " << "ERROR_CHECK_OBJECT(" << bias << "); " << std::endl;
                        ofsCodeC << "    " << "ERROR_CHECK_STATUS(copyTensor(" << bias << ", dataFolder + \"/bias/" + layerName + ".f32\"));" << std::endl;
                    }
                    else if(codeType == "release") {
                        ofsCodeC << "    " << "ERROR_CHECK_STATUS(vxReleaseTensor(&" << bias << "));" << std::endl;
                    }
                    declare_tensor_check[bias] = true;
                }
                if(codeType == "initialize") {
                    ofsCodeC << "    vx_nn_convolution_params_t " << layerName << "_params;" << std::endl;
                    ofsCodeC << "    " << layerName + "_params.padding_x = " << pad_w << ";" << std::endl;
                    ofsCodeC << "    " << layerName + "_params.padding_y = " << pad_h << ";" << std::endl;
                    ofsCodeC << "    " << layerName + "_params.overflow_policy = " << convertPolicy << ";" << std::endl;
                    ofsCodeC << "    " << layerName + "_params.rounding_policy = " << roundPolicy << ";" << std::endl;
                    ofsCodeC << "    " << layerName + "_params.down_scale_size_rounding = " << "VX_NN_DS_SIZE_ROUNDING_FLOOR ;" << std::endl;
                    ofsCodeC << "    " << layerName + "_params.dilation_x = " << dilation_w - 1 << ";" << std::endl;
                    ofsCodeC << "    " << layerName + "_params.dilation_y = " << dilation_h - 1 << ";" << std::endl;
                    ofsCodeC << "    vx_node " << layerName << "_node;" << std::endl;
                    ofsCodeC << "    " << layerName + "_node = " << "vxConvolutionLayer(graph, " << inputName << ", " << weights << ", " << bias << ", &" << layerName + "_params, " << "sizeof(" << layerName + "_params ), " << layerName << ");" << std::endl;
                    ofsCodeC << "    " << "ERROR_CHECK_OBJECT(" + layerName + "_node);" << std::endl;
                    ofsCodeC << "    " << "ERROR_CHECK_STATUS(vxReleaseNode(&" << layerName + "_node));" << std::endl;
                }
            }
        }
        else if(type == "Deconvolution") {
            std::stringstream ss(params);
            int k, kernel_w, kernel_h, stride_w, stride_h, pad_w, pad_h, dilation_w, dilation_h, bias_term;
            ss >> k >> kernel_w >> kernel_h >> stride_w >> stride_h >> pad_w >> pad_h >> dilation_w >> dilation_h >> bias_term;
            std::string weights = layerName + "_W";
            std::string dim_weights = output + "_W";
            auto&& dim = tensorMap[dim_weights];
            if(codeType == "initialize") {
                ofsCodeC << "    vx_size " << weights << "_dims[4] = { " << dim[3] << ", " << dim[2] << ", " << dim[1] << ", " << dim[0] << " };" << std::endl;
                ofsCodeC << "    vx_tensor " << weights << ";" << std::endl;
                ofsCodeC << "    " << weights + "= vxCreateTensor(context,4, " << weights + "_dims, " << tensorType << ", " << fixedPosition << ");" << std::endl;
                ofsCodeC << "    " << "ERROR_CHECK_OBJECT(" << weights << "); " << std::endl;
                ofsCodeC << "    " << "ERROR_CHECK_STATUS(copyTensor(" << weights << ", dataFolder + \"/weights/" + layerName + ".f32\"));" << std::endl;
            }
            else if(codeType == "release") {
                ofsCodeC << "    " << "vxReleaseTensor(&" << weights << " );" << std::endl;
            }
            declare_tensor_check[weights] = true;
            std::string bias = "NULL";
            if(bias_term) {
                bias = layerName + "_B";
                if(codeType == "initialize") {
                    ofsCodeC << "    vx_size " << bias << "_dims[1] = { " << k << " };" << std::endl;
                    ofsCodeC << "    vx_tensor " << bias << ";" << std::endl;
                    ofsCodeC << "    " << bias + " = vxCreateTensor(context,1, " << bias + "_dims, " << tensorType << ", " << fixedPosition << ");" << std::endl;
                    ofsCodeC << "    " << "ERROR_CHECK_OBJECT(" << bias << "); " << std::endl;
                    ofsCodeC << "    " << "ERROR_CHECK_STATUS(copyTensor(" << bias << ", dataFolder + \"/bias/" + layerName + ".f32\"));" << std::endl;
                }
                else if(codeType == "release") {
                    ofsCodeC << "    " << "ERROR_CHECK_STATUS(vxReleaseTensor(&" << bias << "));" << std::endl;
                }
                declare_tensor_check[bias] = true;
            }
            if(codeType == "initialize") {
                ofsCodeC << "    vx_nn_deconvolution_params_t " << layerName << "_params;" << std::endl;
                ofsCodeC << "    " << layerName + "_params.padding_x = " << pad_w << ";" << std::endl;
                ofsCodeC << "    " << layerName + "_params.padding_y = " << pad_h << ";" << std::endl;
                ofsCodeC << "    " << layerName + "_params.overflow_policy = " << convertPolicy << ";" << std::endl;
                ofsCodeC << "    " << layerName + "_params.rounding_policy = " << roundPolicy << ";" << std::endl;
                ofsCodeC << "    " << layerName + "_params.a_x = " << dilation_w - 1 << ";" << std::endl;
                ofsCodeC << "    " << layerName + "_params.a_y = " << dilation_h - 1 << ";" << std::endl;
                ofsCodeC << "    vx_node " << layerName << "_node;" << std::endl;
                ofsCodeC << "    " << layerName + "_node = " << " vxDeconvolutionLayer(graph, " << inputName << ", " << weights << ", " << bias << ", &" << layerName + "_params, sizeof(" + layerName + "_params), " << layerName << ");" << std::endl;
                ofsCodeC << "    " << "ERROR_CHECK_OBJECT(" + layerName + "_node);" << std::endl;
                ofsCodeC << "    " << "ERROR_CHECK_STATUS(vxReleaseNode(&" << layerName + "_node));" << std::endl;
            }
        }
        else if(type == "Pooling") {
            std::stringstream ss(params);
            int kernel_w, kernel_h, stride_w, stride_h, pad_w, pad_h, pool;
            ss >> kernel_w >> kernel_h >> stride_w >> stride_h >> pad_w >> pad_h >> pool;
            if((pool != 0 && pool != 1)) error("writeGDF: pooling_layer supports only MAX and AVG\n");
            if(codeType == "initialize") {
                ofsCodeC << "    vx_enum " << layerName << "_type = " << (pool == 0 ? "VX_NN_POOLING_MAX" : "VX_NN_POOLING_AVG") << ";" << std::endl;
                ofsCodeC << "    vx_size " << layerName << "_kernel_w = " << kernel_w << ";" << std::endl;
                ofsCodeC << "    vx_size " << layerName << "_kernel_h = " << kernel_h << ";" << std::endl;
                ofsCodeC << "    vx_size " << layerName << "_pad_w = " << pad_w << ";" << std::endl;
                ofsCodeC << "    vx_size " << layerName << "_pad_h = " << pad_h << ";" << std::endl;
                ofsCodeC << "    vx_enum " << layerName << "_roundPolicy = " << roundPolicy << ";" << std::endl;
                ofsCodeC << "    vx_node " << layerName << "_node;" << std::endl;
                ofsCodeC << "    " << layerName + "_node = " << "vxPoolingLayer(graph, " << inputName << ", " << layerName + "_type" << ", " << layerName + "_kernel_w, " << layerName + "_kernel_h, "
                                   << layerName + "_pad_w, " << layerName + "_pad_h, " << layerName + "_roundPolicy, " << layerName << " );" << std::endl;
                ofsCodeC << "    " << "ERROR_CHECK_OBJECT(" + layerName + "_node);" << std::endl;
                ofsCodeC << "    " << "ERROR_CHECK_STATUS(vxReleaseNode(&" << layerName + "_node));" << std::endl;
            }
        }
        else if(type == "InnerProduct") {
            std::stringstream ss(params);
            int k,bias_term;
            ss >> k >> bias_term;
            std::string weights = layerName + "_W";
            std::string dim_weights = output + "_W";
            auto&& dim = tensorMap[dim_weights];
            if(codeType == "initialize") {
                ofsCodeC << "    vx_size " << weights << "_dims[4] = { " << dim[3] << ", " << dim[2] << ", " << dim[1] << ", " << dim[0] << " };" << std::endl;
                ofsCodeC << "    vx_tensor " << weights << ";" << std::endl;
                ofsCodeC << "    " << weights << "= vxCreateTensor(context,4," << weights + "_dims, " << tensorType << ", " << fixedPosition << ");" << std::endl;
                ofsCodeC << "    " << "ERROR_CHECK_OBJECT(" << weights << "); " << std::endl;
                ofsCodeC << "    " << "ERROR_CHECK_STATUS(copyTensor(" << weights << ", dataFolder + \"/weights/" + layerName + ".f32\"));" << std::endl;
            }
            else if(codeType == "release") {
                ofsCodeC << "    " << "ERROR_CHECK_STATUS(vxReleaseTensor(&" << weights << "));" << std::endl;
            }
            declare_tensor_check[weights]= true;
            std::string bias= "NULL";
            if(bias_term) {
                bias = layerName + "_B";
                if(codeType == "initialize") {
                    ofsCodeC << "    vx_size " << bias << "_dims[1] = { " << k << " };" << std::endl;
                    ofsCodeC << "    vx_tensor " << bias << ";" << std::endl;
                    ofsCodeC << "    " << bias << "= vxCreateTensor(context,1," << bias + "_dims, " << tensorType << ", " << fixedPosition << ");" << std::endl;
                    ofsCodeC << "    " << "ERROR_CHECK_OBJECT(" << bias << "); " << std::endl;
                    ofsCodeC << "    " << "ERROR_CHECK_STATUS(copyTensor(" << bias << ", dataFolder + \"/bias/" + layerName + ".f32\"));" << std::endl;
                }
                else if(codeType == "release") {
                    ofsCodeC << "    " << "ERROR_CHECK_STATUS(vxReleaseTensor(&" << bias << "));" << std::endl;
                }
                declare_tensor_check[bias]= true;
            }
            if(codeType == "initialize") {
                ofsCodeC << "    vx_enum " << layerName << "_convertPolicy = " << convertPolicy << ";" << std::endl;
                ofsCodeC << "    vx_enum " << layerName << "_roundPolicy = " << roundPolicy << ";" << std::endl;
                ofsCodeC << "    vx_node " << layerName << "_node;" << std::endl;
                ofsCodeC << "    " << layerName + "_node = " << "vxFullyConnectedLayer( graph, " << inputName << ", " << weights << ", " << bias << ", " << layerName + "_convertPolicy, " << layerName + "_roundPolicy, " << layerName + ");" << std::endl;
                ofsCodeC << "    " << "ERROR_CHECK_OBJECT(" + layerName + "_node);" << std::endl;
                ofsCodeC << "    " << "ERROR_CHECK_STATUS(vxReleaseNode(&" << layerName + "_node));" << std::endl;
            }
        }
        else if(type == "ReLU") {
            if(codeType == "initialize") {
                ofsCodeC << "    vx_enum " << layerName << "_mode = " << "VX_NN_ACTIVATION_RELU ; " << std::endl;
                ofsCodeC << "    vx_float32 " << layerName << "_param_a = 0;" << std::endl;
                ofsCodeC << "    vx_float32 " << layerName << "_param_b = 0;" << std::endl;
                ofsCodeC << "    vx_node " << layerName << "_node;" << std::endl;
                ofsCodeC << "    " << layerName + "_node = " << "vxActivationLayer(graph, " << inputName << ", " << layerName + "_mode, " << layerName + "_param_a, " << layerName + "_param_b, " << layerName << ");" << std::endl;
                ofsCodeC << "    " << "ERROR_CHECK_OBJECT(" + layerName + "_node);" << std::endl;
                ofsCodeC << "    " << "ERROR_CHECK_STATUS(vxReleaseNode(&" << layerName + "_node));" << std::endl;
            }
        }
        else if(type == "LRN") {
            int normalization_size; float alpha,beta,k;
            std::string norm_region;
            std::stringstream ss(params);
            ss >> normalization_size >> alpha >> beta >> norm_region >> k;
            std::string lrnType;
            lrnType =  (norm_region == "1") ? "VX_NN_NORMALIZATION_SAME_MAP" : "VX_NN_NORMALIZATION_ACROSS_MAPS";
            if(codeType == "initialize") {
                ofsCodeC << "    vx_enum " << layerName << "_mode = " << lrnType << ";" << std::endl;
                ofsCodeC << "    vx_size " << layerName << "_size = "  << normalization_size << ";" << std::endl;
                ofsCodeC << "    vx_float32 " << layerName << "_alpha = " << alpha << ";" << std::endl;
                ofsCodeC << "    vx_float32 " << layerName << "_beta = " << beta << ";" << std::endl;
                ofsCodeC << "    vx_float32 " << layerName << "_bias = " << k << ";" << std::endl;
                ofsCodeC << "    vx_node " << layerName << "_node;" << std::endl;
                ofsCodeC << "    " << layerName + "_node = " << "vxNormalizationLayer( graph, " << inputName << ", " << layerName + "_mode, " << layerName + "_size, " << layerName + "_alpha, " << layerName + "_beta, "
                         << layerName << " );" << std::endl;
                ofsCodeC << "    " << "ERROR_CHECK_OBJECT(" + layerName + "_node);" << std::endl;
                ofsCodeC << "    if(" << layerName << "_bias != 1) {" << std::endl;
                ofsCodeC << "        vx_scalar s_bias = vxCreateScalarWithSize(context, VX_TYPE_FLOAT32, &" << layerName << "_bias, sizeof(" << layerName << "_bias));" << std::endl;
                ofsCodeC << "        ERROR_CHECK_OBJECT(s_bias);" << std::endl;
                ofsCodeC << "        ERROR_CHECK_STATUS(vxSetParameterByIndex(" << layerName << "_node, 6, (vx_reference) s_bias));" << std::endl;
                ofsCodeC << "        ERROR_CHECK_STATUS(vxReleaseScalar(&s_bias));" << std::endl;
                ofsCodeC << "    }" << std::endl;
                ofsCodeC << "    " << "ERROR_CHECK_STATUS(vxReleaseNode(&" << layerName + "_node));" << std::endl;
            }
        }
        else if(type == "BatchNorm") {
            int use_global_stats;
            std::stringstream ss(params);
            float eps;
            ss >> eps >> use_global_stats;
            std::string weights = layerName + "_W";
            std::string dim_weights = output + "_W";
            auto&& dim = tensorMap[dim_weights];
            if(codeType == "initialize") {
                ofsCodeC << "    vx_size " << weights << "_dims[1] = { " << dim[0] << " };" << std::endl;
                ofsCodeC << "    vx_float32 " << layerName << "_eps = " << eps << ";" << std::endl;
                ofsCodeC << "    vx_tensor " << weights << ";" << std::endl;
                ofsCodeC << "    " << weights << " = vxCreateTensor(context,1, " << weights + "_dims, " << tensorType << ", " << fixedPosition << ");" << std::endl;
                ofsCodeC << "    " << "ERROR_CHECK_OBJECT(" << weights << "); " << std::endl;
                ofsCodeC << "    " << "ERROR_CHECK_STATUS(copyTensor(" << weights << ", dataFolder + \"/weights/" + layerName + ".f32\"));" << std::endl;
            }
            else if(codeType == "release") {
                ofsCodeC << "    " << "ERROR_CHECK_STATUS(vxReleaseTensor(&" << weights << "));" << std::endl;
            }
            declare_tensor_check[weights] = true;
            std::string bias = layerName + "_B";
            std::string dim_bias = output + "_B";
            dim = tensorMap[dim_bias];
            if(codeType == "initialize") {
                ofsCodeC << "    vx_size " << bias << "_dims[1] = { " << dim[0] << " };" << std::endl;
                ofsCodeC << "    vx_tensor " << bias << ";" << std::endl;
                ofsCodeC << "    " << bias << " = vxCreateTensor(context,1, " << bias + "_dims, " << tensorType << ", " << fixedPosition << ");" << std::endl;
                ofsCodeC << "    " << "ERROR_CHECK_OBJECT(" << bias << "); " << std::endl;
                ofsCodeC << "    " << "ERROR_CHECK_STATUS(copyTensor(" << bias << ", dataFolder + \"/bias/" + layerName + ".f32\"));" << std::endl;
            }
            else if(codeType == "release") {
                ofsCodeC << "    " << "ERROR_CHECK_STATUS(vxReleaseTensor(&" << bias << "));" << std::endl;
            }
            declare_tensor_check[bias] = true;
            bias = "NULL";

            if (bfuse_scale_layer) {
                // check next node. If scale extract weight and bias paramters for scale layer.
                int bias_term;
                auto& next_node = *std::next(&node);
                auto&& next_output = next_node[3];
                auto&& nn_params = next_node[1];
                std::string nn_layer_name = getIdentifierName(next_node[3]);
                weights = nn_layer_name + "_W";
                std::string dim_weights = next_output + "_W";
                dim = tensorMap[dim_weights];
                if(codeType == "initialize") {
                    ofsCodeC << "    vx_size " << weights << "_dims[1] = { " << dim[0] << " };" << std::endl;
                    ofsCodeC << "    vx_tensor " << weights << ";" << std::endl;
                    ofsCodeC << "    " << weights << " = vxCreateTensor(context,1, " << weights + "_dims, " << tensorType << ", " << fixedPosition << ");" << std::endl;
                    ofsCodeC << "    " << "ERROR_CHECK_OBJECT(" << weights << "); " << std::endl;
                    ofsCodeC << "    " << "ERROR_CHECK_STATUS(copyTensor(" << weights << ", dataFolder + \"/weights/" + nn_layer_name + ".f32\"));" << std::endl;
                }
                else if(codeType == "release") {
                    ofsCodeC << "    " << "ERROR_CHECK_STATUS(vxReleaseTensor(&" << weights << "));" << std::endl;
                }
                declare_tensor_check[weights] = true;

                std::stringstream ss(nn_params);
                ss >> bias_term;
                if(bias_term) {
                    bias = nn_layer_name + "_B";
                    std::string dim_bias = next_output + "_B";
                    dim = tensorMap[dim_bias];
                    if(codeType == "initialize") {
                        ofsCodeC << "    vx_size " << bias << "_dims[1] = { " << dim[0] << " };" << std::endl;
                        ofsCodeC << "    vx_tensor " << bias << ";" << std::endl;
                        ofsCodeC << "    " << bias << " = vxCreateTensor(context,1, " << bias + "_dims, " << tensorType << ", " << fixedPosition << ");" << std::endl;
                        ofsCodeC << "    " << "ERROR_CHECK_OBJECT(" << bias << "); " << std::endl;
                        ofsCodeC << "    " << "ERROR_CHECK_STATUS(copyTensor(" << bias << ", dataFolder + \"/bias/" + nn_layer_name + ".f32\"));" << std::endl;
                    }
                    else if(codeType == "release") {
                        ofsCodeC << "    " << "ERROR_CHECK_STATUS(vxReleaseTensor(&" << bias << "));" << std::endl;
                    }
                    declare_tensor_check[bias] = true;
                }
                if(codeType == "initialize") {
                    ofsCodeC << "    vx_node " << layerName << "_node;" << std::endl;
                    ofsCodeC << "    " << layerName + "_node = " << "vxBatchNormalizationLayer(graph, "
                             << inputName +", "
                             << layerName + "_W, "
                             << layerName + "_B, "
                             << weights+", "
                             << bias+", "
                             << layerName + "_eps, "
                             << nn_layer_name << ");" << std::endl;
                    ofsCodeC << "    " << "ERROR_CHECK_OBJECT(" + layerName + "_node);" << std::endl;
                    ofsCodeC << "    " << "ERROR_CHECK_STATUS(vxReleaseNode(&" << layerName + "_node));" << std::endl;
                }
                else if(codeType == "release") {
                }
            }
            else{
                // put default scale and bias term
                std::vector<float> scale_arr(dim[0]);
                std::fill(scale_arr.begin(), scale_arr.end(), 1.0);
                std::string fileName_weights = outputFolder + "/weights/scale_init.f32";
                FILE *fp = fopen(fileName_weights.c_str(), "wb");
                if (fp) {
                    fwrite(scale_arr.data(), sizeof(float), dim[0], fp);
                    fclose(fp);
                }
                weights = layerName +"_W1";
                if(codeType == "initialize") {
                    ofsCodeC << "    vx_size " << weights << "_dims[1] = { " << dim[0] << " };" << std::endl;
                    ofsCodeC << "    vx_tensor " << weights << ";" << std::endl;
                    ofsCodeC << "    " << weights << " = vxCreateTensor(context,1, " << weights + "_dims, " << tensorType << ", " << fixedPosition << ");" << std::endl;
                    ofsCodeC << "    " << "ERROR_CHECK_OBJECT(" << weights << "); " << std::endl;
                    ofsCodeC << "    " << "ERROR_CHECK_STATUS(copyTensor(" << weights << ", dataFolder + \"/weights/scale_init.f32\"));" << std::endl;
                }
                else if(codeType == "release") {
                    ofsCodeC << "    " << "ERROR_CHECK_STATUS(vxReleaseTensor(&" << weights << "));" << std::endl;
                }
                declare_tensor_check[weights] = true;

                if(codeType == "initialize") {
                    ofsCodeC << "    vx_node " << layerName << "_node;" << std::endl;
                    ofsCodeC << "    " << layerName + "_node = " << "vxBatchNormalizationLayer(graph, "
                             << inputName +", "
                             << layerName + "_W, "
                             << layerName + "_B, "
                             << weights+", "
                             << bias+", "
                             << layerName + "_eps, "
                             << layerName << ");" << std::endl;
                    ofsCodeC << "    " << "ERROR_CHECK_OBJECT(" + layerName + "_node);" << std::endl;
                    ofsCodeC << "    " << "ERROR_CHECK_STATUS(vxReleaseNode(&" << layerName + "_node));" << std::endl;
                }
                else if(codeType == "release") {
                }
            }
        }
        else if(type == "Eltwise") {
            int op;
            std::stringstream ss(params);
            ss >> op;
            auto&& dim = tensorMap[output];
            for(int i=4; i < node.size(); i++) {
                auto&& idim= tensorMap[node[i]];
                if(dim[0]!= idim[0] || dim[1] != idim[1] || dim[2] != idim[2] || dim[3] != idim[3])
                    error("generateCode : Eltwise op=%d requires same dimension inputs : %s[%dx%dx%dx%d] != %s[%dx%dx%dx%d]\n", op, node[i].c_str(),idim[0], idim[1], idim[2], idim[3], node[i-1].c_str(), dim[0],dim[1],dim[2],dim[3]);
                dim = idim;
            }
            std::string tmp = inputName;
            for(int i=5; i < node.size() ; i++) {
                std::string out = layerName;
                if(i < node.size() - 1) {
                    out += "tmp_"+ std::to_string(i-4);
                    if(codeType == "initialize") {
                        ofsCodeC << "    vx_size " << out << "_dim[4] = { " << dim[3] << ", " << dim[2] << ", " << dim[1] << ", " << dim[0] << " };" << std::endl;
                        ofsCodeC << "    vx_tensor " << out << "; " << std::endl;
                        ofsCodeC << "    " << out << "= vxCreateTensor(context,4, " << out + "_dim, " << tensorType << ", " << fixedPosition << ");" << std::endl;
                    }
                    declare_tensor_check[out]= true;
                }
                if(op == 1) {
                    if(codeType == "initialize") {
                        ofsCodeC << "    vx_enum " << layerName << "_convertPolicy = " << convertPolicy << ";" << std::endl;
                        ofsCodeC << "    vx_node    " << layerName <<"_node;" << std::endl;
                        ofsCodeC << "    " << layerName + "_node = " << "vxTensorAddNode(graph, " << tmp << ", " << getIdentifierName(node[i]) << ", " << layerName + "_convertPolicy, " << out << ");" << std::endl;
                        ofsCodeC << "    " << "ERROR_CHECK_OBJECT(" + layerName + "_node);" << std::endl;
                        ofsCodeC << "    " << "ERROR_CHECK_STATUS(vxReleaseNode(&" << layerName + "_node));" << std::endl;
                    }
                    tmp = out;
                }
                else error("generateCode : Eltwise op=%d not supported\n", op);
            }
        }
        else if(type == "Scale") {
            int bias_term;
            std::stringstream ss(params); ss >> bias_term;

            std::string weights = layerName + "_W";
            std::string dim_weights = output + "_W";
            auto&& dim = tensorMap[dim_weights];
            if(codeType == "initialize") {
                ofsCodeC << "    vx_size " << weights << "_dims[1] = { " << dim[0] << " };" << std::endl;
                ofsCodeC << "    vx_tensor " << weights << ";" << std::endl;
                ofsCodeC << "    " << weights << " = vxCreateTensor(context,1, " << weights + "_dims, " << tensorType << ", " << fixedPosition << ");" << std::endl;
                ofsCodeC << "    " << "ERROR_CHECK_OBJECT(" << weights << "); " << std::endl;
                ofsCodeC << "    " << "ERROR_CHECK_STATUS(copyTensor(" << weights << ", dataFolder + \"/weights/" + layerName + ".f32\"));" << std::endl;
            }
            else if(codeType == "release") {
                ofsCodeC << "    " << "ERROR_CHECK_STATUS(vxReleaseTensor(&" << weights << "));" << std::endl;
            }
            declare_tensor_check[weights] = true;
            std::string bias = "NULL";
            if(bias_term) {
                bias = layerName + "_B";
                std::string dim_bias = output + "_B";
                dim = tensorMap[dim_bias];
                if(codeType == "initialize") {
                    ofsCodeC << "    vx_size " << bias << "_dims[1] = { " << dim[0] << " };" << std::endl;
                    ofsCodeC << "    vx_tensor " << bias << ";" << std::endl;
                    ofsCodeC << "    " << bias << " = vxCreateTensor(context,1, " << bias + "_dims, " << tensorType << ", " << fixedPosition << ");" << std::endl;
                    ofsCodeC << "    " << "ERROR_CHECK_OBJECT(" << bias << "); " << std::endl;
                    ofsCodeC << "    " << "ERROR_CHECK_STATUS(copyTensor(" << bias << ", dataFolder + \"/bias/" + layerName + ".f32\"));" << std::endl;
                }
                else if(codeType == "release") {
                    ofsCodeC << "    " << "ERROR_CHECK_STATUS(vxReleaseTensor(&" << bias << "));" << std::endl;
                }
                declare_tensor_check[bias] = true;
            }
            if(codeType == "initialize") {
                ofsCodeC << "    vx_node " << layerName << "_node;" << std::endl;
                ofsCodeC << "    " << layerName + "_node = " << "vxScaleLayer(graph, "
                                   << inputName +", "
                                   << layerName + "_W, "
                                   << bias + ", "
                                   << layerName << ");" << std::endl;
                ofsCodeC << "    " << "ERROR_CHECK_OBJECT(" + layerName + "_node);" << std::endl;
                ofsCodeC << "    " << "ERROR_CHECK_STATUS(vxReleaseNode(&" << layerName + "_node));" << std::endl;
            }
            else if(codeType == "release") {
            }
        }
        else if(type == "Concat") {
            if(codeType == "initialize") {
                ofsCodeC << "    vx_node " << layerName << "_node;" << std::endl;
                ofsCodeC << "    " <<  layerName + "_node = " << "vxConcatLayer(graph, ";
                ofsCodeC << layerName;
                int param_count = 0;
                for(int i = 4; i < node.size(); i++) {
                    std::string layerInputs = getIdentifierName(node[i]);
                    ofsCodeC << ", " << layerInputs;
                    param_count++;
                }
                while(param_count < 8) {
                    ofsCodeC << ", NULL";
                    param_count++;
                }
                ofsCodeC << " );" << std::endl;
                ofsCodeC << "    " << "ERROR_CHECK_OBJECT(" + layerName + "_node);" << std::endl;
                ofsCodeC << "    " << "ERROR_CHECK_STATUS(vxReleaseNode(&" << layerName + "_node));" << std::endl;
            }
        }
        else if(type == "Dropout") {
            //during inference dropout layer propogates input to output .
            if(codeType ==  "initialize") {
                ofsCodeC << "    vx_node " << layerName << "_node;" << std::endl;
                ofsCodeC << "    " << layerName + "_node = " << "vxCopyNode( graph, (vx_reference)" << inputName << ", (vx_reference)" << layerName << ");" << std::endl;
                ofsCodeC << "    " << "ERROR_CHECK_OBJECT(" + layerName + "_node);" << std::endl;
                ofsCodeC << "    " << "ERROR_CHECK_STATUS(vxReleaseNode(&" << layerName + "_node));" << std::endl;
            }
        }
        else if(type == "Softmax") {
            if(codeType == "initialize") {
                ofsCodeC << "    vx_node " << layerName << "_node;" << std::endl;
                ofsCodeC << "    " << layerName + "_node = " << "vxSoftmaxLayer(graph, " << inputName << ", " << layerName << ");" << std::endl;
                ofsCodeC << "    " << "ERROR_CHECK_OBJECT(" + layerName + "_node);" << std::endl;
                ofsCodeC << "    " << "ERROR_CHECK_STATUS(vxReleaseNode(&" << layerName + "_node));" << std::endl;
            }
        }
        else if(type == "Split") {
            if(codeType == "initialize") {
                ofsCodeC << "    vx_node " << layerName << "_node;" << std::endl;
                ofsCodeC << "    " << layerName + "_node = " << "vxCopyNode( graph, (vx_reference)"<< inputName << ", (vx_reference)" << layerName << ");" << std::endl;
                ofsCodeC << "    " << "ERROR_CHECK_OBJECT(" + layerName + "_node);" << std::endl;
                ofsCodeC << "    " << "ERROR_CHECK_STATUS(vxReleaseNode(&" << layerName + "_node));" << std::endl;
            }
        }
        else if(type == "SoftmaxWithLoss") {
            if(codeType == "initialize") {
                ofsCodeC << "    vx_node " << layerName << "_node;" << std::endl;
                ofsCodeC << "    " << layerName + "_node = " << "vxSoftmaxLayer(graph, " << inputName << ", " << layerName << ");" << std::endl;
                ofsCodeC << "    " << "ERROR_CHECK_OBJECT(" + layerName + "_node);" << std::endl;
                ofsCodeC << "    " << "ERROR_CHECK_STATUS(vxReleaseNode(&" << layerName + "_node));" << std::endl;
            }
        }
        else if(type == "Upsamle") {
            if(codeType == "initialize") {
                ofsCodeC << "    vx_node " << layerName << "_node;" << std::endl;
                ofsCodeC << "    " << layerName + "_node = " << "vxUpsampleNearestLayer(graph, " << inputName << ", " << layerName << ");" << std::endl;
                ofsCodeC << "    " << "ERROR_CHECK_OBJECT(" + layerName + "_node);" << std::endl;
                ofsCodeC << "    " << "ERROR_CHECK_STATUS(vxReleaseNode(&" << layerName + "_node));" << std::endl;
            }
        }
        if(codeType== "initialize")
            ofsCodeC << std::endl;
    }
}

void TFUtil::WriteGDF(
    std::ostream& ofsGDF,
    std::vector<std::vector<std::string>>& net,
    std::map<std::string,std::vector<int>>& tensorMap,
    std::string tensorType,
    int fixedPointPosition,
    std::string convertPolicy,
    std::string roundPolicy,
    bool isVirtualEnabled,
    std::string outputFolder,
    bool bFuseScaleLayer)
{
    std::map<std::string,bool> tensorCheck;
    ofsGDF << "import vx_nn" << std::endl;
    bool bfuse_scale_layer = bFuseScaleLayer;

    for(auto& node : net) {
        // create input/output tensor objects
        bool isFirstLayer = (&node == &net.front());
        bool isLastLayer = (&node == &net.back());
        for(size_t i = 4; i < node.size(); i++) {
            if(node[i] != "" && tensorCheck.find(node[i]) == tensorCheck.end()) {
                auto&& dim = tensorMap[node[i]];
                if((isVirtualEnabled && isFirstLayer) || (isVirtualEnabled && isLastLayer)) {
                    ofsGDF << "data " << node[i] << " = tensor:4,{" << dim[3] << "," << dim[2] << "," << dim[1] << "," << dim[0] << "}," << tensorType << "," << fixedPointPosition << std::endl;
                    tensorCheck[node[i]] = true;
                    if(!isLastLayer) {
                        ofsGDF << "read " << node[i] << " input.f32" << std::endl;
                    }
                }
                else {
                    if(isVirtualEnabled) {
                        ofsGDF << "data " << node[i] << " = virtual-tensor:4,{" << dim[3] << "," << dim[2] << "," << dim[1] << "," << dim[0] << "}," << tensorType << "," << fixedPointPosition << std::endl;
                        tensorCheck[node[i]] = true;
                    }
                    else {
                        ofsGDF << "data " << node[i] << " = tensor:4,{" << dim[3] << "," << dim[2] << "," << dim[1] << "," << dim[0] << "}," << tensorType << "," << fixedPointPosition << std::endl;
                        tensorCheck[node[i]]= true;
                        if(isFirstLayer) ofsGDF << "read data input.f32" << std::endl;
                    }
                }
            }
        }
        auto&& output = node[3];
        if (node[0] == "BatchNorm" && !isLastLayer && bfuse_scale_layer) {
            auto& next_node = *std::next(&node);
            if (next_node[0] == "Scale") {
                auto&& next_output = next_node[3];
                auto&& odim = tensorMap[next_output];
                tensorCheck[output] = true; // make sure next node doesn't create input tensor
                if(!tensorCheck[next_output]) {
                    if(!isVirtualEnabled) {
                        ofsGDF << "data " << next_output << " = tensor:4,{" << odim[3] << "," << odim[2] << "," << odim[1] << "," << odim[0] << "}," << tensorType << "," << fixedPointPosition << std::endl;
                    }
                    else {
                        if(!isLastLayer) {
                            ofsGDF << "data " << next_output << " = virtual-tensor:4,{" << odim[3] << "," << odim[2] << "," << odim[1] << "," << odim[0] << "}," << tensorType << "," << fixedPointPosition << std::endl;
                        }
                        else {
                            ofsGDF << "data " << next_output << " = tensor:4,{" << odim[3] << "," << odim[2] << "," << odim[1] << "," << odim[0] << "}," << tensorType << "," << fixedPointPosition << std::endl;
                        }
                    }
#if ENABLE_DIRECTIVE
                    ofsGDF << "directive " << next_output << " VX_DIRECTIVE_AMD_COPY_TO_OPENCL" << std::endl;
#endif
                }
                tensorCheck[next_output] = true;
                bfuse_scale_layer = true;
            }
        }

        if (node[0] == "Scale" && !isFirstLayer && bfuse_scale_layer) {
            auto& prev_node = *std::prev(&node);
            if (prev_node[0]=="BatchNorm")
            continue;
        }

        auto&& odim = tensorMap[output];
        if(!tensorCheck[output]) {
            if(!isVirtualEnabled) {
                ofsGDF << "data " << output << " = tensor:4,{" << odim[3] << "," << odim[2] << "," << odim[1] << "," << odim[0] << "}," << tensorType << "," << fixedPointPosition << std::endl;
            } else {
                if(!isLastLayer) {
                    ofsGDF << "data " << output << " = virtual-tensor:4,{" << odim[3] << "," << odim[2] << "," << odim[1] << "," << odim[0] << "}," << tensorType << "," << fixedPointPosition << std::endl;
                }
                else {
                    ofsGDF << "data " << output << " = tensor:4,{" << odim[3] << "," << odim[2] << "," << odim[1] << "," << odim[0] << "}," << tensorType << "," << fixedPointPosition << std::endl;
                }
            }
#if ENABLE_DIRECTIVE
            ofsGDF << "directive " << output << " VX_DIRECTIVE_AMD_COPY_TO_OPENCL" << std::endl;
#endif
        }
        tensorCheck[output] = true;

        // create node object
        auto&& type = node[0];
        auto&& params = node[1];
        std::string layer_name = getIdentifierName(node[3]);
        if(type == "Convolution") {
            std::stringstream ss(params);
            int k, kernel_w, kernel_h, stride_w, stride_h, pad_w, pad_h, dilation_w, dilation_h, bias_term, group;
            ss >> k >> kernel_w >> kernel_h >> stride_w >> stride_h >> pad_w >> pad_h >> dilation_w >> dilation_h >> bias_term >> group;

            if(group > 1) {
                // Slice the input tensor into group tensors
                auto&& dim_ip_grp = tensorMap[node[4]];

                for(int g = 0; g < group; g++) {
                    if(!isVirtualEnabled) {
                        ofsGDF << "data " << node[4] << "_grp" << g << " = tensor:4,{" << dim_ip_grp[3] << "," << dim_ip_grp[2] << "," << dim_ip_grp[1]/group << "," << dim_ip_grp[0] << "}," << tensorType << "," << fixedPointPosition << std::endl;
                    }
                    else {
                        ofsGDF << "data " << node[4] << "_grp" << g << " = virtual-tensor:4,{" << dim_ip_grp[3] << "," << dim_ip_grp[2] << "," << dim_ip_grp[1]/group << "," << dim_ip_grp[0] << "}," << tensorType << "," << fixedPointPosition << std::endl;
                    }
                }

                // Conv
                auto&& dim_op_grp = tensorMap[node[3]];
                auto&& dim_w = tensorMap[output + "_W"];

                for(int g = 0; g < group; g++) {
                    if(!isVirtualEnabled) {
                        ofsGDF << "data " << output << "_grp" << g << " = tensor:4,{" << dim_op_grp[3] << "," << dim_op_grp[2] << "," << dim_op_grp[1]/group << "," << dim_op_grp[0] << "}," << tensorType << "," << fixedPointPosition << std::endl;
                    }
                    else {
                        ofsGDF << "data " << output << "_grp" << g << " = virtual-tensor:4,{" << dim_op_grp[3] << "," << dim_op_grp[2] << "," << dim_op_grp[1]/group << "," << dim_op_grp[0] << "}," << tensorType << "," << fixedPointPosition << std::endl;
                    }

                    ofsGDF << "data " << output << "_grp" << g << "_W" << " = tensor:4,{" << dim_w[3] << "," << dim_w[2] << "," << dim_w[1]/group << "," << dim_w[0]/group << "}," << tensorType << "," << fixedPointPosition << std::endl;
                    ofsGDF << "init " << output << "_grp" << g << "_W weights/" << layer_name << "_grp" << g << ".f32" << std::endl;
#if ENABLE_DIRECTIVE
                    ofsGDF << "directive " << output << "_grp" << g << "_W" << " VX_DIRECTIVE_AMD_COPY_TO_OPENCL" << std::endl;
#endif

                    if(bias_term){
                        ofsGDF << "data " << output << "_grp" << g << "_B" << " = tensor:1,{" << k / group << "}," << tensorType << "," << fixedPointPosition << std::endl;
                        ofsGDF << "init " << output << "_grp" << g << "_B bias/" << layer_name << "_grp" << g << ".f32" << std::endl;
#if ENABLE_DIRECTIVE
                        ofsGDF << "directive " << output << "_grp" << g << "_B" << " VX_DIRECTIVE_AMD_COPY_TO_OPENCL" << std::endl;
#endif
                    }
                }

                ofsGDF << "data " << node[3] << "_params = " << " scalar:VX_TYPE_NN_CONVOLUTION_PARAMS,{" << pad_w << "," << pad_h << "," << convertPolicy << "," << roundPolicy << ",VX_NN_DS_SIZE_ROUNDING_FLOOR," << dilation_w-1 << "," << dilation_h-1 << "}" << std::endl;
                tensorCheck[output + "_W"] = true;
                if(bias_term) tensorCheck[output + "_B"] = true;

                ofsGDF << "node com.amd.nn_extension.slice_layer ";
                ofsGDF << node[4];
                for(int g = 0; g < group; g++) {
                    ofsGDF << " " << node[4] << "_grp" << g;
                }
                ofsGDF << std::endl;
#if ENABLE_DUMP_LAYER_DATA
                for(int g = 0; g < group; g++) {
                    ofsGDF << "write "<< node[4] << "_grp" << g << " out/"<< node[4] << "_grp" << g << ".f32" << std::endl;
                }
#endif

                for(int g = 0; g < group; g++) {
                    ofsGDF << "node org.khronos.nn_extension.convolution_layer ";
                    ofsGDF << node[4] << "_grp" << g << " ";
                    ofsGDF << node[3] << "_grp" << g << "_W ";
                    if(bias_term)
                        ofsGDF << node[3] << "_grp" << g << "_B ";
                    else
                        ofsGDF << "NULL ";
                    ofsGDF << node[3] << "_params ";
                    ofsGDF << node[3] << "_grp" << g << std::endl;

#if ENABLE_DUMP_LAYER_DATA
                    ofsGDF << "write "<< node[3] << "_grp" << g << " out/"<< layer_name << ".f32" << std::endl;
#endif
                }

                ofsGDF << "node com.amd.nn_extension.concat_layer ";
                ofsGDF << node[3];
                for(int g = 0; g < group; g++) {
                    ofsGDF << " " << node[3] << "_grp" << g;
                }
                ofsGDF << std::endl;
#if ENABLE_DUMP_LAYER_DATA
                for(int g = 0; g < group; g++) {
                    ofsGDF << "write "<< node[3] << "_grp" << g << " out/"<< node[3] << "_grp" << g << ".f32" << std::endl;
                }
#endif
            }
            else {
                std::string weights = output + "_W";
                auto&& dim = tensorMap[weights];
                ofsGDF << "data " << weights << " = tensor:4,{" << dim[3] << "," << dim[2] << "," << dim[1] << "," << dim[0] << "}," << tensorType << "," << fixedPointPosition << std::endl;
                ofsGDF << "init " << weights << " ";
                ofsGDF << "weights/" << layer_name << ".f32" << std::endl;
#if ENABLE_DIRECTIVE
                ofsGDF << "directive " << weights << " VX_DIRECTIVE_AMD_COPY_TO_OPENCL" << std::endl;
#endif
                tensorCheck[weights] = true;
                std::string bias = "NULL";
                if(bias_term) {
                    bias = output + "_B";
                    ofsGDF << "data " << bias << " = tensor:1,{" << k << "}," << tensorType << "," << fixedPointPosition << std::endl;
                    ofsGDF << "init " << bias << " ";
                    ofsGDF << "bias/"<< layer_name << ".f32" << std::endl;
#if ENABLE_DIRECTIVE
                    ofsGDF << "directive " << bias << " VX_DIRECTIVE_AMD_COPY_TO_OPENCL" << std::endl;
#endif
                    tensorCheck[bias] = true;
                }

                ofsGDF << "data " << node[3] << "_params = " << " scalar:VX_TYPE_NN_CONVOLUTION_PARAMS,{" << pad_w << "," << pad_h << "," << convertPolicy << "," << roundPolicy << ",VX_NN_DS_SIZE_ROUNDING_FLOOR," << dilation_w-1 << "," << dilation_h-1 << "}" << std::endl;
                ofsGDF << "node org.khronos.nn_extension.convolution_layer " << node[4] << " " << node[3] << "_W" << " " << bias << " "
                       << node[3] <<"_params"
                       << " " << node[3]
                       << std::endl;
#if ENABLE_DUMP_LAYER_DATA
                ofsGDF << "write "<< node[3] << " out/"<< layer_name << ".f32" << std::endl;
#endif
            }
        }
        else if (type == "Deconvolution") {
            std::stringstream ss(params);
            int k, kernel_w, kernel_h, stride_w, stride_h, pad_w, pad_h, dilation_w, dilation_h, bias_term;
            ss >> k >> kernel_w >> kernel_h >> stride_w >> stride_h >> pad_w >> pad_h >> dilation_w >> dilation_h >> bias_term;
            std::string weights = output + "_W";
            auto&& dim = tensorMap[weights];
            ofsGDF << "data " << weights << " = tensor:4,{" << dim[3] << "," << dim[2] << "," << dim[1] << "," << dim[0] << "}," << tensorType << "," << fixedPointPosition << std::endl;
            ofsGDF << "init " << weights << " weights/" << layer_name << ".f32" << std::endl;
#if ENABLE_DIRECTIVE
            ofsGDF << "directive " << weights << " VX_DIRECTIVE_AMD_COPY_TO_OPENCL" << std::endl;
#endif
            tensorCheck[weights] = true;
            std::string bias = "NULL";
            if(bias_term) {
                bias = output + "_B";
                ofsGDF << "data " << bias << " = tensor:1,{" << k << "}," << tensorType << "," << fixedPointPosition << std::endl;
                ofsGDF << "init " << bias << " bias/"<< layer_name << ".f32" << std::endl;
#if ENABLE_DIRECTIVE
                ofsGDF << "directive " << bias << " VX_DIRECTIVE_AMD_COPY_TO_OPENCL" << std::endl;
#endif
                tensorCheck[bias] = true;
            }

            ofsGDF << "data " << node[3] << "_params = " << " scalar:VX_TYPE_NN_DECONVOLUTION_PARAMS,{" << pad_w << "," << pad_h << "," << convertPolicy << "," << roundPolicy << "," << dilation_w-1 << "," << dilation_h-1 << "}" << std::endl;
            ofsGDF << "node org.khronos.nn_extension.deconvolution_layer " << node[4] << " " << node[3] << "_W" << " " << bias << " "
                   << node[3] <<"_params"
                   << " " << node[3]
                   << std::endl;
#if ENABLE_DUMP_LAYER_DATA
            ofsGDF << "write "<< node[3] << " out/"<< layer_name << ".f32" << std::endl;
#endif
        }
        else if(type == "Pooling") {
            std::stringstream ss(params);
            int kernel_w, kernel_h, stride_w, stride_h, pad_w, pad_h, pool;
            ss >> kernel_w >> kernel_h >> stride_w >> stride_h >> pad_w >> pad_h >> pool;
            if((pool != 0 && pool != 1)) error("writeGDF: pooling_layer supports only MAX and AVG\n");
            ofsGDF << "data " << node[3] <<"_type = " <<  " scalar:VX_TYPE_ENUM," << (pool == 0 ? "VX_NN_POOLING_MAX" : "VX_NN_POOLING_AVG")<< std::endl;
            ofsGDF << "data " << node[3] <<"_kernel_w = " << "scalar:VX_TYPE_SIZE," << kernel_w << std::endl;
            ofsGDF << "data " << node[3] <<"_kernel_h = " << "scalar:VX_TYPE_SIZE," << kernel_h << std::endl;
            ofsGDF << "data " << node[3] <<"_pad_w = " << "scalar:VX_TYPE_SIZE," << pad_w << std::endl;
            ofsGDF << "data " << node[3] <<"_pad_h = " << "scalar:VX_TYPE_SIZE," << pad_h << std::endl;
            ofsGDF << "data " << node[3] <<"_roundPolicy = " << " scalar:VX_TYPE_ENUM," << roundPolicy << std::endl;
            ofsGDF << "node org.khronos.nn_extension.pooling_layer " << node[4] << " "
                   << node[3] << "_type" << " "
                   << node[3] << "_kernel_w "
                   << node[3] << "_kernel_h "
                   << node[3] << "_pad_w "
                   << node[3] << "_pad_h "
                   << node[3] << "_roundPolicy"
                   << " " << node[3]
                   << std::endl;
#if ENABLE_DUMP_LAYER_DATA
            ofsGDF << "write "<< node[3] << " out/"<< layer_name << ".f32" << std::endl;
#endif
        }
        else if(type == "InnerProduct") {
            std::stringstream ss(params);
            int k, bias_term;
            ss >> k >> bias_term;
            std::string weights = output + "_W";
            auto&& dim = tensorMap[weights];
            ofsGDF << "data " << weights << " = tensor:4,{" << dim[3] << "," << dim[2] << "," << dim[1] << "," << dim[0] << "}," << tensorType << "," << fixedPointPosition << std::endl;
            ofsGDF << "init " << weights << " weights/"<< layer_name << ".f32" << std::endl;
#if ENABLE_DIRECTIVE
            ofsGDF << "directive " << weights << " VX_DIRECTIVE_AMD_COPY_TO_OPENCL" << std::endl;
#endif
            tensorCheck[weights] = true;
            std::string bias = "NULL";
            if(bias_term) {
                bias = output + "_B";
                ofsGDF << "data " << bias << " = tensor:1,{" << k << "}," << tensorType << "," << fixedPointPosition << std::endl;
                ofsGDF << "init " << bias << " bias/"<< layer_name << ".f32" << std::endl;
#if ENABLE_DIRECTIVE
                ofsGDF << "directive " << bias << " VX_DIRECTIVE_AMD_COPY_TO_OPENCL" << std::endl;
#endif
                tensorCheck[bias] = true;
            }
            ofsGDF << "data " << node[3] <<"_convertPolicy = " << " scalar:VX_TYPE_ENUM," << convertPolicy << std::endl;
            ofsGDF << "data " << node[3] <<"_roundPolicy =" << " scalar:VX_TYPE_ENUM,VX_" << roundPolicy << std::endl;
            ofsGDF << "node org.khronos.nn_extension.fully_connected_layer " << node[4] << " " << node[3] << "_W" << " " << bias << " "
                   << node[3] << "_convertPolicy "
                   << node[3] << "_roundPolicy"
                   << " " << node[3]
                   << std::endl;
#if ENABLE_DUMP_LAYER_DATA
            ofsGDF << "write "<< node[3] << " out/"<< layer_name << ".f32" << std::endl;
#endif
        }
        else if(type == "ReLU") {
            ofsGDF << "data " << node[3] << "_mode = " << " scalar:VX_TYPE_ENUM,VX_NN_ACTIVATION_RELU" << std::endl;
            ofsGDF << "data " << node[3] << "_param_a =" << " scalar:VX_TYPE_FLOAT32,0" << std::endl;
            ofsGDF << "data " << node[3] << "_param_b =" << " scalar:VX_TYPE_FLOAT32,0" << std::endl;
            ofsGDF << "node org.khronos.nn_extension.activation_layer " << node[4] << " "
                   << node[3] << "_mode "
                   << node[3] << "_param_a "
                   << node[3] << "_param_b"
                   << " " << node[3]
                   << std::endl;
#if ENABLE_DUMP_LAYER_DATA
            ofsGDF << "write "<< node[3] << " out/"<< layer_name << ".f32" << std::endl;
#endif
        }
        else if(type == "LRN") {
            int normalization_size;
            float alpha, beta, k;
            std::string norm_region;
            std::stringstream ss(params);
            ss >> normalization_size >> alpha >> beta >> norm_region >> k;
            std::string lrnType;
            if(norm_region == "1") lrnType = "VX_NN_NORMALIZATION_SAME_MAP";
            else lrnType = "VX_NN_NORMALIZATION_ACROSS_MAPS";
            ofsGDF << "data " << node[3] << "_mode = " << " scalar:VX_TYPE_ENUM," << lrnType << std::endl;
            ofsGDF << "data " << node[3] << "_size = " << " scalar:VX_TYPE_SIZE," << normalization_size << std::endl;
            ofsGDF << "data " << node[3] << "_alpha =" << " scalar:VX_TYPE_FLOAT32," << alpha << std::endl;
            ofsGDF << "data " << node[3] << "_beta ="  << " scalar:VX_TYPE_FLOAT32," << beta << std::endl;
            ofsGDF << "data " << node[3] << "_bias ="  << " scalar:VX_TYPE_FLOAT32," << k << std::endl;
            ofsGDF << "node org.khronos.nn_extension.normalization_layer " << node[4] << " "
                   << node[3] << "_mode "
                   << node[3] << "_size "
                   << node[3] << "_alpha "
                   << node[3] << "_beta "
                   << node[3] << " "
                   << node[3] << "_bias"
                   << std::endl;
#if ENABLE_DUMP_LAYER_DATA
            ofsGDF << "write "<< node[3] << " out/"<< layer_name << ".f32" << std::endl;
#endif
        }
        else if(type == "BatchNorm") {
            int use_global_stats, bias_term;
            float eps;
            std::stringstream ss(params);
            ss >> eps >> use_global_stats;
            std::string weights = output + "_W";
            auto&& dim = tensorMap[weights];
            ofsGDF << "data " << weights << " = tensor:1,{" << dim[0] << "}," << tensorType << "," << fixedPointPosition << std::endl;
            ofsGDF << "init " << weights << " weights/" << layer_name << ".f32" << std::endl;
#if ENABLE_DIRECTIVE
            ofsGDF << "directive " << weights << " VX_DIRECTIVE_AMD_COPY_TO_OPENCL" << std::endl;
#endif
            tensorCheck[weights] = true;
            std::string bias = output + "_B";
            dim = tensorMap[bias];
            ofsGDF << "data " << bias << " = tensor:1,{" << dim[0] << "}," << tensorType << "," << fixedPointPosition << std::endl;
            ofsGDF << "init " << bias << " bias/" << layer_name << ".f32" << std::endl;
#if ENABLE_DIRECTIVE
            ofsGDF << "directive " << bias << " VX_DIRECTIVE_AMD_COPY_TO_OPENCL" << std::endl;
#endif
            tensorCheck[bias] = true;
            bias = "NULL";
            if (bfuse_scale_layer) {
                // check next node. If scale extract weight and bias paramters for scale layer.
                auto& next_node = *std::next(&node);
                auto&& next_output = next_node[3];
                auto&& nn_params = next_node[1];
                std::string nn_layer_name = getIdentifierName(next_node[3]);
                weights = next_output + "_W";
                std::stringstream ss(nn_params);
                ss >> bias_term;
                dim = tensorMap[weights];
                ofsGDF << "data " << weights << " = tensor:1,{" << dim[0] << "}," << tensorType << "," << fixedPointPosition << std::endl;
                ofsGDF << "init " << weights << " weights/" << nn_layer_name << ".f32" << std::endl;
                tensorCheck[weights] = true;
                if(bias_term) {
                    bias = next_output + "_B";
                    ofsGDF << "data " << bias << " = tensor:1,{" << dim[0] << "}," << tensorType << "," << fixedPointPosition << std::endl;
                    ofsGDF << "init " << bias << " bias/"<< nn_layer_name << ".f32" << std::endl;
#if ENABLE_DIRECTIVE
                    ofsGDF << "directive " << bias << " VX_DIRECTIVE_AMD_COPY_TO_OPENCL" << std::endl;
#endif
                    tensorCheck[bias] = true;
                }
                ofsGDF << "data " << node[3] << "_eps ="  << " scalar:VX_TYPE_FLOAT32," << eps << std::endl;
                ofsGDF << "node com.amd.nn_extension.batch_normalization_layer " << node[4] << " " << node[3] << "_W "
                       << node[3] << "_B "
                       << weights << " "
                       << bias << " "
                       << node[3] << "_eps "
                       << next_node[3]
                       << std::endl;
#if ENABLE_DUMP_LAYER_DATA
                ofsGDF << "write "<< next_node[3] << " out/"<< layer_name << ".f32" << std::endl;
#endif
            }
            else {
                weights = output +"_W1";
                ofsGDF << "data " << weights << " = tensor:1,{" << dim[0] << "}," << tensorType << "," << fixedPointPosition << std::endl;
                // put default scale and bias term
                std::vector<float> scale_arr(dim[0]);
                std::fill(scale_arr.begin(), scale_arr.end(), 1.0);
                std::string fileName_weights = outputFolder + "/scale_init.f32";
                FILE *fp = fopen(fileName_weights.c_str(), "wb");
                if (fp) {
                    fwrite(scale_arr.data(), sizeof(float), dim[0], fp);
                    fclose(fp);
                }
                ofsGDF << "init " << weights << " scale_init.f32" << std::endl;
                ofsGDF << "data " << node[3] << "_eps ="  << " scalar:VX_TYPE_FLOAT32," << eps << std::endl;
                ofsGDF << "node com.amd.nn_extension.batch_normalization_layer " << node[4] << " " << node[3] << "_W "
                       << node[3] << "_B "
                       << weights << " "
                       << bias << " "
                       << node[3] << "_eps "
                       << output
                       << std::endl;
#if ENABLE_DUMP_LAYER_DATA
                ofsGDF << "write "<< output << " out/"<< layer_name << ".f32" << std::endl;
#endif
            }
        }
        else if(type == "Eltwise") {
            int op;
            std::stringstream ss(params);
            ss >> op;
            auto&& dim = tensorMap[node[3]];
            for(int i = 4; i < node.size(); i++) {
                auto&& idim = tensorMap[node[i]];
                if(dim[0] != idim[0] || dim[1] != idim[1] || dim[2] != idim[2] || dim[3] != idim[3])
                    error("writeGDF: Eltwise op=%d requires same dimension inputs: %s[%dx%dx%dx%d] != %s[%dx%dx%dx%d]\n", op, node[i].c_str(), idim[0], idim[1], idim[2], idim[3], node[i-1].c_str(), dim[0], dim[1], dim[2], dim[3]);
                dim = idim;
            }
            std::string tmp = node[4];
            for(int i = 5; i < node.size(); i++) {
                std::string out = node[3];
                if(i < node.size()-1) {
                    out += "tmp_" + std::to_string(i-4);
                    ofsGDF << "data " << out << " = tensor:4,{" << dim[3] << "," << dim[2] << "," << dim[1] << "," << dim[0] << "}," << tensorType << "," << fixedPointPosition << std::endl;
                    tensorCheck[out] = true;
                }
                if(op == 1) {
                    ofsGDF << "data " << node[3] <<"_convertPolicy =" << " scalar:VX_TYPE_ENUM," << convertPolicy << std::endl;
                    ofsGDF << "node org.khronos.openvx.tensor_add " << tmp << " " << getIdentifierName(node[i]) << " "
                           << node[3] << "_convertPolicy"
                           << " " << out
                           << std::endl;
                    tmp = out;
#if ENABLE_DUMP_LAYER_DATA
                    ofsGDF << "write "<< node[3] << " out/"<< layer_name << ".f32" << std::endl;
#endif
                }
                else error("writeGDF: Eltwise op=%d not supported\n", op);
            }
        }
        else if(type == "Scale") {
            int bias_term;
            auto&& type = node[0];
            auto&& params = node[1];
            std::string layer_name = getIdentifierName(node[3]);
            std::string weights = output + "_W";
            std::stringstream ss(params); ss >> bias_term;
            auto&& dim = tensorMap[weights];
            ofsGDF << "data " << weights << " = tensor:1,{" << dim[0] << "}," << tensorType << "," << fixedPointPosition << std::endl;
            ofsGDF << "init " << weights << " weights/" << layer_name << ".f32" << std::endl;
            tensorCheck[weights] = true;
#if ENABLE_DIRECTIVE
            ofsGDF << "directive " << weights << " VX_DIRECTIVE_AMD_COPY_TO_OPENCL" << std::endl;
#endif
            std::string bias = "NULL";
            if(bias_term) {
                bias = output + "_B ";
                ofsGDF << "data " << bias << " = tensor:1,{" << dim[0] << "}," << tensorType << "," << fixedPointPosition << std::endl;
                ofsGDF << "init " << bias << " bias/"<< layer_name << ".f32" << std::endl;
#if ENABLE_DIRECTIVE
                ofsGDF << "directive " << bias << " VX_DIRECTIVE_AMD_COPY_TO_OPENCL" << std::endl;
#endif
                tensorCheck[bias] = true;
            }

            ofsGDF << "node com.amd.nn_extension.scale_layer " << node[4] << " "
                   << node[3] << "_W "
                   << node[3] << "_B "
                   << node[3]
                   << std::endl;
#if ENABLE_DUMP_LAYER_DATA
            ofsGDF << "write "<< node[3] << " out/"<< layer_name << ".f32" << std::endl;
#endif
        }
        else if(type == "Concat") {
            ofsGDF << "node com.amd.nn_extension.concat_layer";
            ofsGDF << " " << node[3];
            for(int i = 4; i < node.size(); i++) {
                ofsGDF << " " << node[i];
            }
            ofsGDF << std::endl;
#if ENABLE_DUMP_LAYER_DATA
            ofsGDF << "write "<< node[3] << " out/"<< layer_name << ".f32" << std::endl;
#endif
        }
        else if(type == "Dropout") {
            //during inference dropout layer copies its input to output.
            ofsGDF << "node org.khronos.openvx.copy " << node[4] << " " << node[3] << std::endl;
#if ENABLE_DUMP_LAYER_DATA
            ofsGDF << "write "<< node[3] << " out/"<< layer_name << ".f32" << std::endl;
#endif
        }
        else if(type == "Softmax") {
            ofsGDF << "node org.khronos.nn_extension.softmax_layer " << node[4]
                   << " " << node[3]
                   << std::endl;
#if ENABLE_DUMP_LAYER_DATA
            ofsGDF << "write "<< node[3] << " out/"<< layer_name << ".f32" << std::endl;
#endif
        }
        else if(type == "Split") {
            ofsGDF << "node org.khronos.openvx.copy " << node[4] << " " << node[3] << std::endl;
#if ENABLE_DUMP_LAYER_DATA
            ofsGDF << "write "<< node[3] << " out/"<< layer_name << ".f32" << std::endl;
#endif
        }
        else if(type == "SoftmaxWithLoss") {
            ofsGDF << "node org.khronos.nn_extension.softmax_layer " << node[4]
                   << " " << node[5]
                   << std::endl;
#if ENABLE_DUMP_LAYER_DATA
            ofsGDF << "write "<< node[3] << " out/"<< layer_name << ".f32" << std::endl;
#endif
        }
        else if(type == "Upsamle") {
                    ofsGDF << "node com.amd.nn_extension.upsample_nearest_layer " << node[4]
                           << " " << node[3]
                           << std::endl;
        #if ENABLE_DUMP_LAYER_DATA
                    ofsGDF << "write "<< node[3] << " out/"<< layer_name << ".f32" << std::endl;
        #endif
                }
        else {
            ofsGDF << "# "
                   << std::left << std::setw(16) << node[0]
                   << std::left << std::setw(24) << node[1]
                   << std::left << std::setw(32) << node[3]
                      ;
            for(size_t i = 4; i < node.size(); i++)
                ofsGDF << std::left << std::setw(32) << node[i];
            ofsGDF << std::endl;
        }
        if(isLastLayer) {
                    ofsGDF << "write " << node[3] << " output.f32" << std::endl;
                    auto&& odim = tensorMap[node[3]];
                    printf("#OUTPUT-TENSOR: %s %d %d %d %d\n", node[3].c_str(), odim[0], odim[1], odim[2], odim[3]);
                }
        ofsGDF << std::endl;
    }
}

void TFUtil::GenerateCopyTensorCode(std::ostream& ofsCodeC)
{
    ofsCodeC << "static vx_status copyTensor(vx_tensor tensor, std::string fileName, vx_enum usage = VX_WRITE_ONLY)" << std::endl
             << "{" << std::endl
             << "    vx_size num_of_dims = 4, dims[4] = { 1, 1, 1, 1 }, stride[4];" << std::endl
             << "    vxQueryTensor(tensor, VX_TENSOR_NUMBER_OF_DIMS, &num_of_dims, sizeof(num_of_dims));" << std::endl
             << "    vxQueryTensor(tensor, VX_TENSOR_DIMS, &dims, sizeof(dims[0])*num_of_dims);" << std::endl
             << "    vx_size count = dims[0] * dims[1] * dims[2] * dims[3];" << std::endl
             << "    vx_map_id map_id;" << std::endl
             << "    float * ptr;" << std::endl
             << "    vx_status status = vxMapTensorPatch(tensor, num_of_dims, nullptr, nullptr, &map_id, stride, (void **)&ptr, usage, VX_MEMORY_TYPE_HOST, 0);" << std::endl
             << "    if(status) {" << std::endl
             << "        std::cerr << \"ERROR: vxMapTensorPatch() failed for \" << fileName << std::endl;" << std::endl
             << "        return -1;" << std::endl
             << "    }" << std::endl
             << "    FILE * fp = fopen(fileName.c_str(), usage == VX_WRITE_ONLY ? \"rb\" : \"wb\");" << std::endl
             << "    if(!fp) {" << std::endl
             << "        std::cerr << \"ERROR: unable to open: \" << fileName << std::endl;" << std::endl
             << "        return -1;" << std::endl
             << "    }" << std::endl
             << "    if(usage == VX_WRITE_ONLY) {" << std::endl
             << "        vx_size n = fread(ptr, sizeof(float), count, fp);" << std::endl
             << "        if(n != count) {" << std::endl
             << "            std::cerr << \"ERROR: expected float[\" << count << \"], but got float[\" << n << \"] in \" << fileName << std::endl;" << std::endl
             << "            return -1;" << std::endl
             << "        }" << std::endl
             << "    }" << std::endl
             << "    else {" << std::endl
             << "        fwrite(ptr, sizeof(float), count, fp);" << std::endl
             << "    }" << std::endl
             << "    fclose(fp);" << std::endl
             << "    status = vxUnmapTensorPatch(tensor, map_id);" << std::endl
             << "    if(status) {" << std::endl
             << "        std::cerr << \"ERROR: vxUnmapTensorPatch() failed for \" << fileName << std::endl;" << std::endl
             << "        return -1;" << std::endl
             << "    }" << std::endl
             << "    return 0;" << std::endl
             << "}" << std::endl << std::endl;
}

void TFUtil::GenerateCode(
    std::ostream& ofsCodeH,
    std::ostream& ofsCodeC,
    std::ofstream& ofsCodeM,
    std::ofstream& ofsCodeA,
    std::ofstream& ofsCodeD,
    std::vector<std::vector<std::string>>& net,
    std::map<std::string,std::vector<int>>& tensorMap,
    std::string tensorType,
    int fixedPointPosition,
    std::string convertPolicy,
    std::string roundPolicy,
    bool isVirtualEnabled,
    std::string outputFolder,
    bool bFuseScaleLayer)
{
    ////
    // generate .h file
    //
    ofsCodeH << "#ifndef annmodule_h" <<  std::endl
             << "#define annmodule_h" <<  std::endl
             <<                         std::endl
             << "#include <VX/vx.h>" << std::endl
             <<                         std::endl
             << "extern \"C\" {"     << std::endl
             << "    VX_API_ENTRY void     VX_API_CALL annGetTensorDimensions(vx_size dimInput[4], vx_size dimOutput[4]);" << std::endl
             << "    VX_API_ENTRY vx_graph VX_API_CALL annCreateGraph(vx_context context, vx_tensor input, vx_tensor output, const char * options);" << std::endl
             << "};"                 << std::endl
             <<                         std::endl
             << "#endif" <<  std::endl;

    ////
    // generate .cpp file
    //
    ofsCodeC << "#include \"annmodule.h\"" << std::endl << std::endl;
    ofsCodeC << "#include <vx_ext_amd.h>" << std::endl;
    ofsCodeC << "#include <VX/vx_khr_nn.h>" << std::endl;
    ofsCodeC << "#include <vx_amd_nn.h>" << std::endl<< std::endl;
    ofsCodeC << "#include <iostream>" << std::endl;
    ofsCodeC << "#include <stdio.h>" << std::endl;
    ofsCodeC << "#include <stdlib.h>" << std::endl << std::endl;

    ofsCodeC << "#define ERROR_CHECK_STATUS(call) { vx_status status = (call); if(status != VX_SUCCESS) { vxAddLogEntry(NULL, status, \"ERROR: failed with status = (%d) at \" __FILE__ \"#%d\", status, __LINE__); return nullptr; } }" << std::endl;
    ofsCodeC << "#define ERROR_CHECK_OBJECT(obj) { vx_status status = vxGetStatus((vx_reference)(obj)); if(status != VX_SUCCESS) { vxAddLogEntry((vx_reference)(obj), status, \"ERROR: failed with status = (%d) at \" __FILE__ \"#%d\", status, __LINE__); return nullptr; } }" << std::endl << std::endl;

    GenerateCopyTensorCode(ofsCodeC);

    auto&& input = net[0][4];
    auto&& output = net[net.size()-1][3];
    auto&& idim = tensorMap[input];
    auto&& odim = tensorMap[output];
    ofsCodeC << "VX_API_ENTRY void VX_API_CALL annGetTensorDimensions(vx_size dimInput[4], vx_size dimOutput[4])" << std::endl
             << "{" << std::endl
             << "    dimInput[0] = " << idim[3] << ";" << std::endl
             << "    dimInput[1] = " << idim[2] << ";" << std::endl
             << "    dimInput[2] = " << idim[1] << ";" << std::endl
             << "    dimInput[3] = " << idim[0] << ";" << std::endl
             << "    dimOutput[0] = " << odim[3] << ";" << std::endl
             << "    dimOutput[1] = " << odim[2] << ";" << std::endl
             << "    dimOutput[2] = " << odim[1] << ";" << std::endl
             << "    dimOutput[3] = " << odim[0] << ";" << std::endl
             << "}" << std::endl << std::endl;

    ofsCodeC << "VX_API_ENTRY vx_graph VX_API_CALL annCreateGraph(vx_context context, vx_tensor " << input << ", vx_tensor " << output << ", const char * dataFolder_)" << std::endl;
    ofsCodeC << "{" << std::endl;
    ofsCodeC << "    // load neural network extension kernels" << std::endl;
    ofsCodeC << "    ERROR_CHECK_STATUS(vxLoadKernels(context,\"vx_nn\"));" << std::endl;
    ofsCodeC << std::endl;
    ofsCodeC << "    // create graph" << std::endl;
    ofsCodeC << "    vx_graph graph = vxCreateGraph(context); " << std::endl;
    ofsCodeC << "    ERROR_CHECK_OBJECT(graph);" << std::endl;
    ofsCodeC << std::endl;
    ofsCodeC << "    // get dataFolder option" << std::endl;
    ofsCodeC << "    std::string dataFolder = dataFolder_ ? dataFolder_ : \".\", fileName;" << std::endl;
    ofsCodeC << std::endl;
    ofsCodeC << "    ////" << std::endl;
    ofsCodeC << "    // initialize the graph" << std::endl;
    WriteVXCode(ofsCodeC, net, tensorMap, tensorType, fixedPointPosition, convertPolicy, roundPolicy, isVirtualEnabled, bFuseScaleLayer, outputFolder, "initialize");
    ofsCodeC << "    ////" << std::endl;
    ofsCodeC << "    // release intermediate objects" << std::endl;
    WriteVXCode(ofsCodeC, net, tensorMap, tensorType, fixedPointPosition, convertPolicy, roundPolicy, isVirtualEnabled, bFuseScaleLayer, outputFolder, "release");
    ofsCodeC << std::endl;
    ofsCodeC << "    ////" << std::endl;
    ofsCodeC << "    // verify the built graph" << std::endl;
    ofsCodeC << "    ERROR_CHECK_STATUS(vxVerifyGraph(graph));" << std::endl;
    ofsCodeC << std::endl;
    ofsCodeC << "    return graph;" << std::endl;
    ofsCodeC << "}" << std::endl;

    /////
    // generate CMakeLists.txt
    //
    ofsCodeM << "cmake_minimum_required (VERSION 2.8)" << std::endl;
    ofsCodeM << "project (annmodule)" << std::endl;
    ofsCodeM << "set (CMAKE_CXX_STANDARD 11)" << std::endl;
    ofsCodeM << "list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)" << std::endl;
    ofsCodeM << "find_package(OpenCL     REQUIRED)" << std::endl;
    ofsCodeM << "include_directories (${OpenCL_INCLUDE_DIRS} ${OpenCL_INCLUDE_DIRS}/Headers )" << std::endl;
    ofsCodeM << "include_directories (/opt/rocm/include)" << std::endl;
    ofsCodeM << "link_directories    (/opt/rocm/lib)" << std::endl;
    ofsCodeM << "list(APPEND SOURCES annmodule.cpp)" << std::endl;
    ofsCodeM << "add_library(${PROJECT_NAME} SHARED ${SOURCES})" << std::endl;
    ofsCodeM << "set(CMAKE_CXX_FLAGS \"${CMAKE_CXX_FLAGS} -msse4.2 -std=c++11\")" << std::endl;
    ofsCodeM << "target_link_libraries(${PROJECT_NAME} openvx vx_nn pthread)" << std::endl;
    ofsCodeM << "add_executable(annunit annunit.cpp)" << std::endl;
    ofsCodeM << "target_link_libraries(annunit openvx vx_nn pthread ${PROJECT_NAME})" << std::endl;

    /////
    // generate simple application
    //
    ofsCodeA << "#include \"annmodule.h\"" << std::endl ;
    ofsCodeA << "#include <vx_ext_amd.h>" << std::endl;
    ofsCodeA << "#include <iostream>" << std::endl;
    ofsCodeA << "#include <stdio.h>" << std::endl;
    ofsCodeA << "#include <string>" << std::endl;
    ofsCodeA << std::endl;

    GenerateCopyTensorCode(ofsCodeA);

    ofsCodeA << "int main(int argc , char ** argv)" << std::endl;
    ofsCodeA << "{" << std::endl;
    ofsCodeA << "    // get module configuration" << std::endl;
    ofsCodeA << "    vx_size dimInput[4] = { 0 }, dimOutput[4] = { 0 };" << std::endl;
    ofsCodeA << "    annGetTensorDimensions(dimInput, dimOutput);" << std::endl;
    ofsCodeA << "    printf(\"OK: annGetTensorDimensions() => [input %ldx%ldx%ldx%ld] [output %ldx%ldx%ldx%ld]\\n\", dimInput[0], dimInput[1], dimInput[2], dimInput[3], dimOutput[0], dimOutput[1], dimOutput[2], dimOutput[3]);" << std::endl;
    ofsCodeA << std::endl;
    ofsCodeA << "    // create context, input, output, and graph" << std::endl;
    ofsCodeA << "    vx_context context = vxCreateContext();" << std::endl;
    ofsCodeA << "    if(vxGetStatus((vx_reference)context)) {" << std::endl;
    ofsCodeA << "        printf(\"ERROR: vxCreateContext() failed\\n\");" << std::endl;
    ofsCodeA << "        return -1;" << std::endl;
    ofsCodeA << "    }" << std::endl;
    ofsCodeA << "    vx_tensor input = vxCreateTensor(context, 4, dimInput, VX_TYPE_FLOAT32, 0);" << std::endl;
    ofsCodeA << "    if(vxGetStatus((vx_reference)input)) {" << std::endl;
    ofsCodeA << "        printf(\"ERROR: vxCreateTensor(input,4,{%ld,%ld,%ld,%ld}) failed\\n\", dimInput[0], dimInput[1], dimInput[2], dimInput[3]);" << std::endl;
    ofsCodeA << "        return -1;" << std::endl;
    ofsCodeA << "    }" << std::endl;
    ofsCodeA << "    vx_tensor output = vxCreateTensor(context, 4, dimOutput, VX_TYPE_FLOAT32, 0);" << std::endl;
    ofsCodeA << "    if(vxGetStatus((vx_reference)output)) {" << std::endl;
    ofsCodeA << "        printf(\"ERROR: vxCreateTensor(output,4,{%ld,%ld,%ld,%ld}) failed\\n\", dimOutput[0], dimOutput[1], dimOutput[2], dimOutput[3]);" << std::endl;
    ofsCodeA << "        return -1;" << std::endl;
    ofsCodeA << "    }" << std::endl;
    ofsCodeA << std::endl;
    ofsCodeA << "    // build graph from the module" << std::endl;
    ofsCodeA << "    vx_graph graph = annCreateGraph(context, input, output, argc > 1 ? argv[1] : nullptr);" << std::endl;
    ofsCodeA << "    if(vxGetStatus((vx_reference)graph)) {" << std::endl;
    ofsCodeA << "        printf(\"ERROR: annCreateGraph(...,%s) failed\\n\", argv[1]);" << std::endl;
    ofsCodeA << "        return -1;" << std::endl;
    ofsCodeA << "    }" << std::endl;
    ofsCodeA << std::endl;
    ofsCodeA << "    if(argc > 2 && copyTensor(input, argv[2], VX_WRITE_ONLY) < 0) {" << std::endl;
    ofsCodeA << "        return -1;" << std::endl;
    ofsCodeA << "    }" << std::endl;
    ofsCodeA << std::endl;
    ofsCodeA << "    vx_status status = vxProcessGraph(graph);" << std::endl;
    ofsCodeA << "    if(status != VX_SUCCESS) {" << std::endl;
    ofsCodeA << "        printf(\"ERROR: vxProcessGraph() failed (%d)\\n\", status);" << std::endl;
    ofsCodeA << "        return -1;" << std::endl;
    ofsCodeA << "    }" << std::endl;
    ofsCodeA << std::endl;
    ofsCodeA << "    if(argc > 3 && copyTensor(output, argv[3], VX_READ_ONLY) < 0) {" << std::endl;
    ofsCodeA << "        return -1;" << std::endl;
    ofsCodeA << "    }" << std::endl;
    ofsCodeA << std::endl;
    ofsCodeA << "    // release resources" << std::endl;
    ofsCodeA << "    vxReleaseGraph(&graph);" << std::endl;
    ofsCodeA << "    vxReleaseTensor(&input);" << std::endl;
    ofsCodeA << "    vxReleaseTensor(&output);" << std::endl;
    ofsCodeA << "    vxReleaseContext(&context);" << std::endl;
    ofsCodeA << std::endl;
    ofsCodeA << "    return 0;"<< std::endl;
    ofsCodeA << "}" << std::endl;

    ofsCodeD << "find_path(OPENCL_INCLUDE_DIRS"  << std::endl;
    ofsCodeD << "NAMES OpenCL/cl.h CL/cl.h" << std::endl;
    ofsCodeD << "HINTS" << std::endl;
    ofsCodeD << "${OPENCL_ROOT}/include" << std::endl;
    ofsCodeD << "$ENV{AMDAPPSDKROOT}/include" << std::endl;
    ofsCodeD << "PATHS" << std::endl;
    ofsCodeD << "/usr/include" << std::endl;
    ofsCodeD << "/usr/local/include" << std::endl;
    ofsCodeD << "/opt/rocm/opencl/include" << std::endl;
    ofsCodeD << "DOC \"OpenCL header file path\"" << std::endl;
    ofsCodeD << ")" << std::endl;
    ofsCodeD << "mark_as_advanced( OPENCL_INCLUDE_DIRS )" << std::endl << std::endl;
    ofsCodeD << "if(\"${CMAKE_SIZEOF_VOID_P}\" EQUAL \"8\")" << std::endl;
    ofsCodeD << "   find_library( OPENCL_LIBRARIES" << std::endl;
    ofsCodeD << "       NAMES OpenCL" << std::endl;
    ofsCodeD << "       HINTS" << std::endl;
    ofsCodeD << "       ${OPENCL_ROOT}/lib" << std::endl;
    ofsCodeD << "       $ENV{AMDAPPSDKROOT}/lib" << std::endl;
    ofsCodeD << "       DOC \"OpenCL dynamic library path\"" << std::endl;
    ofsCodeD << "       PATH_SUFFIXES x86_64 x64 x86_64/sdk" << std::endl;
    ofsCodeD << "       PATHS" << std::endl;
    ofsCodeD << "       /usr/lib" << std::endl;
    ofsCodeD << "       /opt/rocm/opencl/lib" << std::endl;
    ofsCodeD << "       )" << std::endl;
    ofsCodeD << "else( )" << std::endl;
    ofsCodeD << "   find_library( OPENCL_LIBRARIES" << std::endl;
    ofsCodeD << "       NAMES OpenCL" << std::endl;
    ofsCodeD << "       HINTS" << std::endl;
    ofsCodeD << "       ${OPENCL_ROOT}/lib" << std::endl;
    ofsCodeD << "       $ENV{AMDAPPSDKROOT}/lib" << std::endl;
    ofsCodeD << "       DOC \"OpenCL dynamic library path\"" << std::endl;
    ofsCodeD << "       PATH_SUFFIXES x86 Win32" << std::endl;
    ofsCodeD << "       PATHS" << std::endl;
    ofsCodeD << "       /usr/lib" << std::endl;
    ofsCodeD << "       )" << std::endl;
    ofsCodeD << "endif( )" << std::endl;
    ofsCodeD << "mark_as_advanced( OPENCL_LIBRARIES )" << std::endl << std::endl;
    ofsCodeD << "include( FindPackageHandleStandardArgs )" << std::endl;
    ofsCodeD << "find_package_handle_standard_args( OPENCL DEFAULT_MSG OPENCL_LIBRARIES OPENCL_INCLUDE_DIRS )" << std::endl;
    ofsCodeD << "set(OpenCL_FOUND ${OPENCL_FOUND} CACHE INTERNAL \"\")" << std::endl;
    ofsCodeD << "set(OpenCL_LIBRARIES ${OPENCL_LIBRARIES} CACHE INTERNAL \"\")" << std::endl;
    ofsCodeD << "set(OpenCL_INCLUDE_DIRS ${OPENCL_INCLUDE_DIRS} CACHE INTERNAL \"\")" << std::endl;
    ofsCodeD << "if( NOT OPENCL_FOUND )" << std::endl;
    ofsCodeD << "   message( STATUS \"FindOpenCL looked for libraries named: OpenCL\" )" << std::endl;
    ofsCodeD << "endif()" << std::endl;
}

int TFUtil::GenerateGDF(int inputDim[4], std::string outputFolder, const char* tensorType, const char* convertPolicy, const char* roundPolicy, int fixedPointPosition, bool isVirtualEnabled,  bool bFuseScaleWithBatchNorm){

    //Parse the given model
    this->ParseTFModel(inputDim, outputFolder);

    // generate tensorMap for given input dimensions
    if(this->CalculateTensorDim(inputDim, mTensorMap) < 0) {
        return -1;
    }

    //Write GDF
    mTensorType = tensorType;
    mConvertPolicy = convertPolicy;
    mRoundPolicy = roundPolicy;
    mFixedPointPosition = fixedPointPosition;
    mIsVirtualEnabled = isVirtualEnabled;
    mBFuseScaleWithBatchNorm = bFuseScaleWithBatchNorm;

    std::ofstream ofsGDF(outputFolder + "/net.gdf", std::ios::binary);
    this->WriteGDF(ofsGDF, this->mNet, mTensorMap, tensorType, fixedPointPosition, convertPolicy, roundPolicy, isVirtualEnabled, outputFolder, bFuseScaleWithBatchNorm);

    //temp
    GenerateVXC(outputFolder);
    return 0;
}

void TFUtil::GenerateVXC(std::string outputFolder){
    std::ofstream ofsCodeH(outputFolder + "/annmodule.h", std::ios::binary);
    std::ofstream ofsCodeC(outputFolder + "/annmodule.cpp", std::ios::binary);
    std::ofstream ofsCodeM(outputFolder + "/CMakeLists.txt", std::ios::binary);
    std::ofstream ofsCodeA(outputFolder + "/annunit.cpp", std::ios::binary);
    std::string dir = outputFolder + "/cmake";
    mkdir(dir.c_str(), 0777);
    std::ofstream ofsCodeD(dir + "/FindOpenCL.cmake", std::ios::binary);
    GenerateCode(ofsCodeH, ofsCodeC, ofsCodeM, ofsCodeA, ofsCodeD, this->mNet, mTensorMap, mTensorType, mFixedPointPosition, mConvertPolicy, mRoundPolicy, mIsVirtualEnabled, outputFolder, mBFuseScaleWithBatchNorm);
}
