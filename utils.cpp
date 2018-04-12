#include "tfutil.h"
#include <regex>




// These are all common classes it's handy to reference with no namespace.
using tensorflow::Flag;
using tensorflow::Tensor;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::int32;

// Takes a file name, and loads a list of labels from it, one per line, and
// returns a vector of the strings. It pads with empty strings so the length
// of the result is a multiple of 16, because our model expects that.
Status ReadLabelsFile(const string& file_name, std::vector<string>* result,
                      size_t* found_label_count) {
  std::ifstream file(file_name);
  if (!file) {
    return tensorflow::errors::NotFound("Labels file ", file_name,
                                        " not found.");
  }
  result->clear();
  string line;
  while (std::getline(file, line)) {
    result->push_back(line);
  }
  *found_label_count = result->size();
  const int padding = 16;
  while (result->size() % padding) {
    result->emplace_back();
  }
  return Status::OK();
}

static Status ReadEntireFile(tensorflow::Env* env, const string& filename,
                             Tensor* output) {
  tensorflow::uint64 file_size = 0;
  TF_RETURN_IF_ERROR(env->GetFileSize(filename, &file_size));

  string contents;
  contents.resize(file_size);

  std::unique_ptr<tensorflow::RandomAccessFile> file;
  TF_RETURN_IF_ERROR(env->NewRandomAccessFile(filename, &file));

  tensorflow::StringPiece data;
  TF_RETURN_IF_ERROR(file->Read(0, file_size, &data, &(contents)[0]));
  if (data.size() != file_size) {
    return tensorflow::errors::DataLoss("Truncated read of '", filename,
                                        "' expected ", file_size, " got ",
                                        data.size());
  }
  output->scalar<string>()() = data.ToString();
  return Status::OK();
}

// Given an image file name, read in the data, try to decode it as an image,
// resize it to the requested size, and then scale the values as desired.
Status ReadTensorFromImageFile(const string& file_name, const int input_height,
                               const int input_width, const float input_mean,
                               const float input_std,
                               std::vector<Tensor>* out_tensors) {
  auto root = tensorflow::Scope::NewRootScope();
  using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

  string input_name = "file_reader";
  string output_name = "normalized";

  // read file_name into a tensor named input
  Tensor input(tensorflow::DT_STRING, tensorflow::TensorShape());
  TF_RETURN_IF_ERROR(
      ReadEntireFile(tensorflow::Env::Default(), file_name, &input));

  // use a placeholder to read input data
  auto file_reader =
      Placeholder(root.WithOpName("input"), tensorflow::DataType::DT_STRING);

  std::vector<std::pair<string, tensorflow::Tensor>> inputs = {
      {"input", input},
  };

  // Now try to figure out what kind of file it is and decode it.
  const int wanted_channels = 3;
  tensorflow::Output image_reader;
  if (tensorflow::StringPiece(file_name).ends_with(".png")) {
    image_reader = DecodePng(root.WithOpName("png_reader"), file_reader,
                             DecodePng::Channels(wanted_channels));
  } else if (tensorflow::StringPiece(file_name).ends_with(".gif")) {
    // gif decoder returns 4-D tensor, remove the first dim
    image_reader =
        Squeeze(root.WithOpName("squeeze_first_dim"),
                DecodeGif(root.WithOpName("gif_reader"), file_reader));
  } /*else if (tensorflow::StringPiece(file_name).ends_with(".bmp")) {
    image_reader = DecodeBmp(root.WithOpName("bmp_reader"), file_reader);
  }*/ else {
    // Assume if it's neither a PNG nor a GIF then it must be a JPEG.
    image_reader = DecodeJpeg(root.WithOpName("jpeg_reader"), file_reader,
                              DecodeJpeg::Channels(wanted_channels));
  }
  // Now cast the image data to float so we can do normal math on it.
  auto float_caster =
      Cast(root.WithOpName("float_caster"), image_reader, tensorflow::DT_FLOAT);
  // The convention for image ops in TensorFlow is that all images are expected
  // to be in batches, so that they're four-dimensional arrays with indices of
  // [batch, height, width, channel]. Because we only have a single image, we
  // have to add a batch dimension of 1 to the start with ExpandDims().
  auto dims_expander = ExpandDims(root, float_caster, 0);
  // Bilinearly resize the image to fit the required dimensions.
  auto resized = ResizeBilinear(
      root, dims_expander,
      Const(root.WithOpName("size"), {input_height, input_width}));
  // Subtract the mean and divide by the scale.
  Div(root.WithOpName(output_name), Sub(root, resized, {input_mean}),
      {input_std});

  // This runs the GraphDef network definition that we've just constructed, and
  // returns the results in the output tensor.
  tensorflow::GraphDef graph;
  TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));

  std::unique_ptr<tensorflow::Session> session(
      tensorflow::NewSession(tensorflow::SessionOptions()));
  TF_RETURN_IF_ERROR(session->Create(graph));
  TF_RETURN_IF_ERROR(session->Run({inputs}, {output_name}, {}, out_tensors));
  return Status::OK();
}

// Reads a model graph definition from disk, and creates a session object you
// can use to run it.
Status LoadGraph(const string& graph_file_name,
                 std::unique_ptr<tensorflow::Session>* session) {
  tensorflow::GraphDef graph_def;
  Status load_graph_status =
      ReadBinaryProto(tensorflow::Env::Default(), graph_file_name, &graph_def);
  if (!load_graph_status.ok()) {
    return tensorflow::errors::NotFound("Failed to load compute graph at '",
                                        graph_file_name, "'");
  }
  session->reset(tensorflow::NewSession(tensorflow::SessionOptions()));
  Status session_create_status = (*session)->Create(graph_def);
  if (!session_create_status.ok()) {
    return session_create_status;
  }
  return Status::OK();
}

// Analyzes the output of the Inception graph to retrieve the highest scores and
// their positions in the tensor, which correspond to categories.
Status GetTopLabels(const std::vector<Tensor>& outputs, int how_many_labels,
                    Tensor* indices, Tensor* scores) {
  auto root = tensorflow::Scope::NewRootScope();
  using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

  string output_name = "top_k";
  TopK(root.WithOpName(output_name), outputs[0], how_many_labels);
  // This runs the GraphDef network definition that we've just constructed, and
  // returns the results in the output tensors.
  tensorflow::GraphDef graph;
  TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));

  std::unique_ptr<tensorflow::Session> session(
      tensorflow::NewSession(tensorflow::SessionOptions()));
  TF_RETURN_IF_ERROR(session->Create(graph));
  // The TopK node returns two outputs, the scores and their original indices,
  // so we have to append :0 and :1 to specify them both.
  std::vector<Tensor> out_tensors;
  TF_RETURN_IF_ERROR(session->Run({}, {output_name + ":0", output_name + ":1"},
                                  {}, &out_tensors));
  *scores = out_tensors[0];
  *indices = out_tensors[1];
  return Status::OK();
}

// Given the output of a model run, and the name of a file containing the labels
// this prints out the top five highest-scoring values.
Status PrintTopLabels(const std::vector<Tensor>& outputs,
                      const string& labels_file_name) {
  std::vector<string> labels;
  size_t label_count;
  Status read_labels_status =
      ReadLabelsFile(labels_file_name, &labels, &label_count);
  if (!read_labels_status.ok()) {
    LOG(ERROR) << read_labels_status;
    return read_labels_status;
  }
  const int how_many_labels = std::min(5, static_cast<int>(label_count));
  Tensor indices;
  Tensor scores;
  TF_RETURN_IF_ERROR(GetTopLabels(outputs, how_many_labels, &indices, &scores));
  tensorflow::TTypes<float>::Flat scores_flat = scores.flat<float>();
  tensorflow::TTypes<int32>::Flat indices_flat = indices.flat<int32>();
  for (int pos = 0; pos < how_many_labels; ++pos) {
    const int label_index = indices_flat(pos);
    const float score = scores_flat(pos);
    LOG(INFO) << labels[label_index] << " (" << label_index << "): " << score;
  }
  return Status::OK();
}

// This is a testing function that returns whether the top label index is the
// one that's expected.
Status CheckTopLabel(const std::vector<Tensor>& outputs, int expected,
                     bool* is_expected) {
  *is_expected = false;
  Tensor indices;
  Tensor scores;
  const int how_many_labels = 1;
  TF_RETURN_IF_ERROR(GetTopLabels(outputs, how_many_labels, &indices, &scores));
  tensorflow::TTypes<int32>::Flat indices_flat = indices.flat<int32>();
  if (indices_flat(0) != expected) {
    LOG(ERROR) << "Expected label #" << expected << " but got #"
               << indices_flat(0);
    *is_expected = false;
  } else {
    *is_expected = true;
  }
  return Status::OK();
}

struct tensorStats_t {
    typedef unsigned long long intStat_t;
    typedef double floatStat_t;

    intStat_t maxDimensions = 0;
    intStat_t minDimensions = 0;

    intStat_t maxNValues = 0;
    intStat_t minNValues = 0;

    floatStat_t maxValue = -std::numeric_limits<float>::infinity();
    floatStat_t minValue =  std::numeric_limits<float>::infinity();

    floatStat_t maxRange = 0;
};

void analyzeRanges(const tensorflow::GraphDef* graph) {
    int consts = 0;

    int typeCounter[11];
    string typeStrings[11] = {"NOT_SET", "list", "string", "int", "float", "bool", "Type", "Shape", "Tensor", "Placeholder", "Function"};

    int tensorTypeCounter[20];
    string tensorTypeStrings[] = {"invalid", "float", "double", "int32", "uint8", "int16", "int8", "string", "complex64", "int64", "bool", "qint8", "quint8", "qint32", "bfloat16", "qint16", "quint16", "uint16", "complex128", "half"};

    std::fill_n(typeCounter, 11, 0);
    std::fill_n(tensorTypeCounter, 20, 0);

    unsigned long long numberOfFloatConstants = 0;
    unsigned long long numberOfNonFloatConstants = 0;

    tensorStats_t stats;

    std::map<const string, int> nodeToNumberOfConstants;

    int totalNumberOfConsts = 23886351;

    for(auto& node: graph->node()) {
        if(node.op() == "Const") {
            consts++;
            const google::protobuf::Map<string, tensorflow::AttrValue>& attr = node.attr();

            for(auto pair : attr) {
                ++typeCounter[pair.second.value_case()];
                //cout << pair.first << ":" << pair.second.value_case() << endl;
                /* if(pair.first == "value") {
                    cout << "found a value" << endl;
                    break;
                } */
            }

            google::protobuf::Map<string, tensorflow::AttrValue>::const_iterator data = attr.find("dtype");

            if(data != attr.end()) {

                tensorflow::DataType dataT = data->second.type();

                if(dataT > 100) {
                    std::cout << "ERROR. Found ref type " << dataT << " on node " << node.name() << std::endl;
                    exit(1);
                }

                if(dataT != tensorflow::DataType::DT_FLOAT) {
                    std::cout << "Found datatype " << DataType_Name(dataT) << " (" << dataT << ") on node " << node.name() << std::endl;

                    tensorflow::Tensor tensor;
                    if(!tensor.FromProto(attr.at("value").tensor())) {
                        std::cout << "Warning: failed to parse tensor for const node " << node.name() << std::endl;
                    } else {
                        numberOfNonFloatConstants += tensor.NumElements();

                        if (tensor.NumElements() > (0.01 * totalNumberOfConsts)) {
                          std::cout << "Node with many non-float Consts: " << node.name() << " --- " << tensor.NumElements() << std::endl; // << " -- dimensions: " << tensor.DebugString() << endl;
                        }
                    }
                } else {
                    tensorflow::Tensor tensor;
                    if(!tensor.FromProto(attr.at("value").tensor())) {
                        std::cout << "Warning: failed to parse tensor for const node " << node.name() << std::endl;
                    } else {
                        numberOfFloatConstants += tensor.NumElements();

                        if (tensor.NumElements() > (0.01 * totalNumberOfConsts)) {
                          std::cout << "Node with many float Consts: " << node.name() << " --- " << tensor.NumElements() << std::endl; //  << " -- dimensions: " << tensor.DebugString() << endl;
                        }

                        auto flattened = tensor.flat<float>();

                        //We could write a function that takes a function pointer and a tensor, plus a member pointer for min and one for max respectivley, but no...

                        //float max = flattened.maxCoeff();
                        //float min = flattened.minCoeff();
                        float min =  std::numeric_limits<float>::infinity();
                        float max = -std::numeric_limits<float>::infinity();

                        for(int i=0; i<flattened.size(); ++i) {
                            if(flattened(i) < min) {
                                min = flattened(i);
                            }
                            if(flattened(i) > max) {
                                max = flattened(i);
                            }
                        }

                        if(min < stats.minValue) {
                            stats.minValue = min;
                        }

                        if(max > stats.maxValue) {
                            stats.maxValue = max;
                        }

                        if(max-min > stats.maxRange) {
                            stats.maxRange = max-min;
                        }

                        //tensor.
                    }
                }

                ++tensorTypeCounter[dataT];
            }
        }
    }

    /*
     * Information about the types assigned to const nodes
    */
    std::cout << "Attribute map type information: (attribute types supplied to const nodes)\n" << std::endl;

    for(size_t i=0; i<sizeof(typeCounter)/sizeof(decltype(typeCounter[0])); ++i) {
        std::cout << typeStrings[i] << " : " << typeCounter[i] << std::endl;
    }

    //types used inside tensors
    std::cout << "Information about types inside tensors:" << std::endl;

    for(size_t i=0; i<sizeof(tensorTypeCounter)/sizeof(decltype(tensorTypeCounter[0])); ++i) {
        std::cout << tensorTypeStrings[i] << " : " << tensorTypeCounter[i] << std::endl;
    }

    //Tensor info
    std::cout << "Number of float constants: " << numberOfFloatConstants << std::endl;
    std::cout << "Number of non-float constants: " << numberOfNonFloatConstants << std::endl;
    std::cout << "Total number of constants/weights: " << numberOfFloatConstants + numberOfNonFloatConstants << std::endl;
    std::cout << "The parameters are in the range " << stats.minValue << " to " << stats.maxValue << " (maxRange:" << stats.maxRange << ")" << std::endl;

    // Print how many constants/weights each node carries
    // for(auto elem : nodeToNumberOfConstants)
    // {
    //    cout << elem.first << " --- " << elem.second << endl;
    // }

}


void statistics(const tensorflow::GraphDef* graph, std::map<const string, int>* nodeIdMap) {
    std::cout << "Version (producer):" << graph->versions().producer() << "\n"
                                                                     "\n";
    std::cout << "Library           \n"
            "  Functions       :" << graph->library().function_size() << "\n"
            "  Gradients       :" << graph->library().gradient_size() << "\n"
                                                                         "\n";
    std::cout << "Nodes             :" << graph->node_size() << "\n" << std::endl;

    std::map<const string, int> operations;

    std::regex  nodeNameRegex("[A-Za-z0-9.][A-Za-z0-9_./]*");
    std::smatch match;

    int conv2Dtensors  = 0;
    int maxOutChannels = 0;
    int totalOutChannels = 0;
    int same = 0;
    int valid = 0;

    int strideOccurences[4];

    std::fill_n(strideOccurences, 4, 0);

    for(auto node: graph->node()) {
        auto counter = operations.find(node.op());
        if(counter == operations.end()) {
            //operations.emplace(piecewise_construct, node.op(), 1);
            operations.emplace(node.op(), 1);
        } else {
            counter->second++;
        }

        std::cout << "Node: " << node.name() << std::endl;
        std::cout << "Node OP: " << node.op() << std::endl;


        if(node.op() == "Conv2D") {
            regex_search(node.input(1), match, nodeNameRegex); //Input 1 == convolution parameters
            if(!match.empty()) {
                auto src = graph->node(nodeIdMap->at(match.str()));
                tensorflow::DataType dataT = src.attr().at("T").type();

                if(dataT != tensorflow::DataType::DT_FLOAT) {
                    std::cout << "Found datatype " << DataType_Name(dataT) << " (" << dataT << ") on node " << node.name() << std::endl;
                } else {
                    tensorflow::Tensor tensor;
                    if(!tensor.FromProto(graph->node(nodeIdMap->at(match.str())+1).attr().at("value").tensor())) {
                        std::cout << "Warning: failed to parse tensor for const node " << graph->node(nodeIdMap->at(match.str())+1).name() << std::endl;
                    } else {
                        std::cout << "Elements: " << tensor.NumElements() << std::endl;
                        std::cout << "Dimensions: " << tensor.shape().dim_size(0) * tensor.shape().dim_size(1) * tensor.shape().dim_size(2)  << ":" << tensor.shape().dim_size(3) << std::endl;
                        ++conv2Dtensors;
                        if(tensor.shape().dim_size(3) > maxOutChannels) {
                            maxOutChannels = tensor.shape().dim_size(3);
                        }
                        totalOutChannels += tensor.shape().dim_size(3);
                    }
                }
            } else {
                std::cout << "Failed to find source node " << match.str() << std::endl;
            }

            if(node.attr().at("padding").s() == "SAME") {
                ++same;
            } else {
                ++valid;
            }

            const google::protobuf::RepeatedField<google::protobuf::int64 >& strides = node.attr().at("strides").list().i();

            int maxStride = 0;
            for(int i=0; i<strides.size(); ++i) {
                //cout << "..  " << setw(4) << strides.Get(i) << "\n";
                if(strides.Get(i) > maxStride) {
                    maxStride = strides.Get(i);
                }
            }

            if(maxStride > 3) {
                std::cout << "Stride error. Stride " << maxStride << " to large for analyzer." << std::endl;
                exit(5);
            }

            ++strideOccurences[maxStride];
        }
    }

    std::cout << "Found " << conv2Dtensors << " Tensors for Conv2D operations.\n\n";
    std::cout << "Maximum out channels : " << maxOutChannels << "\n";
    std::cout << "Total out channels : "   << totalOutChannels << std::endl;
    std::cout << "Padding Same : " << same << "\n";
    std::cout << "Padding Valid: " << valid << "\n\n";
    std::cout << "Stride - Occurences\n";

    for(int i=0; i<4; ++i) {
        std::cout << i << " - " << strideOccurences[i] << "\n";
    }

    for(auto nodetype: operations) {
        std::cout << nodetype.first << " : " << nodetype.second << "\n";
    }
}

void PrintNodeInfo(const tensorflow::NodeDef* node) {
  std::string shape_description = "None";
  if (node->attr().count("shape")) {
    tensorflow::TensorShapeProto shape_proto = node->attr().at("shape").shape();
    Status shape_status = tensorflow::PartialTensorShape::IsValidShape(shape_proto);
    if (shape_status.ok()) {
      shape_description = tensorflow::PartialTensorShape(shape_proto).DebugString();
    } else {
      shape_description = shape_status.error_message();
    }
  }
  tensorflow::DataType dtype = tensorflow::DT_INVALID;
  if (node->attr().count("dtype")) {
    dtype = node->attr().at("dtype").type();
  }
  std::cout << "(name=" << node->name();
  std::cout << ", type=" << DataTypeString(dtype) << "(" << dtype << ")";
  std::cout << ", shape=" << shape_description << ") ";
}

