// Persistent CPU QNN context with native typed application buffers.
#include <cstdio>
#include <string>
#include <QnnInterface.h>
#include "QnnWrapperUtils.hpp"
#include "QnnTypeMacros.hpp"
#include <dlfcn.h>
#include <cstdarg>
#include <cstdio>
#include <cstdint>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
thread_local std::string last_error;
using Compose = qnn_wrapper_api::ModelError_t (*)(Qnn_BackendHandle_t, QNN_INTERFACE_VER_TYPE,
    Qnn_ContextHandle_t, const qnn_wrapper_api::GraphConfigInfo_t**, uint32_t,
    qnn_wrapper_api::GraphInfo_t***, uint32_t*, bool, QnnLog_Callback_t, QnnLog_Level_t);
using FreeGraphs = qnn_wrapper_api::ModelError_t (*)(qnn_wrapper_api::GraphInfo_t***, uint32_t);
void check(Qnn_ErrorHandle_t code, const char* operation) {
  if (code != QNN_SUCCESS) throw std::runtime_error(std::string(operation) + " error " + std::to_string(code));
}
void logger(const char* format, QnnLog_Level_t, uint64_t, va_list args) {
  vfprintf(stderr, format, args); fputc('\n', stderr);
}
template<class T> T symbol(void* lib, const char* name) {
  auto value = reinterpret_cast<T>(dlsym(lib, name));
  if (!value) throw std::runtime_error(std::string("missing symbol ") + name);
  return value;
}
std::string quote(const char* value) {
  std::ostringstream s; s << '"';
  for (const unsigned char* p = reinterpret_cast<const unsigned char*>(value); *p; ++p) {
    if (*p == '"' || *p == '\\') s << '\\' << char(*p);
    else if (*p < 32) { char hex[7]; snprintf(hex, sizeof(hex), "\\u%04x", *p); s << hex; }
    else s << char(*p);
  }
  s << '"'; return s.str();
}
size_t width(Qnn_DataType_t type) {
  switch(type) {
    case QNN_DATATYPE_INT_8: case QNN_DATATYPE_UINT_8: case QNN_DATATYPE_BOOL_8: return 1;
    case QNN_DATATYPE_INT_16: case QNN_DATATYPE_UINT_16: case QNN_DATATYPE_FLOAT_16: return 2;
    case QNN_DATATYPE_INT_32: case QNN_DATATYPE_UINT_32: case QNN_DATATYPE_FLOAT_32: return 4;
    case QNN_DATATYPE_INT_64: case QNN_DATATYPE_UINT_64: case QNN_DATATYPE_FLOAT_64: return 8;
    default: throw std::runtime_error("unsupported native tensor dtype " + std::to_string(type));
  }
}
size_t bytes(const Qnn_Tensor_t& tensor) {
  size_t count = width(getQnnTensorDataType(tensor));
  for(uint32_t i=0; i<getQnnTensorRank(tensor); ++i) {
    auto dim=getQnnTensorDimensions(tensor)[i];
    if (!dim || count > UINT32_MAX / dim) throw std::runtime_error("invalid or oversized client buffer");
    count *= dim;
  }
  return count;
}
struct Session {
  void* backend_lib=nullptr; void* model_lib=nullptr;
  QNN_INTERFACE_VER_TYPE api{}; Qnn_BackendHandle_t backend=nullptr; Qnn_ContextHandle_t context=nullptr;
  Qnn_LogHandle_t log=nullptr; FreeGraphs free_graphs=nullptr;
  qnn_wrapper_api::GraphInfo_t** graphs=nullptr; uint32_t graph_count=0;
  std::vector<Qnn_Tensor_t> inputs, outputs; std::string description;
  ~Session() {
    if (graphs && free_graphs) free_graphs(&graphs, graph_count);
    if (context && api.contextFree) api.contextFree(context, nullptr);
    if (backend && api.backendFree) api.backendFree(backend);
    if (log && api.logFree) api.logFree(log);
    if (model_lib) dlclose(model_lib);
    if (backend_lib) dlclose(backend_lib);
  }
  void init(const char* backend_path, const char* model_path) {
    backend_lib=dlopen(backend_path, RTLD_NOW|RTLD_LOCAL);
    if (!backend_lib) throw std::runtime_error(dlerror());
    auto providers=symbol<decltype(&QnnInterface_getProviders)>(backend_lib,"QnnInterface_getProviders");
    const QnnInterface_t** list=nullptr; uint32_t count=0;
    check(providers(&list,&count),"getProviders"); bool found=false;
    for(uint32_t i=0;i<count;++i) {
      if(list[i]->apiVersion.coreApiVersion.major == QNN_API_VERSION_MAJOR &&
          list[i]->apiVersion.coreApiVersion.minor >= QNN_API_VERSION_MINOR) {
        api=list[i]->QNN_INTERFACE_VER_NAME; found=true; break;
      }
    }
    if(!found) throw std::runtime_error("compatible QNN interface not found");
    check(api.logCreate(logger,QNN_LOG_LEVEL_ERROR,&log),"logCreate");
    check(api.backendCreate(log,nullptr,&backend),"backendCreate");
    check(api.contextCreate(backend,nullptr,nullptr,&context),"contextCreate");
    model_lib=dlopen(model_path,RTLD_NOW|RTLD_LOCAL);
    if(!model_lib) throw std::runtime_error(dlerror());
    auto compose=symbol<Compose>(model_lib,"QnnModel_composeGraphs");
    free_graphs=symbol<FreeGraphs>(model_lib,"QnnModel_freeGraphsInfo");
    auto code=compose(backend,api,context,nullptr,0,&graphs,&graph_count,false,logger,QNN_LOG_LEVEL_ERROR);
    if(code != qnn_wrapper_api::MODEL_NO_ERROR) throw std::runtime_error("composeGraphs model error " + std::to_string(code));
    if(graph_count != 1 || !graphs || !*graphs) throw std::runtime_error("expected exactly one graph");
    auto& graph=(*graphs)[0];
    check(api.graphFinalize(graph.graph,nullptr,nullptr),"graphFinalize");
    inputs.assign(graph.inputTensors,graph.inputTensors+graph.numInputTensors);
    outputs.assign(graph.outputTensors,graph.outputTensors+graph.numOutputTensors);
    std::ostringstream s; s << "{\"graph_name\":" << quote(graph.graphName);
    for(auto pair: {std::make_pair("inputs",&inputs),std::make_pair("outputs",&outputs)}) {
      s << ",\"" << pair.first << "\":["; bool first=true;
      for(auto& tensor: *pair.second) {
        if(!validateTensorVersion(tensor)) throw std::runtime_error("unsupported tensor version");
        if(!first) s << ','; first=false;
        s << "{\"name\":" << quote(getQnnTensorName(tensor)) << ",\"dtype_code\":" << getQnnTensorDataType(tensor)
          << ",\"bytes\":" << bytes(tensor) << ",\"shape\":[";
        for(uint32_t i=0;i<getQnnTensorRank(tensor);++i) {if(i) s << ','; s << getQnnTensorDimensions(tensor)[i];}
        s << "]}";
      }
      s << ']';
    }
    s << '}'; description=s.str();
  }
  void execute(void** in, const uint32_t* in_bytes, uint32_t in_count,
               void** out, const uint32_t* out_bytes, uint32_t out_count) {
    if(in_count!=inputs.size() || out_count!=outputs.size()) throw std::runtime_error("buffer count differs");
    for(auto pair: {std::make_pair(&inputs,std::make_pair(in,in_bytes)),std::make_pair(&outputs,std::make_pair(out,out_bytes))}) {
      for(size_t i=0;i<pair.first->size();++i) {
        auto& t=(*pair.first)[i];
        if(!pair.second.first[i] || pair.second.second[i]!=bytes(t)) throw std::runtime_error("buffer extent differs");
        setQnnTensorMemType(t,QNN_TENSORMEMTYPE_RAW);
        setQnnTensorClientBuf(t,Qnn_ClientBuffer_t{pair.second.first[i],pair.second.second[i]});
      }
    }
    check(api.graphExecute((*graphs)[0].graph,inputs.data(),inputs.size(),outputs.data(),outputs.size(),nullptr,nullptr),"graphExecute");
  }
};
}
extern "C" {
const char* q4_qnn_error() { return last_error.c_str(); }
void* q4_qnn_create(const char* backend, const char* model) {
  try { last_error.clear(); auto s=std::make_unique<Session>(); s->init(backend,model); return s.release(); }
  catch(const std::exception& e) { last_error=e.what(); return nullptr; }
}
const char* q4_qnn_describe(void* handle) { return static_cast<Session*>(handle)->description.c_str(); }
int q4_qnn_execute(void* handle,void** in,const uint32_t* sizes,uint32_t count,void** out,const uint32_t* out_sizes,uint32_t out_count) {
  try { last_error.clear(); static_cast<Session*>(handle)->execute(in,sizes,count,out,out_sizes,out_count); return 0; }
  catch(const std::exception& e) { last_error=e.what(); return -1; }
}
void q4_qnn_destroy(void* handle) { delete static_cast<Session*>(handle); }
}
