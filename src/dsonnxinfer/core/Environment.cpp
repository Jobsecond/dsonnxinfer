#include "Environment.h"

#include <flowonnx/environment.h>

namespace fs = std::filesystem;

DSONNXINFER_BEGIN_NAMESPACE

static Environment *g_env = nullptr;

constexpr flowonnx::ExecutionProvider to_flowonnx_ep(ExecutionProvider ep) {
    switch (ep) {
        case EP_CPU:
            return flowonnx::EP_CPU;
        case EP_CUDA:
            return flowonnx::EP_CUDA;
        case EP_DirectML:
            return flowonnx::EP_DirectML;
        case EP_CoreML:
            return flowonnx::EP_CoreML;
    }
    return flowonnx::EP_CPU;
}

constexpr ExecutionProvider from_flowonnx_ep(flowonnx::ExecutionProvider ep) {
    switch (ep) {
        case flowonnx::EP_CPU:
            return EP_CPU;
        case flowonnx::EP_CUDA:
            return EP_CUDA;
        case flowonnx::EP_DirectML:
            return EP_DirectML;
        case flowonnx::EP_CoreML:
            return EP_CoreML;
    }
    return EP_CPU;
}

class Environment::Impl {
public:
    bool load(const fs::path &path, ExecutionProvider ep, std::string *errorMessage) {
        return _env.load(path, to_flowonnx_ep(ep), errorMessage);
    }

    flowonnx::Environment _env;
};

Environment::Environment() : _impl(std::make_unique<Impl>()) {
    g_env = this;
}

Environment::~Environment() {
    g_env = nullptr;
}

bool Environment::load(const fs::path &path, ExecutionProvider ep, std::string *errorMessage) {
    auto &impl = *_impl;
    return impl.load(path, ep, errorMessage);
}

bool Environment::isLoaded() const {
    auto &impl = *_impl;
    return impl._env.isLoaded();
}

Environment *Environment::instance() {
    return g_env;
}

fs::path Environment::runtimePath() const {
    auto &impl = *_impl;
    return impl._env.runtimePath();
}

ExecutionProvider Environment::executionProvider() const {
    auto &impl = *_impl;
    return from_flowonnx_ep(impl._env.executionProvider());
}

std::string Environment::versionString() const {
    auto &impl = *_impl;
    return impl._env.versionString();
}

DSONNXINFER_END_NAMESPACE