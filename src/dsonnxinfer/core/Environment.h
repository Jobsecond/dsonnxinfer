#ifndef DSONNXINFER_ENVIRONMENT_H
#define DSONNXINFER_ENVIRONMENT_H

#include <filesystem>
#include <string>
#include <vector>
#include <memory>
#include <dsonnxinfer/dsonnxinfer_global.h>
#include <dsonnxinfer/dsonnxinfer_common.h>

#define dsEnv (DSONNXINFER_NAMESPACE::Environment::instance())

DSONNXINFER_BEGIN_NAMESPACE

class DSONNXINFER_EXPORT Environment {
public:
    Environment();
    ~Environment();

    static Environment *instance();

public:
    bool load(const std::filesystem::path &path, ExecutionProvider ep, std::string *errorMessage);
    bool isLoaded() const;

    std::filesystem::path runtimePath() const;

    int deviceIndex() const;
    void setDeviceIndex(int deviceIndex);

    int defaultSteps() const;
    void setDefaultSteps(int defaultSteps);

    float defaultDepth() const;
    void setDefaultDepth(float defaultDepth);

    void setLoggerCallback(DsLoggingCallback callback);

    ExecutionProvider executionProvider() const;
    std::string versionString() const;

protected:
    class Impl;
    std::unique_ptr<Impl> _impl;
};

DSONNXINFER_END_NAMESPACE

#endif // DSONNXINFER_ENVIRONMENT_H