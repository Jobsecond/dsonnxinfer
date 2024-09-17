#ifndef DSONNXINFER_GLOBAL_H
#define DSONNXINFER_GLOBAL_H

#ifdef _MSC_VER
#  define DSONNXINFER_DECL_EXPORT __declspec(dllexport)
#  define DSONNXINFER_DECL_IMPORT __declspec(dllimport)
#else
#  define DSONNXINFER_DECL_EXPORT __attribute__((visibility("default")))
#  define DSONNXINFER_DECL_IMPORT __attribute__((visibility("default")))
#endif

#ifndef DSONNXINFER_EXPORT
#  ifdef DSONNXINFER_STATIC
#    define DSONNXINFER_EXPORT
#  else
#    ifdef DSONNXINFER_LIBRARY
#      define DSONNXINFER_EXPORT DSONNXINFER_DECL_EXPORT
#    else
#      define DSONNXINFER_EXPORT DSONNXINFER_DECL_IMPORT
#    endif
#  endif
#endif

#define DSONNXINFER_DISABLE_COPY(Class)                                                        \
    Class(const Class &) = delete;                                                             \
    Class &operator=(const Class &) = delete;

#define DSONNXINFER_DISABLE_MOVE(Class)                                                        \
    Class(Class &&) = delete;                                                                  \
    Class &operator=(Class &&) = delete;

#define DSONNXINFER_DISABLE_COPY_MOVE(Class)                                                   \
    DSONNXINFER_DISABLE_COPY(Class)                                                            \
    DSONNXINFER_DISABLE_MOVE(Class)

#define DSONNXINFER_NAMESPACE dsonnxinfer
#define DSONNXINFER_BEGIN_NAMESPACE namespace DSONNXINFER_NAMESPACE {
#define DSONNXINFER_END_NAMESPACE }

#endif // DSONNXINFER_GLOBAL_H
