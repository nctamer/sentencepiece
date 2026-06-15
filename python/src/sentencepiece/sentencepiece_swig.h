#include <sentencepiece_processor.h>
#include <sentencepiece_trainer.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <thread>
#include <vector>

#include "third_party/absl/status/status.h"

namespace {

PyObject* kUnicodeInput = reinterpret_cast<PyObject*>(0x1);
PyObject* kByteInput = reinterpret_cast<PyObject*>(0x2);

using BytesArray = std::vector<sentencepiece::util::bytes>;

inline void ReleaseResultObject(PyObject* obj) {
  if (obj != nullptr && obj != kUnicodeInput && obj != kByteInput) {
    Py_XDECREF(obj);
  }
}

class PyInputString {
 public:
  explicit PyInputString(PyObject* obj) {
    if (PyUnicode_Check(obj)) {
      str_ = const_cast<char*>(PyUnicode_AsUTF8AndSize(obj, &size_));
      input_type_ = kUnicodeInput;
    } else if (PyBytes_Check(obj)) {
      PyBytes_AsStringAndSize(obj, &str_, &size_);
      input_type_ = kByteInput;
    } else {
      str_ = nullptr;
    }
  }
  absl::string_view str() const { return absl::string_view(data(), size()); }
  const char* data() const { return str_; }
  Py_ssize_t size() const { return size_; }
  bool IsAvalable() const { return str_ != nullptr; }
  PyObject* input_type() const { return input_type_; }

  static bool IsUnicode(PyObject* resultobj) {
    return (resultobj == nullptr || resultobj == kUnicodeInput);
  }

 private:
  PyObject* input_type_ = nullptr;
  char* str_ = nullptr;
  Py_ssize_t size_ = 0;
};

PyObject* MakePyOutputString(const std::string& output, PyObject* resultobj) {
  if (PyInputString::IsUnicode(resultobj)) {
    return PyUnicode_FromStringAndSize(output.data(), output.size());
  }
  return PyBytes_FromStringAndSize(output.data(), output.size());
}

PyObject* MakePyOutputBytes(const sentencepiece::util::bytes& output) {
  return PyBytes_FromStringAndSize(output.data(), output.size());
}

int ToSwigError(absl::StatusCode code) {
  switch (code) {
    case absl::StatusCode::kNotFound:
      return SWIG_IOError;
    case absl::StatusCode::kOutOfRange:
      return SWIG_IndexError;
    case absl::StatusCode::kInvalidArgument:
      return SWIG_SyntaxError;
    default:
      return SWIG_RuntimeError;
  }
  return SWIG_RuntimeError;
}

#ifndef Py_GIL_DISABLED
// RAII class to release GIL.
// Release GIL in contractor and acquire GIL in destractor.
class ScopedGILRelease {
 public:
  ScopedGILRelease() { save_ = PyEval_SaveThread(); }
  ~ScopedGILRelease() { PyEval_RestoreThread(save_); }

 private:
  PyThreadState* save_;
};

// RAII class to aquire GIL.
// Acquire GIL in contractor and release GIL in destractor.
class ScopedGILAcquire {
 public:
  ScopedGILAcquire() { state_ = PyGILState_Ensure(); }
  ~ScopedGILAcquire() { PyGILState_Release(state_); }

 private:
  PyGILState_STATE state_;
};
#else
class ScopedGILRelease {
 public:
  ScopedGILRelease() {}
  ~ScopedGILRelease() {}
};

class ScopedGILAcquire {
 public:
  ScopedGILAcquire() {}
  ~ScopedGILAcquire() {}
};
#endif  // Py_GIL_DISABLED

class PySentenceIterator : public sentencepiece::SentenceIterator {
 public:
  PySentenceIterator(PyObject* iter) : iter_(iter) {
    ScopedGILAcquire aquire;
    item_ = PyIter_Next(iter_);
    CopyValue();
  }

  ~PySentenceIterator() {
    // Py_XDECREF(iter_);
  }

  bool done() const override { return item_ == nullptr; }

  void Next() override {
    ScopedGILAcquire aquire;
    item_ = PyIter_Next(iter_);
    CopyValue();
  }

  const std::string& value() const override { return value_; }

  absl::Status status() const override { return status_; }

 private:
  void CopyValue() {
    if (item_ == nullptr) return;
    const PyInputString ustring(item_);
    if (ustring.IsAvalable()) {
      absl::string_view data(ustring.data(), ustring.size());
      while (!data.empty()) {
        if (data.back() == '\r' || data.back() == '\n')
          data.remove_suffix(1);
        else
          break;
      }
      value_.assign(data.data(), data.size());
    } else {
      status_ = absl::Status(absl::StatusCode::kInternal, "Not a string.");
    }
    Py_XDECREF(item_);
  }
  PyObject* iter_ = nullptr;
  PyObject* item_ = nullptr;
  std::string value_;
  absl::Status status_;
};

inline void RewriteIds(const sentencepiece::SentencePieceProcessor& sp,
                       std::vector<int>* ids, bool add_bos, bool add_eos,
                       bool reverse, bool emit_unk_piece) {
  if (!add_bos && !add_eos && !reverse) return;
  if (reverse) std::reverse(ids->begin(), ids->end());
  if (add_bos) ids->insert(ids->begin(), sp.bos_id());
  if (add_eos) ids->push_back(sp.eos_id());
}

inline void RewriteIds(const sentencepiece::SentencePieceProcessor& sp,
                       std::vector<std::string>* pieces, bool add_bos,
                       bool add_eos, bool reverse, bool emit_unk_piece) {
  if (!add_bos && !add_eos && !reverse && !emit_unk_piece) return;
  if (reverse) std::reverse(pieces->begin(), pieces->end());
  if (add_bos) pieces->insert(pieces->begin(), sp.IdToPiece(sp.bos_id()));
  if (add_eos) pieces->push_back(sp.IdToPiece(sp.eos_id()));
  if (emit_unk_piece) {
    const auto& unk = sp.IdToPiece(sp.unk_id());
    for (auto& piece : *pieces) {
      const int id = sp.PieceToId(piece);
      if (id == sp.unk_id()) {
        piece = unk;
      }
    }
  }
}

inline void RewriteIds(const sentencepiece::SentencePieceProcessor& sp,
                       sentencepiece::util::bytes* proto, bool add_bos,
                       bool add_eos, bool reverse, bool emit_unk_piece) {
  if (add_bos || add_eos || reverse || emit_unk_piece) {
    throw absl::Status(
        absl::StatusCode::kUnimplemented,
        "add_bos, add_eos, reverse, and emit_unk_piece is not supported in "
        "proto API");
  }
}

inline void RewriteIds(const sentencepiece::SentencePieceProcessor& sp,
                       sentencepiece::ImmutableSentencePieceText* proto,
                       bool add_bos, bool add_eos, bool reverse,
                       bool emit_unk_piece) {
  if (add_bos || add_eos || reverse || emit_unk_piece) {
    throw absl::Status(
        absl::StatusCode::kUnimplemented,
        "add_bos, add_eos, reverse, and emit_unk_piece is not supported in "
        "proto API");
  }
}

inline void CheckIds(const std::vector<int>& ids, int num_pieces) {
  for (int id : ids) {
    if (id < 0 || id >= num_pieces) {
      throw absl::Status(absl::StatusCode::kOutOfRange,
                         "piece id is out of range.");
    }
  }
}

inline void CheckIds(const std::vector<absl::string_view>& ids,
                     int num_pieces) {}

inline void CheckIdsBatch(const std::vector<std::vector<int>>& ids,
                          int num_pieces) {
  for (const auto& v : ids) CheckIds(v, num_pieces);
}

template <typename T>
inline void ConvertToUnicodeSpans(T* proto) {}

template <>
inline void ConvertToUnicodeSpans(
    sentencepiece::ImmutableSentencePieceText* proto) {
  proto->ConvertToUnicodeSpans();
}

template <>
inline void ConvertToUnicodeSpans(
    sentencepiece::ImmutableNBestSentencePieceText* proto) {
  proto->ConvertToUnicodeSpans();
}

inline int GetNumThreads(int num_threads) {
  if (num_threads < 0) {
    return std::thread::hardware_concurrency();
  }
  return std::max<int>(1, std::min<int>(num_threads, 65536));
}

#define INIT_THREAD_POOL                                                  \
  num_threads = GetNumThreads(num_threads);                               \
  std::unique_ptr<sentencepiece::ThreadPool> pool_impl;                   \
  if (!thread_pool) {                                                     \
    pool_impl = std::make_unique<sentencepiece::ThreadPool>(num_threads); \
  }                                                                       \
  auto* pool = thread_pool ? thread_pool : pool_impl.get();

#define DEFINE_ENCODE_BATCH_FUNC_IMPL(FuncName, InType, OutType)              \
  std::vector<OutType> outs(ins.size());                                      \
  INIT_THREAD_POOL;                                                           \
  auto status = sentencepiece::RunBatch(                                      \
      ins.size(),                                                             \
      [&](size_t i) {                                                         \
        try {                                                                 \
          auto out = enable_sampling                                          \
                         ? self->Sample##FuncName(ins[i], nbest_size, alpha)  \
                         : self->FuncName(ins[i]);                            \
          RewriteIds(*self, &out, add_bos, add_eos, reverse, emit_unk_piece); \
          ConvertToUnicodeSpans(&out);                                        \
          outs[i] = std::move(out);                                           \
        } catch (const absl::Status& s) {                                     \
          return s;                                                           \
        }                                                                     \
        return absl::OkStatus();                                              \
      },                                                                      \
      *pool);                                                                 \
  if (!status.ok()) throw status;                                             \
  return outs;

#define DEFINE_DECODE_BATCH_FUNC_IMPL(FuncName, InType, OutType) \
  std::vector<OutType> outs(ins.size());                         \
  INIT_THREAD_POOL;                                              \
  auto status = sentencepiece::RunBatch(                         \
      ins.size(),                                                \
      [&](size_t i) {                                            \
        try {                                                    \
          outs[i] = self->FuncName(ins[i]);                      \
          ConvertToUnicodeSpans(&outs[i]);                       \
        } catch (const absl::Status& s) {                        \
          return s;                                              \
        }                                                        \
        return absl::OkStatus();                                 \
      },                                                         \
      *pool);                                                    \
  if (!status.ok()) throw status;                                \
  return outs;

}  // namespace
