#pragma once
#include "dl_cls_postprocessor.hpp"

// EuroSAT classification postprocessor. Identical to Espressif's
// ImageNetClsPostprocessor pattern -- the base class does dequantize + softmax +
// top-k argmax; we only swap in the 10 EuroSAT category names.
namespace dl {
namespace cls {
class EuroSatClsPostprocessor : public ClsPostprocessor {
public:
    EuroSatClsPostprocessor(Model *model,
                            const int top_k,
                            const float score_thr,
                            bool need_softmax,
                            const std::string &output_name = "");
};
} // namespace cls
} // namespace dl
