#include "eurosat_cls_postprocessor.hpp"
#include "eurosat_category_name.hpp"

namespace dl {
namespace cls {
EuroSatClsPostprocessor::EuroSatClsPostprocessor(
    Model *model, const int top_k, const float score_thr, bool need_softmax, const std::string &output_name) :
    ClsPostprocessor(model, top_k, score_thr, need_softmax, output_name)
{
    m_cat_names = eurosat_cat_names;
}
} // namespace cls
} // namespace dl
