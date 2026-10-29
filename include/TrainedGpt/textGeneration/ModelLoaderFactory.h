//
// Created by moinshaikh on 10/29/26.
//

#ifndef LARGELANGUAGEMODELCPP_MODELLOADERFACTORY_H
#define LARGELANGUAGEMODELCPP_MODELLOADERFACTORY_H

#include<torch/torch.h>
#include<string>
#include<unordered_map>

namespace gpt2 {

    // Reads model.ckpt.index + model.ckpt.data-* from model_dir (no TensorFlow)
    // and saves all float32 tensors as buffers of a TorchScript module.
    // Buffer names: "wte", "wpe", "ln_f__g", "h0__attn__c_attn__w", ...
    //
    // Declared here, defined in src/TrainedGpt/textGeneration/ModelLoaderFactory.cpp.
    // Keep these as declarations only: this header is included by several
    // translation units, so out-of-line definitions would cause
    // "multiple definition of gpt2::convert_checkpoint/load_params" at link time.
    void convert_checkpoint(const std::string& model_dir, const std::string& out_path);

    // Loads the converted file back into a name -> tensor map.
    std::unordered_map<std::string, torch::Tensor> load_params(const std::string& torchscript_path);

} // namespace gpt2

#endif //LARGELANGUAGEMODELCPP_MODELLOADERFACTORY_H
