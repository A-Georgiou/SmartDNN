#ifndef TRANSFORMER_BLOCK_HPP
#define TRANSFORMER_BLOCK_HPP

#include "smart_dnn/Activations/GELU.hpp"
#include "smart_dnn/Layer.hpp"
#include "smart_dnn/Layers/ActivationLayer.hpp"
#include "smart_dnn/Layers/FullyConnectedLayer.hpp"
#include "smart_dnn/Layers/MultiHeadAttentionLayer.hpp"
#include "smart_dnn/Regularisation/LayerNormalizationLayer.hpp"
#include <initializer_list>
#include <vector>

namespace smart_dnn {

// GPT-style pre-norm block on (batch, length, embeddingSize): h = x + attention(norm(x)), then h + MLP(norm(h)) with a 4x GELU MLP.
template <typename T = float>
class TransformerBlock : public Layer<T> {
public:
    TransformerBlock(int embeddingSize, int heads, bool causal = true)
        : attentionNorm(embeddingSize), attention(embeddingSize, heads, causal), mlpNorm(embeddingSize),
          expand(embeddingSize, 4 * embeddingSize), activation(GELU<T>()), contract(4 * embeddingSize, embeddingSize) {}

    Tensor<T> forward(const Tensor<T>& input) override {
        const Tensor<T> hidden = input + attention.forward(attentionNorm.forward(input));
        return hidden + contract.forward(activation.forward(expand.forward(mlpNorm.forward(hidden))));
    }

    Tensor<T> backward(const Tensor<T>& gradOutput) override {
        const Tensor<T> gradHidden = gradOutput + mlpNorm.backward(expand.backward(activation.backward(contract.backward(gradOutput))));
        return gradHidden + attentionNorm.backward(attention.backward(gradHidden));
    }

    std::vector<Parameter<T>> parameters() override {
        std::vector<Parameter<T>> all;
        for (Layer<T>* layer : std::initializer_list<Layer<T>*>{&attentionNorm, &attention, &mlpNorm, &expand, &contract}) {
            for (const Parameter<T>& parameter : layer->parameters()) {
                all.push_back(parameter);
            }
        }
        return all;
    }

    void save(std::ostream& os) const override {
        for (const Layer<T>* layer : std::initializer_list<const Layer<T>*>{&attentionNorm, &attention, &mlpNorm, &expand, &contract}) {
            layer->save(os);
        }
    }

    void load(std::istream& is) override {
        for (Layer<T>* layer : std::initializer_list<Layer<T>*>{&attentionNorm, &attention, &mlpNorm, &expand, &contract}) {
            layer->load(is);
        }
    }

private:
    LayerNormalizationLayer<T> attentionNorm;
    MultiHeadAttentionLayer<T> attention;
    LayerNormalizationLayer<T> mlpNorm;
    FullyConnectedLayer<T> expand;
    ActivationLayer<T> activation;
    FullyConnectedLayer<T> contract;
};

} // namespace smart_dnn

#endif // TRANSFORMER_BLOCK_HPP
