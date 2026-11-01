#ifndef DATA_PARALLEL_TRAINER_HPP
#define DATA_PARALLEL_TRAINER_HPP

#include "smart_dnn/SmartDNN.hpp"
#include "smart_dnn/Regularisation/DropoutLayer.hpp"
#include <algorithm>
#include <exception>
#include <functional>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

namespace smart_dnn {

// Splits each batch by rows across threads, each running its own copy of a compiled model, then applies the row-weighted mean gradient with the model's optimizer.
template <typename T = float>
class DataParallelTrainer {
public:
    // build adds the model's layers to an empty model; loss must be a mean over rows, like MSELoss and SoftmaxCrossEntropyLoss.
    template <typename LossType>
    DataParallelTrainer(SmartDNN<T>& model, const std::function<void(SmartDNN<T>&)>& build, const LossType& loss, int threads)
        : model(model) {
        if (threads < 1) {
            throw std::invalid_argument("DataParallelTrainer needs at least one thread");
        }
        for (size_t i = 0; i < model.getLayerCount(); ++i) {
            if (dynamic_cast<DropoutLayer<T>*>(model.getLayer(i))) {
                throw std::invalid_argument("Dropout layers share one random engine, so they cannot train on several threads");
            }
        }
        const std::vector<Parameter<T>> parameters = parametersOf(model);
        for (int i = 0; i < threads; ++i) {
            losses.push_back(std::make_unique<LossType>(loss));
            if (i == 0) {
                continue;
            }
            replicas.push_back(std::make_unique<SmartDNN<T>>());
            build(*replicas.back());
            const std::vector<Parameter<T>> copies = parametersOf(*replicas.back());
            const bool same = copies.size() == parameters.size() && std::equal(parameters.begin(), parameters.end(), copies.begin(),
                [](const Parameter<T>& a, const Parameter<T>& b) { return a.value.get().getShape() == b.value.get().getShape(); });
            if (!same) {
                throw std::invalid_argument("build must add the same layers as the model");
            }
        }
    }

    // Returns the batch loss; learningRate overrides the optimizer's rate when positive, as in SmartDNN::trainStep.
    T trainStep(const Tensor<T>& inputs, const Tensor<T>& targets, T learningRate = T(-1)) {
        const int rows = inputs.getShape()[0];
        if (targets.getShape()[0] != rows) {
            throw std::invalid_argument("Inputs and targets need the same number of rows");
        }
        const int workers = std::min(static_cast<int>(replicas.size()) + 1, rows);
        auto firstRow = [&](int worker) { return rows * worker / workers; };
        std::vector<T> shares(static_cast<size_t>(workers));
        std::vector<T> shardLosses(static_cast<size_t>(workers));
        std::vector<std::vector<Parameter<T>>> parameters{parametersOf(model)};
        for (int worker = 1; worker < workers; ++worker) {
            parameters.push_back(parametersOf(*replicas[static_cast<size_t>(worker - 1)]));
        }
        runAcross(workers, [&](int worker) {
            SmartDNN<T>& copy = worker == 0 ? model : *replicas[static_cast<size_t>(worker - 1)];
            for (size_t p = 0; worker > 0 && p < parameters[0].size(); ++p) {
                const auto& values = parameters[0][p].value.get().getData();
                std::copy(values.begin(), values.end(), parameters[worker][p].value.get().getData().begin());
            }
            const int first = firstRow(worker);
            const int last = firstRow(worker + 1);
            const Tensor<T> prediction = copy.predict(rowsOf(inputs, first, last));
            const Tensor<T> target = rowsOf(targets, first, last);
            shares[worker] = T(last - first) / T(rows);
            shardLosses[worker] = losses[worker]->compute(prediction, target)[0] * shares[worker];
            copy.backward(losses[worker]->gradient(prediction, target));
        });

        // Each thread sums a disjoint set of parameters into the model's gradients.
        runAcross(workers, [&](int worker) {
            for (size_t p = static_cast<size_t>(worker); p < parameters[0].size(); p += static_cast<size_t>(workers)) {
                T* total = parameters[0][p].gradient.get().value().getData().data();
                const size_t size = parameters[0][p].value.get().getShape().size();
                for (size_t i = 0; i < size; ++i) {
                    total[i] *= shares[0];
                }
                for (size_t w = 1; w < parameters.size(); ++w) {
                    const T* part = parameters[w][p].gradient.get().value().getData().data();
                    for (size_t i = 0; i < size; ++i) {
                        total[i] += shares[w] * part[i];
                    }
                }
            }
        });
        model.updateWeights(learningRate);

        T loss = T(0);
        for (T part : shardLosses) {
            loss += part;
        }
        return loss;
    }

private:
    static std::vector<Parameter<T>> parametersOf(SmartDNN<T>& net) {
        std::vector<Parameter<T>> all;
        for (size_t i = 0; i < net.getLayerCount(); ++i) {
            for (const Parameter<T>& parameter : net.getLayer(i)->parameters()) {
                all.push_back(parameter);
            }
        }
        return all;
    }

    static Tensor<T> rowsOf(const Tensor<T>& tensor, int first, int last) {
        std::vector<int> dimensions = tensor.getShape().getDimensions();
        const size_t rowSize = tensor.getShape().size() / static_cast<size_t>(dimensions[0]);
        dimensions[0] = last - first;
        return Tensor<T>(Shape(dimensions), tensor.getData().data() + static_cast<size_t>(first) * rowSize);
    }

    // Runs work(0) here and work(1..workers-1) on new threads, then rethrows the first failure.
    template <typename Work>
    static void runAcross(int workers, Work work) {
        std::vector<std::exception_ptr> errors(static_cast<size_t>(workers));
        auto guarded = [&](int worker) {
            try {
                work(worker);
            } catch (...) {
                errors[static_cast<size_t>(worker)] = std::current_exception();
            }
        };
        std::vector<std::thread> threads;
        for (int worker = 1; worker < workers; ++worker) {
            threads.emplace_back(guarded, worker);
        }
        guarded(0);
        for (std::thread& thread : threads) {
            thread.join();
        }
        for (const std::exception_ptr& error : errors) {
            if (error) {
                std::rethrow_exception(error);
            }
        }
    }

    SmartDNN<T>& model;
    std::vector<std::unique_ptr<SmartDNN<T>>> replicas;
    std::vector<std::unique_ptr<Loss<T>>> losses;
};

} // namespace smart_dnn

#endif // DATA_PARALLEL_TRAINER_HPP
