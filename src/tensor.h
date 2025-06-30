
#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_NEURAL_NETWORK_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_NEURAL_NETWORK_H

#include "nn_interfaces.h"
#include "nn_loss.h"
#include "nn_optimizer.h"
#include <memory>
#include <vector>
#include <iostream>
#include <iomanip>

namespace utec {
namespace neural_network {

  // Red neuronal que permite entrenar y predecir resultados
  template<typename T>
  class NeuralNetwork {
    std::vector<std::unique_ptr<ILayer<T>>> layers;
  public:
    void add_layer(std::unique_ptr<ILayer<T>> layer) {
      layers.emplace_back(std::move(layer));
    }

    // Propagacion completa
    Tensor<T,2> forward(const Tensor<T,2>& X) {
      Tensor<T,2> out = X;
      for (auto& l : layers)
        out = l->forward(out);
      return out;
    }

    // Prediccion sin entrenamiento
    Tensor<T,2> predict(const Tensor<T,2>& X) {
      return forward(X);
    }

    // Entrena usando una función de pérdida y un optimizador
    template <template<typename> class Loss, template<typename> class Optim>
    void train(const Tensor<T,2>& X, const Tensor<T,2>& Y, size_t epochs, size_t bs, T lr) {
      Optim<T> opt(lr);
      for (size_t ep = 0; ep < epochs; ++ep) {
        auto y_pred = forward(X);
        Loss<T> loss(y_pred, Y);
        auto grad = loss.loss_gradient();
        for (auto it = layers.rbegin(); it != layers.rend(); ++it)
          grad = (*it)->backward(grad);
        for (auto& l : layers)
          l->update_params(opt);
        opt.step();
      }
    }

    // Entrena usando la función de perdida y el optimizador por defecto (SGD)
    template <template<typename> class Loss>
    void train(const Tensor<T,2>& X, const Tensor<T,2>& Y, size_t epochs, size_t bs, T lr) {
      train<Loss, SGD>(X, Y, epochs, bs, lr);
    }
  };

  // Muestra la prediccion para cada entrada
  template<typename T>
  void print_predictions(NeuralNetwork<T>& nn, const std::vector<Tensor<T,2>>& inputs) {
    for (const auto& x : inputs) {
      auto y = nn.forward(x);
      std::cout << "Input: (" << x(0,0) << "," << x(0,1) << ") -> Pred: " << y(0,0) << "";
    }
  }

  // Calcula cuantas predicciones fueron correctas
  template<typename T>
  void print_accuracy(NeuralNetwork<T>& nn, const Tensor<T,2>& X, const Tensor<T,2>& Y) {
    size_t hits = 0;
    for (size_t i = 0; i < X.shape()[0]; ++i) {
      Tensor<T,2> sample({1, X.shape()[1]});
      for (size_t j = 0; j < X.shape()[1]; ++j)
        sample(0, j) = X(i, j);
      auto pred = nn.forward(sample);
      if ((pred(0,0) >= 0.5) == (Y(i,0) >= 0.5)) ++hits;
    }
    std::cout << std::fixed << std::setprecision(6)
              << static_cast<double>(hits) / X.shape()[0] << "";
  }

}
}

#endif //PROG3_NN_FINAL_PROJECT_V2025_01_NEURAL_NETWORK_H
