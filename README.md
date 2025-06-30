# Task #PF - 2: Proyecto Final - Neural Network  
**course:** Programación III  
**unit:** final project  
**cmake project:** prog3_nn_final_project_v2025_01
## Indicaciones Específicas
El tiempo límite para la evaluación es 2 semanas.

Las preguntas deberán ser respondidas en archivos cabeceras (.h) correspondientes:

- `nn_interfaces.h`
- `nn_dense.h`
- `nn_activation.h`
- `nn_loss.h`
- `nn_optimizer.h`
- `tensor.h`
- `neural_network.h`

Deberás subir estos archivos directamente a www.gradescope.com o se puede crear un `.zip` que contenga todos ellos y subirlo.

## Question #1 - Activation - RELU y Sigmoid (2 points)

```c++
  template<typename T>
  class ReLU final : public ILayer<T> {
  public:
    Tensor<T,2> forward(const Tensor<T,2>& z) override { ... }
    Tensor<T,2> backward(const Tensor<T,2>& g) override { ... }
  };
```

```c++
  template<typename T>
  class Sigmoid final : public ILayer<T> {
  public:
    Tensor<T,2> forward(const Tensor<T,2>& z) override { ... }
    Tensor<T,2> backward(const Tensor<T,2>& g) override { ... }
  };
```
  
**Use Case: ReLu**  
```c++
using T = float;
auto relu = utec::neural_network::ReLU<T>();
// Tensores
Tensor<T, 2> M({2,2}); M = {-1, 2, 0, -3};
Tensor<T, 2> GR({2,2}); GR.fill(1.0f);
// Forward
auto R = relu.forward(M);
std::cout << R(0,1) << "\n"; // espera 2
// Backward
const auto dM = relu.backward(GR);
std::cout << dM;
```

```c++
auto sigmoid = utec::neural_network::Sigmoid<T>();
// Tensores
constexpr int rows = 5;
constexpr int cols = 4;
Tensor<T, 2> M({rows, cols});
M.fill(-100.0);
for (int i = 0; i < rows; ++i)
    for (int j = 0; j < cols; ++j) {
        if (i == j) M(i, j) = 100.0;
        if (i == rows - 1 - j) M(i, j) = 100.0;
    }
std::cout << std::fixed << std::setprecision(1);
std::cout << M << std::endl;
// Forward
const auto S = sigmoid.forward(M);
std::cout << S << std::endl;
// Backward
Tensor<T, 2> GR({rows,cols}); GR.fill(1.0);
const auto dM = sigmoid.backward(GR);
std::cout << dM << std::endl;
```

## Question #2 - Loss Function - MSE y BCE (2 points)

```c++
  template<typename T>
  class MSELoss final: public ILoss<T, 2> {
  public:
    MSELoss(const Tensor<T,2>& y_prediction, const Tensor<T,2>& y_true) { ... }
    T loss() const override { ... }
    Tensor<T,2> loss_gradient() const override { ... }
  };
```

```c++
  template<typename T>
  class BCELoss final: public ILoss<T, 2> {
  public:
    MSELoss(const Tensor<T,2>& y_prediction, const Tensor<T,2>& y_true) { ... }
    T loss() const override { ... }
    Tensor<T,2> loss_gradient() const override { ... }
  };
```
 
**Use Case: MSE**  
```c++
using T = double;
// Tensores
Tensor<T,2> y_predicted({1,2}); y_predicted = {1, 2};
Tensor<T,2> y_expected({1,2}); y_expected = {0, 4};

const utec::neural_network::MSELoss<T> mse_loss(y_predicted, y_expected);
// Forward
const T loss = mse_loss.loss();
std::cout << loss << "\n";
// Backward
const Tensor<T,2> dP = mse_loss.loss_gradient();
std::cout << dP;
```
**Use Case: BCE**
```c++
using T = double;
// Tensores
Tensor<T,2> y_predicted({1,2}); y_predicted = {0.9, 0.1};
Tensor<T,2> y_expected({1,2}); y_expected = {0, 1};

const utec::neural_network::BCELoss<T> bce_loss(y_predicted, y_expected);
// Forward
const T loss = bce_loss.loss();
std::cout << loss << "\n";
// Backward
const Tensor<T,2> dP = bce_loss.loss_gradient();
std::cout << dP;
```

## Question #3 - Dense Layer (6 points)

```c++
  template<typename T>
  class Dense final : public ILayer<T> {
  public:
    template<typename InitWFun, typename InitBFun>
    Dense(size_t in_f, size_t out_f, InitWFun init_w_fun, InitBFun init_b_fun) { ... }
    Tensor<T,2> forward(const Tensor<T,2>& x) override { ... }
    Tensor<T,2> backward(const Tensor<T,2>& dZ) override { ... }
    void update_params(IOptimizer<T>& optimizer) override { ... }
  };
```

**Use Case: Using Identity Initializer and Zero**
```c++
using T = double;

// Inicializador identidad
auto init_identity = [](Tensor<T,2>& M) {
    const auto shape = M.shape();
    const size_t rows = shape[0];
    const size_t cols = shape[1];
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            M(i,j) = (i == j ? 1.0 : 0.0);
};

// Inicializador de ceros
auto init_zero = [](Tensor<T,2>& M) {
    for (auto& v : M) v = 0.0;
};
constexpr int n_batches = 2;
constexpr int in_features = 4;
constexpr int out_features = 3;
Dense<double> layer(size_t{in_features}, size_t{out_features},init_identity, init_zero);

Tensor<T,2> X1({n_batches, in_features});
std::iota(X1.begin(), X1.end(), 1);
// Forward
Tensor<T,2> Y = layer.forward(X1);
std::cout << Y << std::endl;

Tensor<T,2> Z({n_batches, out_features});
std::iota(Z.begin(), Z.end(), 1);
auto Z_adjusted = Z / static_cast<T>(Z.size());

Tensor<T,2> X_adjusted = layer.backward(Z_adjusted);
// X ajustado
std::cout << X_adjusted << std::endl;
```
**Use Case: Using Xavier Initializer**
```c++
using T = double;

// Inicializador Xavier
std::mt19937 gen(4);
auto xavier_init = [&](auto& parameter) {
    const double limit = std::sqrt(6.0 / (parameter.shape()[0] + parameter.shape()[1]));
    std::uniform_real_distribution<> dist(-limit, limit);
    for (auto& v : parameter) v = dist(gen);
};

constexpr int n_batches = 2;
constexpr int in_features = 4;
constexpr int out_features = 3;
Dense<double> layer(size_t{in_features}, size_t{out_features},xavier_init, xavier_init);

Tensor<T,2> X1({n_batches, in_features});
std::iota(X1.begin(), X1.end(), 1);
// Forward
Tensor<T,2> Y = layer.forward(X1);
std::cout << Y << std::endl;

Tensor<T,2> Z({n_batches, out_features});
std::iota(Z.begin(), Z.end(), 1);
auto Z_adjusted = Z / static_cast<T>(Z.size());

Tensor<T,2> X_adjusted = layer.backward(Z_adjusted);
// X ajustado
std::cout << X_adjusted << std::endl;
```

## Question #4 - Entrenamiento - XOR (8 points)
  
```c++
template<typename T>
class NeuralNetwork {
public:
    void add_layer(std::unique_ptr<ILayer<T>> layer) { ...}
    template <template <typename ...> class LossType, 
        template <typename ...> class OptimizerType = SGD>
    void train( const Tensor<T,2>& X,const Tensor<T,2>& Y, 
        const size_t epochs, const size_t batch_size, T learning_rate) { ... }
    // Para realizar predicciones
    Tensor<T,2> predict(const Tensor<T,2>& X) { ... }
};
```  

**Use Case: Using He Initializer y MSE Loss Function**
```c++
constexpr size_t batch_size = 4;
Tensor<double,2> X({batch_size, 2});
Tensor<double,2> Y({batch_size, 1});

// Datos XOR
X = { 0, 0,
      0, 1,
      1, 0,
      1, 1};
Y = { 0, 1, 1, 0};

// Inicializador He
std::mt19937 gen(42);

auto init_he = [&](Tensor<double,2>& M) {
    const double last = 2.0/(static_cast<double>(M.shape()[0]+ M.shape()[1]));
    std::normal_distribution<double> dist(
        0.0,
        std::sqrt(last));
    for (auto& v : M) v = dist(gen);
};

// Construcci?n de la red
NeuralNetwork<double> net;
net.add_layer(std::make_unique<Dense<double>>(
    size_t{2}, size_t{4}, init_he, init_he));
net.add_layer(std::make_unique<ReLU<double>>());
net.add_layer(std::make_unique<Dense<double>>(
    size_t{4}, size_t{1}, init_he, init_he));

// Entrenamiento
constexpr size_t epochs = 3000;
constexpr double learning_rate = 0.08;
net.train<MSELoss> (X, Y, epochs, batch_size, learning_rate);

// Predicci?n
Tensor<double,2> Y_prediction = net.predict(X);

// Verificaci?n
for (size_t i = 0; i < batch_size; ++i) {
    const double p = Y_prediction(i,0);
    std::cout
        << std::fixed << std::setprecision(0)
        << "Input: (" << X(i,0) << "," << X(i,1)
        << std::fixed << std::setprecision(4)
        << ") -> Prediction: " << p << std::endl;
    if (Y(i,0) < 0.5) {
        assert(p < 0.5); // Expected output close to 0
    } else {
        assert(p >= 0.6); // Expected output close to 1
    }
}
```
**Use Case: Using Xavier Initializer y BCE Loss Function**
```c++
    constexpr size_t batch_size = 4;
    Tensor<double,2> X({batch_size, 2});
    Tensor<double,2> Y({batch_size, 1});

    // Datos XOR
    X = { 1, 0,
          0, 1,
          0, 0,
          1, 1};
    Y = { 1, 1, 0, 0};

    // Inicializador Xavier
    std::mt19937 gen(4);
    auto xavier_init = [&](auto& parameter) {
        const double limit = std::sqrt(6.0 / (parameter.shape()[0] + parameter.shape()[1]));
        std::uniform_real_distribution<> dist(-limit, limit);
        for (auto& v : parameter) v = dist(gen);
    };

    // Construcci?n de la red
    NeuralNetwork<double> net;
    net.add_layer(std::make_unique<Dense<double>>(
        size_t{2}, size_t{4}, xavier_init, xavier_init));
    net.add_layer(std::make_unique<Sigmoid<double>>());
    net.add_layer(std::make_unique<Dense<double>>(
        size_t{4}, size_t{1}, xavier_init, xavier_init));
    net.add_layer(std::make_unique<Sigmoid<double>>());

    // Entrenamiento
    constexpr size_t epochs = 4000;
    constexpr double lr = 0.08;
    net.train<BCELoss>(X, Y, epochs, batch_size, lr);

    // Predicci?n
    Tensor<double,2> Y_prediction = net.predict(X);

    // Verificaci?n
    for (size_t i = 0; i < batch_size; ++i) {
        const double p = Y_prediction(i, 0);
        std::cout
            << std::fixed << std::setprecision(0)
            << "Input: (" << X(i,0) << "," << X(i,1)
            << std::fixed << std::setprecision(4)
            << ") -> Prediction: " << p << std::endl;
        if (Y(i,0) < 0.5) {
            assert(p < 0.5); // Expected output close to 0
        } else {
            assert(p >= 0.6); // Expected output close to 1
        }
    }
```

## Question #5 - Optimización (SGD y Adam) (2 points)

```c++
template<typename T>
class SGD final : public IOptimizer<T> {
public:
    explicit SGD(T learning_rate = 0.01) { ... }
    void update(Tensor<T, 2>& params, const Tensor<T, 2>& grads) override { ... }
};
```
```c++
    template<typename T>
    class Adam final : public IOptimizer<T> {
    public:
        explicit Adam(T learning_rate = 0.001, T beta1 = 0.9, T beta2 = 0.999, T epsilon = 1e-8) { ... }
        void update(Tensor<T, 2>& params, const Tensor<T, 2>& grads) override { ... }
        void step() override { ... }
    }
  
**Use Case: SGD**  
```c++
    using T = float;

    Tensor<T,2> W({2,2}); W.fill(1.0f);
    Tensor<T,2> dW({2,2}); dW.fill(0.5f);
    utec::neural_network::SGD<T> opt(0.1f);

    opt.update(W, dW);
    std::cout
        << std::fixed << std::setprecision(6)
        << W(0,0) << "\n";
```
**Use Case: Adam**
```c++
    using T = float;

    Tensor<T,2> W({20,25}); W.fill(1.0f);
    Tensor<T,2> dW({20,25}); dW.fill(0.2f);
    utec::neural_network::Adam opt(0.01f, 0.009f, 9.00f);

    opt.update(W, dW);
    std::cout << W(0,0) << "\n";
```
