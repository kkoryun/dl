#pragma once

#include "data.h"
#include <random>


class Layer
{
public:
    virtual ~Layer() = default;
    virtual void forward(const Vecf& inputs) = 0;
    virtual size_t outputSize() const = 0;
    virtual size_t inputSize() const = 0;
    virtual const Vecf& getOutputs() const = 0;
    //virtual const Vecf& getRawOutputs() const = 0;
    using Ptr = std::shared_ptr<Layer>;
};

class WeightInitializer
{
public:
    virtual void initialize(const Matf& weights) {
        std::fill(weights.data_.get(), 
                  weights.data_.get() + weights.rows_ * weights.columns_, 
                    1);
    };
    virtual ~WeightInitializer() = default;
    using Ptr = std::shared_ptr<WeightInitializer>;
};

class RandomWeightInitializer : public WeightInitializer
{
public:
    RandomWeightInitializer() : WeightInitializer(),generator_(device_()), distribution_(1,2){

    }
    virtual void initialize(const Matf& weights) {
        std::for_each(weights.data_.get(),
            weights.data_.get() + weights.rows_ * weights.columns_,
            [this](double& x) {x = distribution_(generator_); });
    };
    static WeightInitializer::Ptr create() {
        return WeightInitializer::Ptr(new RandomWeightInitializer);
    }
    virtual ~RandomWeightInitializer() = default;
    
    std::random_device device_;
    std::mt19937 generator_;
    //std::normal_distribution<> distribution_;
    std::uniform_real_distribution<> distribution_;

};


using Activation = std::function<void(const Vecf&, Vecf&)>;
using ActivationDerivative = std::function<void(const Vecf&, Vecf&)>;

class FC_Layer : public Layer
{
public:
    FC_Layer(size_t input_size, size_t output_size, 
            Activation activation, 
            ActivationDerivative activation_derivative,
            WeightInitializer::Ptr w_init = WeightInitializer::Ptr(new WeightInitializer)) {
        output_size_ = output_size;
        input_size_ = input_size;
        weights_.resize(output_size, input_size);
        w_init->initialize(weights_);
        raw_outputs_.resize(output_size);
        activation_func_ = activation;
        activation_derivative_func_ = activation_derivative;
        outputs_.resize(output_size);
        delta_.resize(output_size, input_size);
    }
    ~FC_Layer() = default;
    size_t inputSize() const override  { return input_size_; };
    size_t outputSize() const override { return output_size_; };
    const Vecf& getOutputs() const override { return outputs_; };
    //const Vecf& getRawOutputs() const override { return raw_outputs_; };


    size_t output_size_;
    size_t input_size_;
    void forward(const Vecf& inputs) override {
        MULT(weights_, inputs, raw_outputs_);
        activation_func_(raw_outputs_, outputs_);
    }

    void back() {

    }

    friend class WeightInitializer;

    Matf weights_;
    Vecf raw_outputs_;
    Activation activation_func_;
    Vecf outputs_;

    ActivationDerivative activation_derivative_func_;

    Matf delta_;
};

