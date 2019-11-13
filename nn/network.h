#pragma once

#include <list>
#include "layer.h"
#include "data.h"

class Network
{
public:
    Network(std::function<Vecf(Vecf, Vecf)> loss_derivative, double learning_step = 0.01) :loss_derivative_(loss_derivative), learning_step_(learning_step){};
    ~Network() = default;
    void train(const std::vector<Vecf>& data, const std::vector<Vecf>& labels, int batch_size, int epoch_count = 30) {
        if (data.size() != labels.size())
            throw std::logic_error("data.size() != labels.size()");
#if 0
        {
            FC_Layer *first = dynamic_cast<FC_Layer*>(net_.front().get());
            if (!first)
                throw std::exception();
            FC_Layer *second = dynamic_cast<FC_Layer*>(net_.back().get());
            std::cout << "First layer weight: " << first->weights_;
            std::cout << "Second layer weight: " << second->weights_;

        }
#endif // 1

        for (size_t i = 0; i < epoch_count; i++)
        {
            auto start_ = std::chrono::system_clock::now();
            bool low_error = epoch(data, labels, batch_size);
            float dur = (std::chrono::duration<float>(std::chrono::system_clock::now() - start_)).count();
            std::cout << "Epoch duration in second: " << dur << std::endl;
            if (low_error) {
                return;
            };
            
        }
        
    }
    bool epoch(const std::vector<Vecf>& data, const std::vector<Vecf>& labels, int batch_size) {
        auto itd = data.begin();
        auto itl = labels.begin();
        int iteration_count = data.size() / batch_size;
        static std::vector<int> index_vector(iteration_count);
        std::for_each(index_vector.begin(), index_vector.end(), [batch_size](int& x) {x = rand() % batch_size; });
        bool res = false;
        for (size_t i = 0; i < iteration_count; i++)
        {
            res = learn(itd, itl, data, labels, i, batch_size);
            if (res) {
                return true;
            }

        }
    }

    bool learn(std::vector<Vecf>::const_iterator itd, std::vector<Vecf>::const_iterator itl, 
        const std::vector<Vecf>& data, const std::vector<Vecf>& labels, int i, int batch_size) {
        //auto it = std::next(data.begin(),index_vector[std::distance(itd, data.begin())]);
        for (; itd != std::next(data.begin(), (i + 1) * batch_size); ++itd, ++itl) {
            //std::cout << "------------------------ Train Iteration --------------------------" << std::endl;
            forward(*itd);
            Vecf debug_vec = net_.back()->getOutputs();
            //std::cout << "After forward: " << debug_vec;
            back(*itl, *itd);
        };
        static double last_error = 0;

        double error = check_error(std::next(data.begin(), i* batch_size), itd,
            std::next(labels.begin(), i * batch_size));
        double err_diff = 0;
        std::cout << "Error: " << error << std::endl;
        if (error < 0.07) {
            std::cout << "Error: " << error << std::endl;

            return true;
        }
#if 0


        else if (0) {
            if (last_error == 0) {
                last_error = error;
                continue;
            }
            if (error < last_error) {
                //err_diff = (error - last_error);
            }
            if (abs(error / last_error - 1) < 0.1 || error > 1.0) {
                float c = 1;
                if (error > last_error) {
                    c = -1;
                }
                err_diff = c * 0.1;//(error - last_error);
            }
            learning_step_ += err_diff;
            last_error = error;

        }
#endif // 0

        return false;
    }
    double check_error(std::vector<Vecf>::const_iterator itd_s, std::vector<Vecf>::const_iterator itd_e, std::vector<Vecf>::const_iterator itl_s) {
        Vecf infer_res(itl_s->size_);
        double sum = 0;
        size_t dist = std::distance(itd_s, itd_e);
        for (; std::distance(itd_s, itd_e); ++itd_s, ++itl_s)
        {
            infer(*itd_s, infer_res);
            //std::cout << "After forward: " << infer_res;
            //std::cout << "Label: " << *itl_s;
            size_t index = std::max_element(itl_s->data_.get(), itl_s->data_.get() + infer_res.size_) - itl_s->data_.get();
            sum -= log(infer_res.data_.get()[index]);
        }
        return abs(sum) / double(dist);
    }
    void test(const std::vector<Vecf>& data, const std::vector<Vecf>& labels) {
        if (data.size() != labels.size())
            throw std::logic_error("data.size() != labels.size()");

        auto itd = data.begin();
        auto itl = labels.begin();
        
            size_t positive = 0;
        for (; itd != data.end(); ++itd, ++itl) {
            //std::cout << "------------------------ Test Iteration --------------------------" << std::endl;
            //std::cout << "Label: " << *itl;

            forward(*itd);
            Vecf debug_vec = net_.back()->getOutputs();
            //std::cout << "After forward: " << debug_vec;
            //Vecf res(itl->size_);
            //TENS_MULT(debug_vec, *itl, res);
            //std::cout << "Res: " << res;
            double* pr = debug_vec.data_.get();
            int ind = std::max_element(pr,pr+debug_vec.size_) - pr;
            if (itl->data_.get()[ind] > 0.5) {
                positive++;
            }
            /*for (size_t i = 0; i < res.size_; i++)
            {
                if (pr[i] > thresh) {
                    ++success;
                    break;
                }

            }*/
        }
            std::cout << "Accuracy: " << float(positive)/labels.size() << std::endl;
    }

    void infer(const Vecf& v, Vecf& r) {
        forward(v);
        if (r.size_ != net_.back()->outputSize())
            r.resize(net_.back()->outputSize());
        r.fill(net_.back()->getOutputs().data_.get());
    }
    void addLayer(const Layer::Ptr& l) {
        if (net_.size())
        {
            if (net_.back()->outputSize() != l->inputSize())
                throw std::logic_error("Can not link layers output_size_ != l.input_size_");
        }
        net_.push_back(l);
    }
private:
    void forward(const Vecf& inputs) {
        auto input_layer = net_.begin();
        if (inputs.size_ != (*input_layer)->inputSize())
            throw std::logic_error("Input data size != input layer size");
        const Vecf* p_inputs = &inputs;
        for (auto layer_it = net_.begin(); layer_it != net_.end(); ++layer_it) {
            (*layer_it)->forward(*p_inputs);
            p_inputs = &(*layer_it)->getOutputs();
        }
    }
    void back(const Vecf& label, const Vecf& inputs) {
        if (net_.size() > 2) throw std::exception();
        
        FC_Layer *first = dynamic_cast<FC_Layer*>(net_.front().get());
        if (!first)
            throw std::exception();
        FC_Layer *second = dynamic_cast<FC_Layer*>(net_.back().get());
        if (!second)
            throw std::exception();

        if (label.size_ != second->outputs_.size_)
            throw std::logic_error("label.size_ != second->outputs_.size_");
        
        Vecf loss_derivative = loss_derivative_(label, second->outputs_);
        //std::cout << "Loss derivative: " << loss_derivative ;

        Vecf second_activation_derivative(second->raw_outputs_.size_);
        second->activation_derivative_func_(second->raw_outputs_, second_activation_derivative);
        //std::cout << "Second layer activation derivative: " << second_activation_derivative;

        Vecf tmp(loss_derivative.size_);
        TENS_MULT(loss_derivative, second_activation_derivative, tmp);
        //std::cout << "tmp: " << tmp;

        //Matf Dv(second->raw_outputs_.size_, first->outputs_.size_);
        static Matf Dv(second->output_size_, second->input_size_);
        MULT(tmp, first->outputs_, Dv);
        //std::cout << "V weight derivative: " << Dv;

        Vecf tmp2(second->weights_.columns_);
        MULT(tmp, second->weights_, tmp2);
        //std::cout << "tmp2: " << tmp2;

        Vecf tmp3(tmp2.size_);
        Vecf activation_der2(first->raw_outputs_.size_); 
        first->activation_derivative_func_(first->raw_outputs_, activation_der2);
        //std::cout << "First layer activation derivative: " << activation_der2;
        TENS_MULT(tmp2, activation_der2, tmp3);
        //std::cout << "tmp3: " << tmp3;

        static Matf Dw(first->output_size_, first->input_size_);
        MULT(tmp3, inputs, Dw);
        //std::cout << "W weight derivative: " << Dw;

        //std::cout << "Before correction first->weights_: " << first->weights_;
        ADD(Dw, first->weights_, -learning_step_);
        //std::cout << "After correction first->weights_: " << first->weights_;

        //std::cout << "Before correction second->weights_: " << second->weights_;
        ADD(Dv, second->weights_, -learning_step_);
        //std::cout << "After correction second->weights_: " << second->weights_;
    }

    void correct() {

    }

    std::function<Vecf(Vecf, Vecf)> loss_derivative_;
    std::list<Layer::Ptr> net_;
    double learning_step_;
};

