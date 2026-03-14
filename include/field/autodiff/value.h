#ifndef FIELD_AUTODIFF_VALUE_H_
#define FIELD_AUTODIFF_VALUE_H_

#include <cmath>
#include <functional>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace field::autodiff {

class Value {
private:
    struct Node {
        double data = 0.0;
        double grad = 0.0;
        std::string label;
        std::vector<std::shared_ptr<Node>> parents;
        std::function<void()> backward;
    };
public:
    Value() : node_(std::make_shared<Node>()) {}
    explicit Value(double data, std::string label="") : node_(std::make_shared<Node>()) {node_->data = data; node_->label = std::move(label);}
    Value Parameter(double data, const std::string& label="") {return Value(data, label);}
    double Data() const {return node_->data;}
    double Grad() const {return node_->grad;}
    const std::string& Label() const {return node_->label;}
    void SetData(double data) {node_->data = data;}
    void ZeroGrad() {node_->grad = 0.0;}
    void Backward() {
        std::vector<std::shared_ptr<Node>> topo;
        std::set<const Node*> visited;
        BuildTopo(node_, visited, topo);
        for (const std::shared_ptr<Node>& curr : topo) {
            curr->grad = 0.0;
        }
        node_->grad = 1.0;
        for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
            if ((*it)->backward) {
                (*it)->backward();
            }
        }
    }
    friend Value operator+(const Value& lhs, const Value& rhs) {
        auto out = Value(lhs.Data() + rhs.Data());
        out.node_->parents = {lhs.node_, rhs.node_};
        std::weak_ptr<Node> out_weak = out.node_;
        std::shared_ptr<Node> left = lhs.node_;
        std::shared_ptr<Node> right  = rhs.node_;
        out.node_->backward = [out_weak, left, right]() {
            std::shared_ptr<Node> out_locked = out_weak.lock();
            if (!out_locked) return;
            left->grad += out_locked->grad;
            right->grad += out_locked->grad;
        };
        return out;
    }
    friend Value operator-(const Value& lhs, const Value& rhs) {
        auto out = Value(lhs.Data() - rhs.Data());
        out.node_->parents = {lhs.node_, rhs.node_};
        std::weak_ptr<Node> out_weak = out.node_;
        std::shared_ptr<Node> left = lhs.node_;
        std::shared_ptr<Node> right = rhs.node_;

        out.node_->backward = [out_weak, left, right]() {
            std::shared_ptr<Node> out_locked = out_weak.lock();
            if (!out_locked) return;
            left->grad += out_locked->grad;
            right->grad -= out_locked->grad;
        };
        return out;
    }
    friend Value operator*(const Value& lhs, const Value& rhs) {
        auto out = Value(lhs.Data() * rhs.Data());
        out.node_->parents = {lhs.node_, rhs.node_};
        std::weak_ptr<Node> out_weak = out.node_;
        std::shared_ptr<Node> left = lhs.node_;
        std::shared_ptr<Node> right = rhs.node_;

        out.node_->backward = [out_weak, left, right]() {
            std::shared_ptr<Node> out_locked = out_weak.lock();
            if (!out_locked) return;
            left->grad += right->data * out_locked->grad;
            right->grad += left->data * out_locked->grad;
        };
        return out;
    }
    friend Value operator/(const Value& lhs, const Value& rhs) {return lhs * Pow(rhs, -1.0);}
    friend Value operator+(const Value& lhs, double rhs) {return lhs + Value(rhs); }
    friend Value operator+(double lhs, const Value& rhs) {return Value(lhs) + rhs;}
    friend Value operator-(const Value& lhs, double rhs) {return lhs - Value(rhs);}
    friend Value operator-(double lhs, const Value& rhs) {return Value(lhs) - rhs;}
    friend Value operator*(const Value& lhs, double rhs) {return lhs * Value(rhs);}
    friend Value operator*(double lhs, const Value& rhs) {return Value(lhs) * rhs;}
    friend Value operator/(const Value& lhs, double rhs) {return lhs / Value(rhs);}
    friend Value operator/(double lhs, const Value& rhs) {return Value(lhs)/rhs;}
    friend Value operator-(const Value& v) {return Value(0.0) - v;}
    friend Value Tanh(const Value& x) {
        const double t = std::tanh(x.Data());
        auto out = Value(t);
        out.node_->parents = {x.node_};
        std::weak_ptr<Node> out_weak = out.node_;
        std::shared_ptr<Node> input = x.node_;

        out.node_->backward = [out_weak, input]() {
            std::shared_ptr<Node> out_locked = out_weak.lock();
            if (!out_locked) return;
            input->grad += (out_locked->grad > 0.0 ? 1.0 : 0.0) * out_locked->grad;
        };
        return out;
    }
    friend Value ReLu(const Value& x) {
        auto out = Value(x.Data() > 0.0 ? x.Data() : 0.0);
        out.node_->parents = {x.node_};
        std::weak_ptr<Node> out_weak = out.node_;
        std::shared_ptr<Node> input = x.node_;

        out.node_->backward = [out_weak, input]() {
            std::shared_ptr<Node> out_locked = out_weak.lock();
            if (!out_locked) return;
            input->grad += (out_locked->data > 0.0 ? 1.0 : 0.0) * out_locked->grad;
        };
        return out;
    }
    friend Value Exp(const Value& x) {
        const double e = std::exp(x.Data());
        auto out = Value(e);
        out.node_->parents = {x.node_};
        std::weak_ptr<Node> out_weak = out.node_;
        std::shared_ptr<Node> input = x.node_;

        out.node_->backward = [out_weak, input]() {
            std::shared_ptr<Node> out_locked = out_weak.lock();
            if (!out_locked) return;
            input->grad += out_locked->data * out_locked->grad;
        };
        return out;
    }
    friend Value Log(const Value& x) {
        if (x.Data() <= 0.0) throw std::domain_error("Log requires a positive input.");
        auto out = Value(std::log(x.Data()));
        out.node_->parents = {x.node_};
        std::weak_ptr<Node> out_weak = out.node_;
        std::shared_ptr<Node> input = x.node_;

        out.node_->backward = [out_weak, input]() {
            std::shared_ptr<Node> out_locked = out_weak.lock();
            if (!out_locked) return;
            input->grad += (1.0 / input->data) * out_locked->grad;
        };
        return out;
    }    
    friend Value Pow(const Value& x, double power) {
        auto out = Value(std::pow(x.Data(), power));
        out.node_->parents = {x.node_};
        std::weak_ptr<Node> out_weak = out.node_;
        std::shared_ptr<Node> input = x.node_;

        out.node_->backward = [out_weak, input, power]() {
            std::shared_ptr<Node> out_locked = out_weak.lock();
            if (!out_locked) return;
            input->grad += power * std::pow(input->data, power - 1.0) * out_locked->grad;
        };
        return out;
    }
private:
    static void BuildTopo(const std::shared_ptr<Node>& node, std::set<const Node*>& vis,
                            std::vector<std::shared_ptr<Node>>& topo) {
        if (!vis.insert(node.get()).second) return;
        for (const std::shared_ptr<Node&> p : node->parents) {
            BuildTopo(p, vis, topo);
        }
        topo.push_back(node);
    } 
    std::shared_ptr<Node> node_;
};

}

#endif  // FIELD_AUTODIFF_VALUE_H_
