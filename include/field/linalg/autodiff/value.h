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
