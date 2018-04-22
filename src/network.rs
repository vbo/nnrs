use math::Matrix;
use math::Vector;
use std::fmt;
use std::mem;
use std::collections::HashSet;

pub struct Network {
    layers: Vec<Layer>,
    input_layer: LayerID,
    output_layer: LayerID,
}

#[derive(Debug, Copy, Clone, Hash, Eq, PartialEq)]
pub struct LayerID(usize);
impl fmt::Display for LayerID {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "LayerID:{}", self.0)
    }
}

enum LayerKind {
    Input,
    Output,
    Hidden,
}

fn sigmoid(x: f64) -> f64 { 1.0 / ((-x).exp() + 1.0) }

impl Network {
    pub fn new(input_size: usize, output_size: usize) -> Self {
        Network {
            layers: {
                let mut v = Vec::new();
                v.push(Layer::new(LayerKind::Input, input_size));
                v.push(Layer::new(LayerKind::Output, output_size));
                v
            },
            input_layer: LayerID(0),
            output_layer: LayerID(1),
        }
    }

    pub fn input_layer(&self) -> LayerID { self.input_layer }
    pub fn output_layer(&self) -> LayerID { self.output_layer }
    fn get_layer_mut(&mut self, id: LayerID) -> &mut Layer { &mut self.layers[id.0] }
    fn get_layer_activations_mut(&mut self, id: LayerID) -> &mut Vector {
        &mut self.layers[id.0].activations
    }
    fn get_layer(&self, id: LayerID) -> &Layer { &self.layers[id.0] }

    fn calc_layers_order_internal(
        &self,
        scheduled_currently: &mut HashSet<LayerID>,
        layers_order: &mut Vec<LayerID>,
        current_id: LayerID
    ) {
        scheduled_currently.insert(current_id);
        for dependency in &self.get_layer(current_id).dependencies {
            if layers_order.contains(&dependency.id) {
                continue;
            }
            if scheduled_currently.contains(&dependency.id) {
                panic!("Dependency cycle!");
            } else {
                self.calc_layers_order_internal(
                    scheduled_currently, layers_order, dependency.id);
            }
        }
        layers_order.push(current_id);
    }

    fn calc_layers_order(&self) -> Vec<LayerID> {
        let mut scheduled_currently = HashSet::new();
        let mut layers_order = Vec::new();
        let output_layer = self.output_layer();
        self.calc_layers_order_internal(
            &mut scheduled_currently,
            &mut layers_order,
            output_layer);

        return layers_order;
    }

    fn get_layer_activations_owned(&mut self, layer_id: LayerID) -> Vector {
        let layer = &mut self.layers[layer_id.0];
        let mut tmp = Vector::empty();
        mem::swap(&mut tmp, &mut layer.activations);
        return tmp;
    }

    fn set_layer_activations_owned(&mut self, layer_id: LayerID, mut activations: Vector) {
        let layer = &mut self.layers[layer_id.0];
        mem::swap(&mut activations, &mut layer.activations);
    }

    fn compute_layer_forward(&mut self, layer_id: LayerID) {
        let mut activations = self.get_layer_activations_owned(layer_id);
        println!("A:   {:?}", activations);
        {
            let layer = self.get_layer(layer_id);
            let layer_dependencies = &self.get_layer(layer_id).dependencies;
            let mut activations_from_dep = Vector::new(activations.rows)
                                                  .init_with(0.0);

            for dependency in layer_dependencies {
                let dep_activations = &self.get_layer(dependency.id).activations;
                let weights = &dependency.weights;
                weights.dot_vec(dep_activations, &mut activations_from_dep);
                activations.add_to_me(&activations_from_dep);
            }
            activations.apply(|x| {x + layer.bias});
            activations.apply(sigmoid);
            println!("A:   {:?}", activations);
        }
        self.set_layer_activations_owned(layer_id, activations);
    }

    pub fn forward_propagation(&mut self, input_data: &[f64]) {
        let layers_order = self.calc_layers_order();

        {
            let input_id = self.input_layer();
            let mut input_layer = self.get_layer_mut(input_id);
            assert!(input_data.len() == input_layer.activations.rows, "Invalid input dimensions.");
            input_layer.activations.copy_from_slice(input_data);
        }

        println!("Layers order: {:?}", layers_order);
        for layer_id in layers_order {
            println!("Computing layer {}", layer_id);
            match self.get_layer_mut(layer_id).kind {
                LayerKind::Input => continue,
                _ => self.compute_layer_forward(layer_id),
            }
        }
    }

    pub fn add_hidden_layer(&mut self, rows: usize) -> LayerID {
        self.layers.push(Layer::new(LayerKind::Hidden, rows));
        return LayerID(self.layers.len() - 1);
    }

    pub fn add_layer_dependency(&mut self, source_id: LayerID, target_id: LayerID) {
        assert!(source_id.0 != target_id.0, "Self dependency is not allowed for {}", source_id);
        assert!(source_id.0 < self.layers.len(), "Invalid dependency source");
        assert!(target_id.0 < self.layers.len(), "Invalid dependency target");
        // TODO(parallel): Matrices can be shared between threads readonly, while
        // activations we need to copy.
        let weights = {
            let source = &self.layers[source_id.0];
            let target = &self.layers[target_id.0];
            Matrix::new(source.activations.rows, target.activations.rows).init_rand()
        };
        let layer_dependency = LayerDependency {
            id: target_id,
            weights: weights,
        };
        self.layers[source_id.0].dependencies.push(layer_dependency);
    }
}

struct Layer {
    kind: LayerKind,
    activations: Vector,
    dependencies: Vec<LayerDependency>,
    bias: f64
}

impl Layer {
    fn new(kind: LayerKind, rows: usize) -> Self {
        Layer {
            kind: kind,
            activations: Vector::new(rows).init_with(0.0),
            dependencies: Vec::new(),
            bias: 0.0f64
        }
    }
}

struct LayerDependency {
    id: LayerID,
    weights: Matrix,
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layer_order_forward() {
        let mut nn = Network::new(5, 2);
        let inputs_id = nn.input_layer();
        let l1_id = nn.add_hidden_layer(2);
        let l2_id = nn.add_hidden_layer(2);
        let outputs_id = nn.output_layer();
        nn.add_layer_dependency(outputs_id, l2_id);
        nn.add_layer_dependency(l2_id, l1_id);
        nn.add_layer_dependency(l1_id, inputs_id);

        let layer_order = nn.calc_layers_order();
        assert_eq!(layer_order, vec!(inputs_id, l1_id, l2_id, outputs_id))
    }

    #[test]
    fn layer_order_skip() {
        let mut nn = Network::new(5, 2);
        let inputs_id = nn.input_layer();
        let l1_id = nn.add_hidden_layer(2);
        let l2_id = nn.add_hidden_layer(2);
        let outputs_id = nn.output_layer();
        nn.add_layer_dependency(outputs_id, l2_id);
        nn.add_layer_dependency(outputs_id, l1_id);
        nn.add_layer_dependency(l2_id, l1_id);
        nn.add_layer_dependency(l2_id, inputs_id);
        nn.add_layer_dependency(l1_id, inputs_id);

        let layer_order = nn.calc_layers_order();
        assert_eq!(layer_order, vec!(inputs_id, l1_id, l2_id, outputs_id))
    }

    #[test]
    #[should_panic(expected="Dependency cycle")]
    fn layer_order_cycle() {
        let mut nn = Network::new(5, 2);
        let inputs_id = nn.input_layer();
        let outputs_id = nn.output_layer();
        let l1_id = nn.add_hidden_layer(2);
        nn.add_layer_dependency(outputs_id, l1_id);
        nn.add_layer_dependency(l1_id, inputs_id);
        nn.add_layer_dependency(inputs_id, outputs_id);

        let layer_order = nn.calc_layers_order();
    }
}
