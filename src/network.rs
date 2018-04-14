use math::Matrix;
use math::Vector;
use std::fmt;
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
    fn get_layer(&self, id: LayerID) -> &Layer { &self.layers[id.0] }

    fn calc_layers_order_internal(
        &mut self,
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

    fn calc_layers_order(&mut self) -> Vec<LayerID> {
        let mut scheduled_currently = HashSet::new();
        let mut layers_order = Vec::new();

        self.calc_layers_order_internal(
            &mut scheduled_currently,
            &mut layers_order,
            self.output_layer());

        return layers_order;
    }

    pub fn forward_propagation(&mut self, input_data: &[f64]) {
        let layers_order = self.calc_layers_order();
        let input_id = self.input_layer();
        {
            let mut input_layer = self.get_layer_mut(input_id);
            assert!(input_data.len() == input_layer.activations.rows, "Invalid input dimensions.");
            input_layer.activations.copy_from_slice(input_data);
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
            Matrix::new(target.activations.rows, source.activations.rows).init_rand()
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
}

impl Layer {
    fn new(kind: LayerKind, rows: usize) -> Self {
        Layer {
            kind: kind,
            activations: Vector::new(rows).init_with(0.0),
            dependencies: Vec::new(),
        }
    }
}

struct LayerDependency {
    id: LayerID,
    weights: Matrix,
}

