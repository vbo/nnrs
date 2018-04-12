use math::Matrix;
use math::Vector;
use std::fmt;

pub struct Network {
    layers: Vec<Layer>,
    input_layer: LayerID,
    output_layer: LayerID,
}

#[derive(Debug, Copy, Clone)]
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

    pub fn forward_propagation(&mut self, input_data: &[f64]) {
        // TODO: implement dependency traversal.
        let mut layers_order: Vec<LayerID> = Vec::new();
        let i: usize = 0;

        let mut stack: Vec<LayerID> = Vec::new();
        stack.push(current_id);

        loop {
            let current_id = layers_order[i];
            let deps = &self.get_layer(current_id).dependencies;
            for dep_id in deps {
                layer_order.push(dep_id);
            }
            i += 1;
        }

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
            target: target_id,
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
    target: LayerID,
    weights: Matrix,
}

