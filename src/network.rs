use math::Matrix;
use math::Vector;

pub struct Network {
    layers: Vec<Layer>,
}

type LayerID = usize;

impl Network {
    pub fn new() -> Self {
        Network {
            layers: Vec::new(),
        }
    }

    pub fn add_layer(&mut self, rows: usize) -> LayerID {
        self.layers.push(Layer {
            activations: Vector::new(rows).init_with(0.0),
            dependencies: Vec::new(),
        });
        return self.layers.len() - 1;
    }

    pub fn add_layer_dependency(&mut self, source_id: LayerID, target_id: LayerID) {
        assert!(source_id != target_id, "Self dependency is not allowed for layer {}", source_id);
        assert!(source_id < self.layers.len(), "Invalid dependency source");
        assert!(target_id < self.layers.len(), "Invalid dependency target");
        let weights = {
            let source = &self.layers[source_id];
            let target = &self.layers[target_id];
            Matrix::new(target.activations.rows, source.activations.rows).init_rand()
        };

        self.layers[source_id].dependencies.push((target_id, weights));
    }
}

struct Layer {
    activations: Vector,
    dependencies: Vec<(LayerID, Matrix)>,
}

impl Layer {
    fn new(rows: usize) -> Self {
        Layer {
            activations: Vector::new(rows),
            dependencies: Vec::new(),
        }
    }
}


