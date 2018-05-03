use std::fmt;
use std::mem;
use std::collections::HashSet;
use std::fs::File;
use serde_json;
use math::Matrix;
use math::Vector;

const LEARNING_RATE: f64 = 0.3;
pub const BATCH_SIZE: usize = 1000;

#[derive(Serialize, Deserialize)]
pub struct Network {
    layers: Vec<Layer>,
    input_layer: LayerID,
    output_layer: LayerID,
}

#[derive(Debug, Copy, Clone, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct LayerID(usize);
impl fmt::Display for LayerID {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "LayerID:{}", self.0)
    }
}

#[derive(Serialize, Deserialize)]
enum LayerKind {
    Input,
    Output,
    Hidden,
}

fn sigmoid(x: f64) -> f64 {
    1.0 / ((-x).exp() + 1.0)
}

fn avg_by_batch(x: f64) -> f64 {
    x * LEARNING_RATE / BATCH_SIZE as f64
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

    pub fn input_layer(&self) -> LayerID {
        self.input_layer
    }
    pub fn output_layer(&self) -> LayerID {
        self.output_layer
    }
    fn get_layer_mut(&mut self, id: LayerID) -> &mut Layer {
        &mut self.layers[id.0]
    }
    fn get_layer_activations_mut(&mut self, id: LayerID) -> &mut Vector {
        &mut self.layers[id.0].activations
    }
    fn get_layer(&self, id: LayerID) -> &Layer {
        &self.layers[id.0]
    }

    /// Layers vector is parallel to layer.dependencies.
    fn get_layer_and_deps_mut(&mut self, layer_id: LayerID) -> (&mut Layer, Vec<&mut Layer>) {
        // NOTE(vbo): maybe separate Nodes and Weights to different vectors.
        let layers_buf: *mut Layer = self.layers.as_mut_ptr();
        let cur_layer: *mut Layer = unsafe { layers_buf.offset(layer_id.0 as isize) };
        let mut deps_layers_vec = Vec::with_capacity(self.get_layer(layer_id).dependencies.len());
        for dep in &mut self.get_layer_mut(layer_id).dependencies {
            let dep_layer: *mut Layer = unsafe { layers_buf.offset(dep.id.0 as isize) };
            deps_layers_vec.push(unsafe { &mut *dep_layer });
        }

        return (unsafe { &mut *cur_layer }, deps_layers_vec);
    }

    fn calc_layers_order_internal(
        &self,
        scheduled_currently: &mut HashSet<LayerID>,
        layers_order: &mut Vec<LayerID>,
        current_id: LayerID,
    ) {
        scheduled_currently.insert(current_id);
        for dependency in &self.get_layer(current_id).dependencies {
            if layers_order.contains(&dependency.id) {
                continue;
            }
            if scheduled_currently.contains(&dependency.id) {
                panic!("Dependency cycle!");
            } else {
                self.calc_layers_order_internal(scheduled_currently, layers_order, dependency.id);
            }
        }
        layers_order.push(current_id);
    }

    fn calc_layers_order(&self) -> Vec<LayerID> {
        let mut scheduled_currently = HashSet::with_capacity(self.layers.len());
        let mut layers_order = Vec::with_capacity(self.layers.len());
        let output_layer = self.output_layer();
        self.calc_layers_order_internal(&mut scheduled_currently, &mut layers_order, output_layer);

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
        let (layer, dep_layers) = self.get_layer_and_deps_mut(layer_id);
        let layer_bias = layer.bias;
        let activations = &mut layer.activations;
        activations.fill_with(0.0);
        for (dep_index, dependency) in layer.dependencies.iter_mut().enumerate() {
            let dep_activations = &dep_layers[dep_index].activations;
            let weights = &dependency.weights;
            weights.add_dot_vec(dep_activations, activations);
        }
        activations.apply(|x| x + layer_bias);
        activations.apply(sigmoid);
    }

    pub fn predict(&mut self, input_data: &[f64]) -> &Vector {
        let layers_order = self.calc_layers_order();

        {
            let input_id = self.input_layer();
            let input_layer = self.get_layer_mut(input_id);
            assert!(
                input_data.len() == input_layer.activations.rows,
                "Invalid input dimensions."
            );
            input_layer.activations.copy_from_slice(input_data);
        }

        for layer_id in layers_order {
            match self.get_layer_mut(layer_id).kind {
                LayerKind::Input => continue,
                _ => self.compute_layer_forward(layer_id),
            }
        }

        return &self.get_layer(self.output_layer()).activations;
    }

    pub fn backward_propagation(&mut self, true_outputs: &Vector) {
        // TODO(vbo): calculate order once and reuse, invalidate on adding
        // layers / dependencies
        let mut layers_order = self.calc_layers_order();
        layers_order.reverse();

        for layer in &mut self.layers {
            layer.error.fill_with(0.0);
        }

        // Compute output error.
        let output_layer_id = self.output_layer();
        {
            let output_layer = self.get_layer_mut(output_layer_id);
            assert!(
                true_outputs.rows == output_layer.activations.rows,
                "Labels size mismatch with output layer: {} != {}",
                true_outputs.rows,
                output_layer.activations.rows
            );

            true_outputs.sub(&output_layer.activations, &mut output_layer.error);
        }

        for layer_id in layers_order {
            match self.get_layer(layer_id).kind {
                LayerKind::Input => continue,
                _ => self.compute_layer_backward(layer_id),
            }
        }
    }

    fn calc_grad_prefix(activations: &Vector, error: &Vector, res: &mut Vector) {
        assert!(
            activations.rows == error.rows && error.rows == res.rows,
            "Invalid dimentions for output error grad prefix PD: {}x1, {}x1, {}x1",
            activations.rows,
            error.rows,
            res.rows
        );

        for row in 0..activations.rows {
            res.mem[row] =
                2.0 * error.mem[row] * activations.mem[row] * (1.0 - activations.mem[row]);
        }
    }

    fn calc_weights_pd(
        error_grad_prefix: &Vector,
        previous_activations: &Vector,
        res: &mut Matrix,
    ) {
        assert!(
            error_grad_prefix.rows == res.rows && res.cols == previous_activations.rows,
            "Invalid dimentions for output weights PD: {}x1, {}x1, {}x{}",
            error_grad_prefix.rows,
            previous_activations.rows,
            res.rows,
            res.cols
        );

        for row in 0..res.rows {
            let row_start = row * res.cols;
            for col in 0..res.cols {
                res.mem[row_start + col] =
                    error_grad_prefix.mem[row] * previous_activations.mem[col];
            }
        }
    }

    fn compute_layer_backward(&mut self, layer_id: LayerID) {
        let (layer, mut dep_layers) = self.get_layer_and_deps_mut(layer_id);
        let mut grad_prefix = Vector::new(layer.activations.rows).init_with(0.0);
        Network::calc_grad_prefix(&layer.activations, &layer.error, &mut grad_prefix);
        layer.bias_batch_pd += grad_prefix.calc_sum();
        for (dep_index, dependency) in layer.dependencies.iter_mut().enumerate() {
            let mut weights_pd = Matrix::new_same_dim(&dependency.weights_batch_pd).init_with(0.0);
            Network::calc_weights_pd(
                &grad_prefix,
                &dep_layers[dep_index].activations,
                &mut weights_pd,
            );
            dependency.weights_batch_pd.add(&weights_pd);
            let mut weights_t =
                Matrix::new(dependency.weights.cols, dependency.weights.rows).init_with(0.0);
            dependency.weights.transpose(&mut weights_t);
            weights_t.add_dot_vec(&grad_prefix, &mut dep_layers[dep_index].error);
        }
    }

    pub fn add_hidden_layer(&mut self, rows: usize) -> LayerID {
        self.layers.push(Layer::new(LayerKind::Hidden, rows));
        return LayerID(self.layers.len() - 1);
    }

    pub fn add_layer_dependency(&mut self, source_id: LayerID, target_id: LayerID) {
        assert!(
            source_id.0 != target_id.0,
            "Self dependency is not allowed for {}",
            source_id
        );
        assert!(source_id.0 < self.layers.len(), "Invalid dependency source");
        assert!(target_id.0 < self.layers.len(), "Invalid dependency target");

        let mut dup_found = false;
        for dep in &self.layers[source_id.0].dependencies {
            if dep.id == target_id {
                dup_found = true;
            }
        }
        assert!(!dup_found, "Duplicate dependency");

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
            weights_batch_pd: Matrix::new(
                self.layers[source_id.0].activations.rows,
                self.layers[target_id.0].activations.rows,
            ).init_with(0.0),
        };
        self.layers[source_id.0].dependencies.push(layer_dependency);
    }

    /// Weights adjustment from info accumulated during backward propagation calls.
    /// Will also reset this info, preparing for a new batch.
    pub fn apply_batch(&mut self) {
        for layer in &mut self.layers {
            layer.bias += avg_by_batch(layer.bias_batch_pd);
            layer.bias_batch_pd = 0.0;

            for dependency in &mut layer.dependencies {
                dependency.weights_batch_pd.apply(&avg_by_batch);
                dependency.weights.add(&dependency.weights_batch_pd);
                dependency.weights_batch_pd.fill_with(0.0);
            }
        }
    }

    pub fn write_to_file(&self, output_path: &str) {
        let file = File::create(&output_path).unwrap();
        serde_json::to_writer(file, &self).unwrap();
    }

    pub fn load_from_file(path: &str) -> Self {
        let file = File::open(path).unwrap();
        serde_json::from_reader(&file).unwrap()
    }
}

#[derive(Serialize, Deserialize)]
struct Layer {
    kind: LayerKind,
    dependencies: Vec<LayerDependency>,
    bias: f64,
    // The following is needed only at evaluation and training time
    activations: Vector,
    // Following is needed only at training time
    error: Vector,
    bias_batch_pd: f64,
}

impl Layer {
    fn new(kind: LayerKind, rows: usize) -> Self {
        Layer {
            kind: kind,
            activations: Vector::new(rows).init_with(0.0),
            dependencies: Vec::new(),
            bias: 0.0,
            error: Vector::new(rows).init_with(0.0),
            bias_batch_pd: 0.0,
        }
    }
}

#[derive(Serialize, Deserialize)]
struct LayerDependency {
    id: LayerID,
    weights: Matrix,
    weights_batch_pd: Matrix,
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
        assert_eq!(layer_order, vec![inputs_id, l1_id, l2_id, outputs_id])
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
        assert_eq!(layer_order, vec![inputs_id, l1_id, l2_id, outputs_id])
    }

    #[test]
    #[should_panic(expected = "Dependency cycle")]
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
